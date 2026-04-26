// lunara/passes/operator_fusion.cpp
// Operator fusion:  producer-consumer element-wise chains, GEMM epilogues,
// and FlashAttention-style attention kernel fusion.

#include "lunara/passes/operator_fusion.h"
#include "lunara/utils/logging.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace lunara {
namespace passes {

// ─────────────────────────────────────────────────────────────────────────────
// Helper predicates
// ─────────────────────────────────────────────────────────────────────────────

static bool isElementWise(linalg::GenericOp op) {
  // All indexing maps are identity -> element-wise
  return llvm::all_of(op.getIndexingMapsArray(), [](AffineMap m) {
    return m.isIdentity();
  });
}

static bool isMatmulLike(Operation *op) {
  return isa<linalg::MatmulOp, linalg::BatchMatmulOp,
             linalg::MatmulTransposeBOp>(op);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: Fuse element-wise consumer into element-wise producer
// ─────────────────────────────────────────────────────────────────────────────

struct FuseElementWiseConsumer : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    if (!isElementWise(consumer)) return failure();

    // Look for a single element-wise producer feeding all inputs.
    for (auto operand : consumer->getOperands()) {
      auto defOp = operand.getDefiningOp<linalg::GenericOp>();
      if (!defOp || !isElementWise(defOp)) continue;
      // Ensure producer has single use.
      if (!defOp->hasOneUse()) continue;

      LUNARA_LOG(debug) << "Fusing element-wise ops: "
                        << defOp->getName().getStringRef().str()
                        << " -> " << consumer->getName().getStringRef().str();

      // Use MLIR's built-in elementwise fusion utility.
      auto fusedOp = linalg::fuseElementwiseOps(rewriter, consumer);
      if (succeeded(fusedOp)) return success();
    }
    return failure();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: Fuse GEMM epilogue  (matmul -> bias-add -> activation)
// ─────────────────────────────────────────────────────────────────────────────

struct FuseGemmEpilogue : OpRewritePattern<linalg::GenericOp> {
  FusionOptions opts;
  FuseGemmEpilogue(MLIRContext *ctx, FusionOptions o)
      : OpRewritePattern(ctx), opts(o) {}

  LogicalResult matchAndRewrite(linalg::GenericOp epilogue,
                                PatternRewriter &rewriter) const override {
    if (!opts.fuse_gemm_epilogue) return failure();
    if (!isElementWise(epilogue)) return failure();

    // Find matmul-like op producing one of our inputs.
    for (auto operand : epilogue->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (!defOp || !isMatmulLike(defOp)) continue;
      if (!defOp->hasOneUse()) continue;

      LUNARA_LOG(debug) << "Fusing GEMM epilogue";
      // todo: rewrite to a linalg.generic that contains both the matmul accumulation and the epilogue computation.
      rewriter.setInsertionPoint(epilogue);
      return failure();
    }
    return failure();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: Attention fusion  Q*K^T -> scale -> softmax -> *V
// ─────────────────────────────────────────────────────────────────────────────

struct FuseAttentionKernel : RewritePattern {
  FuseAttentionKernel(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Detect the QK^T matmul by checking op name / attributes set by the
    // frontend.  In production this checks a `lunara.attention` marker attr.
    if (!op->hasAttr("lunara.qk_matmul")) return failure();

    LUNARA_LOG(info) << "Fusing attention kernel (Q*K^T → softmax → *V)";
    // Full FlashAttention tiling strategy is implemented in the Triton
    // codegen backend (codegen/triton_emitter.py).  This pass marks the
    // matmul chain for the backend to recognise.
    op->setAttr("lunara.fused_attention", rewriter.getUnitAttr());
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass implementations
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct OperatorFusionPass
    : PassWrapper<OperatorFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OperatorFusionPass)

  FusionOptions opts;
  OperatorFusionPass(FusionOptions o) : opts(o) {}

  StringRef getArgument()  const override { return "lunara-fuse-ops"; }
  StringRef getDescription() const override {
    return "Fuse element-wise chains and GEMM epilogues";
  }

  void runOnOperation() override {
    auto mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<FuseElementWiseConsumer>(ctx);
    patterns.add<FuseGemmEpilogue>(ctx, opts);

    int fused = 0;
    mod.walk([&](linalg::GenericOp op) {
      if (isElementWise(op)) ++fused;
    });
    LUNARA_LOG(info) << "  Fusion pass: " << fused
                     << " element-wise ops in module";

    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

struct AttentionFusionPass
    : PassWrapper<AttentionFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttentionFusionPass)

  StringRef getArgument()  const override { return "lunara-fuse-attention"; }
  StringRef getDescription() const override {
    return "Fuse Q*K^T → softmax → *V into a single kernel";
  }

  void runOnOperation() override {
    auto mod = getOperation();
    RewritePatternSet patterns(mod.getContext());
    patterns.add<FuseAttentionKernel>(mod.getContext());
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

}

std::unique_ptr<Pass> createOperatorFusionPass(FusionOptions opts) {
  return std::make_unique<OperatorFusionPass>(opts);
}

std::unique_ptr<Pass> createAttentionFusionPass() {
  return std::make_unique<AttentionFusionPass>();
}

} // namespace passes
} // namespace lunara
