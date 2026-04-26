// lunara/passes/layout_transform.cpp
#include "lunara/passes/layout_transform.h"
#include "lunara/utils/logging.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace lunara {
namespace passes {

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: NCHW convolution -> NHWC by inserting transposes around conv op.
// ─────────────────────────────────────────────────────────────────────────────

struct NCHWToNHWC : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp conv,
                                PatternRewriter &rewriter) const override {
    LUNARA_LOG(debug) << "Inserting NCHW->NHWC layout transpose";
    // todo:
    //  1. Insert tensor.transpose (N,C,H,W) → (N,H,W,C) on input + filter.
    //  2. Replace conv with linalg.conv_2d_nhwc_hwcf.
    //  3. Insert reverse transpose on output.
    (void)rewriter;
    return failure();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: Tensor Core alignment — pad innermost dims to multiple of tile_size
// ─────────────────────────────────────────────────────────────────────────────

struct PadForTensorCore : OpRewritePattern<linalg::MatmulOp> {
  int tile_size;
  PadForTensorCore(MLIRContext *ctx, int tile)
      : OpRewritePattern(ctx), tile_size(tile) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    if (tile_size <= 0) return failure();

    auto aType = matmul.getInputs()[0].getType().dyn_cast<RankedTensorType>();
    if (!aType) return failure();

    // Check if already aligned
    auto shape = aType.getShape();
    bool needs_pad = false;
    for (auto d : shape)
      if (d > 0 && d % tile_size != 0) { needs_pad = true; break; }

    if (!needs_pad) return failure();

    LUNARA_LOG(debug) << "Padding matmul dims to multiple of " << tile_size;
    // Full: emit tensor.pad on A, B; slice output to original size.
    return failure();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Passes
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct LayoutTransformPass
    : PassWrapper<LayoutTransformPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutTransformPass)

  LayoutOptions opts;
  LayoutTransformPass(LayoutOptions o) : opts(o) {}

  StringRef getArgument()  const override { return "lunara-layout-transform"; }
  StringRef getDescription() const override {
    return "Canonicalise tensor layouts for GPU efficiency";
  }

  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    RewritePatternSet patterns(ctx);

    if (opts.conv_layout == LayoutPreference::NHWC)
      patterns.add<NCHWToNHWC>(ctx);
    if (opts.tensor_core_tile > 0)
      patterns.add<PadForTensorCore>(ctx, opts.tensor_core_tile);

    // Tensor pack patterns for cache-tiled layouts
    tensor::populateSimplifyTensorPackAndUnPackPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns)))) {
      LUNARA_LOG(warn) << "Layout transform: some patterns failed (non-fatal)";
    }
  }
};

struct TensorPackingPass
    : PassWrapper<TensorPackingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorPackingPass)

  int block_size;
  TensorPackingPass(int bs) : block_size(bs) {}

  StringRef getArgument()  const override { return "lunara-tensor-pack"; }
  StringRef getDescription() const override {
    return "Pack tensors into cache-friendly blocked layouts";
  }

  void runOnOperation() override {
    auto mod = getOperation();
    RewritePatternSet patterns(mod.getContext());
    // Populate linalg data-layout propagation patterns
    linalg::populateDataLayoutPropagationPatterns(
        patterns, [](OpOperand &op) { return true; });
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLayoutTransformPass(LayoutOptions opts) {
  return std::make_unique<LayoutTransformPass>(opts);
}

std::unique_ptr<Pass> createTensorPackingPass(int block_size) {
  return std::make_unique<TensorPackingPass>(block_size);
}

} // namespace passes
} // namespace lunara
