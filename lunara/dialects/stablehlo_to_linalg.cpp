// lunara/dialects/stablehlo_to_linalg.cpp
// Pattern implementations for StableHLO -> linalg lowering.

#include "lunara/dialects/stablehlo_to_linalg.h"
#include "lunara/utils/logging.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace lunara {
namespace dialects {

namespace {

// ── Utility: make an indexing maps pair for matmul (m,k) x (k,n) -> (m,n) ──
static SmallVector<AffineMap> matmulIndexingMaps(MLIRContext *ctx) {
  // (d0, d1, d2) -> ...
  auto map_A = AffineMap::getMultiDimIdentityMap(3, ctx).getSliceMap(0, 2); // d0,d2
  auto map_B = AffineMap::get(3, 0, {getAffineDimExpr(2, ctx),
                                      getAffineDimExpr(1, ctx)}, ctx);
  auto map_C = AffineMap::getMultiDimIdentityMap(3, ctx).getSliceMap(0, 2); // d0,d1
  return {map_A, map_B, map_C};
}

// ── Pattern: lower arith.constant -> linalg-compatible constant tensors ────────
struct ConstantFoldPattern : OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    // todo: match stablehlo::ConstantOp and emit tensor ops.
    return failure();
  }
};

}

// ── Public API ────────────────────────────────────────────────────────────────

void populateStableHLOToLinalgPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  // In a real integration, register stablehlo-to-linalg patterns from
  // the stablehlo project:
  //   stablehlo::populateStablehloToLinalgConversionPatterns(
  //       patterns.getContext(), &typeConverter, &patterns);
  //
  // Here we register our extra patterns on top:
  patterns.add<ConstantFoldPattern>(patterns.getContext());
  LUNARA_LOG(debug) << "Populated StableHLO→linalg patterns";
}

// ── Pass implementation ───────────────────────────────────────────────────────

namespace {

struct ConvertStableHLOToLinalgPass
    : PassWrapper<ConvertStableHLOToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertStableHLOToLinalgPass)

  StringRef getArgument()  const override { return "lunara-lower-stablehlo"; }
  StringRef getDescription() const override {
    return "Lower StableHLO dialect to linalg-on-tensors";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect,
                    arith::ArithDialect,
                    tensor::TensorDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    auto mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    TypeConverter typeConverter;
    // Identity conversion – real implementation maps HLO types to memref/tensor
    typeConverter.addConversion([](Type t) { return t; });

    RewritePatternSet patterns(ctx);
    populateStableHLOToLinalgPatterns(patterns, typeConverter);

    ConversionTarget target(*ctx);
    target.addLegalDialect<linalg::LinalgDialect,
                           arith::ArithDialect,
                           tensor::TensorDialect,
                           func::FuncDialect>();
    // target.addIllegalDialect<stablehlo::StablehloDialect>(); // when linked

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      LUNARA_LOG(error) << "StableHLO→linalg conversion failed";
      signalPassFailure();
    }
  }
};

struct LinalgCleanupPass
    : PassWrapper<LinalgCleanupPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgCleanupPass)

  StringRef getArgument()  const override { return "lunara-linalg-cleanup"; }
  StringRef getDescription() const override {
    return "Canonicalise and DCE after linalg lowering";
  }

  void runOnOperation() override {
    auto mod = getOperation();
    RewritePatternSet patterns(mod.getContext());
    // Collect all canonicalization patterns
    for (auto *dialect : mod.getContext()->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

}

std::unique_ptr<Pass> createConvertStableHLOToLinalgPass() {
  return std::make_unique<ConvertStableHLOToLinalgPass>();
}

std::unique_ptr<Pass> createLinalgCleanupPass() {
  return std::make_unique<LinalgCleanupPass>();
}

} // namespace dialects
} // namespace lunara
