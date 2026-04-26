// lunara/passes/tiling.cpp
#include "lunara/passes/tiling.h"
#include "lunara/utils/logging.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace lunara {
namespace passes {

// ─────────────────────────────────────────────────────────────────────────────
// Tiling utilities
// ─────────────────────────────────────────────────────────────────────────────

static SmallVector<OpFoldResult> makeTiles(MLIRContext *ctx,
                                           ArrayRef<int64_t> sizes) {
  SmallVector<OpFoldResult> tiles;
  for (auto s : sizes)
    tiles.push_back(IntegerAttr::get(IndexType::get(ctx), s));
  return tiles;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: tile linalg.matmul with (M, N, K) tile sizes
// ─────────────────────────────────────────────────────────────────────────────

struct TileMatmul : OpRewritePattern<linalg::MatmulOp> {
  TileSizes sizes;
  TileMatmul(MLIRContext *ctx, TileSizes s)
      : OpRewritePattern(ctx), sizes(s) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    // Skip already-tiled ops (marked by inner-most loop attr)
    if (matmul->hasAttr("lunara.tiled")) return failure();

    LUNARA_LOG(debug) << "Tiling matmul ["
                      << sizes.gemm[0] << "x" << sizes.gemm[1]
                      << "x" << sizes.gemm[2] << "]";

    auto tiles = makeTiles(rewriter.getContext(),
                           {sizes.gemm[0], sizes.gemm[1], sizes.gemm[2]});

    scf::SCFTilingOptions opts;
    opts.setTileSizes(tiles);

    auto tiled = scf::tileUsingSCF(rewriter, cast<TilingInterface>(matmul.getOperation()), opts);
    if (failed(tiled)) return failure();

    // Mark resulting inner op so we don't tile again.
    tiled->tiledOps.front()->setAttr("lunara.tiled", rewriter.getUnitAttr());
    rewriter.replaceOp(matmul, tiled->replacements);
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pattern: tile linalg.generic (element-wise) for vectorisation
// ─────────────────────────────────────────────────────────────────────────────

struct TileElementWise : OpRewritePattern<linalg::GenericOp> {
  int64_t eltwise_tile;
  TileElementWise(MLIRContext *ctx, int64_t t)
      : OpRewritePattern(ctx), eltwise_tile(t) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("lunara.tiled")) return failure();

    // Only tile 1-D element-wise (others handled by separate patterns)
    if (op.getNumLoops() != 1) return failure();

    LUNARA_LOG(debug) << "Tiling element-wise op [" << eltwise_tile << "]";
    auto tiles = makeTiles(rewriter.getContext(), {eltwise_tile});
    scf::SCFTilingOptions opts;
    opts.setTileSizes(tiles);

    auto tiled = scf::tileUsingSCF(rewriter, cast<TilingInterface>(op.getOperation()), opts);
    if (failed(tiled)) return failure();

    tiled->tiledOps.front()->setAttr("lunara.tiled", rewriter.getUnitAttr());
    rewriter.replaceOp(op, tiled->replacements);
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Passes
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct TilingPass
    : PassWrapper<TilingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPass)

  TileSizes sizes;
  TilingPass(TileSizes s) : sizes(s) {}

  StringRef getArgument()  const override { return "lunara-tile"; }
  StringRef getDescription() const override {
    return "Tile linalg ops for cache hierarchy and thread-block granularity";
  }

  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<scf::SCFDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TileMatmul>(ctx, sizes);
    patterns.add<TileElementWise>(ctx, sizes.eltwise);

    LUNARA_LOG(info) << "  Tiling: GEMM=" << sizes.gemm[0] << "x"
                     << sizes.gemm[1] << "x" << sizes.gemm[2];
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

struct VectorizationPass
    : PassWrapper<VectorizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizationPass)

  int vector_width;
  VectorizationPass(int w) : vector_width(w) {}

  StringRef getArgument()  const override { return "lunara-vectorize"; }
  StringRef getDescription() const override {
    return "Vectorise tiled inner loops";
  }

  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(func.getContext());
    linalg::populateLinalgToVectorPatterns(patterns);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

struct LoopHoistingPass
    : PassWrapper<LoopHoistingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopHoistingPass)

  StringRef getArgument()  const override { return "lunara-loop-hoist"; }
  StringRef getDescription() const override {
    return "Hoist loop-invariant slice ops out of tiled loops";
  }

  void runOnOperation() override {
    auto func = getOperation();
    // LICM is already in mlir::createLoopInvariantCodeMotionPass();
    // here we add linalg-specific hoisting of tensor.extract_slice.
    RewritePatternSet patterns(func.getContext());
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

}

std::unique_ptr<Pass> createTilingPass(TileSizes sizes) {
  return std::make_unique<TilingPass>(sizes);
}
std::unique_ptr<Pass> createVectorizationPass(int vector_width) {
  return std::make_unique<VectorizationPass>(vector_width);
}
std::unique_ptr<Pass> createLoopHoistingPass() {
  return std::make_unique<LoopHoistingPass>();
}

} // namespace passes
} // namespace lunara
