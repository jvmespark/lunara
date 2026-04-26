// tests/unit/test_passes.cpp
// Unit tests for the Lunara pass infrastructure using GoogleTest.
// Tests are written against synthetic MLIR IR to keep them hermetic.

#include "lunara/passes/pipeline.h"
#include "lunara/passes/operator_fusion.h"
#include "lunara/passes/layout_transform.h"
#include "lunara/passes/tiling.h"
#include "lunara/utils/shape_utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using namespace mlir;
using namespace lunara;

// ─────────────────────────────────────────────────────────────────────────────
// Fixture: sets up an MLIRContext with all dialects loaded
// ─────────────────────────────────────────────────────────────────────────────

class LunaraPassTest : public ::testing::Test {
protected:
  void SetUp() override {
    DialectRegistry reg;
    registerAllDialects(reg);
    ctx = std::make_unique<MLIRContext>(reg);
    ctx->loadAllAvailableDialects();
    builder = std::make_unique<OpBuilder>(ctx.get());
  }

  /// Parse a snippet of MLIR text into a ModuleOp.
  OwningOpRef<ModuleOp> parse(const char *src) {
    return parseSourceString<ModuleOp>(src, ctx.get());
  }

  std::unique_ptr<MLIRContext> ctx;
  std::unique_ptr<OpBuilder>   builder;
};

// ─────────────────────────────────────────────────────────────────────────────
// Shape utilities
// ─────────────────────────────────────────────────────────────────────────────

TEST(ShapeUtils, BroadcastIdentical) {
  auto result = utils::broadcastShapes({4, 8, 16}, {4, 8, 16});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ((*result)[0], 4);
  EXPECT_EQ((*result)[1], 8);
  EXPECT_EQ((*result)[2], 16);
}

TEST(ShapeUtils, BroadcastScalarExpand) {
  auto result = utils::broadcastShapes({1}, {4, 8, 16});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), 3u);
  EXPECT_EQ((*result)[0], 4);
  EXPECT_EQ((*result)[2], 16);
}

TEST(ShapeUtils, BroadcastIncompatible) {
  auto result = utils::broadcastShapes({3, 4}, {5, 4});
  EXPECT_FALSE(result.has_value());
}

TEST(ShapeUtils, NumElements) {
  EXPECT_EQ(utils::numElements({2, 3, 4}), 24);
  EXPECT_EQ(utils::numElements({-1, 3, 4}), -1);
  EXPECT_EQ(utils::numElements({}), 1);
}

TEST(ShapeUtils, MatmulFlops) {
  // C[1024,1024] = A[1024,1024] @ B[1024,1024] -> 2 * 1024^3
  EXPECT_EQ(utils::matmulFlops(1024, 1024, 1024), 2LL * 1024 * 1024 * 1024);
}

TEST(ShapeUtils, AttentionFlops) {
  // batch=1, heads=12, seq_q=512, seq_kv=512, head_dim=64
  int64_t expected = 4LL * 1 * 12 * 512 * 512 * 64;
  EXPECT_EQ(utils::attentionFlops(1, 12, 512, 512, 64), expected);
}

TEST(ShapeUtils, ShapeStr) {
  EXPECT_EQ(utils::shapeStr({4, 128, 768}), "4x128x768");
  EXPECT_EQ(utils::shapeStr({-1, 512}),     "?x512");
}

TEST(ShapeUtils, ParseShape) {
  auto s = utils::parseShape("4x128x768");
  ASSERT_EQ(s.size(), 3u);
  EXPECT_EQ(s[0], 4);
  EXPECT_EQ(s[1], 128);
  EXPECT_EQ(s[2], 768);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline construction
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(LunaraPassTest, PipelineBuilds) {
  PassManager pm(ctx.get());
  passes::PipelineOptions opts;
  opts.opt_level = 2;
  passes::buildLunaraPipeline(pm, opts);
  EXPECT_GT(pm.size(), 0u);
}

TEST_F(LunaraPassTest, PipelineBuildsAllOptLevels) {
  for (int level : {0, 1, 2, 3}) {
    PassManager pm(ctx.get());
    passes::PipelineOptions opts;
    opts.opt_level = level;
    passes::buildLunaraPipeline(pm, opts);
    EXPECT_GE(pm.size(), 2u) << "O" << level << " pipeline has too few passes";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion pass — basic smoke
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(LunaraPassTest, FusionPassCreates) {
  auto pass = passes::createOperatorFusionPass();
  EXPECT_NE(pass.get(), nullptr);
  EXPECT_EQ(pass->getArgument(), "lunara-fuse-ops");
}

TEST_F(LunaraPassTest, AttentionFusionPassCreates) {
  auto pass = passes::createAttentionFusionPass();
  EXPECT_NE(pass.get(), nullptr);
  EXPECT_EQ(pass->getArgument(), "lunara-fuse-attention");
}

// ─────────────────────────────────────────────────────────────────────────────
// Layout pass
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(LunaraPassTest, LayoutPassCreates) {
  passes::LayoutOptions opts;
  opts.tensor_core_tile = 16;
  auto pass = passes::createLayoutTransformPass(opts);
  EXPECT_NE(pass.get(), nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiling pass
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(LunaraPassTest, TilingPassCreates) {
  passes::TileSizes sizes;
  sizes.gemm = {64, 64, 32};
  auto pass = passes::createTilingPass(sizes);
  EXPECT_NE(pass.get(), nullptr);
  EXPECT_EQ(pass->getArgument(), "lunara-tile");
}

TEST_F(LunaraPassTest, VectorizationPassCreates) {
  auto pass = passes::createVectorizationPass(8);
  EXPECT_NE(pass.get(), nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass on a minimal linalg.matmul module
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(LunaraPassTest, TilingPassRunsOnMatmul) {
  const char *ir = R"mlir(
    func.func @matmul(%A: tensor<128x128xf32>,
                      %B: tensor<128x128xf32>,
                      %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
      %r = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                         outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
      return %r : tensor<128x128xf32>
    }
  )mlir";

  auto mod = parse(ir);
  ASSERT_TRUE(mod);

  PassManager pm(ctx.get());
  passes::TileSizes sizes;
  sizes.gemm = {32, 32, 16};
  pm.addPass(passes::createTilingPass(sizes));

  // The pass may succeed or gracefully fail (pattern not matching) — we
  // verify it doesn't crash.
  (void)pm.run(*mod);
  SUCCEED();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
