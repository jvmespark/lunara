// tests/integration/test_pipeline.cpp
// End-to-end integration: parse a small MLIR module, run the full
// Lunara pipeline, and verify that it completes without errors.

#include "lunara/passes/pipeline.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using namespace mlir;

// A trivial linalg-on-tensors program.
static const char kMatmul[] = R"mlir(
func.func @matmul(%A: tensor<128x256xf32>,
                  %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x512xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x512xf32>)
            -> tensor<128x512xf32>
  %r = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                     outs(%fill : tensor<128x512xf32>)
                     -> tensor<128x512xf32>
  return %r : tensor<128x512xf32>
}
)mlir";

class PipelineTest : public ::testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registerAllDialects(registry);
    ctx_ = std::make_unique<MLIRContext>(registry);
    ctx_->loadAllAvailableDialects();
  }
  std::unique_ptr<MLIRContext> ctx_;
};

TEST_F(PipelineTest, RunsOnSimpleMatmul) {
  auto mod = parseSourceString<ModuleOp>(kMatmul, ctx_.get());
  ASSERT_TRUE(mod);

  lunara::passes::PipelineOptions opts;
  opts.opt_level = 1;
  opts.verify_each = true;
  EXPECT_TRUE(lunara::passes::runPipeline(*mod, opts));
}

TEST_F(PipelineTest, RunsAtO2) {
  auto mod = parseSourceString<ModuleOp>(kMatmul, ctx_.get());
  ASSERT_TRUE(mod);

  lunara::passes::PipelineOptions opts;
  opts.opt_level = 2;
  EXPECT_TRUE(lunara::passes::runPipeline(*mod, opts));
}
