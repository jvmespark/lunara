// lunara/passes/pipeline.cpp
#include "lunara/passes/pipeline.h"
#include "lunara/dialects/stablehlo_to_linalg.h"
#include "lunara/utils/logging.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace lunara {
namespace passes {

void buildLunaraPipeline(PassManager &pm, const PipelineOptions &opts) {
  // ── Diagnostics ────────────────────────────────────────────────────────────
  if (opts.print_ir_after_each)
    pm.enableIRPrinting();
  if (opts.verify_each)
    pm.enableVerifier(true);

  LUNARA_LOG(info) << "Building Lunara pipeline (O" << opts.opt_level << ")";

  // ── Stage 1: StableHLO -> linalg-on-tensors ─────────────────────────────────
  pm.addPass(dialects::createConvertStableHLOToLinalgPass());
  pm.addPass(dialects::createLinalgCleanupPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // ── Stage 2: Layout transformation ─────────────────────────────────────────
  pm.addPass(createLayoutTransformPass(opts.layout));
  pm.addPass(createTensorPackingPass(opts.layout.tensor_core_tile));
  pm.addPass(createCanonicalizerPass());

  // ── Stage 3: Operator fusion ────────────────────────────────────────────────
  if (opts.opt_level >= 1) {
    pm.addPass(createOperatorFusionPass(opts.fusion));
    pm.addPass(createAttentionFusionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // ── Stage 4: Tiling ─────────────────────────────────────────────────────────
  if (opts.opt_level >= 1) {
    pm.addPass(createTilingPass(opts.tiles));
    pm.addPass(createLoopHoistingPass());
    pm.addPass(createCanonicalizerPass());
  }

  // ── Stage 5: Vectorisation ──────────────────────────────────────────────────
  if (opts.opt_level >= 2)
    pm.addNestedPass<func::FuncOp>(
        createVectorizationPass(opts.vector_width));

  // ── Stage 6: Bufferisation (tensors -> memrefs) ──────────────────────────────
  // Required before GPU/Triton lowering.
  pm.addPass(createLinalgBufferizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // ── Stage 7: Triton / PTX codegen (handled externally via Python API) ───────
  // The PassManager stops here in C++; the Python driver calls
  // lunara.codegen.triton_emitter after runPipeline() succeeds.
  LUNARA_LOG(info) << "Pipeline constructed (" << pm.size() << " passes)";
}

bool runPipeline(ModuleOp mod, const PipelineOptions &opts) {
  PassManager pm(mod.getContext());
  buildLunaraPipeline(pm, opts);

  LUNARA_LOG(info) << "Running Lunara compilation pipeline...";
  if (failed(pm.run(mod))) {
    LUNARA_LOG(error) << "Pipeline failed.";
    return false;
  }
  LUNARA_LOG(info) << "Pipeline complete.";
  return true;
}

} // namespace passes
} // namespace lunara
