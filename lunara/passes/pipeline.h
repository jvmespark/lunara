#pragma once
// lunara/passes/pipeline.h
// Constructs and runs the full Lunara compilation pipeline:
//   StableHLO -> linalg -> fused+tiled linalg -> Triton IR -> PTX/CUBIN

#include "lunara/passes/operator_fusion.h"
#include "lunara/passes/layout_transform.h"
#include "lunara/passes/tiling.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include <memory>

namespace lunara {
namespace passes {

struct PipelineOptions {
  FusionOptions  fusion;
  LayoutOptions  layout;
  TileSizes      tiles;
  int            vector_width   = 4;
  bool           enable_triton  = true;
  bool           enable_ptx     = false;
  bool           print_ir_after_each = false;
  bool           verify_each    = true;
  int            opt_level      = 2;     ///< 0=none, 1=O1, 2=O2, 3=O3
};

/// Build the full pass pipeline into \p pm.
void buildLunaraPipeline(mlir::PassManager &pm, const PipelineOptions &opts);

/// Convenience: run the entire pipeline on a module.
/// Returns true on success.
bool runPipeline(mlir::ModuleOp mod, const PipelineOptions &opts);

} // namespace passes
} // namespace lunara
