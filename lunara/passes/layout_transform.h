#pragma once
// lunara/passes/layout_transform.h
// Inserts linalg.transpose / tensor.pack ops to convert between NCHW↔NHWC,
// row-major<->column-major (for BLAS), and tiled memory layouts
// (e.g., 16x16 block layout for Tensor Core alignment).

#include "mlir/Pass/Pass.h"
#include <memory>

namespace lunara {
namespace passes {

enum class LayoutPreference {
  NHWC,      ///< Activation channel-last (best for most CUDA conv)
  NCHW,      ///< Activation channel-first
  RowMajor,  ///< Standard row-major matmul  (A: M*K, B: K*N)
  ColMajor,  ///< Column-major BLAS style
};

struct LayoutOptions {
  LayoutPreference conv_layout    = LayoutPreference::NHWC;
  LayoutPreference matmul_layout  = LayoutPreference::RowMajor;
  /// Tile innermost dims to this multiple for Tensor Core alignment (0=off).
  int tensor_core_tile = 16;
  bool insert_transposes_for_cublas = true;
};

/// Pass: canonicalise tensor layouts and insert necessary transposes.
std::unique_ptr<mlir::Pass> createLayoutTransformPass(
    LayoutOptions opts = {});

/// Pass: pack tensors into blocked layouts for cache-friendly access.
std::unique_ptr<mlir::Pass> createTensorPackingPass(int block_size = 16);

} // namespace passes
} // namespace lunara
