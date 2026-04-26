#pragma once
// lunara/passes/tiling.h
// Tile linalg ops for L1/L2 cache hierarchy and thread-block granularity.

#include "mlir/Pass/Pass.h"
#include <array>
#include <memory>
#include <vector>

namespace lunara {
namespace passes {

struct TileSizes {
  // GEMM / MatMul tiles  (M, N, K)
  std::array<int64_t,3> gemm   = {128, 128, 32};
  // Conv tiles            (N, H, W, C_out, C_in, kH, kW)
  std::array<int64_t,7> conv   = {1, 8, 8, 64, 32, 1, 1};
  // Attention head tile   (batch, heads, seq, seq, dim)
  std::array<int64_t,5> attn   = {1, 1, 64, 64, 64};
  // Element-wise tile per thread (vectorization width)
  int64_t              eltwise = 4;
};

/// Pass: apply linalg.tile_using_for with the given tile sizes.
std::unique_ptr<mlir::Pass> createTilingPass(TileSizes sizes = {});

/// Pass: vectorise tiled inner loops (vector<4xf32> / vector<8xf16>).
std::unique_ptr<mlir::Pass> createVectorizationPass(int vector_width = 4);

/// Pass: hoist loop-invariant tensor.extract_slice / tensor.insert_slice
///       out of tiled loops to reduce redundant loads.
std::unique_ptr<mlir::Pass> createLoopHoistingPass();

} // namespace passes
} // namespace lunara
