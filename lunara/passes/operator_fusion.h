#pragma once
// lunara/passes/operator_fusion.h
// Identifies and fuses element-wise producer-consumer chains and
// matmul+bias+activation patterns into single linalg.generic regions.

#include "mlir/Pass/Pass.h"
#include <memory>

namespace lunara {
namespace passes {

struct FusionOptions {
  /// Maximum number of ops in a single fused region.
  int max_fusion_depth = 8;
  /// Allow fusing reductions (e.g. softmax denominator) into attention kernel.
  bool fuse_reductions = true;
  /// Fuse epilogue (bias + activation) into GEMM.
  bool fuse_gemm_epilogue = true;
  /// Fuse LayerNorm into preceding matmul when possible.
  bool fuse_layernorm = true;
};

/// Pass: fuse element-wise and epilogue ops into producer generics.
std::unique_ptr<mlir::Pass> createOperatorFusionPass(
    FusionOptions opts = {});

/// Pass: fuse scaled-dot-product attention (Q*K^T, softmax, *V) into
/// a single FlashAttention-style kernel region.
std::unique_ptr<mlir::Pass> createAttentionFusionPass();

} // namespace passes
} // namespace lunara
