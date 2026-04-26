#pragma once
// lunara/utils/shape_utils.h
// Utilities for manipulating, broadcasting, and validating tensor shapes.

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>
#include <vector>

namespace lunara {
namespace utils {

/// Compute the numpy-style broadcast shape of two ranked tensor shapes.
/// Returns std::nullopt if shapes are incompatible.
std::optional<llvm::SmallVector<int64_t, 4>>
broadcastShapes(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b);

/// Return true if shape \p a is broadcast-compatible with shape \p b.
bool areBroadcastCompatible(llvm::ArrayRef<int64_t> a,
                             llvm::ArrayRef<int64_t> b);

/// Compute number of elements; returns -1 if any dim is dynamic.
int64_t numElements(llvm::ArrayRef<int64_t> shape);

/// Return the FLOPs for a matmul  C[M,N] = A[M,K] * B[K,N].
inline int64_t matmulFlops(int64_t M, int64_t N, int64_t K) {
  return 2 * M * N * K;
}

/// Return the FLOPs for a 2-D convolution.
int64_t conv2dFlops(int64_t N, int64_t C_out, int64_t C_in,
                    int64_t H_out, int64_t W_out,
                    int64_t kH, int64_t kW);

/// Return the FLOPs for multi-head attention.
int64_t attentionFlops(int64_t batch, int64_t heads,
                       int64_t seq_q, int64_t seq_kv, int64_t head_dim);

/// Pretty-print a shape (e.g. "4x128x768").
std::string shapeStr(llvm::ArrayRef<int64_t> shape);

/// Parse a shape string "4x128x768" -> {4, 128, 768}.
std::vector<int64_t> parseShape(const std::string &s);

} // namespace utils
} // namespace lunara
