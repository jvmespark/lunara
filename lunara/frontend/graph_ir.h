#pragma once
// lunara/frontend/graph_ir.h
// Lightweight in-memory graph that sits between ONNX import and MLIR lowering.
// Nodes carry op-level metadata used by later fusion / tiling passes.

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace lunara {
namespace frontend {

enum class OpKind {
  // Linear algebra
  MatMul, Gemm, Conv2d,
  // Element-wise
  Add, Mul, Sub, Div, Relu, Gelu, Sigmoid, Tanh,
  // Reduction
  Softmax, LayerNorm, BatchNorm, ReduceSum, ReduceMean,
  // Shape
  Reshape, Transpose, Concat, Slice, Gather, Expand,
  // Pooling
  MaxPool, AveragePool,
  // Attention helpers
  ScaledDotProductAttention,
  // Misc
  Constant, Input, Output, Unknown
};

struct TensorType {
  std::vector<int64_t> shape;  // -1 = dynamic dim
  enum class DType { F32, F16, BF16, F64, I32, I64 } dtype = DType::F32;

  int64_t num_elements() const {
    int64_t n = 1;
    for (auto d : shape) { if (d < 0) return -1; n *= d; }
    return n;
  }

  std::string str() const;
};

struct GraphNode {
  std::string              name;
  OpKind                   kind = OpKind::Unknown;
  std::vector<std::string> inputs;   // names of input tensors
  std::vector<std::string> outputs;  // names of output tensors
  TensorType               output_type;
  // Op-specific attributes stored as strings for simplicity
  std::unordered_map<std::string, std::string> attrs;
};

struct Graph {
  std::string              name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<GraphNode>   nodes;

  /// Topological ordering; returns false if a cycle is detected.
  bool topoSort();

  /// Print human-readable summary to stderr.
  void dump() const;
};

} // namespace frontend
} // namespace lunara
