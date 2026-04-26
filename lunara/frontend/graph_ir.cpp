// lunara/frontend/graph_ir.cpp
#include "lunara/frontend/graph_ir.h"
#include "lunara/utils/logging.h"
#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace lunara {
namespace frontend {

// ── TensorType ────────────────────────────────────────────────────────────────
std::string TensorType::str() const {
  std::ostringstream os;
  os << "<";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) os << "x";
    os << (shape[i] < 0 ? "?" : std::to_string(shape[i]));
  }
  os << "x";
  switch (dtype) {
  case DType::F32:  os << "f32";  break;
  case DType::F16:  os << "f16";  break;
  case DType::BF16: os << "bf16"; break;
  case DType::F64:  os << "f64";  break;
  case DType::I32:  os << "i32";  break;
  case DType::I64:  os << "i64";  break;
  }
  os << ">";
  return os.str();
}

// ── Graph topo-sort (Kahn's algorithm) ────────────────────────────────────────
bool Graph::topoSort() {
  // Build tensor -> producer index
  std::unordered_map<std::string, size_t> producer; // tensor name -> node idx
  for (size_t i = 0; i < nodes.size(); ++i)
    for (const auto &out : nodes[i].outputs)
      producer[out] = i;

  // Build in-degree and adjacency
  std::vector<int> indegree(nodes.size(), 0);
  std::vector<std::vector<size_t>> adj(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (const auto &inp : nodes[i].inputs) {
      auto it = producer.find(inp);
      if (it != producer.end()) {
        adj[it->second].push_back(i);
        ++indegree[i];
      }
    }
  }

  // BFS
  std::vector<GraphNode> sorted;
  std::vector<size_t> queue;
  for (size_t i = 0; i < nodes.size(); ++i)
    if (indegree[i] == 0) queue.push_back(i);

  while (!queue.empty()) {
    size_t cur = queue.back(); queue.pop_back();
    sorted.push_back(nodes[cur]);
    for (auto nxt : adj[cur]) {
      if (--indegree[nxt] == 0) queue.push_back(nxt);
    }
  }

  if (sorted.size() != nodes.size()) {
    LUNARA_LOG(error) << "Cycle detected in graph " << name;
    return false;
  }
  nodes = std::move(sorted);
  return true;
}

// ── Dump ──────────────────────────────────────────────────────────────────────
static const char *kindStr(OpKind k) {
  switch (k) {
#define CASE(x) case OpKind::x: return #x
  CASE(MatMul); CASE(Gemm); CASE(Conv2d);
  CASE(Add); CASE(Mul); CASE(Sub); CASE(Div);
  CASE(Relu); CASE(Gelu); CASE(Sigmoid); CASE(Tanh);
  CASE(Softmax); CASE(LayerNorm); CASE(BatchNorm);
  CASE(ReduceSum); CASE(ReduceMean);
  CASE(Reshape); CASE(Transpose); CASE(Concat);
  CASE(Slice); CASE(Gather); CASE(Expand);
  CASE(MaxPool); CASE(AveragePool);
  CASE(ScaledDotProductAttention);
  CASE(Constant); CASE(Input); CASE(Output);
#undef CASE
  default: return "Unknown";
  }
}

void Graph::dump() const {
  LUNARA_LOG(info) << "=== Graph: " << name << " ===";
  LUNARA_LOG(info) << "  Inputs: ";
  for (const auto &n : input_names) LUNARA_LOG(info) << "    " << n;
  LUNARA_LOG(info) << "  Outputs: ";
  for (const auto &n : output_names) LUNARA_LOG(info) << "    " << n;
  LUNARA_LOG(info) << "  Nodes (" << nodes.size() << "):";
  for (const auto &nd : nodes)
    LUNARA_LOG(info) << "    [" << kindStr(nd.kind) << "] " << nd.name
                     << " -> " << nd.output_type.str();
}

} // namespace frontend
} // namespace lunara
