// lunara/utils/shape_utils.cpp
#include "lunara/utils/shape_utils.h"
#include <numeric>
#include <sstream>

namespace lunara {
namespace utils {

std::optional<llvm::SmallVector<int64_t, 4>>
broadcastShapes(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  size_t rank = std::max(a.size(), b.size());
  llvm::SmallVector<int64_t, 4> result(rank);

  for (size_t i = 0; i < rank; ++i) {
    int64_t da = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
    int64_t db = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];

    if (da == db)                { result[i] = da; }
    else if (da == 1)            { result[i] = db; }
    else if (db == 1)            { result[i] = da; }
    else if (da < 0 || db < 0)  { result[i] = -1; } // dynamic
    else                         { return std::nullopt; }          // incompatible
  }
  return result;
}

bool areBroadcastCompatible(llvm::ArrayRef<int64_t> a,
                              llvm::ArrayRef<int64_t> b) {
  return broadcastShapes(a, b).has_value();
}

int64_t numElements(llvm::ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (auto d : shape) {
    if (d < 0) return -1;
    n *= d;
  }
  return n;
}

int64_t conv2dFlops(int64_t N, int64_t C_out, int64_t C_in,
                    int64_t H_out, int64_t W_out,
                    int64_t kH, int64_t kW) {
  // 2 * N * C_out * H_out * W_out * C_in * kH * kW
  return 2 * N * C_out * H_out * W_out * C_in * kH * kW;
}

int64_t attentionFlops(int64_t batch, int64_t heads,
                        int64_t seq_q, int64_t seq_kv, int64_t head_dim) {
  // QK^T: 2 * B * H * seq_q * seq_kv * D
  // *V:   2 * B * H * seq_q * seq_kv * D
  return 4 * batch * heads * seq_q * seq_kv * head_dim;
}

std::string shapeStr(llvm::ArrayRef<int64_t> shape) {
  std::ostringstream os;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) os << "x";
    if (shape[i] < 0) os << "?";
    else os << shape[i];
  }
  return os.str();
}

std::vector<int64_t> parseShape(const std::string &s) {
  std::vector<int64_t> result;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, 'x')) {
    if (tok == "?") result.push_back(-1);
    else            result.push_back(std::stoll(tok));
  }
  return result;
}

} // namespace utils
} // namespace lunara
