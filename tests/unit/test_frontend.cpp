// tests/unit/test_frontend.cpp
// Tests for the ONNX frontend and graph IR.

#include "lunara/frontend/graph_ir.h"
#include "lunara/frontend/onnx_importer.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"

#include <gtest/gtest.h>

using namespace lunara::frontend;

// ─────────────────────────────────────────────────────────────────────────────
// TensorType
// ─────────────────────────────────────────────────────────────────────────────

TEST(TensorType, NumElementsStatic) {
  TensorType t;
  t.shape = {2, 3, 4};
  EXPECT_EQ(t.num_elements(), 24);
}

TEST(TensorType, NumElementsDynamic) {
  TensorType t;
  t.shape = {-1, 3, 4};
  EXPECT_EQ(t.num_elements(), -1);
}

TEST(TensorType, StrF32) {
  TensorType t;
  t.shape = {4, 128, 768};
  t.dtype = TensorType::DType::F32;
  EXPECT_EQ(t.str(), "<4x128x768xf32>");
}

TEST(TensorType, StrDynamic) {
  TensorType t;
  t.shape = {-1, 512};
  t.dtype = TensorType::DType::F16;
  EXPECT_EQ(t.str(), "<?x512xf16>");
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph
// ─────────────────────────────────────────────────────────────────────────────

TEST(Graph, TopoSortLinearChain) {
  Graph g;
  g.name = "test";

  // A -> B -> C (must remain in this order)
  GraphNode a, b, c;
  a.name = "A"; a.outputs = {"t0"};
  b.name = "B"; b.inputs = {"t0"}; b.outputs = {"t1"};
  c.name = "C"; c.inputs = {"t1"}; c.outputs = {"t2"};

  // Insert in reverse order to test sorting
  g.nodes = {c, b, a};
  EXPECT_TRUE(g.topoSort());
  EXPECT_EQ(g.nodes[0].name, "A");
  EXPECT_EQ(g.nodes[1].name, "B");
  EXPECT_EQ(g.nodes[2].name, "C");
}

TEST(Graph, TopoSortDAG) {
  Graph g;

  // A -> C, B -> C (diamond)
  GraphNode a, b, c;
  a.name = "A"; a.outputs = {"t0"};
  b.name = "B"; b.outputs = {"t1"};
  c.name = "C"; c.inputs = {"t0", "t1"}; c.outputs = {"t2"};

  g.nodes = {c, b, a};
  EXPECT_TRUE(g.topoSort());
  // C must come last
  EXPECT_EQ(g.nodes.back().name, "C");
}

TEST(Graph, EmptyGraphSorts) {
  Graph g;
  g.name = "empty";
  EXPECT_TRUE(g.topoSort());
  EXPECT_TRUE(g.nodes.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX importer (stub / no-libonnx path)
// ─────────────────────────────────────────────────────────────────────────────

TEST(OnnxImporter, ImportNonExistentFile) {
  mlir::DialectRegistry reg;
  mlir::registerAllDialects(reg);
  mlir::MLIRContext ctx(reg);

  auto result = importONNXModel(ctx, "/nonexistent/model.onnx");
  // Should return an error, not crash.
  EXPECT_TRUE(static_cast<bool>(result.takeError()));
}

TEST(OnnxImporter, ImportEmptyBytes) {
  mlir::DialectRegistry reg;
  mlir::registerAllDialects(reg);
  mlir::MLIRContext ctx(reg);

  // Empty bytes -> either stub module (no libonnx) or parse error
  std::vector<char> empty;
  auto result = importONNXBytes(ctx, empty);
  // Either succeeds (stub) or fails gracefully — must not crash.
  if (result) {
    EXPECT_NE(result->get(), nullptr);
  } else {
    (void)result.takeError();
    SUCCEED();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
