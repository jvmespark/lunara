// tests/unit/test_graph_ir.cpp
#include "lunara/frontend/graph_ir.h"
#include <gtest/gtest.h>

using namespace lunara::frontend;

static GraphNode makeNode(const std::string &name, OpKind k,
                            std::vector<std::string> ins,
                            std::vector<std::string> outs) {
  GraphNode n;
  n.name = name;
  n.kind = k;
  n.inputs = std::move(ins);
  n.outputs = std::move(outs);
  n.output_type.shape = {1, 4};
  return n;
}

TEST(GraphIR, TopoSort_Linear) {
  Graph g;
  g.name = "linear";
  g.input_names = {"x"};
  g.output_names = {"y"};
  // Out-of-order: B, A
  g.nodes.push_back(makeNode("B", OpKind::Add, {"a"}, {"y"}));
  g.nodes.push_back(makeNode("A", OpKind::Relu, {"x"}, {"a"}));

  ASSERT_TRUE(g.topoSort());
  ASSERT_EQ(g.nodes.size(), 2u);
  EXPECT_EQ(g.nodes[0].name, "A");
  EXPECT_EQ(g.nodes[1].name, "B");
}

TEST(GraphIR, TopoSort_Branched) {
  Graph g;
  // x -> A -> b
  // x -> C -> d
  // b, d -> E -> y
  g.nodes.push_back(makeNode("E", OpKind::Add, {"b","d"}, {"y"}));
  g.nodes.push_back(makeNode("A", OpKind::Relu, {"x"}, {"b"}));
  g.nodes.push_back(makeNode("C", OpKind::Sigmoid, {"x"}, {"d"}));
  ASSERT_TRUE(g.topoSort());
  EXPECT_EQ(g.nodes.back().name, "E");
}

TEST(GraphIR, TensorTypeStr) {
  TensorType t;
  t.shape = {1, 768};
  t.dtype = TensorType::DType::F16;
  EXPECT_EQ(t.str(), "<1x768xf16>");
  t.shape = {-1, 512};
  EXPECT_EQ(t.str(), "<?x512xf16>");
}

TEST(GraphIR, NumElements) {
  TensorType t;
  t.shape = {2, 3, 4};
  EXPECT_EQ(t.num_elements(), 24);
  t.shape = {-1, 3};
  EXPECT_EQ(t.num_elements(), -1);
}
