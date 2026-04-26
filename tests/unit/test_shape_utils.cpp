// tests/unit/test_shape_utils.cpp
#include "lunara/utils/shape_utils.h"
#include <gtest/gtest.h>

using namespace lunara::utils;

TEST(ShapeUtils, Broadcast_Compatible_Same) {
  std::vector<int64_t> a = {3, 4};
  std::vector<int64_t> b = {3, 4};
  auto r = broadcastShapes(a, b);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->size(), 2u);
  EXPECT_EQ((*r)[0], 3);
  EXPECT_EQ((*r)[1], 4);
}

TEST(ShapeUtils, Broadcast_OneDim) {
  std::vector<int64_t> a = {3, 1};
  std::vector<int64_t> b = {1, 4};
  auto r = broadcastShapes(a, b);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ((*r)[0], 3);
  EXPECT_EQ((*r)[1], 4);
}

TEST(ShapeUtils, Broadcast_DifferentRanks) {
  std::vector<int64_t> a = {5};
  std::vector<int64_t> b = {3, 4, 5};
  auto r = broadcastShapes(a, b);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->size(), 3u);
  EXPECT_EQ((*r)[2], 5);
}

TEST(ShapeUtils, Broadcast_Incompatible) {
  std::vector<int64_t> a = {3, 4};
  std::vector<int64_t> b = {2, 4};
  auto r = broadcastShapes(a, b);
  EXPECT_FALSE(r.has_value());
}

TEST(ShapeUtils, NumElements) {
  std::vector<int64_t> a = {2, 3, 4};
  EXPECT_EQ(numElements(a), 24);
  std::vector<int64_t> b = {2, -1, 4};
  EXPECT_EQ(numElements(b), -1);
}

TEST(ShapeUtils, MatmulFlops) {
  EXPECT_EQ(matmulFlops(128, 128, 128), 2 * 128 * 128 * 128);
}

TEST(ShapeUtils, AttentionFlops) {
  // B=1, H=12, seq=512, D=64
  int64_t expected = 4LL * 1 * 12 * 512 * 512 * 64;
  EXPECT_EQ(attentionFlops(1, 12, 512, 512, 64), expected);
}

TEST(ShapeUtils, ShapeStr) {
  EXPECT_EQ(shapeStr({3, 4, 5}), "3x4x5");
  EXPECT_EQ(shapeStr({-1, 4, 5}), "?x4x5");
  EXPECT_EQ(shapeStr({}), "");
}

TEST(ShapeUtils, ParseShape) {
  auto v = parseShape("3x4x5");
  ASSERT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], 3);
  EXPECT_EQ(v[1], 4);
  EXPECT_EQ(v[2], 5);
  v = parseShape("?x128");
  EXPECT_EQ(v[0], -1);
  EXPECT_EQ(v[1], 128);
}
