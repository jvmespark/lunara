add_executable(test_shape_utils test_shape_utils.cpp)
target_link_libraries(test_shape_utils PRIVATE
  lunara-core GTest::gtest GTest::gtest_main)
add_test(NAME test_shape_utils COMMAND test_shape_utils)

add_executable(test_graph_ir test_graph_ir.cpp)
target_link_libraries(test_graph_ir PRIVATE
  lunara-core GTest::gtest GTest::gtest_main)
add_test(NAME test_graph_ir COMMAND test_graph_ir)

add_executable(test_frontend test_frontend.cpp)
target_link_libraries(test_frontend PRIVATE
  lunara-core GTest::gtest GTest::gtest_main
  MLIRIR MLIRSupport)
add_test(NAME test_frontend COMMAND test_frontend)

add_executable(test_passes test_passes.cpp)
target_link_libraries(test_passes PRIVATE
  lunara-core GTest::gtest GTest::gtest_main
  MLIRIR MLIRPass MLIRParser
  MLIRArithDialect MLIRFuncDialect
  MLIRLinalgDialect MLIRTensorDialect)
add_test(NAME test_passes COMMAND test_passes)
