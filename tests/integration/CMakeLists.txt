add_executable(test_pipeline test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE
  lunara-core GTest::gtest GTest::gtest_main
  MLIRIR MLIRPass MLIRParser)
add_test(NAME test_pipeline COMMAND test_pipeline)
