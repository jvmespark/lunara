add_library(lunara-dialects OBJECT
  stablehlo_to_linalg.cpp
)
target_include_directories(lunara-dialects PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(lunara-dialects PUBLIC
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)
