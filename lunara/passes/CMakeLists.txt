add_library(lunara-passes OBJECT
  operator_fusion.cpp
  layout_transform.cpp
  tiling.cpp
  pipeline.cpp
)
target_include_directories(lunara-passes PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(lunara-passes PUBLIC
  lunara-dialects
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRBufferizationDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRPass
)
