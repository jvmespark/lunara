add_library(lunara-frontend OBJECT
  onnx_importer.cpp
  graph_ir.cpp
)
target_include_directories(lunara-frontend PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(lunara-frontend PUBLIC MLIRIR MLIRSupport)
