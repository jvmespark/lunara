add_library(lunara-runtime OBJECT
  executor.cpp
)
target_include_directories(lunara-runtime PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(lunara-runtime PUBLIC LLVMSupport)
if(LUNARA_ENABLE_CUDA)
  target_link_libraries(lunara-runtime PUBLIC CUDA::cuda_driver)
  target_compile_definitions(lunara-runtime PUBLIC LUNARA_ENABLE_CUDA)
endif()
