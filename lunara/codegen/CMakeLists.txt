add_library(lunara-codegen OBJECT
  ptx_backend.cpp
)
target_include_directories(lunara-codegen PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(lunara-codegen PUBLIC
  MLIRGPUDialect
  MLIRNVVMDialect
  MLIRLLVMDialect
  MLIRToLLVMIRTranslationRegistration
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo
  LLVMTarget
  LLVMSupport
)
if(LUNARA_ENABLE_CUDA)
  target_link_libraries(lunara-codegen PUBLIC CUDA::cuda_driver)
endif()

# Python codegen files are installed alongside the C++ library.
install(FILES
  triton_emitter.py
  autotuner.py
  DESTINATION ${CMAKE_INSTALL_PREFIX}/python/lunara/codegen)
