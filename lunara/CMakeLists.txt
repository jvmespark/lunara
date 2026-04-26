add_subdirectory(frontend)
add_subdirectory(dialects)
add_subdirectory(passes)
add_subdirectory(codegen)
add_subdirectory(runtime)
add_subdirectory(profiler)
add_subdirectory(utils)

# ── Aggregate static library ───────────────────────────────────────────────────
add_library(lunara-core STATIC
  $<TARGET_OBJECTS:lunara-frontend>
  $<TARGET_OBJECTS:lunara-dialects>
  $<TARGET_OBJECTS:lunara-passes>
  $<TARGET_OBJECTS:lunara-codegen>
  $<TARGET_OBJECTS:lunara-utils>
)

target_link_libraries(lunara-core PUBLIC
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRBufferizationDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRParser
  MLIRIR
  MLIRSupport
  LLVMSupport
  LLVMCore
)

if(LUNARA_ENABLE_CUDA)
  target_link_libraries(lunara-core PUBLIC CUDA::cudart CUDA::cuda_driver)
endif()
