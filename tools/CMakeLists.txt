add_executable(lunara-opt lunara_opt.cpp)
target_link_libraries(lunara-opt PRIVATE
  lunara-core
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRIR
  MLIRTransforms
  MLIRInitAllDialects
  MLIRInitAllPasses
  LLVMSupport
)
install(TARGETS lunara-opt RUNTIME DESTINATION bin)
