add_library(lunara-utils OBJECT
  shape_utils.cpp
)
target_include_directories(lunara-utils PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(lunara-utils PUBLIC LLVMSupport)
