cmake_minimum_required(VERSION 3.20)
project(Lunara VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ────────────────────────────────────────────────────────────────────
option(LUNARA_ENABLE_CUDA   "Build CUDA/PTX backend"        OFF)
option(LUNARA_ENABLE_TRITON "Build Triton codegen backend"  ON)
option(LUNARA_ENABLE_TESTS  "Build tests"                   ON)
option(LUNARA_ENABLE_BENCH  "Build benchmarks"              ON)

# ── MLIR / LLVM ────────────────────────────────────────────────────────────────
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Found MLIR at: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

# ── ONNX ───────────────────────────────────────────────────────────────────────
find_package(ONNX QUIET)
if(NOT ONNX_FOUND)
  message(STATUS "ONNX not found via find_package; using bundled proto headers")
  include_directories(${PROJECT_SOURCE_DIR}/third_party/onnx)
endif()

# ── Python bindings ────────────────────────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  message(STATUS "Found pybind11 ${pybind11_VERSION}")
else()
  message(STATUS "pybind11 not found; Python bindings disabled")
endif()

# ── CUDA ───────────────────────────────────────────────────────────────────────
if(LUNARA_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ── Subdirectories ─────────────────────────────────────────────────────────────
add_subdirectory(lunara)
add_subdirectory(tools)
if(LUNARA_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ── Install rules ──────────────────────────────────────────────────────────────
install(TARGETS lunara-core lunara-opt
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(DIRECTORY lunara/ DESTINATION include/lunara
  FILES_MATCHING PATTERN "*.h" PATTERN "*.td")
