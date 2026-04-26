# Building Lunara

## Quick reference

```bash
# 1. Get LLVM/MLIR
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && git checkout llvmorg-17.0.6
cmake -B build -G Ninja llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install
cmake --build build --target install

# 2. Build Lunara
cd /path/to/lunara
cmake -B build \
  -DMLIR_DIR=$HOME/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-install/lib/cmake/llvm \
  -DLUNARA_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 3. Install Python side
pip install -e .

# 4. Run tests
ctest --test-dir build --output-on-failure
pytest tests/python -v
```

## CMake options

| Option | Default | Effect |
|---|---|---|
| `LUNARA_ENABLE_CUDA` | OFF | Build PTX backend + runtime executor |
| `LUNARA_ENABLE_TRITON` | ON | Install Triton emitter |
| `LUNARA_ENABLE_TESTS` | ON | Build C++ test executables |
| `LUNARA_ENABLE_BENCH` | ON | Install benchmark harness |
| `CMAKE_BUILD_TYPE` | Release | Use `Debug` for assertions + symbols |

## Common build issues

**`Could NOT find MLIR`**
You need to point CMake at your LLVM install:
```bash
-DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir
-DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm
```

**`fatal error: 'onnx/onnx_pb.h' file not found`**
Either install libonnx (`apt install libonnx-dev` or `pip install onnx
&& cmake --find-package ONNX`) or ignore — Lunara falls back to a
stub frontend that emits an empty module, useful for pipeline
development without real ONNX parsing.

**`undefined reference to cuModuleLoadData`**
Pass `-DLUNARA_ENABLE_CUDA=ON` and ensure `CUDAToolkit_ROOT` points at
your CUDA install.

**Python `ImportError: triton`**
Triton is required for codegen but not for the C++ side. Install with
`pip install triton>=2.1.0`. On platforms without GPU support, use
`pip install -e . --no-deps` and run only the structural tests.

## Cross-compilation

PTX generation is host-independent — you can build on a CPU-only host
and emit PTX for any NVPTX target by setting `--arch=sm_XX` on the
`lunara-opt` command line. The generated PTX runs on any host that
loads it via the CUDA driver.
