#!/usr/bin/env bash
# scripts/build.sh — convenience wrapper around CMake build.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# Detect MLIR
if [[ -z "${MLIR_DIR:-}" ]]; then
  for cand in "$HOME/llvm-install" "/usr/local" "/opt/llvm"; do
    if [[ -d "$cand/lib/cmake/mlir" ]]; then
      export MLIR_DIR="$cand/lib/cmake/mlir"
      export LLVM_DIR="$cand/lib/cmake/llvm"
      break
    fi
  done
fi

if [[ -z "${MLIR_DIR:-}" ]]; then
  echo "ERROR: MLIR_DIR not set and no install found." >&2
  echo "Install LLVM/MLIR or set MLIR_DIR=<path>/lib/cmake/mlir" >&2
  exit 1
fi

# Detect CUDA
ENABLE_CUDA="OFF"
if command -v nvcc &>/dev/null; then
  ENABLE_CUDA="ON"
fi

echo "─ Configuring Lunara ─"
echo "  ROOT       = $ROOT"
echo "  BUILD_DIR  = $BUILD_DIR"
echo "  BUILD_TYPE = $BUILD_TYPE"
echo "  MLIR_DIR   = $MLIR_DIR"
echo "  CUDA       = $ENABLE_CUDA"

cmake -B "$BUILD_DIR" -S "$ROOT" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DMLIR_DIR="$MLIR_DIR" \
  -DLLVM_DIR="${LLVM_DIR:-$MLIR_DIR/../llvm}" \
  -DLUNARA_ENABLE_CUDA="$ENABLE_CUDA"

cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo
echo "✓ Built. Binaries in $BUILD_DIR/tools/"
echo "  Run tests with:  ctest --test-dir $BUILD_DIR --output-on-failure"
