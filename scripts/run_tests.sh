#!/usr/bin/env bash
# scripts/run_tests.sh — run both C++ and Python tests.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"

echo "─── Python tests ───"
cd "$ROOT"
python -m pytest tests/python -v

if [[ -d "$BUILD_DIR" ]]; then
  echo
  echo "─── C++ tests ───"
  ctest --test-dir "$BUILD_DIR" --output-on-failure
else
  echo
  echo "[skip] C++ tests — run scripts/build.sh first"
fi
