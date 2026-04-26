# Lunara — Architecture

This document explains how Lunara is organised and how data flows from
an ONNX model file all the way to a launched GPU kernel.

## High-level dataflow

```
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  ONNX file   │───▶│  Graph IR    │───▶│   StableHLO  │
  │  (.onnx)     │    │ (in-memory)  │    │  MLIR module │
  └──────────────┘    └──────────────┘    └──────┬───────┘
                                                  │ stablehlo_to_linalg
                                                  ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Triton      │◀───│  Tiled +     │◀───│   linalg     │
  │  Python src  │    │  Fused IR    │    │ on tensors   │
  └──────┬───────┘    └──────────────┘    └──────────────┘
         │             ▲           ▲
         │             │ tiling    │ fusion + layout
         │             └───────────┘
         ▼
  ┌──────────────┐    ┌──────────────┐
  │  PTX / CUBIN │───▶│   Runtime    │
  │              │    │  (cuLaunch)  │
  └──────────────┘    └──────────────┘
```

## Stage descriptions

### 1. Frontend (`lunara/frontend/`)

The ONNX importer (`onnx_importer.cpp`) reads the protobuf-encoded model
and walks each `NodeProto`, producing two things:

* An MLIR `ModuleOp` populated with StableHLO ops (when the StableHLO
  Python package is linked in).
* A lightweight `Graph` IR (`graph_ir.h`) that carries op-level metadata
  (shapes, dtypes, attribute strings) which downstream passes use to
  decide whether to fuse, tile, or rewrite layouts.

The split exists because some decisions (e.g. "is this op fusible?") are
much easier to answer on the small in-memory graph than on full MLIR.

### 2. Dialect lowering (`lunara/dialects/`)

`stablehlo_to_linalg.cpp` converts StableHLO operations into:

* `linalg.matmul`, `linalg.batch_matmul`, `linalg.conv_*` for structured
  ops.
* `linalg.generic` with explicit indexing maps for everything else
  (element-wise, reductions, broadcasts).

This is the most important transformation in the entire compiler:
**linalg-on-tensors exposes loop structure** in a way that StableHLO
hides, which is exactly what the tiling and fusion passes need.

### 3. Pass pipeline (`lunara/passes/`)

`pipeline.cpp::buildLunaraPipeline` constructs the following pass
sequence (at `O2`):

1. **Lowering**          — StableHLO → linalg → cleanup (canonicalise + CSE)
2. **Layout transform**  — NCHW↔NHWC swaps, Tensor Core padding,
                            tensor packing for blocked layouts
3. **Operator fusion**   — element-wise chains, GEMM epilogues,
                            Flash-Attention kernel marking
4. **Tiling**            — `linalg.tile_using_for` with target tile sizes,
                            loop hoisting
5. **Vectorisation**     — `linalg → vector` transforms with target width
6. **Bufferisation**     — tensors → memrefs (required before GPU lowering)

Each stage is independent and can be skipped via the `opt_level` knob.

### 4. Codegen (`lunara/codegen/`)

Two backends share the post-pass IR:

**Triton (default)** — `triton_emitter.py` walks the kernel specs that
the passes have annotated and emits readable Triton Python source. Each
kernel kind (GEMM, attention, softmax, layernorm, element-wise) has a
template with placeholders for tile sizes, dtypes, and epilogue ops.

**PTX (optional)** — `ptx_backend.cpp` lowers the MLIR GPU dialect to
LLVM IR, then drives the LLVM NVPTX backend to emit PTX, optionally
piping through `ptxas` to produce a cubin.

### 5. Autotuning (`lunara/codegen/autotuner.py`)

For each kernel, the autotuner enumerates a search space of
`(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)` tuples, prunes
configs that exceed the shared-memory budget, then benchmarks each one
on the target GPU. Results are persisted in
`.lunara_tuning_cache.json` so subsequent runs are instant.

### 6. Profiling (`lunara/profiler/`)

`nsight_profiler.py` wraps `ncu` to capture SM utilisation, HBM
bandwidth, and warp efficiency. When `ncu` is unavailable, it falls
back to a `CUDAEventProfiler` that measures latency only. Either way,
results are formatted into a `KernelMetrics` struct and used to compute
roofline position (memory-bound vs compute-bound, % of peak).

### 7. Runtime (`lunara/runtime/`)

`executor.cpp` provides a thin C++ wrapper around the CUDA driver API:
load PTX/cubin → look up `CUfunction` by name → `cuLaunchKernel`.
This exists so generated kernels can be embedded in non-Python hosts
(e.g. C++ inference servers).

## Module dependency graph

```
                ┌──────────┐
                │   utils  │  (logging, shape_utils)
                └────┬─────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
  ┌─────────┐  ┌──────────┐  ┌────────┐
  │frontend │  │dialects  │  │codegen │
  └────┬────┘  └────┬─────┘  └────────┘
       │            │
       └────┬───────┘
            ▼
       ┌─────────┐
       │  passes │  (depends on dialects)
       └────┬────┘
            ▼
       ┌─────────┐
       │  tools  │  (lunara-opt CLI)
       └─────────┘
            ▼
       ┌─────────┐
       │ runtime │  (depends only on llvm/cuda)
       └─────────┘
```

## Why this split?

The C++ side handles graph-level transformations that benefit from
MLIR's strong type system and pattern-matching infrastructure. The
Python side handles kernel-level codegen because Triton itself is a
Python DSL — generating Python source is the only sane way to interact
with it.

The seam between the two is the optimised MLIR dump (`output.mlir`),
which carries `lunara.kernel_kind` attributes that the Python emitter
reads. This means you can debug and hand-edit either side independently.

## Key invariants

* Every `linalg.matmul` post-tiling has either a `lunara.tiled` attribute
  or has been replaced by an `scf.for` loop nest.
* Fused attention kernels carry `lunara.fused_attention` so the codegen
  knows to emit the FlashAttention template instead of three separate
  ones.
* Tile sizes set in `TileSizes` are upper bounds for what the autotuner
  may explore — never lower bounds.
