# Lunara

> **ML kernel compiler:** ONNX → StableHLO → MLIR linalg → Triton → PTX

Lunara ingests ONNX model graphs, lowers them through StableHLO and MLIR
linalg dialects, applies operator fusion / layout transformation / tiling
passes, and emits optimised Triton kernels and (optionally) PTX assembly.
Generated kernels are profiled with Nsight Compute, with SM utilisation,
memory throughput, and roofline position reported back to the developer.

```
                ┌─────────────┐
   ONNX  ──▶    │   Frontend   │  graph_ir + onnx_importer
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
   StableHLO  ◀──   Dialects  │  stablehlo_to_linalg
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
   linalg     ◀──    Passes   │  fusion · layout · tiling · vectorise
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
   Triton/PTX ◀──   Codegen   │  triton_emitter · autotuner · ptx_backend
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
   Metrics    ◀──  Profiler   │  nsight · roofline · SM/BW analysis
                └─────────────┘
```

## Quick start

```bash
# Build C++ side
cmake -B build -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
                -DLUNARA_ENABLE_CUDA=ON
cmake --build build -j

# Install Python side
pip install -e .

# Emit a transformer block, benchmark it, and produce a roofline
python examples/01_transformer_block.py --hidden 768 --heads 12 --seq 512
```

### From an ONNX model

```python
from lunara import Compiler

compiler  = Compiler(opt_level=2, dtype="float16", autotune=True)
compiled  = compiler.compile("bert-base.onnx", output_dir="bert_kernels/")

# Inspect the generated kernels
for spec in compiled.kernel_specs:
    print(spec.name, spec.kind)

# Run a quick benchmark
results = compiled.benchmark(batch_size=8, seq_len=512)
for name, ms in results.items():
    print(f"  {name:<25s}  {ms:>7.3f} ms")
```

### From the CLI

```bash
# Lower an ONNX model to optimised MLIR
lunara-opt model.onnx --emit=mlir --opt=2 --out=lowered/

# Emit PTX directly (requires LUNARA_ENABLE_CUDA=ON)
lunara-opt model.onnx --emit=ptx --arch=sm_80 --out=model.ptx

# Print the pass pipeline and exit
lunara-opt model.onnx --print-pipeline --opt=3
```

### Profiling

```python
from lunara.profiler import NsightProfiler
import torch

profiler = NsightProfiler()                       # auto-falls-back to cuda events
A = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
B = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")

result = profiler.profile_kernel(
    lambda: torch.mm(A, B),
    flops=2 * 4096**3,
    bytes_accessed=3 * 4096**2 * 2,
    kernel_name="matmul_4kx4kx4k",
)
profiler.print_summary(result)        # SM util, BW, TFLOPS, roofline bound
profiler.plot_roofline(result, "roofline.png")
```

### Autotuning

```python
from lunara import Autotuner

tuner = Autotuner()
result = tuner.tune_gemm(M=4096, N=4096, K=4096, dtype=torch.float16)
print(f"Best: {result.config}, {result.tflops:.2f} TFLOPS")
```

Tuning results are cached in `.lunara_tuning_cache.json`, so subsequent
runs are instant.

---

## Project layout

```
lunara/
├── lunara/
│   ├── frontend/          ONNX → MLIR import + Graph IR
│   ├── dialects/          StableHLO → linalg lowering
│   ├── passes/            fusion · layout · tiling · pipeline
│   ├── codegen/           Triton emitter · autotuner · PTX backend
│   ├── runtime/           CUDA driver-API executor
│   ├── profiler/          Nsight + roofline analysis
│   └── utils/             shape utils, FLOPs counters, logging
├── tools/
│   └── lunara-opt         CLI driver
├── examples/              Worked examples
├── tests/
│   ├── unit/              C++ GoogleTest
│   ├── integration/       End-to-end pipeline tests
│   └── python/            pytest tests
└── docs/
    ├── ARCHITECTURE.md    Design + dataflow
    └── BUILD.md           How to build Lunara 
```

---

## Supported operators

| Category | Ops |
|---|---|
| Linear | MatMul · Gemm · Conv2d (NHWC + NCHW via layout transform) |
| Activations | ReLU · GELU · Sigmoid · Tanh |
| Element-wise | Add · Mul · Sub · Div |
| Normalisation | LayerNorm · BatchNorm |
| Reductions | Softmax · ReduceSum · ReduceMean |
| Shape | Reshape · Transpose · Concat · Slice · Gather · Expand |
| Pooling | MaxPool · AveragePool |
| Composite | ScaledDotProductAttention (Flash-Attention-2 style) |

---

## Performance targets

On an NVIDIA A100 (80 GB), Lunara aims for the following efficiency on
fp16 with Tensor Cores:

| Kernel | Shape | Target | Roofline bound |
|---|---|---|---|
| GEMM | 4096³ | ≥ 70 % of peak | compute |
| Flash Attention | B=2, H=12, seq=2048, D=64 | ≥ 60 % of peak | compute |
| LayerNorm | (B·S, 768) | ≥ 80 % of HBM BW | memory |
| GELU | (B·S, 3072) | ≥ 90 % of HBM BW | memory |

Use `examples/04_roofline_sweep.py` to see where your generated kernels
land for a given GPU.

---

## Requirements

* **C++:** CMake ≥ 3.20, Clang/GCC with C++17, MLIR & LLVM ≥ 17
* **Python:** 3.9+ with PyTorch ≥ 2.1, Triton ≥ 2.1
* **GPU (optional):** NVIDIA SM 7.0+ for FP16, SM 8.0+ for BF16, CUDA 11.8+
* **Profiling (optional):** Nsight Compute (`ncu`) on `$PATH`