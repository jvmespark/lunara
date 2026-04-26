#!/usr/bin/env python3
"""scripts/bench.py — sweep kernel shapes and emit a CSV report.

Usage:
    python scripts/bench.py --output bench.csv --shapes small,medium,large
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lunara.codegen.triton_emitter import (
    KernelKind, KernelSpec, TileConfig, emit_gemm,
)


SHAPE_PRESETS = {
    "small":  [(512, 512, 512), (1024, 1024, 1024)],
    "medium": [(2048, 2048, 2048), (4096, 4096, 4096)],
    "large":  [(8192, 8192, 8192), (4096, 4096, 16384)],
    "all":    [(512, 512, 512), (1024, 1024, 1024),
                (2048, 2048, 2048), (4096, 4096, 4096),
                (8192, 8192, 8192)],
}


def bench_gemm(M: int, N: int, K: int, dtype=torch.float16) -> dict:
    """Bench Lunara GEMM vs torch.mm, return latencies and TFLOPS."""
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")
    C = torch.empty(M, N, dtype=dtype, device="cuda")
    flops = 2 * M * N * K

    # Lunara
    cfg = TileConfig(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                      num_stages=4, num_warps=4)
    spec = KernelSpec(KernelKind.GEMM, "_b", tile=cfg, dtype="float16")
    src = emit_gemm(spec)
    ns: dict = {}
    exec(compile(src, "<bench>", "exec"), ns)
    lunara_launch = ns["_b_launch"]

    def measure(fn):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times)//2]

    try:
        lunara_ms = measure(lambda: lunara_launch(A, B, C))
    except Exception as e:
        lunara_ms = float("nan")
        print(f"  [warn] lunara: {e}")

    torch_ms = measure(lambda: torch.mm(A, B))

    return {
        "M": M, "N": N, "K": K,
        "lunara_ms": lunara_ms,
        "torch_ms":  torch_ms,
        "lunara_tflops": flops / (lunara_ms * 1e-3) / 1e12 if lunara_ms == lunara_ms else 0,
        "torch_tflops":  flops / (torch_ms  * 1e-3) / 1e12,
        "speedup": (torch_ms / lunara_ms) if lunara_ms == lunara_ms else 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="medium",
                   choices=list(SHAPE_PRESETS.keys()))
    p.add_argument("--output", default="bench_results.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    gpu = torch.cuda.get_device_name(0)
    print(f"Benchmarking on: {gpu}\n")

    rows = []
    for M, N, K in SHAPE_PRESETS[args.shapes]:
        print(f"  GEMM {M}×{N}×{K}...", end=" ", flush=True)
        r = bench_gemm(M, N, K)
        rows.append(r)
        print(f"lunara={r['lunara_tflops']:.2f}T  "
              f"torch={r['torch_tflops']:.2f}T  "
              f"({r['speedup']:.2f}× speedup)")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
