#!/usr/bin/env python3
"""
examples/02_autotune_gemm.py
Autotune a GEMM for a specific problem shape and compare the best
config against the default config.

Run:
    python examples/02_autotune_gemm.py --M 4096 --N 4096 --K 4096
"""

import argparse
import sys
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from lunara import Autotuner, KernelSpec, KernelKind, TileConfig
from lunara.codegen.triton_emitter import emit_gemm


def measure(launch_fn, A, B, C, n_warmup=5, n_trials=20):
    for _ in range(n_warmup):
        launch_fn(A, B, C)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); launch_fn(A, B, C); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for this example.")
        return

    M, N, K = args.M, args.N, args.K
    flops = 2 * M * N * K

    print(f"Tuning GEMM {M}×{N}×{K}\n")

    # ── Default config baseline ──
    default_cfg = TileConfig()  # 128/128/32
    default_spec = KernelSpec(KernelKind.GEMM, "_default", default_cfg)
    src = emit_gemm(default_spec)
    ns: dict = {}
    exec(compile(src, "<default>", "exec"), ns)
    default_launch = ns["_default_launch"]

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.empty(M, N, dtype=torch.float16, device="cuda")

    default_lat = measure(default_launch, A, B, C)
    default_tflops = flops / (default_lat * 1e-3) / 1e12
    print(f"  Default 128×128×32: {default_lat:.3f} ms  →  "
          f"{default_tflops:.2f} TFLOPS")

    # ── Tuned config ──
    print("\n  Running autotuner...")
    tuner = Autotuner(verbose=False)
    best = tuner.tune_gemm(M, N, K)
    speedup = default_lat / best.latency_ms
    print(f"\n  Best: {asdict(best.config)}")
    print(f"  Latency: {best.latency_ms:.3f} ms  →  {best.tflops:.2f} TFLOPS")
    print(f"  Speedup: {speedup:.2f}× over default")


if __name__ == "__main__":
    main()
