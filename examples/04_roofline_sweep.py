#!/usr/bin/env python3
"""
examples/04_roofline_sweep.py
Sweep matmul shapes and plot all kernels on a single roofline.

Run:
    python examples/04_roofline_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from lunara.profiler.nsight_profiler import (
    NsightProfiler, KernelMetrics, ProfileResult, detect_gpu,
)


def main():
    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    profiler = NsightProfiler(fallback_to_events=True)
    gpu = detect_gpu()
    print(f"GPU: {gpu.name}")

    shapes = [
        (128,   128,   128),    # tiny
        (512,   512,   512),    # small
        (1024, 1024,  1024),    # medium
        (2048, 2048,  2048),    # large
        (4096, 4096,  4096),    # huge
        (8192, 8192,  1024),    # tall-skinny
        (1024, 8192,  8192),    # short-fat
    ]

    all_kernels = []
    for M, N, K in shapes:
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(K, N, dtype=torch.float16, device="cuda")
        flops = 2 * M * N * K
        bw = (M * K + K * N + M * N) * 2
        r = profiler.profile_kernel(
            lambda a=A, b=B: torch.mm(a, b),
            flops=flops, bytes_accessed=bw,
            kernel_name=f"mm_{M}x{N}x{K}",
        )
        all_kernels.extend(r.kernels)

    combined = ProfileResult(kernels=all_kernels, gpu_spec=gpu,
                              total_duration_us=sum(k.duration_us for k in all_kernels))
    profiler.print_summary(combined)
    profiler.plot_roofline(combined, "output/roofline_sweep.png")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    main()
