#!/usr/bin/env python3
"""
examples/03_flash_attention.py
Generate a Flash Attention forward kernel and validate against
torch.nn.functional.scaled_dot_product_attention.

Run:
    python examples/03_flash_attention.py --batch 2 --heads 8 --seq 1024 --dim 64
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from lunara.codegen.triton_emitter import (
    KernelSpec, KernelKind, TileConfig, emit_attention,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch",  type=int, default=2)
    p.add_argument("--heads",  type=int, default=8)
    p.add_argument("--seq",    type=int, default=1024)
    p.add_argument("--dim",    type=int, default=64)
    p.add_argument("--causal", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    B, H, M, D = args.batch, args.heads, args.seq, args.dim

    # ── Generate kernel ──
    cfg = TileConfig(BLOCK_SEQ=64, HEAD_DIM=D, num_stages=2, num_warps=4)
    spec = KernelSpec(kind=KernelKind.ATTENTION, name="my_flash_attn",
                       tile=cfg, dtype="float16", is_causal=args.causal)
    src = emit_attention(spec)

    # Save
    out = Path("output/flash_attn")
    out.mkdir(parents=True, exist_ok=True)
    (out / "flash_attn.py").write_text(src)
    print(f"Generated: {out/'flash_attn.py'}")

    # Compile and test
    ns: dict = {}
    exec(compile(src, str(out/"flash_attn.py"), "exec"), ns)
    launch = ns["my_flash_attn_launch"]

    Q = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, H, M, D, dtype=torch.float16, device="cuda")

    print(f"\nRunning {B}×{H}×{M}×{D} attention "
          f"(causal={args.causal})...")

    out_lunara = launch(Q, K, V, args.causal)
    out_torch  = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=args.causal)

    diff = (out_lunara - out_torch).abs()
    print(f"  max diff:  {diff.max().item():.4f}")
    print(f"  mean diff: {diff.mean().item():.4f}")

    # Bench
    n_warmup, n_trials = 10, 50
    for _ in range(n_warmup): launch(Q, K, V, args.causal)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); launch(Q, K, V, args.causal); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    lat = times[len(times) // 2]
    flops = 4 * B * H * M * M * D
    tflops = flops / (lat * 1e-3) / 1e12
    print(f"\n  Latency: {lat:.3f} ms  →  {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
