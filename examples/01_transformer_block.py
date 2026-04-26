#!/usr/bin/env python3
"""
examples/01_transformer_block.py
End-to-end demo: emit Triton kernels for a BERT-like transformer block,
benchmark them, and produce a roofline plot.

Run:
    python examples/01_transformer_block.py --hidden 768 --heads 12 --seq 512
"""

import argparse
import sys
from pathlib import Path

# Make package importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from lunara import Compiler, NsightProfiler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=768,
                   help="Hidden dimension (BERT-base = 768)")
    p.add_argument("--heads",  type=int, default=12)
    p.add_argument("--seq",    type=int, default=512)
    p.add_argument("--batch",  type=int, default=8)
    p.add_argument("--dtype",  default="float16")
    p.add_argument("--out",    default="output/transformer_block")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--no-roofline", action="store_true")
    p.add_argument("--tune", action="store_true",
                   help="Run autotuner (slow)")
    args = p.parse_args()

    print("─" * 60)
    print(f"  Lunara — Transformer Block Codegen Demo")
    print(f"  hidden={args.hidden}  heads={args.heads}  "
          f"seq={args.seq}  batch={args.batch}  dtype={args.dtype}")
    print("─" * 60)

    # ── Compile ──────────────────────────────────────────────────────────────
    compiler = Compiler(opt_level=2, dtype=args.dtype, autotune=args.tune)
    compiled = compiler.compile_transformer(
        hidden_dim=args.hidden,
        num_heads=args.heads,
        seq_len=args.seq,
        output_dir=args.out,
    )

    print(f"\n✓ Generated {len(compiled.kernel_specs)} kernels:")
    for s in compiled.kernel_specs:
        print(f"    • {s.name:<25s} kind={s.kind.name}")

    if not torch.cuda.is_available():
        print("\n[!] No CUDA device — skipping benchmark and profiling")
        return

    # ── Benchmark ────────────────────────────────────────────────────────────
    if not args.no_bench:
        print("\n─── Benchmark ───")
        results = compiled.benchmark(batch_size=args.batch, seq_len=args.seq)
        for k, ms in results.items():
            print(f"    {k:<25s}: {ms:>8.3f} ms")
        total = sum(v for v in results.values() if v == v)  # filter NaN
        print(f"    {'TOTAL':<25s}: {total:>8.3f} ms")

    # ── Roofline ─────────────────────────────────────────────────────────────
    if not args.no_roofline:
        print("\n─── Roofline analysis (matmul) ───")
        profiler = NsightProfiler(fallback_to_events=True)
        # Profile the QKV projection as representative GEMM
        M = args.batch * args.seq
        K = args.hidden
        N = 3 * args.hidden
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(K, N, dtype=torch.float16, device="cuda")
        result = profiler.profile_kernel(
            lambda: torch.mm(A, B),
            flops=2 * M * N * K,
            bytes_accessed=(M * K + K * N + M * N) * 2,
            kernel_name=f"qkv_proj_{M}x{N}x{K}",
        )
        profiler.print_summary(result)
        roofline_path = Path(args.out) / "roofline.png"
        profiler.plot_roofline(result, str(roofline_path))


if __name__ == "__main__":
    main()
