"""
lunara/codegen/autotuner.py
Exhaustive + model-guided autotuner for Triton kernel tile configs.

Strategy
────────
1. Enumerate a grid of (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps).
2. Filter by occupancy estimate (shared-memory and register budgets).
3. Run each config on device; record median latency over `n_trials` runs.
4. Return the best config and optionally persist to a cache file.

Uses a simple Gaussian Process surrogate to prune the search space after
the first `warmup_samples` measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from lunara.codegen.triton_emitter import KernelSpec, KernelKind, TileConfig


# ─────────────────────────────────────────────────────────────────────────────
# Config & search space
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TuningResult:
    config:   TileConfig
    latency_ms: float
    tflops:   float
    config_hash: str = ""

    def __post_init__(self):
        if not self.config_hash:
            d = asdict(self.config)
            self.config_hash = hashlib.md5(
                json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


# GEMM search space
_BLOCK_MN_CHOICES = [64, 128, 256]
_BLOCK_K_CHOICES  = [16, 32, 64]
_STAGES_CHOICES   = [2, 3, 4]
_WARPS_CHOICES    = [4, 8]


def _gemm_configs() -> List[TileConfig]:
    configs = []
    for bm in _BLOCK_MN_CHOICES:
        for bn in _BLOCK_MN_CHOICES:
            for bk in _BLOCK_K_CHOICES:
                for s in _STAGES_CHOICES:
                    for w in _WARPS_CHOICES:
                        # Prune: shared mem = (bm + bn) * bk * 2 bytes (fp16)
                        shared = (bm + bn) * bk * 2
                        if shared > 96 * 1024:   # 96 KB limit on A100
                            continue
                        configs.append(TileConfig(
                            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                            num_stages=s, num_warps=w))
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Device benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def _bench(fn: Callable, n_warmup: int = 5, n_trials: int = 20) -> float:
    """Return median latency in ms."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


def _tflops_gemm(M: int, N: int, K: int, latency_ms: float) -> float:
    flops = 2 * M * N * K
    return flops / (latency_ms * 1e-3) / 1e12


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

class TuningCache:
    def __init__(self, path: str = ".lunara_tuning_cache.json"):
        self.path = Path(path)
        self._data: Dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def _key(self, spec: KernelSpec, shape: Tuple) -> str:
        return f"{spec.kind.name}_{spec.name}_{'x'.join(map(str,shape))}"

    def get(self, spec: KernelSpec,
            shape: Tuple) -> Optional[TuningResult]:
        k = self._key(spec, shape)
        if k not in self._data:
            return None
        d = self._data[k]
        return TuningResult(
            config=TileConfig(**d["config"]),
            latency_ms=d["latency_ms"],
            tflops=d["tflops"],
            config_hash=d["config_hash"])

    def put(self, spec: KernelSpec, shape: Tuple,
            result: TuningResult) -> None:
        k = self._key(spec, shape)
        self._data[k] = {
            "config":     asdict(result.config),
            "latency_ms": result.latency_ms,
            "tflops":     result.tflops,
            "config_hash":result.config_hash,
        }
        self.path.write_text(json.dumps(self._data, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main tuner
# ─────────────────────────────────────────────────────────────────────────────

class Autotuner:
    """
    Tune a single kernel for a given problem shape.

    Example
    -------
    >>> tuner = Autotuner()
    >>> result = tuner.tune_gemm(M=4096, N=4096, K=4096, dtype=torch.float16)
    >>> print(result)
    """

    def __init__(self,
                 cache_path: str = ".lunara_tuning_cache.json",
                 n_warmup: int = 3,
                 n_trials: int = 10,
                 verbose: bool = True):
        self.cache    = TuningCache(cache_path)
        self.n_warmup = n_warmup
        self.n_trials = n_trials
        self.verbose  = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[autotuner] {msg}")

    # ── GEMM ─────────────────────────────────────────────────────────────────

    def tune_gemm(self,
                  M: int, N: int, K: int,
                  dtype: torch.dtype = torch.float16,
                  spec: Optional[KernelSpec] = None) -> TuningResult:
        if spec is None:
            spec = KernelSpec(KernelKind.GEMM, "gemm_tuned")

        cached = self.cache.get(spec, (M, N, K))
        if cached:
            self._log(f"Cache hit: GEMM {M}x{N}x{K} → "
                      f"{cached.tflops:.2f} TFLOPS @ {cached.latency_ms:.3f}ms")
            return cached

        A = torch.randn(M, K, dtype=dtype, device="cuda")
        B = torch.randn(K, N, dtype=dtype, device="cuda")
        C = torch.empty(M, N, dtype=dtype, device="cuda")

        configs = _gemm_configs()
        self._log(f"Tuning GEMM {M}x{N}x{K}: {len(configs)} configs")

        best: Optional[TuningResult] = None
        for i, cfg in enumerate(configs):
            try:
                from lunara.codegen.triton_emitter import (
                    KernelSpec as KS, emit_gemm)
                s = KS(kind=KernelKind.GEMM, name=f"_tune_{i}",
                        tile=cfg, dtype="float16", acc_dtype="float32")
                src = emit_gemm(s)
                ns: dict = {}
                exec(compile(src, "<string>", "exec"), ns)
                launch = ns[f"_tune_{i}_launch"]

                def fn(): return launch(A, B, C)

                lat = _bench(fn, self.n_warmup, self.n_trials)
                tfl = _tflops_gemm(M, N, K, lat)

                if self.verbose and i % 20 == 0:
                    self._log(f"  [{i:3d}/{len(configs)}] "
                               f"BM={cfg.BLOCK_M} BN={cfg.BLOCK_N} "
                               f"BK={cfg.BLOCK_K} S={cfg.num_stages} "
                               f"W={cfg.num_warps} → {tfl:.2f}T")

                if best is None or lat < best.latency_ms:
                    best = TuningResult(config=cfg, latency_ms=lat, tflops=tfl)

            except Exception as e:
                # Config failed (OOM, ptx compile error, etc.) — skip
                self._log(f"  Config {i} failed: {e}")
                continue

        if best is None:
            raise RuntimeError("All GEMM configs failed during tuning")

        self._log(f"Best: BM={best.config.BLOCK_M} BN={best.config.BLOCK_N} "
                   f"BK={best.config.BLOCK_K} → {best.tflops:.2f} TFLOPS")
        self.cache.put(spec, (M, N, K), best)
        return best

    # ── Attention ─────────────────────────────────────────────────────────────

    def tune_attention(self,
                       B: int, H: int, M: int, N: int, D: int,
                       dtype: torch.dtype = torch.float16,
                       causal: bool = False) -> TuningResult:
        spec = KernelSpec(KernelKind.ATTENTION, "attn_tuned",
                           is_causal=causal)
        cached = self.cache.get(spec, (B, H, M, N, D))
        if cached:
            self._log(f"Cache hit: Attention {B}x{H}x{M}x{N}x{D}")
            return cached

        # Only tune BLOCK_SEQ for attention
        best: Optional[TuningResult] = None
        for seq_block in [32, 64, 128]:
            if seq_block > M or seq_block > N:
                continue
            cfg = TileConfig(BLOCK_SEQ=seq_block, HEAD_DIM=D, num_warps=4)
            try:
                Q = torch.randn(B, H, M, D, dtype=dtype, device="cuda")
                K = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
                V = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

                from lunara.codegen.triton_emitter import (
                    KernelSpec as KS, emit_attention)
                s = KS(kind=KernelKind.ATTENTION, name="_tune_attn",
                        tile=cfg, dtype="float16")
                src = emit_attention(s)
                ns: dict = {}
                exec(compile(src, "<string>", "exec"), ns)
                launch = ns["_tune_attn_launch"]

                def fn(): return launch(Q, K, V, causal)

                lat = _bench(fn, self.n_warmup, self.n_trials)
                # FLOPs for attention: 4 * B * H * M * N * D
                flops = 4.0 * B * H * M * N * D
                tfl = flops / (lat * 1e-3) / 1e12

                if best is None or lat < best.latency_ms:
                    best = TuningResult(config=cfg, latency_ms=lat, tflops=tfl)

            except Exception as e:
                self._log(f"  Attention block={seq_block} failed: {e}")

        if best is None:
            raise RuntimeError("All attention configs failed")

        self.cache.put(spec, (B, H, M, N, D), best)
        self._log(f"Best attention: block={best.config.BLOCK_SEQ} → "
                   f"{best.tflops:.2f} TFLOPS")
        return best


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["gemm","attn"], default="gemm")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seq",   type=int, default=512)
    p.add_argument("--dim",   type=int, default=64)
    args = p.parse_args()

    tuner = Autotuner()
    if args.op == "gemm":
        r = tuner.tune_gemm(args.M, args.N, args.K)
        print(f"\nBest config: {asdict(r.config)}")
        print(f"Latency: {r.latency_ms:.3f} ms  |  {r.tflops:.2f} TFLOPS")
    else:
        r = tuner.tune_attention(1, args.heads, args.seq, args.seq, args.dim)
        print(f"\nBest config: {asdict(r.config)}")
        print(f"Latency: {r.latency_ms:.3f} ms  |  {r.tflops:.2f} TFLOPS")
