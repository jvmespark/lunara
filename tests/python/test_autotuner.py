"""tests/python/test_autotuner.py
Structural tests for the autotuner (no GPU required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lunara.codegen.autotuner import (
    TuningCache, TuningResult, _gemm_configs,
)
from lunara.codegen.triton_emitter import (
    KernelKind, KernelSpec, TileConfig,
)


def test_gemm_search_space_nonempty():
    configs = _gemm_configs()
    assert len(configs) > 10
    for c in configs:
        assert c.BLOCK_M > 0
        assert c.BLOCK_N > 0
        assert c.BLOCK_K > 0
        assert c.num_warps in (4, 8)


def test_gemm_search_space_respects_shared_mem():
    # 96 KB limit per the autotuner's pruning rule
    for c in _gemm_configs():
        shared = (c.BLOCK_M + c.BLOCK_N) * c.BLOCK_K * 2
        assert shared <= 96 * 1024


def test_tuning_result_hash_stable():
    cfg = TileConfig(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
    r1 = TuningResult(config=cfg, latency_ms=1.0, tflops=100.0)
    r2 = TuningResult(config=cfg, latency_ms=2.0, tflops=50.0)
    # Same config → same hash even with different perf
    assert r1.config_hash == r2.config_hash


def test_tuning_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache = TuningCache(str(cache_path))
    spec = KernelSpec(KernelKind.GEMM, "test")

    cfg = TileConfig(BLOCK_M=64, BLOCK_N=128)
    r = TuningResult(config=cfg, latency_ms=1.5, tflops=200.0)
    cache.put(spec, (4096, 4096, 4096), r)

    # Reload from disk
    cache2 = TuningCache(str(cache_path))
    cached = cache2.get(spec, (4096, 4096, 4096))
    assert cached is not None
    assert cached.latency_ms == 1.5
    assert cached.config.BLOCK_M == 64
    assert cached.config.BLOCK_N == 128


def test_tuning_cache_miss(tmp_path):
    cache = TuningCache(str(tmp_path / "cache.json"))
    spec = KernelSpec(KernelKind.GEMM, "missing")
    assert cache.get(spec, (1, 2, 3)) is None
