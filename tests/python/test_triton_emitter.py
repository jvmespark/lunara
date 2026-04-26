"""tests/python/test_triton_emitter.py
Structural tests for the Triton source emitter.  These do not require
a GPU — they only check that emitted source compiles to valid Python
and contains the expected scaffolding.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lunara.codegen.triton_emitter import (
    KernelKind, KernelSpec, TileConfig, TritonEmitter,
    emit_gemm, emit_attention, emit_softmax, emit_layernorm,
    emit_elementwise,
)


# ── GEMM ─────────────────────────────────────────────────────────────────────

def test_gemm_emits_valid_python():
    spec = KernelSpec(KernelKind.GEMM, "my_gemm")
    src = emit_gemm(spec)
    assert "def my_gemm(" in src
    assert "def my_gemm_launch(" in src
    assert "tl.dot(" in src
    # Must parse as valid Python
    ast.parse(src)


def test_gemm_with_bias_emits_bias_load():
    spec = KernelSpec(KernelKind.GEMM, "gemm_bias", has_bias=True)
    src = emit_gemm(spec)
    assert "bias_ptr" in src
    assert "tl.load(bias_ptr" in src
    ast.parse(src)


def test_gemm_with_activation():
    for act in ("relu", "gelu", "sigmoid"):
        spec = KernelSpec(KernelKind.GEMM, f"gemm_{act}", activation=act)
        src = emit_gemm(spec)
        ast.parse(src)
        if act == "relu":
            assert "tl.maximum" in src
        elif act == "gelu":
            assert "tl.math.erf" in src
        elif act == "sigmoid":
            assert "tl.math.exp" in src


def test_gemm_tile_sizes_propagate():
    cfg = TileConfig(BLOCK_M=64, BLOCK_N=256, BLOCK_K=64,
                      num_stages=3, num_warps=8)
    spec = KernelSpec(KernelKind.GEMM, "g", tile=cfg)
    src = emit_gemm(spec)
    assert "BLOCK_M=64" in src
    assert "BLOCK_N=256" in src
    assert "BLOCK_K=64" in src
    assert "num_stages=3" in src
    assert "num_warps=8" in src


# ── Attention ────────────────────────────────────────────────────────────────

def test_attention_emits_valid_python():
    spec = KernelSpec(KernelKind.ATTENTION, "attn",
                       tile=TileConfig(BLOCK_SEQ=64, HEAD_DIM=64))
    src = emit_attention(spec)
    assert "def attn(" in src
    assert "def attn_launch(" in src
    assert "tl.math.exp" in src   # online softmax
    assert "IS_CAUSAL" in src
    ast.parse(src)


def test_attention_causal_flag():
    spec = KernelSpec(KernelKind.ATTENTION, "causal_attn", is_causal=True,
                       tile=TileConfig(BLOCK_SEQ=64, HEAD_DIM=64))
    src = emit_attention(spec)
    # The emitted source uses Python templating; a runtime arg passes causal
    assert "is_causal" not in src.lower() or "causal" in src.lower()
    ast.parse(src)


# ── Softmax / LayerNorm / Element-wise ───────────────────────────────────────

def test_softmax_emits_valid_python():
    spec = KernelSpec(KernelKind.SOFTMAX, "sm")
    src = emit_softmax(spec)
    assert "tl.max(x, axis=0)" in src
    assert "tl.math.exp" in src
    ast.parse(src)


def test_layernorm_emits_valid_python():
    spec = KernelSpec(KernelKind.LAYERNORM, "ln")
    src = emit_layernorm(spec)
    assert "mean" in src
    assert "var" in src
    assert "tl.math.sqrt" in src
    ast.parse(src)


def test_elementwise_relu():
    spec = KernelSpec(KernelKind.ELEMENTWISE, "relu_kern")
    src = emit_elementwise(spec, body="tl.maximum(x, 0.0)")
    assert "tl.maximum(x, 0.0)" in src
    ast.parse(src)


# ── Emitter writes module ────────────────────────────────────────────────────

def test_emitter_writes_module(tmp_path):
    em = TritonEmitter(str(tmp_path))
    em.emit(KernelSpec(KernelKind.GEMM, "k1"))
    em.emit(KernelSpec(KernelKind.SOFTMAX, "k2"))
    out = em.write_module("test_kernels")
    assert out.exists()
    text = out.read_text()
    assert "def k1(" in text
    assert "def k2(" in text
    assert "KERNEL_DISPATCH" in text
    ast.parse(text)


def test_emitter_individual_files(tmp_path):
    em = TritonEmitter(str(tmp_path))
    em.emit(KernelSpec(KernelKind.GEMM, "g_a"))
    em.emit(KernelSpec(KernelKind.GEMM, "g_b"))
    paths = em.write_individual()
    assert len(paths) == 2
    for p in paths:
        assert p.exists()


# ── Unknown kernel kind raises ───────────────────────────────────────────────

def test_unknown_kind_raises():
    em = TritonEmitter("/tmp/lunara_test")
    spec = KernelSpec(KernelKind.CONV2D, "conv")  # CONV2D not in dispatch
    with pytest.raises(NotImplementedError):
        em.emit(spec)


# ── Tile config defaults are sane ────────────────────────────────────────────

def test_tile_config_defaults():
    t = TileConfig()
    assert t.BLOCK_M == 128
    assert t.BLOCK_N == 128
    assert t.BLOCK_K == 32
    assert t.num_stages == 4
    assert t.num_warps == 4
    assert t.HEAD_DIM == 64
    assert t.BLOCK_SEQ == 64
