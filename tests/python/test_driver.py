"""tests/python/test_driver.py
Tests for the high-level Compiler driver.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lunara.driver import Compiler, CompiledModel, _default_transformer_specs
from lunara.codegen.triton_emitter import KernelKind


def test_default_transformer_specs():
    specs = _default_transformer_specs("test", "float16",
                                         hidden=768, heads=12)
    assert len(specs) == 7
    names = [s.name for s in specs]
    assert "qkv_proj" in names
    assert "flash_attn" in names
    assert "out_proj" in names
    assert "layernorm_attn" in names
    assert "ffn1" in names
    assert "ffn2" in names
    assert "layernorm_ffn" in names


def test_default_transformer_specs_kinds():
    specs = _default_transformer_specs("t", "float16")
    by_name = {s.name: s for s in specs}
    assert by_name["qkv_proj"].kind   == KernelKind.GEMM
    assert by_name["flash_attn"].kind == KernelKind.ATTENTION
    assert by_name["layernorm_attn"].kind == KernelKind.LAYERNORM


def test_default_transformer_specs_dtype():
    specs = _default_transformer_specs("t", "bfloat16")
    for s in specs:
        assert s.dtype == "bfloat16"


def test_default_transformer_specs_head_dim_correct():
    specs = _default_transformer_specs("t", "float16", hidden=512, heads=8)
    by_name = {s.name: s for s in specs}
    assert by_name["flash_attn"].tile.HEAD_DIM == 64  # 512/8


def test_compile_transformer_produces_module(tmp_path):
    compiler = Compiler(opt_level=1, dtype="float16", autotune=False)
    compiled = compiler.compile_transformer(
        hidden_dim=128, num_heads=4, seq_len=64,
        output_dir=str(tmp_path / "compiled"))

    assert isinstance(compiled, CompiledModel)
    assert compiled.kernel_module.exists()
    assert len(compiled.kernel_specs) > 0

    # Generated module must be valid Python
    import ast
    ast.parse(compiled.kernel_module.read_text())


def test_compiled_model_save(tmp_path):
    compiler = Compiler(opt_level=1, dtype="float16")
    compiled = compiler.compile_transformer(
        hidden_dim=64, num_heads=2, seq_len=32,
        output_dir=str(tmp_path / "src"))

    save_dir = tmp_path / "saved"
    compiled.save(str(save_dir))
    assert save_dir.exists()
    assert (save_dir / compiled.kernel_module.name).exists()
