"""tests/python/test_profiler.py
Structural tests for the profiler (no GPU required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lunara.profiler.nsight_profiler import (
    GPU_SPECS, GPUSpec, KernelMetrics, ProfileResult, NsightProfiler,
)


def test_gpu_specs_present():
    for arch in ("A100", "V100", "H100", "RTX4090"):
        assert arch in GPU_SPECS
        spec = GPU_SPECS[arch]
        assert spec.peak_flops_fp16 > 0
        assert spec.peak_bw_hbm > 0
        assert spec.num_sms > 0


def test_kernel_metrics_default():
    m = KernelMetrics()
    assert m.duration_us == 0.0
    assert m.roofline_bound == "unknown"
    assert m.raw_metrics == {}


def test_profile_result_primary():
    a = KernelMetrics(kernel_name="a", duration_us=10.0)
    b = KernelMetrics(kernel_name="b", duration_us=20.0)
    c = KernelMetrics(kernel_name="c", duration_us=5.0)
    pr = ProfileResult(kernels=[a, b, c])
    assert pr.primary is b


def test_profile_result_empty():
    pr = ProfileResult()
    assert pr.primary is None


def test_nsight_profiler_initialises():
    # Should not raise even if ncu isn't available
    p = NsightProfiler(fallback_to_events=True)
    assert p is not None
    assert hasattr(p, "_has_ncu")


def test_print_summary_handles_empty(capsys):
    pr = ProfileResult()
    NsightProfiler.print_summary(pr)
    out = capsys.readouterr().out
    assert "Profile Summary" in out
