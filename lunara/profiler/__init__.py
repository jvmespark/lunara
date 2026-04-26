"""Lunara profiler — Nsight Compute integration + roofline analysis."""

from lunara.profiler.nsight_profiler import (
    NsightProfiler, CUDAEventProfiler,
    ProfileResult, KernelMetrics, GPUSpec, GPU_SPECS, detect_gpu,
)

__all__ = [
    "NsightProfiler", "CUDAEventProfiler",
    "ProfileResult", "KernelMetrics", "GPUSpec", "GPU_SPECS",
    "detect_gpu",
]
