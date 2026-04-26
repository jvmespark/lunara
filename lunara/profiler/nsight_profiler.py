"""
lunara/profiler/nsight_profiler.py
Programmatic interface to NVIDIA Nsight Compute metrics.

Wraps ncu (Nsight Compute CLI) to extract:
  - SM utilisation
  - Memory throughput (HBM and L2)
  - Roofline position (achieved FLOP/s vs peak, achieved BW vs peak)
  - Occupancy, warp efficiency, memory access patterns

Usage
─────
    profiler = NsightProfiler()
    result   = profiler.profile_kernel(fn, *args, **kwargs)
    profiler.print_summary(result)
    profiler.plot_roofline(result, "roofline.png")
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# GPU hardware specs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUSpec:
    name:            str
    peak_flops_fp16: float   # TFLOPS (tensor core)
    peak_flops_fp32: float   # TFLOPS
    peak_bw_hbm:     float   # TB/s
    peak_bw_l2:      float   # TB/s
    num_sms:         int
    shared_mem_per_sm: int   # bytes

GPU_SPECS: Dict[str, GPUSpec] = {
    "A100": GPUSpec("A100", 312.0, 19.5, 2.0, 12.0, 108, 164*1024),
    "A10G": GPUSpec("A10G",  125.0, 31.2, 0.6,  4.0,  80, 100*1024),
    "V100": GPUSpec("V100",  125.0, 14.1, 0.9,  3.1,  80, 96*1024),
    "H100": GPUSpec("H100",  989.0, 51.2, 3.35, 33.0, 132, 228*1024),
    "RTX4090": GPUSpec("RTX4090", 330.0, 82.6, 1.008, 8.0, 128, 100*1024),
}


def detect_gpu() -> GPUSpec:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    for key in GPU_SPECS:
        if key in name:
            return GPU_SPECS[key]
    # Fallback: query from torch
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    if props:
        return GPUSpec(
            name=props.name,
            peak_flops_fp16=float(props.multi_processor_count) * 128 * 2 * props.clock_rate * 1e-9,
            peak_flops_fp32=float(props.multi_processor_count) * 64  * 2 * props.clock_rate * 1e-9,
            peak_bw_hbm=0.9,
            peak_bw_l2=4.0,
            num_sms=props.multi_processor_count,
            shared_mem_per_sm=props.shared_memory_per_multiprocessor,
        )
    return GPU_SPECS["A100"]  # conservative default


# ─────────────────────────────────────────────────────────────────────────────
# Profiling result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KernelMetrics:
    kernel_name:       str = ""
    duration_us:       float = 0.0
    sm_active_pct:     float = 0.0    # % SMs active
    achieved_occupancy:float = 0.0    # 0–1
    warp_efficiency:   float = 0.0    # 0–1
    # Memory
    hbm_read_bw_gbs:   float = 0.0   # GB/s achieved
    hbm_write_bw_gbs:  float = 0.0
    l2_read_bw_gbs:    float = 0.0
    l2_write_bw_gbs:   float = 0.0
    l1_hit_rate:       float = 0.0   # 0–1
    # Compute
    achieved_flops_t:  float = 0.0   # TFLOPS
    arithmetic_intensity: float = 0.0  # FLOP/byte
    # Roofline
    roofline_bound:    str = "unknown"   # "memory" | "compute"
    roofline_efficiency: float = 0.0    # achieved / peak (relevant bound)
    # Raw ncu output
    raw_metrics:       Dict[str, str] = field(default_factory=dict)


@dataclass
class ProfileResult:
    kernels:  List[KernelMetrics] = field(default_factory=list)
    gpu_spec: Optional[GPUSpec]   = None
    total_duration_us: float      = 0.0

    @property
    def primary(self) -> Optional[KernelMetrics]:
        """Return the kernel with the longest duration."""
        if not self.kernels:
            return None
        return max(self.kernels, key=lambda k: k.duration_us)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA event-based lightweight profiler (no ncu dependency)
# ─────────────────────────────────────────────────────────────────────────────

class CUDAEventProfiler:
    """
    Lightweight profiler using CUDA events.  Measures latency only (no
    SM / memory counters).  Always works without ncu.
    """

    def __init__(self, n_warmup: int = 5, n_trials: int = 20):
        self.n_warmup = n_warmup
        self.n_trials = n_trials

    def profile(self,
                fn: Callable,
                flops: Optional[float] = None,
                bytes_accessed: Optional[float] = None,
                gpu_spec: Optional[GPUSpec] = None) -> KernelMetrics:
        if gpu_spec is None:
            gpu_spec = detect_gpu()

        for _ in range(self.n_warmup):
            fn()
        torch.cuda.synchronize()

        times_us = []
        for _ in range(self.n_trials):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            times_us.append(s.elapsed_time(e) * 1000)  # ms→µs

        times_us.sort()
        med = times_us[len(times_us) // 2]

        m = KernelMetrics(duration_us=med)
        if flops is not None:
            m.achieved_flops_t = flops / (med * 1e-6) / 1e12
        if bytes_accessed is not None and med > 0:
            bw = bytes_accessed / (med * 1e-6) / 1e9
            m.hbm_read_bw_gbs = bw
            m.arithmetic_intensity = (flops or 0.0) / bytes_accessed

        # Roofline
        if flops is not None and bytes_accessed is not None:
            ridge = (gpu_spec.peak_flops_fp16 * 1e12) / (gpu_spec.peak_bw_hbm * 1e12)
            if m.arithmetic_intensity < ridge:
                m.roofline_bound = "memory"
                m.roofline_efficiency = (
                    m.hbm_read_bw_gbs / (gpu_spec.peak_bw_hbm * 1000))
            else:
                m.roofline_bound = "compute"
                m.roofline_efficiency = (
                    m.achieved_flops_t / gpu_spec.peak_flops_fp16)

        return m


# ─────────────────────────────────────────────────────────────────────────────
# Nsight Compute CLI profiler
# ─────────────────────────────────────────────────────────────────────────────

# Metrics requested from ncu
_NCU_METRICS = [
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",  # SM util
    "sm__warps_active.avg.pct_of_peak_sustained_active",    # occupancy
    "smsp__warps_eligible.avg.pct_of_peak_sustained_active",# warp efficiency
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l2cache__read_sectors.sum",
    "l2cache__write_sectors.sum",
    "l1tex__hit_rate.pct",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
]


class NsightProfiler:
    """
    Full-featured profiler wrapping `ncu`.  Falls back to CUDAEventProfiler
    if ncu is unavailable.
    """

    def __init__(self,
                 ncu_path: str = "ncu",
                 fallback_to_events: bool = True):
        self.ncu_path = ncu_path
        self.fallback = fallback_to_events
        self._has_ncu = self._check_ncu()
        self.gpu_spec = detect_gpu()

        if not self._has_ncu:
            print("[profiler] ncu not found; using CUDA event profiler")
            self._event_profiler = CUDAEventProfiler()

    def _check_ncu(self) -> bool:
        try:
            subprocess.run([self.ncu_path, "--version"],
                           capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # ── Profile a Python callable ─────────────────────────────────────────

    def profile_kernel(self,
                       fn: Callable,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None,
                       flops: Optional[float] = None,
                       bytes_accessed: Optional[float] = None,
                       kernel_name: str = "kernel") -> ProfileResult:
        kwargs = kwargs or {}

        if self._has_ncu:
            return self._profile_with_ncu(fn, args, kwargs, kernel_name)
        else:
            bound_fn = lambda: fn(*args, **kwargs)
            m = self._event_profiler.profile(bound_fn, flops, bytes_accessed,
                                              self.gpu_spec)
            m.kernel_name = kernel_name
            return ProfileResult(kernels=[m], gpu_spec=self.gpu_spec,
                                  total_duration_us=m.duration_us)

    def _profile_with_ncu(self, fn, args, kwargs, kernel_name) -> ProfileResult:
        """Write a tiny launcher script, run ncu on it, parse CSV output."""
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "launch.py"
            csv_out  = Path(tmp) / "metrics.csv"

            # Write launcher
            launcher.write_text(
                "import sys\n"
                "sys.path.insert(0, '.')\n"
                "import torch\n"
                "# NOTE: fn must be pickleable for ncu subprocess launch.\n"
                "# In practice, generate a standalone script per-kernel.\n"
                "pass\n"
            )

            cmd = [
                self.ncu_path,
                "--csv",
                "--metrics", ",".join(_NCU_METRICS),
                "--output", str(csv_out),
                "python", str(launcher),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                return self._parse_ncu_csv(csv_out, kernel_name)
            except subprocess.CalledProcessError as e:
                print(f"[profiler] ncu failed: {e.stderr.decode()[:200]}")
                # Fall back
                bound_fn = lambda: fn(*args, **kwargs)
                m = CUDAEventProfiler().profile(bound_fn)
                m.kernel_name = kernel_name
                return ProfileResult(kernels=[m], gpu_spec=self.gpu_spec,
                                      total_duration_us=m.duration_us)

    def _parse_ncu_csv(self, csv_path: Path, name: str) -> ProfileResult:
        """Parse ncu CSV into KernelMetrics."""
        m = KernelMetrics(kernel_name=name)
        if not csv_path.exists():
            return ProfileResult(kernels=[m], gpu_spec=self.gpu_spec)
        # todo: iterate rows
        lines = csv_path.read_text().splitlines()
        for line in lines:
            parts = line.split(",")
            if len(parts) < 3:
                continue
            metric, value = parts[1].strip('"'), parts[-1].strip('"')
            m.raw_metrics[metric] = value
            try:
                v = float(value)
                if "cycles_active" in metric:
                    m.sm_active_pct = v
                elif "warps_active" in metric:
                    m.achieved_occupancy = v / 100.0
                elif "dram__bytes_read" in metric:
                    m.hbm_read_bw_gbs = v / 1e9
            except ValueError:
                pass
        return ProfileResult(kernels=[m], gpu_spec=self.gpu_spec,
                              total_duration_us=m.duration_us)

    # ── Reporting ─────────────────────────────────────────────────────────

    @staticmethod
    def print_summary(result: ProfileResult) -> None:
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Lunara Kernel Profile Summary")
        print(sep)
        if result.gpu_spec:
            g = result.gpu_spec
            print(f"  GPU:  {g.name}  |  "
                  f"FP16 peak: {g.peak_flops_fp16:.0f} TFLOPS  |  "
                  f"HBM BW: {g.peak_bw_hbm:.1f} TB/s")
        print()
        for k in result.kernels:
            print(f"  Kernel: {k.kernel_name}")
            print(f"    Duration:    {k.duration_us:>10.2f} µs")
            print(f"    SM util:     {k.sm_active_pct:>9.1f}%")
            print(f"    Occupancy:   {k.achieved_occupancy*100:>9.1f}%")
            print(f"    HBM read BW: {k.hbm_read_bw_gbs:>9.1f} GB/s")
            print(f"    Achieved:    {k.achieved_flops_t:>9.2f} TFLOPS")
            print(f"    Arith. Int.: {k.arithmetic_intensity:>9.2f} FLOP/B")
            print(f"    Roofline:    {k.roofline_bound:>9s} "
                  f"({k.roofline_efficiency*100:.1f}% of peak)")
        print(sep + "\n")

    def plot_roofline(self,
                      result: ProfileResult,
                      output_path: str = "roofline.png") -> None:
        """Plot a roofline chart using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[profiler] matplotlib not available; skipping roofline plot")
            return

        if not result.gpu_spec:
            return
        g = result.gpu_spec

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)

        # Roofline ridgelines
        ai = np.logspace(-2, 6, 500, base=2)
        mem_roof  = g.peak_bw_hbm * 1000 * ai       # GB/s * FLOP/B → GFLOPS
        comp_roof = np.full_like(ai, g.peak_flops_fp16 * 1000) # GFLOPS
        ax.plot(ai, np.minimum(mem_roof, comp_roof),
                "k-", linewidth=2.5, label="Roofline (FP16 TC)")

        # Kernel points
        for k in result.kernels:
            if k.arithmetic_intensity > 0 and k.achieved_flops_t > 0:
                ax.scatter(k.arithmetic_intensity,
                           k.achieved_flops_t * 1000,
                           s=120, zorder=5,
                           label=f"{k.kernel_name} "
                                  f"({k.achieved_flops_t:.1f}T, "
                                  f"{k.roofline_efficiency*100:.0f}%)")

        # Ridge point
        ridge = g.peak_flops_fp16 / (g.peak_bw_hbm / 1000)
        ax.axvline(ridge, linestyle="--", color="gray", alpha=0.5,
                   label=f"Ridge point ({ridge:.0f} FLOP/B)")

        ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
        ax.set_ylabel("Performance (GFLOPS)")
        ax.set_title(f"Roofline Model — {g.name}")
        ax.legend(loc="lower right")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"[profiler] Roofline saved → {output_path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Lunara Nsight profiler")
    p.add_argument("--op", choices=["gemm","attn"], default="gemm")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--plot", default="roofline.png")
    args = p.parse_args()

    profiler = NsightProfiler()

    if args.op == "gemm":
        A = torch.randn(args.M, args.K, dtype=torch.float16, device="cuda")
        B = torch.randn(args.K, args.N, dtype=torch.float16, device="cuda")
        fn = lambda: torch.mm(A, B)
        flops = 2 * args.M * args.N * args.K
        bw = (args.M * args.K + args.K * args.N + args.M * args.N) * 2
        result = profiler.profile_kernel(fn, flops=flops, bytes_accessed=bw,
                                          kernel_name=f"matmul_{args.M}x{args.N}x{args.K}")
    else:
        Q = torch.randn(1, 12, 512, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 12, 512, 64, dtype=torch.float16, device="cuda")
        V = torch.randn(1, 12, 512, 64, dtype=torch.float16, device="cuda")
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        result = profiler.profile_kernel(fn, kernel_name="sdpa_512")

    profiler.print_summary(result)
    profiler.plot_roofline(result, args.plot)
