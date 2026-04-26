"""
Lunara — ML compiler for ONNX -> StableHLO -> MLIR linalg -> Triton/PTX.

Public API
──────────
    from lunara import Compiler, KernelSpec, KernelKind, TileConfig
    from lunara.profiler import NsightProfiler
    from lunara.codegen import Autotuner
"""

__version__ = "0.1.0"
__author__  = "Lunara Contributors"
__license__ = "Apache-2.0"

from lunara.driver import Compiler, CompiledModel
from lunara.codegen.triton_emitter import (
    KernelKind, KernelSpec, TileConfig, TritonEmitter,
    emit_transformer_block,
)
from lunara.codegen.autotuner import Autotuner, TuningResult
from lunara.profiler.nsight_profiler import (
    NsightProfiler, ProfileResult, KernelMetrics, GPUSpec, detect_gpu,
)

__all__ = [
    "Compiler", "CompiledModel",
    "KernelKind", "KernelSpec", "TileConfig", "TritonEmitter",
    "emit_transformer_block",
    "Autotuner", "TuningResult",
    "NsightProfiler", "ProfileResult", "KernelMetrics", "GPUSpec",
    "detect_gpu",
]
