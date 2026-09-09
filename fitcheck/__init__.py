# version, public API
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fitcheck.advisor import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_LORA_RANKS,
    AdvisorReport,
    AxisCeiling,
    AxisPrice,
    FrontierPoint,
    SweepSpec,
    advise,
)
from fitcheck.config_parser import (
    ModelConfig,
    UnsupportedModelError,
    fetch_model_config,
)
from fitcheck.estimator import (
    InferenceReport,
    MemoryReport,
    ServingConfig,
    TrainingConfig,
    estimate,
    estimate_inference,
    estimate_warnings,
)
from fitcheck.gpu_db import GPU_DB, GpuSpec, get_gpu, list_gpus

try:
    __version__ = version("fitcheck-llm")
except PackageNotFoundError: 
    __version__ = "0.0.0.dev0"

__all__ = [
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_LORA_RANKS",
    "GPU_DB",
    "AdvisorReport",
    "AxisCeiling",
    "AxisPrice",
    "FrontierPoint",
    "GpuSpec",
    "InferenceReport",
    "MemoryReport",
    "ModelConfig",
    "ServingConfig",
    "SweepSpec",
    "TrainingConfig",
    "UnsupportedModelError",
    "__version__",
    "advise",
    "estimate",
    "estimate_inference",
    "estimate_warnings",
    "fetch_model_config",
    "get_gpu",
    "list_gpus",
]
