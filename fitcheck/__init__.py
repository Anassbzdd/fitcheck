# version, public API
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.estimator import (
    InferenceReport,
    MemoryReport,
    ServingConfig,
    TrainingConfig,
    estimate,
    estimate_inference,
)
from fitcheck.gpu_db import GPU_DB, GpuSpec, get_gpu, list_gpus

try:
    __version__ = version("fitcheck-llm")
except PackageNotFoundError: 
    __version__ = "0.0.0.dev0"

__all__ = [
    "GPU_DB",
    "GpuSpec",
    "InferenceReport",
    "MemoryReport",
    "ModelConfig",
    "ServingConfig",
    "TrainingConfig",
    "__version__",
    "estimate",
    "estimate_inference",
    "fetch_model_config",
    "get_gpu",
    "list_gpus",
]
