# Copyright (c) 2026 BAAI. All rights reserved.

"""Common V1 model runner entry point for every FL platform."""

from importlib.metadata import version

from packaging.version import Version

if Version(version("vllm")) >= Version("0.32.0"):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as ModelRunnerFL
else:
    from vllm_fl.worker._legacy_model_runner import ModelRunnerFL

__all__ = ["ModelRunnerFL"]
