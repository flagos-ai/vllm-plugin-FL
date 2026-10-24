"""Select the explicit K3 startup profile through FL's existing YAML setting.

This early check has no Torch, vLLM-platform or operator imports. Workers still
validate the model/parallel configuration through K3ExecutionConfig.
"""
from functools import lru_cache
import os
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def is_kimi_k3_profile() -> bool:
    path = os.environ.get("VLLM_FL_CONFIG")
    if not path:
        return False
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("FL dispatch configuration must be a mapping")
    return data.get("model_profile") == "kimi_k3_metax"
