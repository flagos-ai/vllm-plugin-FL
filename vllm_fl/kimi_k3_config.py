"""Immutable K3 execution policy, derived from model and parallel configuration."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path

import yaml


@dataclass(frozen=True)
class K3ExecutionConfig:
    enabled: bool = False
    tp8_compute: bool = False
    host_pp_feedback: bool = False

    @classmethod
    def from_vllm(cls, config, *, vendor):
        model = config.model_config
        hf = getattr(model, "hf_config", None)
        model_type = getattr(hf, "model_type", None)
        # vLLM constructs K3's language submodel with its own kimi_linear HF
        # config. The selected quantization backend carries the family marker.
        family = getattr(getattr(config, "quant_config", None), "model_family", None)
        is_k3 = model_type == "kimi_k3" or (
            model_type == "kimi_linear" and family == "kimi_k3"
        )
        if vendor != "metax" or not is_k3:
            return cls()
        p = config.parallel_config
        if not p.disable_custom_all_reduce:
            raise NotImplementedError(
                "K3 MetaX requires --disable-custom-all-reduce; use MCCL"
            )
        options = config.additional_config.get("k3", {})
        if options:
            raise ValueError(f"Unknown K3 execution options: {sorted(options)}")
        if (
            p.tensor_parallel_size != 8
            or p.data_parallel_size != 1
            or p.enable_expert_parallel
        ):
            raise ValueError("K3 MetaX requires TP8/DP1 with EP disabled")
        return cls(
            enabled=True,
            tp8_compute=p.tensor_parallel_size == 8,
            host_pp_feedback=p.pipeline_parallel_size > 1,
        )


@lru_cache(maxsize=1)
def is_kimi_k3_profile():
    path = os.environ.get("VLLM_FL_CONFIG")
    if not path:
        return False
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("FL dispatch configuration must be a mapping")
    return config.get("model_profile") == "kimi_k3_metax"


def initialize_process_cache():
    root = os.environ.get("K3_CACHE_ROOT")
    if root is None:
        return
    cache = Path(root) / f"process-{os.getpid()}"
    paths = {
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "extensions",
        "FLAGGEMS_CACHE_DIR": "flaggems",
        "VLLM_CACHE_ROOT": "vllm",
        "XDG_CACHE_HOME": "xdg",
        "TMPDIR": "tmp",
    }
    for variable, child in paths.items():
        path = cache / child
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


def verify_dependencies():
    """Reject same-version packages installed from another source revision."""
    lock = json.loads(
        (Path(__file__).with_name("configs") / "kimi_k3_dependencies.json").read_text()
    )
    for name, item in lock["dependencies"].items():
        installed = distribution(item["distribution"])
        provenance = json.loads(installed.read_text("direct_url.json") or "{}")
        vcs = provenance.get("vcs_info", {})
        if vcs.get("vcs") != "git" or vcs.get("commit_id") != item["commit"]:
            raise RuntimeError(
                f"{name}: install the pinned Git source in requirements.txt"
            )
        package = item["import_package"]
        spec = find_spec(package)
        expected = Path(installed.locate_file(package + "/__init__.py")).resolve()
        if (
            spec is None
            or spec.origin is None
            or Path(spec.origin).resolve() != expected
        ):
            raise RuntimeError(
                f"{name}: import path shadows the pinned installed package"
            )
