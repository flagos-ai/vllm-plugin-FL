# SPDX-License-Identifier: Apache-2.0
"""Defaults for the CosyVoice3-0.5B speech LLM export on a single S5000."""

import multiprocessing
import sys

import torch
import vllm
from vllm import envs
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.logger import init_logger

from vllm_fl.utils import use_flaggems_op

logger = init_logger(__name__)
PROFILE = "cosyvoice3-0.5b"
_EXPORT_CONFIG = {
    "model_type": "qwen2",
    "vocab_size": 6761,
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
}


def _profile_options(config):
    options = getattr(config, "additional_config", None)
    return options if isinstance(options, dict) else {}


def _supports_profile(config, platform, device_id=0):
    model = getattr(config, "model_config", None)
    hf_config = getattr(model, "hf_config", None)
    profile = _profile_options(config).get(
        "flagos_profile", getattr(hf_config, "flagos_profile", None)
    )
    if profile != PROFILE:
        return False
    parallel = config.parallel_config
    if (
        platform.device_type != "musa"
        or platform.vendor_name != "mthreads"
        or (
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.data_parallel_size,
        ) != (1, 1, 1)
        or getattr(model, "dtype", None) != torch.bfloat16
        or not getattr(model, "enable_prompt_embeds", False)
        or getattr(model, "quantization", None) is not None
        or tuple(getattr(hf_config, "architectures", None) or ()) != ("Qwen2ForCausalLM",)
        or any(
            getattr(hf_config, key, None) != value
            for key, value in _EXPORT_CONFIG.items()
        )
    ):
        logger.info("CosyVoice3 profile skipped for this model or parallel configuration")
        return False
    # Limit the defaults to the vLLM/MUSA stack validated for mixed capture.
    if (
        vllm.__version__.split("+", 1)[0] != "0.28.0"
        or torch.__version__ != "2.11.0.post2+musa5.2.0"
        or getattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
        or getattr(config, "use_v2_model_runner", False)
    ):
        logger.info("CosyVoice3 profile skipped on an unvalidated vLLM/MUSA runtime")
        return False
    try:
        return "S5000" in platform.torch_device_fn.get_device_name(device_id).split()
    except (AttributeError, RuntimeError):
        logger.info("CosyVoice3 profile skipped because the MUSA device is unavailable")
        return False


def supports_mixed_graph(config, platform):
    """Let the runner check the final Attention backend and capture support."""
    if not _supports_profile(config, platform):
        return False
    backend = getattr(getattr(config, "attention_config", None), "backend", None)
    return (
        not config.model_config.enforce_eager
        and config.compilation_config.mode != CompilationMode.NONE
        and not getattr(config.parallel_config, "enable_dbo", False)
        and config.speculative_config is None
        and config.kv_transfer_config is None
        and not getattr(envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False)
        and (backend is None or getattr(backend, "name", backend) == "TRITON_ATTN")
    )


def apply_graph_default(config, platform):
    compilation = config.compilation_config
    if compilation.cudagraph_mode is None and supports_mixed_graph(config, platform):
        compilation.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        logger.info("CosyVoice3 S5000 default graph mode is FULL_AND_PIECEWISE")


def configure_sampler(config, platform, device_id, sampler):
    """Bind the fused sampler for this model runner only."""
    if (
        not _supports_profile(config, platform, device_id)
        or _profile_options(config).get("flagos_fused_topk", True) is not True
        or not use_flaggems_op("topk_sample_from_logits")
    ):
        return False
    # The current operator wheel sets a process-wide compiler default on first
    # import. vLLM 0.28 starts a private EngineCore process for this TP=DP=1
    # configuration. Shared in-process engines can reuse a loaded provider.
    if (
        sys.modules.get("flaggems_vllm") is None
        and multiprocessing.current_process().name != "EngineCore"
    ):
        logger.info("CosyVoice3 will use native sampling in the shared engine process")
        return False
    from vllm_fl.dispatch.backends.vendor.musa.impl.topk_sampling import (
        enable_fused_topk_sampling,
    )

    return enable_fused_topk_sampling(sampler.topk_topp_sampler)
