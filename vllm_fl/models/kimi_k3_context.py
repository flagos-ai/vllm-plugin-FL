"""Resolve execution policy at model/worker construction, never in GPU kernels."""

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm_fl.kimi_k3_config import K3ExecutionConfig
from vllm_fl.patches.kimi_k3.profile import is_kimi_k3_profile


def execution_config(config=None):
    if config is None:
        config = get_current_vllm_config()
    policy = K3ExecutionConfig.from_vllm(
        config, vendor=getattr(current_platform, "vendor_name", None)
    )
    if policy.enabled != is_kimi_k3_profile():
        raise ValueError("K3 model and FL dispatch profile must be selected together")
    return policy
