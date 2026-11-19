"""Non-CUDA prefill is explicit and retains the normal dtype/dimension checks."""

from types import SimpleNamespace

import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

from vllm_fl.attention.kimi_k3.backend import K3PrefillBackend


def configuration(dtype):
    return SimpleNamespace(dtype=dtype, mla_dimensions=None)


def test_native_backend_retains_capability_requirement():
    cfg = configuration(torch.bfloat16)
    assert MLAPrefillBackend.validate_configuration(DeviceCapability(9, 0), cfg) == []
    assert MLAPrefillBackend.validate_configuration(None, cfg) == [
        "device capability not available"
    ]


def test_k3_accepts_missing_cuda_capability_without_bypassing_dtype_validation():
    assert (
        K3PrefillBackend.validate_configuration(None, configuration(torch.bfloat16))
        == []
    )
    invalid = K3PrefillBackend.validate_configuration(None, configuration(torch.int32))
    assert invalid == ["dtype torch.int32 not supported"]
