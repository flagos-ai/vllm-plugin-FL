# Copyright (c) 2026 BAAI. All rights reserved.
"""Platform guards for the release checks validated on NVIDIA CUDA."""

import pytest
import torch

CUDA_RELEASE_TESTS = {
    "test_vllm_package.py",
    "test_vllm_switch.py",
}


@pytest.fixture(autouse=True)
def require_nvidia_release_runtime(request):
    if request.node.path.name not in CUDA_RELEASE_TESTS:
        return
    platform = request.config.getoption("--platform")
    if platform and platform != "cuda":
        pytest.skip("These fused release checks require NVIDIA CUDA")
    from vllm.platforms import current_platform

    if (
        getattr(current_platform, "vendor_name", None) != "nvidia"
        or not torch.cuda.is_available()
    ):
        pytest.skip("These fused release checks require NVIDIA CUDA")
