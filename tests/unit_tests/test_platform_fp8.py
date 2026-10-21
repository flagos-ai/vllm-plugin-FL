# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.platforms.interface import DeviceCapability

from vllm_fl.platform import PlatformFL


@pytest.mark.parametrize(
    "capability,supported",
    [
        (DeviceCapability(8, 0), False),
        (DeviceCapability(8, 6), False),
        (DeviceCapability(8, 9), True),
        (DeviceCapability(9, 0), True),
        (DeviceCapability(10, 0), True),
        (None, False),
    ],
)
def test_nvidia_fp8_support_requires_device_capability(
    monkeypatch, capability, supported
):
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    monkeypatch.setattr(PlatformFL, "get_device_capability", lambda **kw: capability)

    assert PlatformFL.supports_fp8() is supported
    assert PlatformFL().supports_fp8() is supported


@pytest.mark.parametrize("vendor", ["metax", "ascend", "kunlunxin", "mthreads"])
def test_non_nvidia_fp8_check_does_not_query_cuda(monkeypatch, vendor):
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(
        PlatformFL,
        "get_device_capability",
        lambda **kw: pytest.fail("non-NVIDIA FP8 check reached CUDA"),
    )

    assert not PlatformFL.supports_fp8()
