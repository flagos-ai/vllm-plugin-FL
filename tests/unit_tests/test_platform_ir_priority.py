# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from vllm_fl.platform import PlatformFL


def test_nvidia_hopper_supports_pdl():
    device = SimpleNamespace(get_device_capability=lambda: (9, 0))

    with (
        patch.object(PlatformFL, "vendor_name", "nvidia"),
        patch.object(PlatformFL, "torch_device_fn", device),
    ):
        assert PlatformFL.is_arch_support_pdl()
