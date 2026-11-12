# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PlatformFL capability, UUID and torch_device_fn delegation (E1).

Companion to test_cov_platform.py: device-probe-derived capability and UUID
branches plus the torch_device_fn delegation surface. Identity attributes are
mutated on the platform class and restored via the ``restore_identity``
fixture — the CONTRACT R2 sanctioned boundary for device probes.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms.interface import DeviceCapability

from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)
from vllm_fl.platform import PlatformFL


@pytest.fixture
def restore_identity(monkeypatch):
    """Snapshot vendor/device identity so tests can mutate it freely."""
    saved = (
        PlatformFL.vendor_name,
        PlatformFL.device_type,
        PlatformFL.device_name,
        PlatformFL.dist_backend,
        PlatformFL.torch_device_fn,
    )
    yield monkeypatch
    (
        PlatformFL.vendor_name,
        PlatformFL.device_type,
        PlatformFL.device_name,
        PlatformFL.dist_backend,
        PlatformFL.torch_device_fn,
    ) = saved


class TestCapabilityAndUuid:
    """Capability and UUID vendor branches (NVML-free ones)."""

    @pytest.mark.parametrize(
        ("vendor_name", "capability", "expected"),
        [
            ("nvidia", DeviceCapability(8, 9), True),
            ("nvidia", DeviceCapability(8, 0), False),
            ("nvidia", None, False),
            ("mthreads", DeviceCapability(3, 1), True),
            ("mthreads", DeviceCapability(3, 0), False),
            ("mthreads", None, False),
            ("gcu", DeviceCapability(4, 0), True),
            ("enflame", DeviceCapability(4, 0), True),
            ("kunlunxin", DeviceCapability(9, 9), False),
            ("hygon", None, False),
        ],
    )
    def test_supports_fp8_thresholds(self, restore_identity, vendor_name, capability, expected):
        """supports_fp8 enforces each vendor's minimum DeviceCapability."""
        PlatformFL.vendor_name = vendor_name
        if capability is None:
            PlatformFL.get_device_capability = classmethod(
                lambda cls, device_id=0: None
            )
        else:
            PlatformFL.get_device_capability = classmethod(
                lambda cls, device_id=0, cap=capability: cap
            )
        assert PlatformFL.supports_fp8() is expected

    def test_fp8_dtype_is_e4m3fn_for_nvidia(self, restore_identity):
        """Non-AMD vendors use the standard e4m3fn FP8 storage dtype."""
        PlatformFL.vendor_name = "nvidia"
        assert PlatformFL.is_fp8_fnuz() is False
        assert PlatformFL.fp8_dtype() == torch.float8_e4m3fn

    def test_get_device_capability_none_for_npu_ptpu_txda(self, restore_identity):
        """NPU/PTPU/txda deliberately report no CUDA-style capability."""
        PlatformFL.vendor_name = "ascend"
        for device_type in ("npu", "ptpu", "txda"):
            PlatformFL.device_type = device_type
            assert PlatformFL.get_device_capability() is None, device_type

    def test_get_device_uuid_branches(self, restore_identity, monkeypatch):
        """UUID synthesis covers iluvatar, NPU (with/without visibility) and others."""
        PlatformFL.vendor_name = "iluvatar"
        PlatformFL.device_type = "cuda"
        assert PlatformFL.get_device_uuid(3) == "ILUVATAR-3"

        PlatformFL.vendor_name = "ascend"
        PlatformFL.device_type = "npu"
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
        assert PlatformFL.get_device_uuid(2) == "NPU-2"

        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6")
        assert PlatformFL.get_device_uuid(1) == "NPU-5"

        PlatformFL.vendor_name = "tsingmicro"
        PlatformFL.device_type = "tsingmicro"
        assert PlatformFL.get_device_uuid(9) == "tsingmicro-9"

    def test_is_fully_connected_false_for_non_nvidia(self, restore_identity):
        """Non-NVIDIA/NVML platforms report no full NVLink connectivity."""
        PlatformFL.vendor_name = "amd"
        PlatformFL.device_type = "cuda"
        assert PlatformFL.is_fully_connected([0, 1]) is False

    def test_support_deep_gemm_false_off_cuda(self, restore_identity):
        """DeepGEMM stays off for non-CUDA device types."""
        PlatformFL.device_type = "npu"
        assert PlatformFL.support_deep_gemm() is False


class TestDeviceFnDelegation:
    """torch_device_fn delegation: memory usage, seeding, block copies."""

    def test_get_current_memory_usage_resets_then_reports(self, restore_identity):
        """Memory usage empties cache, resets peak stats, then reads the max."""
        calls = []

        class Recording:
            @staticmethod
            def empty_cache(*args):
                calls.append(("empty_cache", args))

            @staticmethod
            def reset_peak_memory_stats(*args):
                calls.append(("reset_peak", args))

            @staticmethod
            def max_memory_allocated(*args):
                calls.append(("max_mem", args))
                return 42.5

        PlatformFL.torch_device_fn = Recording
        assert PlatformFL.get_current_memory_usage(None) == 42.5
        assert [name for name, _ in calls] == [
            "empty_cache",
            "reset_peak",
            "max_mem",
        ]
        assert calls[1][1] == (None,)

    def test_set_device_and_empty_cache_delegate(self, restore_identity):
        """set_device and empty_cache forward verbatim to torch_device_fn."""
        calls = []

        class Recording:
            @staticmethod
            def set_device(*args):
                calls.append(("set_device", args))

            @staticmethod
            def empty_cache(*args):
                calls.append(("empty_cache", args))

        PlatformFL.torch_device_fn = Recording
        device = torch.device("cuda", 0)
        PlatformFL.set_device(device)
        assert calls == [("set_device", (device,))]
        PlatformFL.empty_cache()
        assert calls[-1] == ("empty_cache", ())

    def test_get_device_total_memory_and_num_compute_units(self, restore_identity):
        """Total memory and SM count come from device properties per index."""
        properties = {
            0: type("P0", (), {"total_memory": 40 * 1024**3, "multi_processor_count": 108})(),
            1: type("P1", (), {"total_memory": 80 * 1024**3, "multi_processor_count": 132})(),
        }
        fake = type(
            "Fn",
            (),
            {"get_device_properties": staticmethod(lambda i: properties[i])},
        )
        PlatformFL.torch_device_fn = fake
        assert PlatformFL.get_device_total_memory(0) == 40 * 1024**3
        assert PlatformFL.num_compute_units(1) == 132

    def test_get_device_name_returns_class_device_name(self):
        """get_device_name reports the probe-resolved device name."""
        assert PlatformFL.get_device_name(0) == PlatformFL.device_name

    def test_manual_seed_all_direct_path(self, restore_identity):
        """A device fn exposing manual_seed_all is used directly."""
        calls = []

        class WithManualSeedAll:
            @staticmethod
            def manual_seed_all(seed):
                calls.append(("direct", seed))

        PlatformFL.torch_device_fn = WithManualSeedAll
        PlatformFL.manual_seed_all(1234)
        assert calls == [("direct", 1234)]

    def test_manual_seed_all_fallback_seeds_each_generator(self, restore_identity, monkeypatch):
        """Without manual_seed_all, torch + every per-device generator is seeded."""

        class Generator:
            def __init__(self, index):
                self.index = index

            def manual_seed(self, seed):
                calls.append(("gen", self.index, seed))

        calls = []
        generators = {i: Generator(i) for i in range(3)}

        class PtpuLike:
            device_count = staticmethod(lambda: 3)

            @staticmethod
            def _get_or_create_default_generator(index):
                return generators[index]

        PlatformFL.torch_device_fn = PtpuLike
        monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("torch", seed)))
        PlatformFL.manual_seed_all(77)
        assert calls == [
            ("torch", 77),
            ("gen", 0, 77),
            ("gen", 1, 77),
            ("gen", 2, 77),
        ]

    def test_manual_seed_all_bare_device_fn_seeds_torch_only(self, restore_identity, monkeypatch):
        """A device fn without device_count seeds only the global generator."""
        calls = []
        PlatformFL.torch_device_fn = type("Bare", (), {})
        monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("torch", seed)))
        PlatformFL.manual_seed_all(88)
        assert calls == [("torch", 88)]

    def test_insert_blocks_to_device_copies_selected_blocks(self):
        """Device insertion copies exactly the src blocks into dst slots."""
        src = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        dst = torch.zeros(2, 5, 4)
        PlatformFL.insert_blocks_to_device(
            src, dst, torch.tensor([0, 2]), torch.tensor([1, 4])
        )
        assert torch.equal(dst[:, 1], src[:, 0])
        assert torch.equal(dst[:, 4], src[:, 2])
        assert torch.equal(dst[:, 0], torch.zeros(2, 4))

    def test_swap_out_blocks_to_host_copies_selected_blocks(self):
        """Host swap-out copies exactly the src blocks to the CPU tensor."""
        src = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        host = torch.zeros(2, 2, 4)
        PlatformFL.swap_out_blocks_to_host(
            src, host, torch.tensor([1, 2]), torch.tensor([0, 1])
        )
        assert torch.equal(host[:, 0], src[:, 1])
        assert torch.equal(host[:, 1], src[:, 2])
