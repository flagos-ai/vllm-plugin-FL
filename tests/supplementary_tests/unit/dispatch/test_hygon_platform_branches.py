# Copyright 2025 BAAI. All rights reserved.

"""Hygon branches of the shared platform / startup code.

Hygon DCUs are a CUDA-shaped platform that reports ``is_cuda() == False``:
``device_type`` is ``"cuda"`` (HCU goes through the torch.cuda API), yet the
vendor must stay out of every CUDA-alike path upstream. These checks pin the
hygon-specific decisions spread through the platform class, the MoE priority
list and the vendor startup hook, against non-hygon contrasts.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_fl
import vllm_fl.platform as platform_module
import vllm_fl.utils as utils_module
from vllm_fl.ops.fused_moe import fused_moe_utils

pytestmark = pytest.mark.supplementary


class TestIsCudaAlike:
    """``PlatformFL.is_cuda_alike`` must exclude the CUDA-shaped vendors."""

    def _is_cuda_alike(self, vendor_name, device_type):
        # ``is_cuda_alike`` is an instance method that reads only
        # ``self.vendor_name`` / ``self.device_type``; call it unbound with a
        # fake self so the real (already resolved) PlatformFL is not touched.
        return platform_module.PlatformFL.is_cuda_alike(
            SimpleNamespace(vendor_name=vendor_name, device_type=device_type)
        )

    @pytest.mark.platforms("hygon")
    def test_hygon_is_not_cuda_alike_despite_cuda_device_type(self):
        # device_type is "cuda" (HCU goes through the CUDA-shaped API) yet the
        # vendor must stay out of upstream's CUDA-alike paths.
        assert self._is_cuda_alike("hygon", "cuda") is False

    @pytest.mark.platforms("hygon")
    def test_iluvatar_is_not_cuda_alike(self):
        assert self._is_cuda_alike("iluvatar", "cuda") is False

    @pytest.mark.platforms("hygon")
    def test_nvidia_cuda_is_cuda_alike(self):
        assert self._is_cuda_alike("nvidia", "cuda") is True

    @pytest.mark.platforms("hygon")
    def test_musa_device_type_is_cuda_alike(self):
        assert self._is_cuda_alike("mthreads", "musa") is True


class TestSupportStaticGraphMode:
    """Hygon DCUs can capture static graphs; the vendor list says so."""

    @pytest.mark.platforms("hygon")
    def test_hygon_supports_static_graph_mode(self, monkeypatch):
        monkeypatch.setattr(platform_module.PlatformFL, "vendor_name", "hygon")
        assert platform_module.PlatformFL.support_static_graph_mode() is True

    @pytest.mark.platforms("hygon")
    def test_nvidia_supports_static_graph_mode(self, monkeypatch):
        monkeypatch.setattr(platform_module.PlatformFL, "vendor_name", "nvidia")
        assert platform_module.PlatformFL.support_static_graph_mode() is True

    @pytest.mark.platforms("hygon")
    def test_a_vendor_outside_the_list_does_not(self, monkeypatch):
        monkeypatch.setattr(platform_module.PlatformFL, "vendor_name", "sunrise")
        assert platform_module.PlatformFL.support_static_graph_mode() is False


class TestUseCustomAllreduce:
    """Hygon (like AMD/ROCm) must leave reductions to RCCL."""

    @pytest.mark.platforms("hygon")
    def test_hygon_does_not_use_custom_allreduce(self, monkeypatch):
        monkeypatch.setattr(platform_module.PlatformFL, "vendor_name", "hygon")
        assert platform_module.PlatformFL.use_custom_allreduce() is False

    @pytest.mark.platforms("hygon")
    def test_nvidia_nccl_uses_custom_allreduce(self, monkeypatch):
        monkeypatch.setattr(platform_module.PlatformFL, "vendor_name", "nvidia")
        monkeypatch.setattr(platform_module.PlatformFL, "dist_backend", "nccl")
        assert platform_module.PlatformFL.use_custom_allreduce() is True

    @pytest.mark.platforms("hygon")
    def test_flagcx_also_disables_custom_allreduce(self, monkeypatch):
        # vLLM's custom all-reduce assumes NCCL; a flagcx backend must not
        # enable it even for a vendor on the allowed list.
        monkeypatch.setattr(platform_module.PlatformFL, "vendor_name", "nvidia")
        monkeypatch.setattr(platform_module.PlatformFL, "dist_backend", "flagcx")
        assert platform_module.PlatformFL.use_custom_allreduce() is False


class TestMoePriorityBackends:
    """Hygon falls into the Triton-only MoE branch, behind CUDA."""

    @pytest.mark.platforms("hygon")
    def test_hygon_gets_exactly_the_triton_pair_in_order(self, monkeypatch):
        # is_cuda()/is_rocm()/is_xpu()/is_cpu() all answer False for hygon,
        # so without the vendor branch the function would fall through with
        # no assignment (UnboundLocalError on read). Pin the list and order.
        monkeypatch.setattr(
            fused_moe_utils,
            "current_platform",
            SimpleNamespace(
                is_rocm=lambda: False,
                is_cuda=lambda: False,
                is_xpu=lambda: False,
                is_cpu=lambda: False,
                vendor_name="hygon",
            ),
        )
        backends = fused_moe_utils._get_priority_backends(SimpleNamespace())
        assert backends == [
            fused_moe_utils.UnquantizedMoeBackend.TRITON,
            fused_moe_utils.UnquantizedMoeBackend.BATCHED_TRITON,
        ]

    @pytest.mark.platforms("hygon")
    def test_cuda_like_platform_takes_the_cuda_branch_first(self, monkeypatch):
        # Contrast: a CUDA platform (nvidia; HCU is excluded by is_cuda())
        # resolves in the is_cuda() branch and never reaches the hygon
        # fallback, so the FlashInfer entries stay ahead of the Triton pair.
        monkeypatch.setattr(
            fused_moe_utils,
            "current_platform",
            SimpleNamespace(
                is_rocm=lambda: False,
                is_cuda=lambda: True,
                is_xpu=lambda: False,
                is_cpu=lambda: False,
                is_device_capability_family=lambda family: False,
                vendor_name="nvidia",
            ),
        )
        config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(dp_size=1),
            activation=object(),
        )
        backends = fused_moe_utils._get_priority_backends(config)
        assert backends == [
            fused_moe_utils.UnquantizedMoeBackend.FLASHINFER_TRTLLM,
            fused_moe_utils.UnquantizedMoeBackend.FLASHINFER_CUTLASS,
            fused_moe_utils.UnquantizedMoeBackend.TRITON,
            fused_moe_utils.UnquantizedMoeBackend.BATCHED_TRITON,
        ]


class TestInitVendorDeviceHygonBranch:
    """``_init_vendor_device`` routes hygon to ``apply_hygon_patches``."""

    HYGON_PATCH_NAME = "vllm_fl.dispatch.backends.vendor.hygon.patch"

    def _run_with(self, vendor_name, patch_entry):
        """Run ``_init_vendor_device`` under a stubbed DeviceInfo and a
        seeded sys.modules entry for the hygon patch module, restoring both.

        ``patch_entry`` is what ``sys.modules[HYGON_PATCH_NAME]`` becomes for
        the call (a fake module, or ``None`` to poison the from-import).
        Every restore is manual try/finally so no stub survives the test.
        """
        original_module = sys.modules.get(self.HYGON_PATCH_NAME)
        original_device_info = utils_module.DeviceInfo
        sys.modules[self.HYGON_PATCH_NAME] = patch_entry
        utils_module.DeviceInfo = lambda: SimpleNamespace(vendor_name=vendor_name)
        try:
            vllm_fl._init_vendor_device()
        finally:
            if original_module is None:
                sys.modules.pop(self.HYGON_PATCH_NAME, None)
            else:
                sys.modules[self.HYGON_PATCH_NAME] = original_module
            utils_module.DeviceInfo = original_device_info

    @pytest.mark.platforms("hygon")
    def test_hygon_routes_to_apply_hygon_patches(self):
        # Seed a fake patch module so the from-import inside the branch binds
        # the Mock and the REAL patch set (which rebinds torch.accelerator
        # and vllm.utils.platform_utils) never runs.
        apply_mock = Mock(return_value=None)
        fake_patch = ModuleType(self.HYGON_PATCH_NAME)
        fake_patch.apply_hygon_patches = apply_mock

        self._run_with("hygon", fake_patch)

        apply_mock.assert_called_once_with()

    @pytest.mark.platforms("hygon")
    def test_non_hygon_vendor_never_imports_the_hygon_patch(self):
        # A None entry in sys.modules makes any from-import of that module
        # raise ImportError: if the hygon branch were taken for nvidia, this
        # test would fail instead of silently passing.
        self._run_with("nvidia", None)
