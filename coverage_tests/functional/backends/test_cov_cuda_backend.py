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

"""Single-GPU coverage for the vendor.cuda backend registration surface.

The CUDA backend delegates every registered operator to an ``impl`` module.
These tests pin the identity of that delegation: each op resolves to
``vendor.cuda`` under an explicit per-op policy, the resolved callable is the
bound ``CudaBackend`` method, and the class-level availability cache is
populated from the live platform probe (with the sanctioned cache reset so
the negative probes recompute instead of reading a stale True).
"""

import pytest
import torch

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from vllm_fl.dispatch import get_default_manager
from vllm_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

pytestmark = [pytest.mark.gpu]

# Every op registered by vllm_fl/dispatch/backends/vendor/cuda/register_ops.py
# with a CudaBackend delegate method.
DELEGATE_OPS = [
    "silu_and_mul",
    "gelu_and_mul",
    "rms_norm",
    "rotary_embedding",
    "apply_rotary_emb",
    "attention_backend",
    "moe_align_block_size",
    "moe_sum",
    "topk_softmax",
    "invoke_fused_moe_triton_kernel",
    "grouped_topk",
]


@pytest.fixture(autouse=True)
def _reset_backend_cache():
    """Reset the class-level availability cache before and after each test.

    CudaBackend._available is a module-level cache; resetting it to the
    initial value (None) around tests that probe availability is sanctioned
    cache hygiene, not behavior modification.
    """
    CudaBackend._available = None
    yield
    CudaBackend._available = None


class TestCudaBackendIdentity:
    """Backend name/vendor properties and delegate method wiring."""

    def test_name_and_vendor_properties(self):
        """The backend reports its name and vendor."""
        backend = CudaBackend()
        assert backend.name == "cuda"
        assert backend.vendor == "nvidia"

    def test_is_available_true_on_nvidia_gpu(self, device):
        """On a live NVIDIA device the probe returns True and memoizes it."""
        assert device.type == "cuda"
        assert CudaBackend().is_available() is True
        # The probe result is memoized after the first successful check.
        assert CudaBackend._available is True

    def test_is_available_false_without_cuda_runtime(self, monkeypatch):
        """torch.cuda.is_available() False short-circuits availability."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert CudaBackend().is_available() is False
        assert CudaBackend._available is False

    def test_is_available_false_when_no_visible_device(self, monkeypatch):
        """Zero visible devices fails availability."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert CudaBackend().is_available() is False
        assert CudaBackend._available is False

    def test_is_available_false_for_non_nvidia_platform(self, monkeypatch):
        """A non-NVIDIA platform name fails availability."""
        import vllm.platforms as platforms_module

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(
            platforms_module,
            "current_platform",
            type("OtherPlatform", (), {"device_name": "ascend"})(),
        )
        assert CudaBackend().is_available() is False
        assert CudaBackend._available is False

    def test_is_available_false_when_platform_probe_raises(self, monkeypatch):
        """A raising platform probe is caught and reported unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

        import vllm.platforms as platforms_module

        class _RaisingPlatform:
            @property
            def device_name(self):
                raise RuntimeError("platform probe failure")

        monkeypatch.setattr(platforms_module, "current_platform", _RaisingPlatform())
        assert CudaBackend().is_available() is False
        assert CudaBackend._available is False


class TestCudaBackendDelegates:
    """Every op resolves to the bound CudaBackend delegate under policy."""

    @pytest.mark.parametrize("op", DELEGATE_OPS)
    def test_op_selects_vendor_cuda_impl(self, op):
        """Each registered op resolves to the vendor.cuda bound delegate."""
        policy = SelectionPolicy.from_dict(per_op_order={op: ["vendor:cuda"]})
        with policy_context(policy):
            manager = get_default_manager()
            assert manager.get_selected_impl_id(op) == "vendor.cuda"
            # OpManager.resolve returns the selected implementation callable
            # directly: the availability-binding wrapper around the bound
            # CudaBackend delegate method.
            fn = manager.resolve(op)
            assert callable(fn)
            assert getattr(fn, "__name__", None) == op
            delegate = getattr(CudaBackend(), op)
            wrapped = getattr(fn, "__wrapped__", None)
            assert wrapped is not None
            assert wrapped.__func__ is delegate.__func__

    @pytest.mark.parametrize("op", DELEGATE_OPS)
    def test_delegate_availability_binding_is_live(self, op):
        """The resolved wrapper exposes a live availability binding."""
        policy = SelectionPolicy.from_dict(per_op_order={op: ["vendor:cuda"]})
        with policy_context(policy):
            fn = get_default_manager().resolve(op)
            assert fn._is_available() is True

    def test_attention_backend_paths(self):
        """attention_backend maps flags to the expected backend paths."""
        backend = CudaBackend()
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        default_path = backend.attention_backend()
        assert default_path == AttentionBackendEnum.FLASH_ATTN.get_path()
        mla_path = backend.attention_backend(use_mla=True)
        assert mla_path == AttentionBackendEnum.FLASHMLA.get_path()
        sparse_path = backend.attention_backend(use_mla=True, use_sparse=True)
        assert sparse_path == AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
        # use_sparse without use_mla keeps the FLASH_ATTN default.
        plain_sparse = backend.attention_backend(use_sparse=True)
        assert plain_sparse == AttentionBackendEnum.FLASH_ATTN.get_path()

    def test_attention_backend_op_routes_to_delegate(self):
        """The routed attention_backend op returns the delegate's path."""

        op = "attention_backend"
        policy = SelectionPolicy.from_dict(per_op_order={op: ["vendor:cuda"]})
        with policy_context(policy):
            fn = get_default_manager().resolve(op)
            assert fn(False, False).endswith("FlashAttentionBackend")


class TestCudaOpRegistrationSurface:
    """The vendor.cuda impl ids exist in the shared default registry."""

    @pytest.mark.parametrize("op", DELEGATE_OPS)
    def test_impl_registered_with_vendor_kind(self, op):
        """Each op has exactly one vendor.cuda impl of VENDOR kind."""
        manager = get_default_manager()
        manager.ensure_initialized()
        impls = manager._registry.get_implementations(op)
        cuda_impls = [i for i in impls if i.impl_id == "vendor.cuda"]
        assert len(cuda_impls) == 1
        impl = cuda_impls[0]
        from vllm_fl.dispatch.types import BackendImplKind

        assert impl.kind == BackendImplKind.VENDOR
        assert impl.vendor == "cuda"
