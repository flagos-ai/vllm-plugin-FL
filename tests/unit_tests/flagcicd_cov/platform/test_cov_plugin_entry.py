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

"""vllm_fl.__init__ plugin entry points and envs lazy variables (E1).

Covers the plugin registration entry (register), lazy attribute access,
the CPU platform hook, transformers compat, custom-op schema fallbacks and
the vllm_fl.envs lazy environment-variable surface. On hosts without the
vLLM CUDA extension (the Windows collection host) the fallback op registry
is the production path and is asserted directly.
"""

from __future__ import annotations

import os

import pytest
import torch

from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)

import vllm_fl
import vllm_fl.envs as fl_envs


class TestModuleEntry:
    """Top-level module surface and lazy attribute access."""

    def test_version_and_entry_points_exposed(self):
        """The plugin exposes its version and registration callables."""
        assert hasattr(vllm_fl, "version")
        assert callable(vllm_fl.register)
        assert callable(vllm_fl.register_model)
        assert callable(vllm_fl.register_quant_linear)
        assert callable(vllm_fl.register_router)

    def test_lazy_distributed_attribute_imports_module(self):
        """__getattr__('distributed') lazily imports and caches the submodule."""
        module = vllm_fl.__getattr__("distributed")
        assert module.__name__ == "vllm_fl.distributed"
        # Cached in globals: a second access returns the same module object.
        assert vllm_fl.__getattr__("distributed") is module

    def test_unknown_attribute_raises(self):
        """Any other attribute name raises a standard AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute 'no_such_attr'"):
            vllm_fl.__getattr__("no_such_attr")
        assert not hasattr(vllm_fl, "no_such_attr")

    def test_float4_sentinel_injected_on_torch(self):
        """CPU torch builds get the uint8 sentinel vllm.ir.tolerances needs."""
        assert hasattr(torch, "float4_e2m1fn_x2")

    def test_arm_cpu_platform_reported_per_host_architecture(self):
        """The CPU platform hook reports a platform only on AArch64 hosts."""
        machine = __import__("platform").machine().lower()
        if machine in {"aarch64", "arm64"}:
            # x86-only CI images never take this branch; on ARM the hook
            # returns vllm's CPU platform object (a platform, never None).
            assert vllm_fl._arm_cpu_platform() is not None
        else:
            assert vllm_fl._arm_cpu_platform() is None


class TestRegister:
    """The plugin register() entry point."""

    def test_register_returns_platform_path(self, monkeypatch):
        """register() resolves to the FL platform class path."""
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
        assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"

    def test_register_defaults_multiproc_method_to_spawn(self, monkeypatch):
        """An unset VLLM_WORKER_MULTIPROC_METHOD becomes spawn."""
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
        vllm_fl.register()
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

    def test_register_preserves_explicit_multiproc_method(self, monkeypatch):
        """An operator-provided multiproc method is never overwritten."""
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
        vllm_fl.register()
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


class TestCompatPatches:
    """Runtime compatibility patches applied by the plugin."""

    def test_patch_transformers_compat_defines_allowed_layer_types(self):
        """ALLOWED_LAYER_TYPES is present after the compat patch (idempotent)."""
        import transformers.configuration_utils as configuration

        vllm_fl._patch_transformers_compat()
        assert hasattr(configuration, "ALLOWED_LAYER_TYPES")
        vllm_fl._patch_transformers_compat()
        assert hasattr(configuration, "ALLOWED_LAYER_TYPES")

    def test_patch_flash_attn_import_installs_stub_when_unavailable(self):
        """Without CUDA FA extensions a non-CUDA fallback package is installed."""
        import sys

        vllm_fl._patch_flash_attn_import()
        module = sys.modules["vllm.vllm_flash_attn"]
        assert module.FA2_AVAILABLE is False
        assert module.FA3_AVAILABLE is False
        assert module.is_fa_version_supported() is False
        assert "not available" in module.fa_version_unsupported_reason()

    def test_init_vendor_device_tolerates_missing_flashinfer(self):
        """_init_vendor_device treats FlashInfer as optional on every host."""
        vllm_fl._init_vendor_device()
        # The vendor device init completed without acquiring any device.
        assert isinstance(vllm_fl.__dict__.get("_INITIALIZED", None), (bool, type(None)))

    def test_flagcx_backend_disabled_without_package(self, monkeypatch):
        """Without flagcx.api and FLAGCX_PATH the communicator backend is off."""
        import vllm_fl.platform as fl_platform

        monkeypatch.delenv("FLAGCX_PATH", raising=False)
        assert fl_platform._flagcx_backend_enabled() is False


class TestCustomOpsFallback:
    """The _C op-schema fallback registry used without vLLM CUDA extensions."""

    @pytest.fixture(autouse=True)
    def _register_ops(self):
        from vllm_fl.ops._C_ops_registry import register_op_schemas

        register_op_schemas()

    @pytest.mark.parametrize(
        ("op", "capability", "expected"),
        [
            ("cutlass_scaled_mm_supports_fp8", 89, True),
            ("cutlass_scaled_mm_supports_fp8", 80, False),
            ("cutlass_scaled_mm_supports_block_fp8", 100, True),
            ("cutlass_scaled_mm_supports_block_fp8", 99, False),
            ("cutlass_group_gemm_supported", 90, True),
            ("cutlass_scaled_mm_supports_fp4", 100, True),
        ],
    )
    def test_capability_query_ops(self, op, capability, expected):
        """Each capability query op compares against its arch threshold."""
        assert bool(getattr(torch.ops._C, op)(capability)) is expected

    def test_weak_ref_tensor_returns_same_object(self):
        """weak_ref_tensor keeps the tensor identity/storage contract."""
        tensor = torch.arange(6).reshape(2, 3)
        assert torch.ops._C.weak_ref_tensor(tensor) is tensor

    def test_get_cuda_view_from_cpu_tensor_passthrough(self):
        """The CPU view helper passes the tensor through untouched."""
        tensor = torch.ones(2, 2)
        assert torch.ops._C.get_cuda_view_from_cpu_tensor(tensor) is tensor

    def test_repetition_penalty_fallback_math(self):
        """The pure-torch repetition penalty divides/penalizes masked logits."""
        from vllm_fl.ops._C_ops_registry import _apply_repetition_penalties_impl

        logits = torch.tensor([[2.0, -2.0, 0.5]])
        prompt_mask = torch.zeros(1, 3, dtype=torch.bool)
        prompt_mask[0, 0] = True
        output_mask = torch.zeros(1, 3, dtype=torch.bool)
        output_mask[0, 1] = True
        _apply_repetition_penalties_impl(
            logits, prompt_mask, output_mask, torch.tensor([2.0])
        )
        # Positive logit 2.0 with penalty 2.0 -> 1.0; negative -2.0 -> -4.0.
        assert logits.tolist() == [[1.0, -4.0, 0.5]]

    def test_registry_registration_is_idempotent(self):
        """Re-running register_op_schemas does not raise or duplicate."""
        from vllm_fl.ops._C_ops_registry import register_op_schemas

        register_op_schemas()
        register_op_schemas()
        assert torch.ops._C.weak_ref_tensor is not None


class TestConnectorAndRegistrationEntries:
    """Registration helpers driven by the current platform and env gates."""

    def test_register_flagcx_connector_aliases(self):
        """Both FlagCX connector aliases land in the factory registry."""
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )

        vllm_fl._register_flagcx_connector()
        for alias in ("FlagCXConnector", "FlagcxConnector"):
            assert callable(KVConnectorFactory._registry[alias])
        first = KVConnectorFactory._registry["FlagCXConnector"]
        vllm_fl._register_flagcx_connector()
        assert KVConnectorFactory._registry["FlagCXConnector"] is first

    def test_register_router_early_returns_when_oot_disabled(self, monkeypatch):
        """With OOT registration off, no fused-MoE import happens."""
        monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "0")
        import sys

        before = set(sys.modules)
        vllm_fl.register_router()
        assert "vllm_fl.ops.fused_moe.router" not in set(sys.modules) - before

    def test_register_quant_linear_runs_on_non_restricted_platforms(self):
        """On CPU/CUDA device types the OOT quant kernel registration runs."""
        from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
        from vllm.platforms import PlatformEnum

        vllm_fl.register_quant_linear()
        assert PlatformEnum.OOT in _POSSIBLE_KERNELS
        assert _POSSIBLE_KERNELS[PlatformEnum.OOT] is not None

    def test_register_gdn_packed_decode_patch_capability_check(self):
        """The GDN patch reports a boolean without raising either way."""
        assert isinstance(vllm_fl._register_gdn_packed_decode_patch(), bool)


class TestEnvsModule:
    """vllm_fl.envs lazy environment-variable surface."""

    ENV_NAMES = (
        "VLLM_FL_PREFER_ENABLED",
        "USE_FLAGGEMS",
        "VLLM_FL_USE_FLAGGEMS_VLLM",
        "FLAGGEMS_ENABLE_OPLIST_PATH",
    )

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for name in self.ENV_NAMES:
            monkeypatch.delenv(name, raising=False)

    def test_dir_lists_exactly_the_lazy_variables(self):
        """dir() enumerates the four lazy environment variables."""
        assert sorted(dir(fl_envs)) == sorted(self.ENV_NAMES)

    def test_prefer_enabled_defaults_true(self):
        """VLLM_FL_PREFER_ENABLED defaults to enabled."""
        assert fl_envs.VLLM_FL_PREFER_ENABLED is True

    def test_use_flaggems_defaults_true(self):
        """USE_FLAGGEMS mirrors use_flaggems' default."""
        assert fl_envs.USE_FLAGGEMS is True

    def test_use_flaggems_vllm_defaults_false(self):
        """VLLM_FL_USE_FLAGGEMS_VLLM defaults to disabled ('0')."""
        assert fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM is False

    def test_flaggems_oplist_default_path(self):
        """FLAGGEMS_ENABLE_OPLIST_PATH defaults to the /tmp oplist file."""
        assert (
            fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH == "/tmp/flaggems_enable_oplist.txt"
        )

    def test_env_values_reflect_current_environment(self, monkeypatch):
        """Lazy attributes re-read the environment on every access."""
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
        assert fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM is True
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
        assert fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM is False

        monkeypatch.setenv("FLAGGEMS_ENABLE_OPLIST_PATH", "/custom/path.txt")
        assert fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH == "/custom/path.txt"

    def test_is_set_tracks_environment_only(self, monkeypatch):
        """is_set is True exactly when the variable is in os.environ."""
        assert fl_envs.is_set("VLLM_FL_PREFER_ENABLED") is False
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "true")
        assert fl_envs.is_set("VLLM_FL_PREFER_ENABLED") is True

    def test_is_set_rejects_unknown_names(self):
        """An unknown name raises AttributeError, matching __getattr__."""
        with pytest.raises(AttributeError, match="NO_SUCH_VAR"):
            fl_envs.is_set("NO_SUCH_VAR")
        assert not hasattr(fl_envs, "NO_SUCH_VAR")

    def test_getattr_rejects_unknown_names(self):
        """Attribute access on an unknown name raises AttributeError."""
        with pytest.raises(AttributeError, match="NO_SUCH_VAR"):
            fl_envs.NO_SUCH_VAR
        assert not hasattr(fl_envs, "NO_SUCH_VAR")
