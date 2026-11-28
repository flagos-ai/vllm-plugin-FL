# Copyright 2025 BAAI. All rights reserved.

"""Hygon branches of the shared platform / startup code.

Hygon DCUs are a CUDA-shaped platform that reports ``is_cuda() == False``:
``device_type`` is ``"cuda"`` (HCU goes through the torch.cuda API), yet the
vendor must stay out of every CUDA-alike path upstream. These checks pin the
hygon-specific decisions spread through the platform class, the MoE priority
list and the vendor startup hook, against non-hygon contrasts.

The last class exercises the defensive paths of
``vllm_fl/dispatch/backends/vendor/hygon/patch.py`` that the baseline
``tests/unit_tests/test_hygon_patch_rollback.py`` cannot reach: the two
idempotence early-returns guarded by marker attributes (patch.py lines 91
and 148) and the hostile-setattr rollback inside
``_install_accelerator_missing_attrs`` (lines 166-168). In production the
first ``apply_hygon_patches`` call goes through ``lru_cache(maxsize=1)`` and
never re-enters the function body, so those guards only fire when the
cache is cleared (test fixtures, or a failed application retried in the
same process). They are load-bearing as a second, cache-independent
idempotence layer — an ``lru_cache`` removal must not turn a second
``apply_hygon_patches()`` into a partially re-patched process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
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
        # Two independent guards that the hygon branch is not taken for a
        # non-hygon vendor:
        # 1. An observable one — a seeded Mock entry (the same fake module
        #    the positive test uses) must never be called, so a routing
        #    regression that applies hygon patches under nvidia fails on
        #    call_count, not just on an import side effect.
        # 2. The implicit poison from before is kept separately below:
        #    a None entry in sys.modules makes any from-import raise
        #    ImportError, so an accidental import fails loudly too.
        apply_mock = Mock(return_value=None)
        fake_patch = ModuleType(self.HYGON_PATCH_NAME)
        fake_patch.apply_hygon_patches = apply_mock

        self._run_with("nvidia", fake_patch)

        apply_mock.assert_not_called()

    @pytest.mark.platforms("hygon")
    def test_non_hygon_vendor_importing_the_hygon_patch_raises(self):
        # The poisoning twin: sys.modules[...] = None turns any from-import
        # of the hygon patch module into an ImportError. If the hygon branch
        # were taken for nvidia, this test fails instead of silently
        # passing; the two tests together cover both the call and the
        # import side of the regression.
        self._run_with("nvidia", None)


class TestHygonPatchDefensivePaths:
    """Marker-guard idempotence and hostile-setattr rollback in patch.py.

    These cover the five statements the baseline rollback suite misses
    (patch.py 91, 148, 166-168). Production reachability requires
    ``apply_hygon_patches``'s ``lru_cache`` to have been cleared, which the
    fixture below does — the guards are the second, cache-independent
    idempotence layer described in the module docstring. The module is
    loaded via ``spec_from_file_location`` under fake torch/vllm modules,
    mirroring ``tests/unit_tests/test_hygon_patch_rollback.py``, so the real
    process state is never touched.
    """

    pytestmark = pytest.mark.platforms("hygon")

    HYGON_PATCH_SOURCE = (
        Path(__file__).resolve().parents[4]
        / "vllm_fl/dispatch/backends/vendor/hygon/patch.py"
    )

    @pytest.fixture
    def patch_module(self, monkeypatch):
        """A standalone copy of hygon/patch.py over a fake torch module.

        The patch functions import torch and the vllm tree lazily inside
        their bodies, so seeding fake modules in sys.modules (restored by
        monkeypatch) redirects every snapshot and rollback to test
        doubles. ``lru_cache`` is cleared around every test so each call
        re-enters the function body, which is the only way to reach the
        marker guards in a single process.
        """
        spec = importlib.util.spec_from_file_location(
            "hygon_patch_defensive", self.HYGON_PATCH_SOURCE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.apply_hygon_patches.cache_clear()

        monkeypatch.setitem(sys.modules, "torch", self._fake_torch())
        yield module
        module.apply_hygon_patches.cache_clear()

    @staticmethod
    def _fake_torch():
        """A torch double with an empty accelerator namespace."""
        torch = ModuleType("torch")
        torch.accelerator = SimpleNamespace()
        torch.cuda = SimpleNamespace(mem_get_info=lambda index=None: (4096, 8192))
        return torch

    @staticmethod
    def _fake_vllm():
        """A vllm double whose platform_utils reports UVA available.

        Only the sys.modules entries installed by ``_seed_vllm_tree``
        matter (the patch does a plain ``import
        vllm.utils.platform_utils``); the parent attributes are set for
        symmetry with how a real package namespace looks.
        """
        vllm = ModuleType("vllm")
        vllm.__path__ = []
        utils = ModuleType("vllm.utils")
        utils.__path__ = []
        platform_utils = ModuleType("vllm.utils.platform_utils")
        platform_utils.is_uva_available = lambda: True
        vllm.utils = utils
        vllm.utils.platform_utils = platform_utils
        return vllm

    def _seed_vllm_tree(self, monkeypatch, vllm):
        """Install the vllm double and its platform_utils under real names.

        ``_install_uva_unavailable`` does
        ``import vllm.utils.platform_utils as platform_utils``, so the
        parent packages must be reachable through sys.modules.
        """
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        monkeypatch.setitem(sys.modules, "vllm.utils", vllm.utils)
        monkeypatch.setitem(
            sys.modules, "vllm.utils.platform_utils", vllm.utils.platform_utils
        )

    def test_uva_marker_guard_short_circuits_a_reapplication(
        self, patch_module, monkeypatch
    ):
        # patch.py:91 — with ``_hcu_uva_patched`` already True on
        # platform_utils, a re-entered install returns the no-op rollback
        # without rebinding anything. Production reaches this only after a
        # cache_clear (see class docstring); the guard keeps a second
        # application from re-patching an already-patched process.
        vllm = self._fake_vllm()
        platform_utils = vllm.utils.platform_utils
        platform_utils._hcu_uva_patched = True
        original = platform_utils.is_uva_available
        self._seed_vllm_tree(monkeypatch, vllm)

        rollback = patch_module._install_uva_unavailable()

        # The rollback is the shared no-op (identity, not a fresh closure
        # over an empty applied list) and nothing was rebound.
        assert rollback is patch_module._noop
        assert platform_utils.is_uva_available is original

    def test_accelerator_marker_guard_short_circuits_a_reapplication(
        self, patch_module, monkeypatch
    ):
        # patch.py:148 — with ``_hcu_attrs_patched`` already True on
        # torch.accelerator, a re-entered install returns the no-op
        # rollback without installing get_memory_info again.
        fake_torch = sys.modules["torch"]
        fake_torch.accelerator._hcu_attrs_patched = True
        before = dict(vars(fake_torch.accelerator))

        rollback = patch_module._install_accelerator_missing_attrs()

        assert rollback is patch_module._noop
        assert dict(vars(fake_torch.accelerator)) == before

    def test_hostile_setattr_on_accelerator_rolls_back_and_reraises(
        self, patch_module, monkeypatch
    ):
        # patch.py:166-168 — a torch.accelerator whose setattr raises
        # makes the first _apply fail. The except branch must undo what
        # landed (nothing, here: the failing owner is the first target)
        # and re-raise, so apply_hygon_patches sees the failure and rolls
        # the whole set back (the RuntimeError wrap is covered by the
        # baseline rollback suite; this test pins the inner guard).
        class HostileAccelerator:
            """Rejects every attribute write like a locked C extension."""

            def __setattr__(self, name, value):
                raise RuntimeError(f"read-only attribute: {name}")

        fake_torch = sys.modules["torch"]
        original_accelerator = fake_torch.accelerator
        try:
            fake_torch.accelerator = HostileAccelerator()

            with pytest.raises(RuntimeError, match="read-only attribute"):
                patch_module._install_accelerator_missing_attrs()

            # The hostile owner was the first (and only) target, so nothing
            # could have landed on it.
            assert not hasattr(fake_torch.accelerator, "get_memory_info")
            assert not hasattr(fake_torch.accelerator, "_hcu_attrs_patched")
        finally:
            fake_torch.accelerator = original_accelerator

    def test_hostile_setattr_mid_application_restores_the_earlier_binding(
        self, patch_module, monkeypatch
    ):
        # Variant of the above with one binding already applied before the
        # hostile one fires: the accelerator namespace lacks
        # get_memory_info (so the fallback lands in ``bindings`` and is
        # written first), then the ``_hcu_attrs_patched`` marker write
        # hits a hostile owner. The except branch must restore the written
        # binding — with _restore_all removed this test would see the
        # fallback left installed on a half-patched namespace.
        fake_torch = sys.modules["torch"]
        original_accelerator = fake_torch.accelerator

        class MarkerLocked:
            """Accepts every write except the idempotence marker."""

            def __init__(self):
                # get_memory_info must be genuinely absent so the patch
                # records and writes its fallback binding first; the
                # written log proves the mid-application state existed.
                object.__setattr__(self, "written", [])

            def __setattr__(self, name, value):
                if name == "_hcu_attrs_patched":
                    raise RuntimeError("marker attribute is read-only")
                object.__setattr__(self, name, value)
                written = object.__getattribute__(self, "written")
                object.__setattr__(self, "written", written + [name])

        locked = MarkerLocked()

        try:
            fake_torch.accelerator = locked

            with pytest.raises(RuntimeError, match="marker attribute is read-only"):
                patch_module._install_accelerator_missing_attrs()

            # The fallback binding was written before the marker failed
            # (proving the mid-application state existed)...
            assert locked.written == ["get_memory_info"]
            # ...and the rollback undid it: no binding and no marker
            # survive on the hostile owner.
            assert not hasattr(locked, "get_memory_info")
            assert not hasattr(locked, "_hcu_attrs_patched")
        finally:
            fake_torch.accelerator = original_accelerator
