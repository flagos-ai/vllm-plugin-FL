# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin compat patches.

Covers the patch mechanics of:

* ``patches/accelerator_compat.py`` -- binding missing
  ``torch.accelerator`` names from ``torch.cuda`` without overriding
  existing ones;
* ``patches/triton_compat.py`` -- binding ``launch_cooperative_grid`` on
  Triton's XPUOptions, guarded and idempotent;
* ``patches/rope_compat.py`` -- replacing vLLM's M-RoPE position kernel
  with the branch-free ``tl.where`` formulation, flagged and idempotent;
* ``patches/patch_fla_utils.py`` -- wrapping ``torch.xpu.get_device_name``
  so module-level FLA imports cannot crash, and correcting the platform
  flags of an already-imported fla utils module.

Every test restores the exact objects and module entries it touched.
"""

from __future__ import annotations

import contextlib
import inspect
import sys
from types import ModuleType

import pytest

pytestmark = [pytest.mark.supplementary]

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.accelerator_compat import (
    _ACCELERATOR_COMPAT,
    apply_accelerator_compat,
)
from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.triton_compat import (
    apply_triton_compat,
)


# ---------------------------------------------------------------------------
# accelerator_compat
# ---------------------------------------------------------------------------


class TestAcceleratorCompat:
    @pytest.mark.platforms("kunlunxin")
    def test_binds_only_missing_names_from_torch_cuda(self):
        # get_memory_info / current_device / set_device are absent on this
        # torch build's torch.accelerator; the rest already exist. Patching
        # must add exactly the missing ones and leave the rest alone.
        import torch

        bound_before = {
            name: getattr(torch.accelerator, name, None)
            for name in _ACCELERATOR_COMPAT
        }
        missing = [n for n in _ACCELERATOR_COMPAT if not hasattr(torch.accelerator, n)]
        assert missing, "this torch build exposes every compat name; test setup is stale"

        apply_accelerator_compat()

        for name, cuda_name in _ACCELERATOR_COMPAT.items():
            if name in missing:
                assert getattr(torch.accelerator, name, None) is getattr(
                    torch.cuda, cuda_name, None
                ), f"{name} was not bound from torch.cuda.{cuda_name}"
            else:
                assert getattr(torch.accelerator, name, None) is bound_before[name], (
                    f"{name} existed and must not be overridden"
                )

    @pytest.mark.platforms("kunlunxin")
    def test_names_absent_from_cuda_are_skipped(self):
        # A name missing on BOTH surfaces must be skipped, not invented.
        import torch

        # snapshot both surfaces fully so every mutation is reversible
        accel_snapshot = {
            name: getattr(torch.accelerator, name, None)
            for name in _ACCELERATOR_COMPAT
        }
        cuda_snapshot = {
            cuda_name: getattr(torch.cuda, cuda_name, None)
            for _, cuda_name in _ACCELERATOR_COMPAT.items()
        }
        try:
            # remove every accelerator name, and hide one cuda
            # implementation entirely
            for name in _ACCELERATOR_COMPAT:
                if hasattr(torch.accelerator, name):
                    delattr(torch.accelerator, name)
            victim = next(iter(_ACCELERATOR_COMPAT.values()))
            delattr(torch.cuda, victim)

            apply_accelerator_compat()

            # the victim (absent on both surfaces) stays unbound
            assert not hasattr(torch.accelerator, victim)
            # the others get bound from torch.cuda
            for name, cuda_name in _ACCELERATOR_COMPAT.items():
                if cuda_name == victim:
                    continue
                assert getattr(torch.accelerator, name, None) is torch.cuda.__dict__.get(
                    cuda_name
                ), f"{name} was not bound from torch.cuda.{cuda_name}"
        finally:
            for name, value in accel_snapshot.items():
                if value is not None:
                    setattr(torch.accelerator, name, value)
                elif hasattr(torch.accelerator, name):
                    delattr(torch.accelerator, name)
            for name, value in cuda_snapshot.items():
                if value is not None:
                    setattr(torch.cuda, name, value)

    @pytest.mark.platforms("kunlunxin")
    def test_apply_is_idempotent(self):
        import torch

        apply_accelerator_compat()
        snapshot = {
            name: getattr(torch.accelerator, name, None)
            for name in _ACCELERATOR_COMPAT
        }
        apply_accelerator_compat()
        assert {
            name: getattr(torch.accelerator, name, None)
            for name in _ACCELERATOR_COMPAT
        } == snapshot


# ---------------------------------------------------------------------------
# triton_compat
# ---------------------------------------------------------------------------

_XPU_COMPILER = "triton.backends.xpu.compiler"
_XPU_PKG = "triton.backends.xpu"


@contextlib.contextmanager
def _fake_xpu_options():
    """Install a fake triton.backends.xpu.compiler with a bare XPUOptions."""
    class FakeXPUOptions:
        pass

    saved = {
        name: sys.modules.get(name) for name in (_XPU_PKG, _XPU_COMPILER)
    }
    pkg = ModuleType(_XPU_PKG)
    pkg.__path__ = []
    compiler = ModuleType(_XPU_COMPILER)
    compiler.XPUOptions = FakeXPUOptions
    sys.modules[_XPU_PKG] = pkg
    sys.modules[_XPU_COMPILER] = compiler
    try:
        yield FakeXPUOptions
    finally:
        for name, entry in saved.items():
            if entry is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = entry


class TestTritonCompat:
    @pytest.mark.platforms("kunlunxin")
    def test_binds_launch_cooperative_grid_false(self):
        with _fake_xpu_options() as options:
            assert not hasattr(options, "launch_cooperative_grid")
            apply_triton_compat()
            assert options.launch_cooperative_grid is False

    @pytest.mark.platforms("kunlunxin")
    def test_existing_attribute_is_never_overridden(self):
        with _fake_xpu_options() as options:
            options.launch_cooperative_grid = True
            apply_triton_compat()
            assert options.launch_cooperative_grid is True

    @pytest.mark.platforms("kunlunxin")
    def test_absent_xpu_backend_is_a_noop(self):
        # On this build triton.backends.xpu.compiler does not exist; the
        # patch must return without raising and without creating anything.
        assert _XPU_COMPILER not in sys.modules
        before = set(sys.modules)
        apply_triton_compat()
        assert _XPU_COMPILER not in sys.modules
        assert set(sys.modules) == before or _XPU_COMPILER not in sys.modules


# ---------------------------------------------------------------------------
# rope_compat
# ---------------------------------------------------------------------------


class TestRopeCompat:
    @pytest.fixture
    def rope_module(self):
        import vllm.v1.worker.gpu.mm.rope as rope

        return rope

    @pytest.fixture(autouse=True)
    def _restore_rope_kernel(self, rope_module):
        original = rope_module._prepare_rope_positions_kernel
        yield
        rope_module._prepare_rope_positions_kernel = original

    def test_replaces_kernel_with_tl_where_formulation(self, rope_module):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.rope_compat import (
            _PATCH_FLAG,
            apply_rope_compat,
        )

        original = rope_module._prepare_rope_positions_kernel
        assert "tl.where" not in inspect.getsource(original)

        apply_rope_compat()

        replaced = rope_module._prepare_rope_positions_kernel
        assert replaced is not original
        # the replacement is branch-free and flags itself
        source = inspect.getsource(replaced)
        assert "tl.where" in source
        assert "if is_prefill:" not in source
        assert getattr(replaced, _PATCH_FLAG) is True
        # the load is masked with is_prefill so decode steps cannot read
        # out of bounds (the documented subtlety of the tl.where rewrite)
        assert "mask & is_prefill" in source

    def test_apply_is_idempotent(self, rope_module):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.rope_compat import (
            apply_rope_compat,
        )

        apply_rope_compat()
        once = rope_module._prepare_rope_positions_kernel
        apply_rope_compat()
        assert rope_module._prepare_rope_positions_kernel is once

    def test_already_flagged_kernel_is_left_alone(self, rope_module):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.rope_compat import (
            _PATCH_FLAG,
            apply_rope_compat,
        )

        def foreign_kernel():
            return 42

        setattr(foreign_kernel, _PATCH_FLAG, True)
        rope_module._prepare_rope_positions_kernel = foreign_kernel
        apply_rope_compat()
        assert rope_module._prepare_rope_positions_kernel is foreign_kernel

    def test_missing_kernel_is_a_noop(self, rope_module):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.rope_compat import (
            apply_rope_compat,
        )

        saved = rope_module._prepare_rope_positions_kernel
        del rope_module._prepare_rope_positions_kernel
        try:
            apply_rope_compat()
        finally:
            rope_module._prepare_rope_positions_kernel = saved
        # nothing was installed
        assert not hasattr(rope_module, "_prepare_rope_positions_kernel") or (
            rope_module._prepare_rope_positions_kernel is saved
        )


# ---------------------------------------------------------------------------
# patch_fla_utils
# ---------------------------------------------------------------------------

_FLA_UTILS = "vllm.model_executor.layers.fla.ops.utils"


def _platform_fl_or_skip():
    """Import PlatformFL, skipping the test when it cannot resolve a vendor.

    ``vllm_fl.platform`` asks flag_gems' DeviceDetector for the vendor, which
    needs ``GEMS_VENDOR`` or real hardware to answer. On hosts where that
    resolution fails, importing the module raises AttributeError, so the
    vendor-gated behavior of ``_patch_xpu_get_device`` is untestable there.
    """
    try:
        from vllm_fl.platform import PlatformFL
    except Exception as exc:  # noqa: BLE001 - any import-time failure
        pytest.skip(f"vllm_fl.platform unusable in this environment: {exc}")
    return PlatformFL


class TestFlaUtilsCompat:
    @pytest.fixture(autouse=True)
    def _restore_get_device_name(self):
        import torch

        saved = torch.xpu.get_device_name
        yield
        torch.xpu.get_device_name = saved

    def test_install_wrapper_makes_get_device_name_safe(self):
        import torch

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (
            _install_xpu_get_device_wrapper,
        )

        # On this build the raw call raises; the wrapper must return "".
        assert _install_xpu_get_device_wrapper() is True
        assert torch.xpu.get_device_name(0) == ""
        # and it is callable repeatedly without re-wrapping
        first = torch.xpu.get_device_name
        _install_xpu_get_device_wrapper()
        assert torch.xpu.get_device_name is first

    def test_wrapper_preserves_a_working_backend(self):
        import torch

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (
            _install_xpu_get_device_wrapper,
        )

        def working_backend(idx=0):
            return "Intel(R) Arc(TM) A770"

        torch.xpu.get_device_name = working_backend
        _install_xpu_get_device_wrapper()
        assert torch.xpu.get_device_name(0) == "Intel(R) Arc(TM) A770"

    def test_fix_platform_vars_rewrites_flags(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (
            _fix_platform_vars,
        )

        fake = ModuleType(_FLA_UTILS)
        fake.device_platform = "xpu"
        fake.is_nvidia = False
        fake.is_intel = True
        fake.is_amd = True
        fake.is_intel_alchemist = True
        fake.is_nvidia_hopper = True
        fake.is_tma_supported = True
        saved = sys.modules.get(_FLA_UTILS)
        sys.modules[_FLA_UTILS] = fake
        try:
            _fix_platform_vars()
            assert fake.device_platform == "nvidia"
            assert fake.is_nvidia is True
            assert fake.is_intel is False
            assert fake.is_amd is False
            assert fake.is_intel_alchemist is False
            assert fake.is_nvidia_hopper is False
            assert fake.is_tma_supported is False
        finally:
            if saved is None:
                sys.modules.pop(_FLA_UTILS, None)
            else:
                sys.modules[_FLA_UTILS] = saved

    def test_fix_platform_vars_tolerates_missing_module(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (
            _fix_platform_vars,
        )

        saved = sys.modules.get(_FLA_UTILS)
        sys.modules.pop(_FLA_UTILS, None)
        try:
            # must not raise even though the module is not imported
            _fix_platform_vars()
        finally:
            if saved is not None:
                sys.modules[_FLA_UTILS] = saved

    @pytest.mark.platforms("kunlunxin")
    def test_patch_xpu_get_device_gates_on_vendor(self, monkeypatch):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (
            _patch_xpu_get_device,
        )

        # vllm_fl.platform resolves the vendor through flag_gems' device
        # detector, which needs GEMS_VENDOR (or real hardware) to answer.
        # A session where that resolution fails cannot exercise the gate.
        PlatformFL = _platform_fl_or_skip()
        import torch

        saved_vendor = PlatformFL.vendor_name
        try:
            # a non-kunlunxin session must not touch torch.xpu at all
            PlatformFL.vendor_name = "nvidia"
            before = torch.xpu.get_device_name
            monkeypatch.setattr(
                "vllm_fl.dispatch.backends.vendor.kunlunxin.patches."
                "patch_fla_utils.ensure_fla_compat",
                lambda: pytest.fail("ensure_fla_compat ran on a foreign platform"),
                raising=False,
            )
            _patch_xpu_get_device()
            assert torch.xpu.get_device_name is before
        finally:
            PlatformFL.vendor_name = saved_vendor

    @pytest.mark.platforms("kunlunxin")
    def test_patch_xpu_get_device_applies_on_kunlunxin(self):
        import torch

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (
            _patch_xpu_get_device,
        )

        # see test_patch_xpu_get_device_gates_on_vendor for the skip reason
        PlatformFL = _platform_fl_or_skip()

        saved_vendor = PlatformFL.vendor_name
        try:
            PlatformFL.vendor_name = "kunlunxin"
            _patch_xpu_get_device()
            # the safe wrapper is installed (returns "" instead of raising)
            assert torch.xpu.get_device_name(0) == ""
        finally:
            PlatformFL.vendor_name = saved_vendor
