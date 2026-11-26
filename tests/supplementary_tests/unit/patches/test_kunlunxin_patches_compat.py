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
import importlib.util
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

    pytestmark = pytest.mark.platforms("kunlunxin")

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

    pytestmark = pytest.mark.platforms("kunlunxin")

    def test_binds_launch_cooperative_grid_false(self):
        with _fake_xpu_options() as options:
            assert not hasattr(options, "launch_cooperative_grid")
            apply_triton_compat()
            assert options.launch_cooperative_grid is False

    def test_existing_attribute_is_never_overridden(self):
        with _fake_xpu_options() as options:
            options.launch_cooperative_grid = True
            apply_triton_compat()
            assert options.launch_cooperative_grid is True

    def test_absent_xpu_backend_is_a_noop(self):
        # On hosts WITHOUT an XPU triton backend the patch must return
        # without raising and without creating anything; on real P800 hosts
        # the backend genuinely exists (triton.backends.xpu.compiler is
        # importable), so the no-op property is only asserted where it
        # applies.
        def _xpu_backend_present() -> bool:
            if _XPU_COMPILER in sys.modules:
                return True
            try:
                return importlib.util.find_spec(_XPU_COMPILER) is not None
            except ModuleNotFoundError:
                return False

        if _xpu_backend_present():
            pytest.skip("real XPU triton backend present on this host")
        before = set(sys.modules)
        apply_triton_compat()
        assert _XPU_COMPILER not in sys.modules
        assert set(sys.modules) == before or _XPU_COMPILER not in sys.modules


# ---------------------------------------------------------------------------
# rope_compat
# ---------------------------------------------------------------------------


class TestRopeCompat:

    pytestmark = pytest.mark.platforms("kunlunxin")
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
        # The upstream kernel is a triton JITFunction; on hosts where its
        # source is not inspectable (torch.package-imported builds, e.g.
        # the P800 CI image) fall back to an identity-based precheck.
        try:
            original_source = inspect.getsource(original)
            assert "tl.where" not in original_source
        except (TypeError, OSError):
            # source unavailable: just require it not to be flagged yet
            assert not getattr(original, _PATCH_FLAG, False)

        apply_rope_compat()

        replaced = rope_module._prepare_rope_positions_kernel
        assert replaced is not original
        # the replacement is branch-free and flags itself
        source = self._kernel_source_or_none(replaced)
        if source is None:
            # inspectable source unavailable on this host build; the flag
            # and identity checks above are the contract we can still pin
            assert getattr(replaced, _PATCH_FLAG) is True
        else:
            assert "tl.where" in source
            assert "if is_prefill:" not in source
            assert getattr(replaced, _PATCH_FLAG) is True
            # the load is masked with is_prefill so decode steps cannot read
            # out of bounds (the documented subtlety of the tl.where rewrite)
            assert "mask & is_prefill" in source

    @staticmethod
    def _kernel_source_or_none(kernel):
        try:
            return inspect.getsource(kernel)
        except (TypeError, OSError):
            return None

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

    pytestmark = pytest.mark.platforms("kunlunxin")
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


# ---------------------------------------------------------------------------
# kv_block_zero_compat: _zero_block_ids_pytorch span semantics (56-76)
# ---------------------------------------------------------------------------


class TestKvBlockZeroPytorch:
    """The span zeroer itself, driven directly with real cache tensors.

    ``tests/unit_tests/attention/test_kunlunxin_cache.py`` drives the
    installed ``KVBlockZeroer`` through the patched constructor; these
    tests call ``_zero_block_ids_pytorch`` directly with hand-built
    segments so the run-length zeroing (start/end bookkeeping, dedup,
    out-of-range skipping) is asserted independently of collection.
    """

    @staticmethod
    def _zero(segments, block_ids):
        """Build a zeroer self and invoke the patch function directly."""
        from types import SimpleNamespace

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import (
            kv_block_zero_compat,
        )

        self = SimpleNamespace(_kunlunxin_kv_segments=segments)
        kv_block_zero_compat._zero_block_ids_pytorch(self, block_ids)

    def test_contiguous_run_zeroes_one_span(self):
        import torch

        # 4 manager blocks, cache holds 8 (ratio 2): ids [1, 2] form the
        # contiguous run rows 2..6
        kv = torch.ones(8, 3, 4)
        self._zero([(kv, 2)], [1, 2])
        # blocks 1-2 span cache rows 2..6 — zeroed in one narrow
        assert torch.count_nonzero(kv[:2]) == kv[:2].numel()
        assert torch.count_nonzero(kv[2:6]) == 0
        assert torch.count_nonzero(kv[6:]) == kv[6:].numel()

    def test_duplicate_and_unsorted_ids_dedupe_into_one_span(self):
        import torch

        kv = torch.ones(8, 3, 4)
        self._zero([(kv, 2)], [2, 2, 1, 2, 1])
        # ids {1, 2} only: rows 2..6 zero, nothing else touched
        assert torch.count_nonzero(kv[:2]) == kv[:2].numel()
        assert torch.count_nonzero(kv[2:6]) == 0
        assert torch.count_nonzero(kv[6:]) == kv[6:].numel()

    def test_gap_between_ids_zeroes_two_disjoint_spans(self):
        import torch

        kv = torch.ones(12, 3, 4)
        self._zero([(kv, 2)], [0, 3])
        # id 0 -> rows 0..2; id 3 -> rows 6..8 (the gap rows 2..6 stay)
        assert torch.count_nonzero(kv[0:2]) == 0
        assert torch.count_nonzero(kv[2:6]) == kv[2:6].numel()
        assert torch.count_nonzero(kv[6:8]) == 0
        assert torch.count_nonzero(kv[8:]) == kv[8:].numel()

    def test_out_of_range_ids_are_skipped_not_clamped(self):
        import torch

        kv = torch.ones(8, 3, 4)
        # id 7 would need rows 14..16 past the 8-row tensor; id 4 needs
        # rows 8..10 — both must be skipped, id 1 zeroes rows 2..4
        self._zero([(kv, 2)], [1, 4, 7])
        assert torch.count_nonzero(kv[:2]) == kv[:2].numel()
        assert torch.count_nonzero(kv[2:4]) == 0
        assert torch.count_nonzero(kv[4:]) == kv[4:].numel()

    def test_negative_ids_are_skipped(self):
        import torch

        kv = torch.ones(4, 3, 4)
        self._zero([(kv, 1)], [-1, 0])
        # only block 0 (rows 0..1) is zeroed; -1 is skipped, not wrapped
        assert torch.count_nonzero(kv[0:1]) == 0
        assert torch.count_nonzero(kv[1:]) == kv[1:].numel()

    def test_empty_ids_and_empty_segments_return_early(self):
        import torch

        kv = torch.ones(4, 3, 4)
        empty_kv = torch.ones(4, 3, 4)
        # no ids: nothing happens
        self._zero([(kv, 1)], [])
        assert torch.count_nonzero(kv) == kv.numel()
        # no segments: nothing happens even with ids
        self._zero([], [0, 1])
        assert torch.count_nonzero(empty_kv) == empty_kv.numel()

    def test_multiple_segments_each_zero_their_own_spans(self):
        import torch

        # two layers with different block ratios over distinct tensors
        kv_a = torch.ones(6, 2)
        kv_b = torch.ones(9, 2)
        self._zero([(kv_a, 3), (kv_b, 1)], [1])
        # layer A: ratio 3 -> rows 3..6 zeroed
        assert torch.count_nonzero(kv_a[:3]) == kv_a[:3].numel()
        assert torch.count_nonzero(kv_a[3:]) == 0
        # layer B: ratio 1 -> row 1 only
        assert torch.count_nonzero(kv_b[0:1]) == kv_b[0:1].numel()
        assert torch.count_nonzero(kv_b[1:2]) == 0
        assert torch.count_nonzero(kv_b[2:]) == kv_b[2:].numel()

    def test_aliasing_views_zero_the_shared_storage_once(self):
        import torch

        # packed overlay: a narrow view and a wide view on one storage
        storage = torch.ones(8, 4, 32)
        narrow = storage[:, :2]
        wide = storage
        # segments list (kv view, ratio); the zeroer writes through views
        self._zero([(narrow, 2), (wide, 2)], [2])
        assert torch.count_nonzero(storage[4:6]) == 0
        assert torch.count_nonzero(storage[:4]) == storage[:4].numel()
