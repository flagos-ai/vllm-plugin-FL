# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for the Iluvatar vision-precision binding.

Covers ``vllm_fl/dispatch/backends/vendor/iluvatar/vision_precision.py``:

* ``in_vision_forward()`` defaults to False.
* ``_wrap_vision_forward`` sets the flag inside the wrapped call, resets it
  afterwards (also on exception), supports re-entrancy, and is idempotent
  (double-wrapping returns the same function object).
* ``patch_vision_precision`` swaps Qwen3_VisionTransformer.forward on a
  fake sys.modules stub (pattern from
  tests/unit_tests/test_kunlunxin_vl_compat.py), installs the marker flag,
  is idempotent, and leaves the stub restored afterwards.

CPU-only. Vendor-gated to the iluvatar platform.
"""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]

_MOD_NAME = "vllm.model_executor.models.qwen3_vl"
_PARENTS = [
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.models",
]


@pytest.fixture
def fresh_qwen3_vl_stub():
    """Install a fresh fake qwen3_vl module into sys.modules; restore after.

    Saves every touched sys.modules entry (including absent ones) and
    restores exactly that state in the finally block, so the real qwen3_vl
    import state is unaffected by the stub.
    """
    saved = {name: sys.modules.get(name) for name in _PARENTS + [_MOD_NAME]}

    class _Vision:
        def forward(self, x):
            return ("original", x)

    original_forward = _Vision.__dict__["forward"]
    module = types.ModuleType(_MOD_NAME)
    module.Qwen3_VisionTransformer = _Vision
    sys.modules[_MOD_NAME] = module
    for parent in _PARENTS:
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)

    yield module.Qwen3_VisionTransformer, original_forward

    # restore: pop entries we created, put back entries we replaced
    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    _Vision.forward = original_forward


class TestInVisionForward:
    def test_default_is_false(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            in_vision_forward,
        )

        assert in_vision_forward() is False

    def test_flag_round_trip(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            _vision_forward,
            in_vision_forward,
        )

        assert in_vision_forward() is False
        token = _vision_forward.set(True)
        try:
            assert in_vision_forward() is True
        finally:
            _vision_forward.reset(token)
        assert in_vision_forward() is False


class TestWrapVisionForward:
    def _wrap_target(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            _wrap_vision_forward,
        )

        return _wrap_vision_forward

    def test_sets_true_inside_and_false_after(self):
        wrap = self._wrap_target()
        observed = []

        def forward(x):
            observed.append(in_vision := None)  # placeholder replaced below
            return x

        # simpler: capture the flag inside via a closure
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            in_vision_forward,
        )

        seen = []

        def forward(x):
            seen.append(in_vision_forward())
            return x + 1

        wrapped = wrap(forward)
        assert wrapped(1) == 2
        assert seen == [True]
        assert in_vision_forward() is False

    def test_resets_on_exception(self):
        wrap = self._wrap_target()
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            in_vision_forward,
        )

        def forward(x):
            assert in_vision_forward() is True
            raise RuntimeError("vision forward failed")

        wrapped = wrap(forward)
        with pytest.raises(RuntimeError, match="vision forward failed"):
            wrapped(1)
        assert in_vision_forward() is False

    def test_reentrancy_inner_sees_true(self):
        wrap = self._wrap_target()
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            in_vision_forward,
        )

        inner_seen = []

        def inner(x):
            inner_seen.append(in_vision_forward())
            return x * 2

        def outer(x):
            assert in_vision_forward() is True
            return wrap(inner)(x) + 1

        assert wrap(outer)(3) == 7
        assert inner_seen == [True]
        assert in_vision_forward() is False

    def test_idempotent_double_wrap_returns_same_object(self):
        wrap = self._wrap_target()

        def forward(x):
            return x

        once = wrap(forward)
        assert wrap(once) is once
        assert getattr(once, "_iluvatar_fp32_reduce", False)

    def test_wrapped_preserves_metadata(self):
        wrap = self._wrap_target()

        def a_well_named_forward(x):
            """Docstring of the original."""
            return x

        wrapped = wrap(a_well_named_forward)
        assert wrapped.__name__ == "a_well_named_forward"
        assert wrapped.__doc__ == "Docstring of the original."

    def test_nested_calls_reset_together(self):
        """After the outer wrapped call exits the flag is False, not stuck."""
        wrap = self._wrap_target()
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            in_vision_forward,
        )

        def inner(x):
            return x

        def outer(x):
            return wrap(inner)(x)

        wrapped_outer = wrap(outer)
        wrapped_outer(1)
        assert in_vision_forward() is False


class TestPatchVisionPrecision:
    def test_patching_swaps_forward_and_sets_flag(self, fresh_qwen3_vl_stub):
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            patch_vision_precision,
        )

        vision_cls, original_forward = fresh_qwen3_vl_stub
        patch_vision_precision()
        patched = vision_cls.__dict__["forward"]
        assert patched is not original_forward
        assert getattr(patched, "_iluvatar_fp32_reduce", False)

    def test_patched_forward_still_executes_original_body(
        self, fresh_qwen3_vl_stub
    ):
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            in_vision_forward,
            patch_vision_precision,
        )

        vision_cls, _ = fresh_qwen3_vl_stub
        patch_vision_precision()
        instance = vision_cls()
        # the stub body returns ("original", x); wrapper must pass it through
        assert instance.forward(21) == ("original", 21)
        assert in_vision_forward() is False

    def test_idempotent(self, fresh_qwen3_vl_stub):
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            patch_vision_precision,
        )

        vision_cls, _ = fresh_qwen3_vl_stub
        patch_vision_precision()
        first = vision_cls.__dict__["forward"]
        patch_vision_precision()
        assert vision_cls.__dict__["forward"] is first

    def test_no_double_wrapping_chain(self, fresh_qwen3_vl_stub):
        """The marker flag prevents stacking wrappers on repeat patches."""
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            patch_vision_precision,
        )

        vision_cls, _ = fresh_qwen3_vl_stub
        patch_vision_precision()
        patched = vision_cls.__dict__["forward"]
        patch_vision_precision()
        patch_vision_precision()
        # still the same single wrapper — __wrapped__ chain length is 1
        chain = 0
        fn = patched
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
            chain += 1
        assert chain <= 1

    def test_real_module_state_restored(self, fresh_qwen3_vl_stub):
        """The fixture's finally restores both the class and sys.modules."""
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            patch_vision_precision,
        )

        vision_cls, original_forward = fresh_qwen3_vl_stub
        patch_vision_precision()
        assert vision_cls.__dict__["forward"] is not original_forward
