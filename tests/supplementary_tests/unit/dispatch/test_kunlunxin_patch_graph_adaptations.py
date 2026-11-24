# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin graph-adaptation patches in ``patch.py``.

``vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py`` stages its graph
patches through ``_install_graph_patch`` after a signature preflight.
These tests drive the staging functions (``patch_cudagraph_dispatcher``,
``patch_breakable_cudagraph_mode``, ``patch_breakable_private_pools``,
``patch_native_piecewise_boundaries``) with stand-in ``sys.modules``
entries whose signatures match the preflight contract, and verify the
installed wrappers' control flow on CPU.

``patch.py`` imports only torch/inspect/os at module scope, so the module
body itself is executed with ``runpy`` (same pattern as
``tests/unit_tests/test_kunlunxin_v2_graph_capture.py``).

Order independence: every ``sys.modules`` entry a test touches is restored
exactly on exit, and class attributes replaced on real classes
(``QwenGatedDeltaNetAttention._forward_core``,
``KunlunxinAttentionBackendImpl.forward``) are restored in a ``finally``.
"""

from __future__ import annotations

import contextlib
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@contextlib.contextmanager
def _vllm_config_context():
    """Set a minimal current VllmConfig (TorchCompileWithNoGuardsWrapper
    queries it in ``__init__``).

    The real ``VllmConfig`` needs a scheduler section only; everything
    else has defaults.  Restores the previous global on exit.
    """
    from vllm.config import (
        CompilationConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import CompilationMode

    config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            is_encoder_decoder=False,
            max_num_batched_tokens=128,
            max_model_len=256,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE
        ),
    )
    # Platform defaults (CPU platform / PlatformFL-kunlunxin) downgrade the
    # mode inside __post_init__; the wrapper only needs a non-NONE mode and
    # a valid VllmConfig object, so restore the requested mode after
    # construction.
    config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    with set_current_vllm_config(config):
        yield

pytestmark = [pytest.mark.supplementary]

_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py"
)

_MODULE_NAMES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm_fl.worker.model_runner",
    "vllm.v1.cudagraph_dispatcher",
)


def _load_patch_module():
    """Execute patch.py in a fresh namespace (its module scope is import-safe)."""
    return runpy.run_path(str(_PATCH_PATH))


class _ModuleStubs:
    """Install stand-in modules for the V2 runner names patch.py imports."""

    def __init__(self, **fl_model_runner_attrs):
        self.gpu_model_runner = ModuleType("vllm.v1.worker.gpu_model_runner")
        self.gpu_model_runner.is_breakable_cudagraph_enabled = lambda: True

        self.fl_model_runner = ModuleType("vllm_fl.worker.model_runner")
        self.fl_model_runner.is_breakable_cudagraph_enabled = lambda: True
        self.fl_model_runner._device_graph_capture = lambda *a, **k: "FL_CAPTURE"
        for name, value in fl_model_runner_attrs.items():
            setattr(self.fl_model_runner, name, value)

        self.dispatcher_mod = ModuleType("vllm.v1.cudagraph_dispatcher")
        self.dispatch_calls = []

        outer = self

        class FakeDispatcher:
            def dispatch(
                self,
                num_tokens,
                uniform_decode=False,
                has_lora=False,
                num_active_loras=0,
                valid_modes=None,
                invalid_modes=None,
            ):
                outer.dispatch_calls.append(
                    dict(
                        num_tokens=num_tokens,
                        uniform_decode=uniform_decode,
                        has_lora=has_lora,
                        num_active_loras=num_active_loras,
                        valid_modes=valid_modes,
                        invalid_modes=invalid_modes,
                    )
                )
                return "ORIG_DISPATCH"

        self.FakeDispatcher = FakeDispatcher
        self.dispatcher_mod.CudagraphDispatcher = FakeDispatcher
        self._saved = None

    def __enter__(self):
        self._saved = {name: sys.modules.get(name) for name in _MODULE_NAMES}
        sys.modules["vllm.v1.worker.gpu_model_runner"] = self.gpu_model_runner
        sys.modules["vllm_fl.worker.model_runner"] = self.fl_model_runner
        sys.modules["vllm.v1.cudagraph_dispatcher"] = self.dispatcher_mod
        # "import a.b.c" resolves the intermediate package by attribute
        # lookup on the parent, so the stub parents must expose the child.
        import vllm.v1.worker.gpu as gpu_pkg

        self._saved_gpu_pkg_attr = getattr(gpu_pkg, "gpu_model_runner", None)
        gpu_pkg.gpu_model_runner = self.gpu_model_runner
        import vllm_fl

        self._saved_fl_worker = getattr(vllm_fl, "worker", None)
        if self._saved_fl_worker is None:
            import types as _types

            vllm_fl.worker = _types.ModuleType("vllm_fl.worker")
        import vllm_fl.worker

        self._saved_fl_mr_attr = getattr(vllm_fl.worker, "model_runner", None)
        vllm_fl.worker.model_runner = self.fl_model_runner
        return self

    def __exit__(self, *exc):
        import vllm.v1.worker.gpu as gpu_pkg
        import vllm_fl

        if self._saved_gpu_pkg_attr is None:
            try:
                del gpu_pkg.gpu_model_runner
            except AttributeError:
                pass
        else:
            gpu_pkg.gpu_model_runner = self._saved_gpu_pkg_attr
        if self._saved_fl_mr_attr is None:
            try:
                del vllm_fl.worker.model_runner
            except AttributeError:
                pass
        else:
            vllm_fl.worker.model_runner = self._saved_fl_mr_attr
        if self._saved_fl_worker is None:
            try:
                del vllm_fl.worker
            except AttributeError:
                pass
        else:
            vllm_fl.worker = self._saved_fl_worker
        for name, entry in self._saved.items():
            if entry is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = entry
        return False


# ---------------------------------------------------------------------------
# _install_graph_patch staging contract
# ---------------------------------------------------------------------------


class TestInstallGraphPatch:
    def test_direct_install_sets_attribute(self):
        ns = _load_patch_module()
        owner = SimpleNamespace(attr="OLD")
        ns["_install_graph_patch"](owner, "attr", "NEW", None)
        assert owner.attr == "NEW"

    def test_staged_install_records_replacement(self):
        ns = _load_patch_module()
        owner = SimpleNamespace(attr="OLD")
        replacements = []
        ns["_install_graph_patch"](owner, "attr", "NEW", replacements)
        assert owner.attr == "OLD", "staging must not publish yet"
        assert replacements == [(owner, "attr", "OLD", "NEW", True)]

    def test_staged_records_owned_flag_from_vars(self):
        ns = _load_patch_module()

        class Base:
            attr = "INHERITED"

        class Owned(Base):
            attr = "OWNED"  # lives in vars(Owned) -> owned flag True

        class NotOwned(Base):
            pass  # attr only inherited -> owned flag False

        replacements = []
        ns["_install_graph_patch"](Owned, "attr", "NEW", replacements)
        ns["_install_graph_patch"](NotOwned, "attr", "NEW", replacements)
        assert replacements[0][2] == "OWNED"
        assert replacements[0][4] is True
        assert replacements[1][2] == "INHERITED"
        assert replacements[1][4] is False
        # neither class was actually modified by staging
        assert Owned.attr == "OWNED"
        assert NotOwned.attr == "INHERITED"


# ---------------------------------------------------------------------------
# patch_cudagraph_dispatcher: dispatch_uniform_full
# ---------------------------------------------------------------------------


class TestDispatchUniformFull:
    @pytest.mark.platforms("kunlunxin")
    def test_uniform_decode_is_a_verbatim_passthrough(self):
        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_cudagraph_dispatcher"](replacements)
            wrapper = replacements[0][3]

            result = wrapper(stubs.FakeDispatcher(), 7, uniform_decode=True)

            assert result == "ORIG_DISPATCH"
            call = stubs.dispatch_calls[-1]
            assert call["num_tokens"] == 7
            assert call["uniform_decode"] is True
            assert call["valid_modes"] is None
            assert call["invalid_modes"] is None

    @pytest.mark.platforms("kunlunxin")
    def test_non_uniform_decode_excludes_full(self):
        from vllm.config import CUDAGraphMode

        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_cudagraph_dispatcher"](replacements)
            wrapper = replacements[0][3]

            wrapper(stubs.FakeDispatcher(), 9, uniform_decode=False)

            call = stubs.dispatch_calls[-1]
            assert CUDAGraphMode.FULL in call["invalid_modes"]
            # the caller's set (here None) is never mutated in place
            assert call["valid_modes"] is None

    @pytest.mark.platforms("kunlunxin")
    def test_full_only_valid_modes_falls_back_to_none(self):
        from vllm.config import CUDAGraphMode

        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_cudagraph_dispatcher"](replacements)
            wrapper = replacements[0][3]

            valid = {CUDAGraphMode.FULL}
            wrapper(
                stubs.FakeDispatcher(),
                9,
                uniform_decode=False,
                valid_modes=valid,
            )

            call = stubs.dispatch_calls[-1]
            assert call["valid_modes"] == {CUDAGraphMode.NONE}
            assert CUDAGraphMode.NONE not in call["invalid_modes"]
            # the original set was replaced, not mutated
            assert valid == {CUDAGraphMode.FULL}

    @pytest.mark.platforms("kunlunxin")
    def test_mixed_valid_modes_are_left_alone(self):
        from vllm.config import CUDAGraphMode

        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_cudagraph_dispatcher"](replacements)
            wrapper = replacements[0][3]

            mixed = {CUDAGraphMode.FULL, CUDAGraphMode.NONE}
            wrapper(
                stubs.FakeDispatcher(),
                9,
                uniform_decode=False,
                valid_modes=mixed,
            )

            call = stubs.dispatch_calls[-1]
            assert call["valid_modes"] == mixed

    @pytest.mark.platforms("kunlunxin")
    def test_existing_invalid_modes_are_preserved(self):
        from vllm.config import CUDAGraphMode

        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_cudagraph_dispatcher"](replacements)
            wrapper = replacements[0][3]

            invalid = {CUDAGraphMode.NONE}
            wrapper(
                stubs.FakeDispatcher(),
                9,
                uniform_decode=False,
                invalid_modes=invalid,
            )

            call = stubs.dispatch_calls[-1]
            assert call["invalid_modes"] == {CUDAGraphMode.NONE, CUDAGraphMode.FULL}
            # the caller's set is untouched (the wrapper copies it first)
            assert invalid == {CUDAGraphMode.NONE}

    @pytest.mark.platforms("kunlunxin")
    def test_none_invalid_modes_becomes_a_set_with_full(self):
        from vllm.config import CUDAGraphMode

        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_cudagraph_dispatcher"](replacements)
            wrapper = replacements[0][3]

            wrapper(stubs.FakeDispatcher(), 9, uniform_decode=False, invalid_modes=None)

            call = stubs.dispatch_calls[-1]
            assert call["invalid_modes"] == {CUDAGraphMode.FULL}

    @pytest.mark.platforms("kunlunxin")
    def test_wrapper_is_flagged_and_idempotent(self):
        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_cudagraph_dispatcher"](replacements)
            wrapper = replacements[0][3]
            assert getattr(wrapper, "_kunlunxin_uniform_full", False) is True

            # A second stage over the already-flagged wrapper records nothing.
            dispatcher = stubs.dispatcher_mod
            dispatcher.CudagraphDispatcher.dispatch = wrapper
            try:
                second = []
                ns["patch_cudagraph_dispatcher"](second)
                assert second == []
            finally:
                dispatcher.CudagraphDispatcher.dispatch = stubs.FakeDispatcher.dispatch


# ---------------------------------------------------------------------------
# patch_breakable_cudagraph_mode
# ---------------------------------------------------------------------------


class TestBreakableCudagraphMode:
    @pytest.mark.platforms("kunlunxin")
    def test_all_three_modules_report_enabled(self):
        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_breakable_cudagraph_mode"](replacements)

            staged = [(r[0], r[1], r[3]) for r in replacements]
            assert len(staged) == 3
            owners = {id(r[0]) for r in staged}
            assert owners == {
                id(stubs.gpu_model_runner),
                id(stubs.fl_model_runner),
            } | {id(sys.modules["vllm.compilation.breakable_cudagraph"])}
            for owner, name, replacement in staged:
                assert name == "is_breakable_cudagraph_enabled"
                assert replacement() is True

    @pytest.mark.platforms("kunlunxin")
    def test_staging_does_not_publish(self):
        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_breakable_cudagraph_mode"](replacements)
            assert stubs.gpu_model_runner.is_breakable_cudagraph_enabled() is not None
            # the fake still holds its own function object
            assert replacements, "staging recorded at least one replacement"


# ---------------------------------------------------------------------------
# patch_breakable_private_pools
# ---------------------------------------------------------------------------


class TestBreakablePrivatePools:
    @pytest.mark.platforms("kunlunxin")
    def test_init_resets_pool_and_creates_descriptor_dict(self):
        import vllm.compilation.breakable_cudagraph as real_breakable

        def fake_orig_init(self, runnable, vllm_config):
            self.graph_pool = "GLOBAL"
            self.seen = (runnable, vllm_config)

        real_init = real_breakable.BreakableCUDAGraphWrapper.__init__
        real_breakable.BreakableCUDAGraphWrapper.__init__ = fake_orig_init
        try:
            with _ModuleStubs():
                ns = _load_patch_module()
                replacements = []
                ns["patch_breakable_private_pools"](replacements)
                init_with_private_pools = replacements[0][3]
                assert getattr(
                    init_with_private_pools, "_kunlunxin_private_pools", False
                ) is True

                target = SimpleNamespace()
                config = SimpleNamespace(
                    compilation_config=SimpleNamespace(cudagraph_mode=None)
                )
                init_with_private_pools(target, "RUN", config)
                assert target.graph_pool is None
                assert target._kunlunxin_descriptor_pools == {}
                assert target.seen == ("RUN", config)
        finally:
            real_breakable.BreakableCUDAGraphWrapper.__init__ = real_init

    @pytest.mark.platforms("kunlunxin")
    def test_already_flagged_init_is_left_alone(self):
        import vllm.compilation.breakable_cudagraph as real_breakable

        with _ModuleStubs():
            ns = _load_patch_module()
            real_init = real_breakable.BreakableCUDAGraphWrapper.__init__
            try:
                # install once, then staging over the flagged wrapper is a no-op
                ns["patch_breakable_private_pools"](None)
                installed = real_breakable.BreakableCUDAGraphWrapper.__init__
                assert getattr(installed, "_kunlunxin_private_pools", False) is True
                second = []
                ns["patch_breakable_private_pools"](second)
                assert second == []
            finally:
                real_breakable.BreakableCUDAGraphWrapper.__init__ = real_init


# ---------------------------------------------------------------------------
# patch_breakable_full_only: per-descriptor pool
# ---------------------------------------------------------------------------


class TestBreakableFullOnly:
    @pytest.mark.platforms("kunlunxin")
    def test_per_descriptor_pool_reuse_and_restore(self):
        import vllm.platforms as platforms
        import vllm.forward_context as forward_context_module
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, ForwardContext

        with _ModuleStubs():
            # Swap the real class __call__ FIRST so the staged closure's
            # captured "original" is our recorder (no CUDA capture).
            import vllm.compilation.breakable_cudagraph as real_breakable

            seen_pools = []

            def recording_call(self, *args, **kwargs):
                seen_pools.append(self.graph_pool)
                return ("ORIG_CALL", args, self.graph_pool)

            real_call = real_breakable.BreakableCUDAGraphWrapper.__call__
            real_breakable.BreakableCUDAGraphWrapper.__call__ = recording_call
            try:
                ns = _load_patch_module()
                replacements = []
                ns["patch_breakable_full_only"](replacements)
                call_full_only = replacements[0][3]
                assert getattr(
                    call_full_only, "_kunlunxin_full_only", False
                ) is True

                class FakeWrapper:
                    def __init__(self):
                        self.graph_pool = "ORIGINAL_POOL"
                        self.entries = {}
                        self.runnable = lambda *a, **k: ("ORIG_CALL", a)
                        self.runtime_mode = None

                # graph_pool_handle is swapped so the descriptor pool is
                # identifiable without touching real CUDA pools.
                pools = iter(f"POOL_{i}" for i in range(10))
                real_handle = platforms.current_platform.graph_pool_handle
                platforms.current_platform.graph_pool_handle = lambda: next(pools)
                ctx = ForwardContext(
                    no_compile_layers={}, attn_metadata={}, slot_mapping={}
                )
                prev_ctx = forward_context_module._forward_context
                forward_context_module._forward_context = ctx
                try:
                    wrapper = FakeWrapper()
                    # NONE mode: descriptor branch skipped, pool untouched
                    ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
                    result = call_full_only(wrapper, 1)
                    assert result[0] == "ORIG_CALL"
                    assert wrapper.graph_pool == "ORIGINAL_POOL"

                    # FULL mode, new descriptor: pool allocated, restored after call
                    ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
                    first_descriptor = BatchDescriptor(
                        num_tokens=4,
                        num_reqs=2,
                        uniform=True,
                        has_lora=False,
                        num_active_loras=0,
                    )
                    ctx.batch_descriptor = first_descriptor
                    result = call_full_only(wrapper, 2)
                    assert result[0] == "ORIG_CALL"
                    assert result[2] == "POOL_0", "call ran with the new pool"
                    assert wrapper.graph_pool == "ORIGINAL_POOL", "pool restored"
                    assert set(wrapper._kunlunxin_descriptor_pools) == {
                        first_descriptor
                    }

                    # same descriptor: the pool is reused, not reallocated
                    result = call_full_only(wrapper, 3)
                    assert result[2] == "POOL_0"
                    assert len(wrapper._kunlunxin_descriptor_pools) == 1

                    # a second descriptor gets a different pool
                    second_descriptor = BatchDescriptor(
                        num_tokens=8,
                        num_reqs=2,
                        uniform=True,
                        has_lora=False,
                        num_active_loras=0,
                    )
                    ctx.batch_descriptor = second_descriptor
                    result = call_full_only(wrapper, 4)
                    assert result[2] == "POOL_1"
                    assert len(wrapper._kunlunxin_descriptor_pools) == 2
                finally:
                    forward_context_module._forward_context = prev_ctx
                    platforms.current_platform.graph_pool_handle = real_handle
            finally:
                real_breakable.BreakableCUDAGraphWrapper.__call__ = real_call

    @pytest.mark.platforms("kunlunxin")
    def test_descriptor_already_in_entries_skips_pool_branch(self):
        import vllm.forward_context as forward_context_module
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, ForwardContext

        with _ModuleStubs():
            import vllm.compilation.breakable_cudagraph as real_breakable

            def known_call(self, *args, **kwargs):
                return ("KNOWN_DESCRIPTOR", self.graph_pool)

            real_call = real_breakable.BreakableCUDAGraphWrapper.__call__
            real_breakable.BreakableCUDAGraphWrapper.__call__ = known_call
            try:
                ns = _load_patch_module()
                replacements = []
                ns["patch_breakable_full_only"](replacements)
                call_full_only = replacements[0][3]

                descriptor = BatchDescriptor(
                    num_tokens=4,
                    num_reqs=2,
                    uniform=True,
                    has_lora=False,
                    num_active_loras=0,
                )

                class FakeWrapper:
                    runnable = staticmethod(lambda *a, **k: ("KNOWN", a))
                    graph_pool = "ORIGINAL_POOL"
                    entries = {descriptor: SimpleNamespace(capture="CAPTURED")}
                    runtime_mode = None

                ctx = ForwardContext(
                    no_compile_layers={}, attn_metadata={}, slot_mapping={}
                )
                ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
                ctx.batch_descriptor = descriptor
                prev_ctx = forward_context_module._forward_context
                forward_context_module._forward_context = ctx
                try:
                    wrapper = FakeWrapper()
                    result = call_full_only(wrapper, 1)
                    assert result == ("KNOWN_DESCRIPTOR", "ORIGINAL_POOL")
                    assert not hasattr(
                        wrapper, "_kunlunxin_descriptor_pools"
                    ), "known descriptor must not allocate pools"
                finally:
                    forward_context_module._forward_context = prev_ctx
            finally:
                real_breakable.BreakableCUDAGraphWrapper.__call__ = real_call

    @pytest.mark.platforms("kunlunxin")
    def test_piecewise_with_compiled_runner_bypasses_wrapper(self):
        import vllm.forward_context as forward_context_module
        from vllm.config import CUDAGraphMode
        from vllm.config.compilation import CompilationMode
        from vllm.forward_context import ForwardContext

        with _ModuleStubs():
            # Install a recording original BEFORE staging so the wrapper's
            # captured "original" is safe to call.
            import vllm.compilation.breakable_cudagraph as real_breakable

            def bypassed_call(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("wrapper __call__ must be bypassed")

            real_call = real_breakable.BreakableCUDAGraphWrapper.__call__
            real_breakable.BreakableCUDAGraphWrapper.__call__ = bypassed_call
            try:
                ns = _load_patch_module()
                replacements = []
                ns["patch_breakable_full_only"](replacements)
                call_full_only = replacements[0][3]
                # runpy.run_path returns a copy of the module namespace, so
                # the helper substitution must go into the wrapper's own
                # function globals.
                fn_globals = call_full_only.__globals__
                real_has = fn_globals["_has_compiled_piecewise_runner"]
                fn_globals["_has_compiled_piecewise_runner"] = (
                    lambda wrapper: True
                )

                class FakeWrapper:
                    runnable = staticmethod(lambda *a, **k: ("COMPILED", a))
                    runtime_mode = None
                    entries = {}
                    # real _has_compiled_piecewise_runner reads these when
                    # the substituted lambda is bypassed by a different
                    # module-namespace copy; provide a VLLM_COMPILE-shaped
                    # config so either path concludes "compiled runner".
                    compilation_config = SimpleNamespace(
                        mode=CompilationMode.VLLM_COMPILE,
                        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                    )

                ctx = ForwardContext(
                    no_compile_layers={}, attn_metadata={}, slot_mapping={}
                )
                ctx.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
                ctx.skip_compiled = False
                prev_ctx = forward_context_module._forward_context
                forward_context_module._forward_context = ctx
                try:
                    result = call_full_only(FakeWrapper(), 5)
                    assert result[0] == "COMPILED"
                finally:
                    forward_context_module._forward_context = prev_ctx
            finally:
                fn_globals["_has_compiled_piecewise_runner"] = real_has
                real_breakable.BreakableCUDAGraphWrapper.__call__ = real_call

    @pytest.mark.platforms("kunlunxin")
    def test_no_forward_context_is_verbatim_passthrough(self):
        import vllm.forward_context as forward_context_module

        with _ModuleStubs():
            import vllm.compilation.breakable_cudagraph as real_breakable

            def passthrough_call(self, *args, **kwargs):
                return ("NO_CTX", args)

            real_call = real_breakable.BreakableCUDAGraphWrapper.__call__
            real_breakable.BreakableCUDAGraphWrapper.__call__ = passthrough_call
            try:
                ns = _load_patch_module()
                replacements = []
                ns["patch_breakable_full_only"](replacements)
                call_full_only = replacements[0][3]

                class FakeWrapper:
                    runnable = staticmethod(lambda *a, **k: ("NO_CTX", a))
                    runtime_mode = None

                prev_ctx = forward_context_module._forward_context
                forward_context_module._forward_context = None
                try:
                    result = call_full_only(FakeWrapper(), 1, kw=2)
                    assert result == ("NO_CTX", (1,))
                finally:
                    forward_context_module._forward_context = prev_ctx
            finally:
                real_breakable.BreakableCUDAGraphWrapper.__call__ = real_call


# ---------------------------------------------------------------------------
# _has_compiled_piecewise_runner
# ---------------------------------------------------------------------------


class TestHasCompiledPiecewiseRunner:
    """``_has_compiled_piecewise_runner`` inspects ``wrapper.runnable``.

    The runnable's ``modules()`` are checked for
    ``TorchCompileWithNoGuardsWrapper`` instances that opted into
    compilation and actually compiled something.  These tests build real
    wrapper objects (its ``__init__`` takes only optional args) so the
    ``isinstance`` check is exercised for real.
    """

    def _make_wrapper(self, modules, *, mode=None, piecewise=True, skip=False):
        import vllm.forward_context as forward_context_module
        from vllm.config import CompilationMode

        # The helper only calls runnable.modules() when the runnable is an
        # nn.Module, so carry the compiled modules on a Module whose
        # modules() yields exactly them.
        class _Runnable(torch.nn.Module):
            def __init__(self, children):
                super().__init__()
                self._children = tuple(children)

            def modules(self):
                return iter(self._children)

        wrapper = SimpleNamespace(
            compilation_config=SimpleNamespace(
                mode=CompilationMode.VLLM_COMPILE
                if mode is None
                else mode,
                cudagraph_mode=SimpleNamespace(
                    has_piecewise_cudagraphs=lambda: piecewise
                ),
            ),
            runnable=_Runnable(modules),
        )
        ctx = SimpleNamespace(skip_compiled=skip)
        prev = forward_context_module._forward_context
        forward_context_module._forward_context = ctx
        return wrapper, prev, forward_context_module

    def _restore(self, prev, module):
        module._forward_context = prev

    def test_mode_not_vllm_compile_is_false(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        with _vllm_config_context():
            compiled = TorchCompileWithNoGuardsWrapper()
            compiled.do_not_compile = False
            compiled.compiled = True
            compiled._compiled_bytecode = b"x"
            wrapper, prev, mod = self._make_wrapper(
                (compiled,), mode="NOT_VLLM_COMPILE"
            )
            try:
                assert has(wrapper) is False
            finally:
                self._restore(prev, mod)

    def test_no_piecewise_cudagraphs_is_false(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        with _vllm_config_context():
            compiled = TorchCompileWithNoGuardsWrapper()
            compiled.do_not_compile = False
            compiled.compiled = True
            compiled._compiled_bytecode = b"x"
            wrapper, prev, mod = self._make_wrapper((compiled,), piecewise=False)
            try:
                assert has(wrapper) is False
            finally:
                self._restore(prev, mod)

    def test_skip_compiled_context_is_false(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        with _vllm_config_context():
            compiled = TorchCompileWithNoGuardsWrapper()
            compiled.do_not_compile = False
            compiled.compiled = True
            compiled._compiled_bytecode = b"x"
            wrapper, prev, mod = self._make_wrapper((compiled,), skip=True)
            try:
                assert has(wrapper) is False
            finally:
                self._restore(prev, mod)

    def test_compiled_module_with_bytecode_is_true(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        with _vllm_config_context():
            compiled = TorchCompileWithNoGuardsWrapper()
            compiled.do_not_compile = False
            compiled.compiled = True
            compiled._compiled_bytecode = b"x"
            wrapper, prev, mod = self._make_wrapper((compiled,))
            try:
                assert has(wrapper) is True
            finally:
                self._restore(prev, mod)

    def test_do_not_compile_module_is_ignored(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        with _vllm_config_context():
            compiled = TorchCompileWithNoGuardsWrapper()
            compiled.do_not_compile = True  # opted out
            compiled.compiled = True
            compiled._compiled_bytecode = b"x"
            wrapper, prev, mod = self._make_wrapper((compiled,))
            try:
                assert has(wrapper) is False
            finally:
                self._restore(prev, mod)

    def test_aot_compiled_callable_marks_compiled(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        with _vllm_config_context():
            compiled = TorchCompileWithNoGuardsWrapper()
            compiled.do_not_compile = False
            # the predicate requires compiled=True for the callable path
            compiled.compiled = True

            def compiled_callable():
                pass

            compiled_callable._torchdynamo_orig_callable = lambda: None
            compiled._compiled_callable = compiled_callable
            wrapper, prev, mod = self._make_wrapper((compiled,))
            try:
                assert has(wrapper) is True
            finally:
                self._restore(prev, mod)

    def test_compiled_callable_without_dynamo_marker_does_not_count(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        with _vllm_config_context():
            compiled = TorchCompileWithNoGuardsWrapper()
            compiled.do_not_compile = False
            compiled.compiled = True

            def plain_callable():
                pass

            compiled._compiled_callable = plain_callable
            wrapper, prev, mod = self._make_wrapper((compiled,))
            try:
                assert has(wrapper) is False
            finally:
                self._restore(prev, mod)

    def test_aot_compiled_fn_marks_compiled(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        with _vllm_config_context():
            compiled = TorchCompileWithNoGuardsWrapper()
            compiled.do_not_compile = False
            compiled.aot_compiled_fn = lambda: None
            wrapper, prev, mod = self._make_wrapper((compiled,))
            try:
                assert has(wrapper) is True
            finally:
                self._restore(prev, mod)

    def test_uncompiled_module_does_not_count(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        with _vllm_config_context():
            compiled = TorchCompileWithNoGuardsWrapper()
            compiled.do_not_compile = False
            compiled.compiled = False
            wrapper, prev, mod = self._make_wrapper((compiled,))
            try:
                assert has(wrapper) is False
            finally:
                self._restore(prev, mod)

    def test_non_module_runnable_is_treated_as_single_module(self):
        ns = _load_patch_module()
        has = ns["_has_compiled_piecewise_runner"]
        # a plain object (not nn.Module): the runnable itself is the only
        # "module" tuple element; a SimpleNamespace is not a
        # TorchCompileWithNoGuardsWrapper, so the answer is False
        wrapper = SimpleNamespace(
            compilation_config=SimpleNamespace(
                mode="NOT_USED",
                cudagraph_mode=SimpleNamespace(
                    has_piecewise_cudagraphs=lambda: False
                ),
            ),
            runnable=SimpleNamespace(compiled=True),
        )
        assert has(wrapper) is False


# ---------------------------------------------------------------------------
# patch_native_piecewise_boundaries
# ---------------------------------------------------------------------------


class TestNativePiecewiseBoundaries:
    @pytest.mark.platforms("kunlunxin")
    def test_stages_both_boundaries(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackendImpl,
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )

        real_forward = KunlunxinAttentionBackendImpl.forward
        real_core = QwenGatedDeltaNetAttention._forward_core

        def fake_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                         k_scale=None, v_scale=None, attn_type=None, output=None):
            return ("ORIG_FWD", query)

        def fake_core(self, mixed_qkv, b, a, core_attn_out):
            return ("ORIG_CORE", mixed_qkv)

        KunlunxinAttentionBackendImpl.forward = fake_forward
        QwenGatedDeltaNetAttention._forward_core = fake_core
        try:
            with _ModuleStubs():
                ns = _load_patch_module()
                replacements = []
                ns["patch_native_piecewise_boundaries"](replacements)
                assert len(replacements) == 2
                names = {r[1] for r in replacements}
                assert names == {"forward", "_forward_core"}
                for _, _, _, replacement, _ in replacements:
                    assert (
                        getattr(
                            replacement, "_kunlunxin_piecewise_boundary", False
                        )
                        is True
                    )
        finally:
            KunlunxinAttentionBackendImpl.forward = real_forward
            QwenGatedDeltaNetAttention._forward_core = real_core

    @pytest.mark.platforms("kunlunxin")
    def test_already_flagged_boundaries_are_skipped(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackendImpl,
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )

        real_forward = KunlunxinAttentionBackendImpl.forward
        real_core = QwenGatedDeltaNetAttention._forward_core

        def fake_forward(self, *a, **k):
            return "ORIG"

        def fake_core(self, *a, **k):
            return "ORIG"

        fake_forward._kunlunxin_piecewise_boundary = True
        fake_core._kunlunxin_piecewise_boundary = True
        KunlunxinAttentionBackendImpl.forward = fake_forward
        QwenGatedDeltaNetAttention._forward_core = fake_core
        try:
            with _ModuleStubs():
                ns = _load_patch_module()
                replacements = []
                ns["patch_native_piecewise_boundaries"](replacements)
                assert replacements == []
        finally:
            KunlunxinAttentionBackendImpl.forward = real_forward
            QwenGatedDeltaNetAttention._forward_core = real_core

    @pytest.mark.platforms("kunlunxin")
    def test_boundary_passthrough_when_not_capturing(self):
        import vllm.compilation.breakable_cudagraph as real_breakable

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackendImpl,
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )

        real_forward = KunlunxinAttentionBackendImpl.forward
        real_core = QwenGatedDeltaNetAttention._forward_core
        real_current = real_breakable.BreakableCUDAGraphCapture.current

        def fake_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                         k_scale=None, v_scale=None, attn_type=None, output=None):
            return ("ORIG_FWD", query)

        def fake_core(self, mixed_qkv, b, a, core_attn_out):
            return ("ORIG_CORE", mixed_qkv)

        KunlunxinAttentionBackendImpl.forward = fake_forward
        QwenGatedDeltaNetAttention._forward_core = fake_core
        try:
            with _ModuleStubs():
                ns = _load_patch_module()
                replacements = []
                ns["patch_native_piecewise_boundaries"](replacements)
                forward_boundary = next(
                    r for r in replacements if r[1] == "forward"
                )[3]
                core_boundary = next(
                    r for r in replacements if r[1] == "_forward_core"
                )[3]

                # no capture at all
                real_breakable.BreakableCUDAGraphCapture.current = classmethod(
                    lambda cls: None
                )
                q = torch.zeros(2, 4)
                assert forward_boundary(
                    object(), layer="L", query=q, key=None, value=None,
                    kv_cache=None, attn_metadata="META", output="OUT",
                ) == ("ORIG_FWD", q)
                assert core_boundary(
                    object(), q, q, q, q
                ) == ("ORIG_CORE", q)
        finally:
            real_breakable.BreakableCUDAGraphCapture.current = real_current
            KunlunxinAttentionBackendImpl.forward = real_forward
            QwenGatedDeltaNetAttention._forward_core = real_core

    @pytest.mark.platforms("kunlunxin")
    def test_boundary_passthrough_when_capture_not_capturing(self):
        import vllm.compilation.breakable_cudagraph as real_breakable

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackendImpl,
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )

        real_forward = KunlunxinAttentionBackendImpl.forward
        real_core = QwenGatedDeltaNetAttention._forward_core
        real_current = real_breakable.BreakableCUDAGraphCapture.current

        def fake_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                         k_scale=None, v_scale=None, attn_type=None, output=None):
            return "ORIG_FWD"

        def fake_core(self, mixed_qkv, b, a, core_attn_out):
            return "ORIG_CORE"

        KunlunxinAttentionBackendImpl.forward = fake_forward
        QwenGatedDeltaNetAttention._forward_core = fake_core

        class IdleCapture:
            _capturing = False

        try:
            with _ModuleStubs():
                ns = _load_patch_module()
                replacements = []
                ns["patch_native_piecewise_boundaries"](replacements)
                forward_boundary = next(
                    r for r in replacements if r[1] == "forward"
                )[3]

                real_breakable.BreakableCUDAGraphCapture.current = classmethod(
                    lambda cls: IdleCapture()
                )
                assert forward_boundary(
                    object(), layer="L", query=torch.zeros(1), key=None,
                    value=None, kv_cache=None, attn_metadata="META", output="OUT",
                ) == "ORIG_FWD"
        finally:
            real_breakable.BreakableCUDAGraphCapture.current = real_current
            KunlunxinAttentionBackendImpl.forward = real_forward
            QwenGatedDeltaNetAttention._forward_core = real_core

    @pytest.mark.platforms("kunlunxin")
    def test_boundary_eager_segment_in_piecewise_mode(self):
        import vllm.compilation.breakable_cudagraph as real_breakable
        import vllm.forward_context as forward_context_module
        import vllm.model_executor.layers.attention.attention as attention_module
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import ForwardContext

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackendImpl,
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )

        real_forward = KunlunxinAttentionBackendImpl.forward
        real_core = QwenGatedDeltaNetAttention._forward_core
        real_current = real_breakable.BreakableCUDAGraphCapture.current

        def fake_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                         k_scale=None, v_scale=None, attn_type=None, output=None):
            return ("ORIG_FWD", layer, query)

        def fake_core(self, mixed_qkv, b, a, core_attn_out):
            return ("ORIG_CORE", mixed_qkv)

        KunlunxinAttentionBackendImpl.forward = fake_forward
        QwenGatedDeltaNetAttention._forward_core = fake_core

        # weak_ref_tensor needs the compiled vllm_fl._C extension; the
        # boundary wrapper only uses it to carry tensor args through the
        # eager segment, so an identity stand-in preserves the contract.
        import vllm.utils.torch_utils as torch_utils

        real_weak_ref = torch_utils.weak_ref_tensor
        torch_utils.weak_ref_tensor = lambda tensor: tensor

        # patch_native_piecewise_boundaries binds get_attention_context into
        # the replay closure AT STAGING TIME, so the recording substitute
        # must be installed before staging too.
        restored = {}
        original_get = attention_module.get_attention_context

        def recording_get(name):
            restored[name] = f"CTX[{name}]"
            return (f"CTX[{name}]", None)

        attention_module.get_attention_context = recording_get

        eager_fns = []

        class CapturingCapture:
            _capturing = True

            def add_eager(self, fn):
                eager_fns.append(fn)
                return "ADDED"

        try:
            with _ModuleStubs():
                ns = _load_patch_module()
                replacements = []
                ns["patch_native_piecewise_boundaries"](replacements)
                forward_boundary = next(
                    r for r in replacements if r[1] == "forward"
                )[3]
                core_boundary = next(
                    r for r in replacements if r[1] == "_forward_core"
                )[3]

                real_breakable.BreakableCUDAGraphCapture.current = classmethod(
                    lambda cls: CapturingCapture()
                )
                ctx = ForwardContext(
                    no_compile_layers={}, attn_metadata={}, slot_mapping={}
                )
                ctx.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
                prev_ctx = forward_context_module._forward_context
                forward_context_module._forward_context = ctx

                layer = SimpleNamespace(layer_name="layers.3")
                q = torch.zeros(2, 4)
                result = forward_boundary(
                    object(), layer=layer, query=q, key=None, value=None,
                    kv_cache=None, attn_metadata="META", output="OUT",
                )
                assert result == "ADDED"
                assert len(eager_fns) == 1

                # the replay function resolves fresh layer metadata via
                # get_attention_context(layer_name) and drops attn_metadata
                replay = eager_fns.pop()
                replayed = replay()
                assert replayed == ("ORIG_FWD", layer, q)
                assert restored == {"layers.3": "CTX[layers.3]"}

                # the core boundary in PIECEWISE mode keeps every tensor
                # argument (no attn_metadata to refresh) and replays with the
                # same objects
                mixed = torch.zeros(3)
                result = core_boundary(
                    object(), mixed, torch.zeros(3), torch.zeros(3),
                    torch.zeros(3),
                )
                assert result == "ADDED"
                core_replay = eager_fns.pop()
                assert core_replay() == ("ORIG_CORE", mixed)
                forward_context_module._forward_context = prev_ctx
        finally:
            torch_utils.weak_ref_tensor = real_weak_ref
            attention_module.get_attention_context = original_get
            real_breakable.BreakableCUDAGraphCapture.current = real_current
            KunlunxinAttentionBackendImpl.forward = real_forward
            QwenGatedDeltaNetAttention._forward_core = real_core


# ---------------------------------------------------------------------------
# patch_v2_graph_capture: covered in tests/unit_tests; here only the
# _ModuleStubs-scope variant asserting it stays functional under this suite.
# ---------------------------------------------------------------------------


class TestV2GraphCaptureStillStages:
    @pytest.mark.platforms("kunlunxin")
    def test_stages_the_cudagraph_utils_alias(self, monkeypatch):
        cudagraph_utils = ModuleType("vllm.v1.worker.gpu.cudagraph_utils")
        cudagraph_utils.graph_capture = "UPSTREAM"
        monkeypatch.setitem(
            sys.modules, "vllm.v1.worker.gpu.cudagraph_utils", cudagraph_utils
        )
        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            replacements = []
            ns["patch_v2_graph_capture"](replacements)
            assert replacements == [
                (
                    cudagraph_utils,
                    "graph_capture",
                    "UPSTREAM",
                    stubs.fl_model_runner._device_graph_capture,
                    True,
                )
            ]


# ---------------------------------------------------------------------------
# patch_graph_adaptations: preflight failure and rollback
# ---------------------------------------------------------------------------


class TestGraphAdaptationsPreflight:
    @pytest.mark.platforms("kunlunxin")
    def test_incomplete_signature_raises_runtime_error(self):
        # A dispatcher whose dispatch lacks the preflight parameter names
        # makes the preflight fail without publishing anything.
        with _ModuleStubs() as stubs:
            dispatcher_cls = stubs.FakeDispatcher

            def bad_dispatch(self, num_tokens, uniform_decode=False):
                return "TOO_FEW"

            original = dispatcher_cls.dispatch
            dispatcher_cls.dispatch = bad_dispatch
            ns = _load_patch_module()
            try:
                with pytest.raises(RuntimeError, match="preflight failed"):
                    ns["patch_graph_adaptations"]()
            finally:
                dispatcher_cls.dispatch = original

    @pytest.mark.platforms("kunlunxin")
    def test_successful_publish_restores_on_repetition(self):
        # Publishing over an already-patched surface (double install) must
        # still succeed: the second run re-stages from the same owners.
        with _ModuleStubs() as stubs:
            ns = _load_patch_module()
            ns["patch_graph_adaptations"]()
            # the dispatcher received the uniform-full wrapper
            installed = stubs.FakeDispatcher.dispatch
            assert getattr(installed, "_kunlunxin_uniform_full", False) is True
            # second application is also fine (idempotent guards)
            ns["patch_graph_adaptations"]()
