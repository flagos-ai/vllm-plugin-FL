# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin patch orchestration entry point.

``vllm_fl/.../kunlunxin/patch.py`` organizes its patches through
``apply_kunlunxin_patches`` → ``_apply_kunlunxin_patches`` (the legacy
import chain plus the graph adaptations), guarded by the module-level
``_patch_state`` namespace.  The combined coverage run never executed this
orchestration chain (patch.py:30-48, 51-161): every existing test drives an
individual ``patch_*`` / ``apply_*`` function but never the entry point
itself.

These tests cover:

* the orchestration smoke — every legacy sub-patch is dispatched exactly
  once, the graph adaptations run, ``_patch_state.applied`` flips to True,
  and a second entry call is a no-op (the ``applied`` guard short-circuits
  before any sub-patch can run again); a failure inside the chain leaves
  ``applied`` False and releases the ``initializing`` guard;
* ``patch_graph_adaptations``' publish-failure rollback (238-243) — an
  owner whose attribute set raises mid-publish restores every previously
  published surface (verified through the real function, not a re-derivation);
* the ``RuntimeError`` wrapping around failed sub-patches (the
  ``patch_cudagraph_dispatcher`` try/except at patch.py:398) — a broken
  import surface is re-raised as ``RuntimeError`` with the original cause
  chained;
* ``patch_ssm_cache_update``'s fallback to ``apply_ssm_patch``
  (patch.py:1157-1171) — when the first-stage symbol import fails, the
  second-stage path runs instead of the source rewrite.

The module scope of ``patch.py`` imports only torch/inspect/os, so it is
executed with ``runpy`` (the pattern inherited from
``test_kunlunxin_patch_graph_adaptations.py``).  Sub-patch modules are
stubbed through recorded ``sys.modules`` entries restored exactly on exit;
the graph adaptations run against real vllm classes with every touched
surface snapshotted for restoration.
"""

from __future__ import annotations

import contextlib
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.supplementary]

_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py"
)

_PATCHES_PKG = "vllm_fl.dispatch.backends.vendor.kunlunxin.patches"

#: (module_name, apply_names[, extra_symbols]) for every patches/ module the
#: legacy chain imports.  Each becomes a recording stub in sys.modules; the
#: extras cover symbols other patch.py stages import from the same modules
#: (``patch_graph_all_reduce`` needs ``maybe_reduce_vision_tp``).
_LEGACY_PATCH_MODULES = {
    "accelerator_compat": (
        ("apply_accelerator_compat",),
        {},
    ),
    "triton_compat": (
        ("apply_triton_compat",),
        {},
    ),
    "kv_block_zero_compat": (
        ("apply_kv_block_zero_compat",),
        {},
    ),
    "uva_compat": (
        ("apply_uva_compat",),
        {},
    ),
    "rope_compat": (
        ("apply_rope_compat",),
        {},
    ),
    "penalties_compat": (
        ("apply_penalties_compat",),
        {},
    ),
    "input_batch_compat": (
        ("apply_input_batch_compat",),
        {},
    ),
    "block_table_compat": (
        ("apply_block_table_compat",),
        {},
    ),
    "mamba_utils_compat": (
        ("apply_mamba_utils_compat",),
        {},
    ),
    "structured_outputs_compat": (
        ("apply_structured_outputs_compat",),
        {},
    ),
    "topk_topp_compat": (
        ("apply_topk_topp_compat",),
        {},
    ),
    "vl_compat": (
        ("apply_vision_tp_scope", "apply_vl_compat"),
        # patch_graph_all_reduce imports this at staging time; None means
        # "not inside a vision TP scope", matching every non-vision reduce
        {"maybe_reduce_vision_tp": lambda comm, t: None},
    ),
    "buffer_utils_compat": (
        ("apply_buffer_utils_compat",),
        {},
    ),
    "mamba_hybrid_compat": (
        ("apply_mamba_hybrid_compat",),
        {},
    ),
    "mamba_align_compat": (
        ("apply_mamba_align_compat",),
        {},
    ),
    "v2_greedy_sampler": (
        ("apply_v2_greedy_sampler",),
        {},
    ),
}

#: patch.py-internal legacy functions dispatched by name; re-bound to
#: recorders in the runpy namespace for the orchestration smoke.
_LEGACY_LOCAL_FUNCTIONS = (
    "patch_block_table_slot_mapping",
    "patch_attention_backend_registry",
    "patch_causal_conv1d",
    "patch_fla_ops",
    "patch_fused_gdn_gating",
    "patch_ssm_cache_update",
    "patch_sampler_rng",
)

_RUNNER_MODULE_NAMES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm_fl.worker.model_runner",
)


def _load_patch_module():
    """Execute patch.py in a fresh namespace (import-safe module scope).

    ``runpy.run_path`` returns a *copy* of the module globals, so name
    re-binding for the orchestration tests must target the functions'
    shared ``__globals__`` dict (reachable through any function the module
    defines), not the returned copy.
    """
    ns = runpy.run_path(str(_PATCH_PATH))
    ns["__shared_globals__"] = ns["_apply_legacy_kunlunxin_patches"].__globals__
    return ns


@contextlib.contextmanager
def _recorded_legacy_calls():
    """Stub every legacy sub-patch the orchestrator dispatches.

    The patches/ modules the legacy chain imports are replaced in
    ``sys.modules`` by recording stubs — the orchestration is the object
    under test; the sub-patches themselves each have dedicated suites.
    Yields the call list; restores every module entry exactly on exit.
    """
    calls = []
    saved_modules = {}
    try:
        for module_name, (apply_names, extras) in _LEGACY_PATCH_MODULES.items():
            full = f"{_PATCHES_PKG}.{module_name}"
            saved_modules[full] = sys.modules.get(full)
            stub = ModuleType(full)
            for apply_name in apply_names:

                def recorder(_name=apply_name, _mod=module_name):
                    calls.append(f"{_mod}.{_name}")

                setattr(stub, apply_name, recorder)
            for extra_name, extra_value in extras.items():
                setattr(stub, extra_name, extra_value)
            sys.modules[full] = stub
        yield calls
    finally:
        for name, entry in saved_modules.items():
            if entry is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = entry


@contextlib.contextmanager
def _graph_surfaces_snapshotted():
    """Run the graph adaptations with every touched surface restored.

    Mirrors ``test_successful_publish_restores_on_repetition``: the graph
    adaptations publish onto real vllm classes, so each surface is saved
    before the publish and restored after the assertions.
    """
    import vllm.compilation.breakable_cudagraph as real_breakable
    import vllm.v1.worker.gpu.cudagraph_utils as cudagraph_utils
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    import vllm_fl.distributed.communicator as comm_mod
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
        KunlunxinAttentionBackendImpl,
    )

    surfaces = [
        (real_breakable, "is_breakable_cudagraph_enabled"),
        (real_breakable.BreakableCUDAGraphWrapper, "__init__"),
        (real_breakable.BreakableCUDAGraphWrapper, "__call__"),
        (cudagraph_utils, "graph_capture"),
        (comm_mod.CommunicatorFL, "all_reduce"),
        (comm_mod.CommunicatorFL, "all_gather"),
        (KunlunxinAttentionBackendImpl, "forward"),
        (QwenGatedDeltaNetAttention, "_forward_core"),
    ]
    snapshots = [
        (owner, name, getattr(owner, name, None), name in vars(owner))
        for owner, name in surfaces
    ]
    try:
        yield
    finally:
        for owner, name, saved, existed in snapshots:
            if existed:
                setattr(owner, name, saved)
            else:
                with contextlib.suppress(AttributeError):
                    delattr(owner, name)


@contextlib.contextmanager
def _stubbed_runner_modules(**attrs):
    """Replace the two runner modules the graph preflight imports.

    Follows ``_ModuleStubs`` from the graph-adaptations suite: the runner
    names must also be reachable through their parent packages' attributes
    because ``import a.b.c`` resolves the intermediate package by attribute
    lookup.  Restores everything on exit.
    """
    gpu_runner = ModuleType(_RUNNER_MODULE_NAMES[0])
    fl_runner = ModuleType(_RUNNER_MODULE_NAMES[1])
    for module in (gpu_runner, fl_runner):
        module.is_breakable_cudagraph_enabled = lambda: True
        module._device_graph_capture = lambda *a, **k: "FL_CAPTURE"
    for name, value in attrs.items():
        setattr(gpu_runner, name, value)
        setattr(fl_runner, name, value)

    saved = {name: sys.modules.get(name) for name in _RUNNER_MODULE_NAMES}
    import vllm.v1.worker.gpu as gpu_pkg

    saved_gpu_attr = getattr(gpu_pkg, "gpu_model_runner", None)
    import vllm_fl

    saved_fl_worker = getattr(vllm_fl, "worker", None)
    if saved_fl_worker is None:
        vllm_fl.worker = ModuleType("vllm_fl.worker")
    import vllm_fl.worker

    saved_fl_mr_attr = getattr(vllm_fl.worker, "model_runner", None)
    sys.modules[_RUNNER_MODULE_NAMES[0]] = gpu_runner
    sys.modules[_RUNNER_MODULE_NAMES[1]] = fl_runner
    gpu_pkg.gpu_model_runner = gpu_runner
    vllm_fl.worker.model_runner = fl_runner
    try:
        yield SimpleNamespace(gpu_runner=gpu_runner, fl_runner=fl_runner)
    finally:
        if saved_gpu_attr is None:
            with contextlib.suppress(AttributeError):
                del gpu_pkg.gpu_model_runner
        else:
            gpu_pkg.gpu_model_runner = saved_gpu_attr
        if saved_fl_mr_attr is None:
            with contextlib.suppress(AttributeError):
                del vllm_fl.worker.model_runner
        else:
            vllm_fl.worker.model_runner = saved_fl_mr_attr
        if saved_fl_worker is None:
            with contextlib.suppress(AttributeError):
                del vllm_fl.worker
        else:
            vllm_fl.worker = saved_fl_worker
        for name, entry in saved.items():
            if entry is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = entry


# ---------------------------------------------------------------------------
# apply_kunlunxin_patches orchestration (30-48, 51-161)
# ---------------------------------------------------------------------------


class TestApplyKunlunxinPatchesOrchestration:
    @pytest.mark.platforms("kunlunxin")
    def test_entry_point_dispatches_every_legacy_patch_once(self):
        with _stubbed_runner_modules():
            ns = _load_patch_module()
            state = ns["_patch_state"]
            assert state.applied is False

            # re-bind the patch.py-internal legacy functions to recorders
            # in the functions' shared __globals__ (the runpy copy is not
            # the live namespace the orchestrator resolves names from)
            shared_globals = ns["__shared_globals__"]
            local_calls = []
            for fn_name in _LEGACY_LOCAL_FUNCTIONS:

                def recorder(_name=fn_name):
                    local_calls.append(_name)

                shared_globals[fn_name] = recorder

            with (
                _recorded_legacy_calls() as module_calls,
                _graph_surfaces_snapshotted(),
            ):
                ns["apply_kunlunxin_patches"]()

            assert state.applied is True
            assert state.initializing is False
            # every module-level legacy patch ran exactly once
            expected_module_calls = sorted(
                f"{module}.{name}"
                for module, (names, _extras) in _LEGACY_PATCH_MODULES.items()
                for name in names
            )
            assert sorted(module_calls) == expected_module_calls
            assert len(module_calls) == len(set(module_calls)), (
                "each sub-patch must be dispatched exactly once"
            )
            # every patch.py-internal legacy function ran exactly once
            assert sorted(local_calls) == sorted(_LEGACY_LOCAL_FUNCTIONS)
            assert len(local_calls) == len(set(local_calls))

    @pytest.mark.platforms("kunlunxin")
    def test_second_call_is_guarded_no_op(self):
        with _stubbed_runner_modules():
            ns = _load_patch_module()
            state = ns["_patch_state"]

            with (
                _recorded_legacy_calls() as module_calls,
                _graph_surfaces_snapshotted(),
            ):
                ns["apply_kunlunxin_patches"]()
                assert state.applied is True
                count_after_first = len(module_calls)
                # a second entry must short-circuit on the applied flag
                # before dispatching anything again
                ns["apply_kunlunxin_patches"]()
                assert len(module_calls) == count_after_first

    @pytest.mark.platforms("kunlunxin")
    def test_failed_patch_does_not_flip_the_applied_flag(self):
        with _stubbed_runner_modules():
            ns = _load_patch_module()
            state = ns["_patch_state"]

            def boom():
                raise RuntimeError("sub-patch exploded")

            ns["__shared_globals__"]["patch_sampler_rng"] = boom

            with (
                _recorded_legacy_calls() as module_calls,
                _graph_surfaces_snapshotted(),
                pytest.raises(RuntimeError, match="sub-patch exploded"),
            ):
                ns["apply_kunlunxin_patches"]()

            assert state.applied is False
            assert state.initializing is False, (
                "the initializing guard must be released even on failure"
            )
            # the chain is ordered: patches before the boom still ran
            assert module_calls, "earlier legacy patches ran before the failure"
            # legacy_applied is only set after the whole legacy chain
            # returns — the boom inside it leaves the flag False, so a
            # retry re-runs the legacy patches too
            assert state.legacy_applied is False


# ---------------------------------------------------------------------------
# patch_graph_adaptations publish-failure rollback (238-243)
# ---------------------------------------------------------------------------


class TestGraphAdaptationsPublishRollback:
    @pytest.mark.platforms("kunlunxin")
    def test_mid_publish_failure_rolls_back_every_published_surface(self):
        """A failing setattr mid-publish restores the earlier surfaces.

        The staging order inside ``patch_graph_adaptations`` is
        dispatcher, v2 graph capture, breakable mode (three module
        surfaces), private pools, full-only, collectives, boundaries.
        Replacing one of the breakable-mode runner modules with a module
        whose attribute set raises makes the publish loop fail after the
        dispatcher and graph-capture surfaces are already published; the
        rollback must restore both (the real code path, driven through
        the real ``patch_graph_adaptations``).
        """
        import vllm.compilation.breakable_cudagraph as real_breakable
        import vllm.v1.worker.gpu.cudagraph_utils as cudagraph_utils
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

        class _BoomModule(ModuleType):
            """Resolves attributes dynamically; every setattr raises.

            The dynamic ``__getattr__`` keeps the preflight and staging
            reads working while leaving the module ``__dict__`` empty, so
            the staged entry records ``owned=False`` and the rollback loop
            skips this owner (both of its branches) instead of raising
            while restoring it — the production rollback loop is not
            guarded, so a failing restore would abort the whole rollback.
            """

            def __getattr__(self, name):
                if name == "is_breakable_cudagraph_enabled":
                    return lambda: True
                if name == "_device_graph_capture":
                    return lambda *a, **k: "FL_CAPTURE"
                raise AttributeError(name)

            def __setattr__(self, name, value):
                raise ValueError("simulated publish failure")

        _BoomModule("vllm_fl.worker.model_runner")
        _BoomModule("vllm.v1.worker.gpu_model_runner")

        # snapshots of every surface published BEFORE the boom owner
        dispatch_before = CudagraphDispatcher.dispatch
        graph_capture_before = cudagraph_utils.graph_capture
        # the boom modules themselves are staged by
        # patch_breakable_cudagraph_mode as the third publish step
        real_enabled = real_breakable.is_breakable_cudagraph_enabled

        gpu_runner = ModuleType("vllm.v1.worker.gpu_model_runner")
        fl_runner = _BoomModule("vllm_fl.worker.model_runner")
        gpu_runner.is_breakable_cudagraph_enabled = lambda: True
        gpu_runner._device_graph_capture = lambda *a, **k: "FL_CAPTURE"

        runner_names = _RUNNER_MODULE_NAMES
        saved = {name: sys.modules.get(name) for name in runner_names}
        import vllm.v1.worker.gpu as gpu_pkg

        saved_gpu_attr = getattr(gpu_pkg, "gpu_model_runner", None)
        import vllm_fl

        saved_fl_worker = getattr(vllm_fl, "worker", None)
        if saved_fl_worker is None:
            vllm_fl.worker = ModuleType("vllm_fl.worker")
        import vllm_fl.worker

        saved_fl_mr_attr = getattr(vllm_fl.worker, "model_runner", None)
        sys.modules[runner_names[0]] = gpu_runner
        sys.modules[runner_names[1]] = fl_runner
        gpu_pkg.gpu_model_runner = gpu_runner
        vllm_fl.worker.model_runner = fl_runner
        try:
            ns = _load_patch_module()
            with pytest.raises(RuntimeError, match="installation failed"):
                ns["patch_graph_adaptations"]()

            # The rollback contract: surfaces published before the failing
            # owner are back to their pre-call objects.
            assert CudagraphDispatcher.dispatch is dispatch_before, (
                "the already-published dispatcher surface must be restored"
            )
            assert cudagraph_utils.graph_capture is graph_capture_before, (
                "the already-published graph-capture alias must be restored"
            )
            assert real_breakable.is_breakable_cudagraph_enabled is real_enabled
        finally:
            # restore the runner modules and parent attributes regardless
            if saved_gpu_attr is None:
                with contextlib.suppress(AttributeError):
                    del gpu_pkg.gpu_model_runner
            else:
                gpu_pkg.gpu_model_runner = saved_gpu_attr
            if saved_fl_mr_attr is None:
                with contextlib.suppress(AttributeError):
                    del vllm_fl.worker.model_runner
            else:
                vllm_fl.worker.model_runner = saved_fl_mr_attr
            if saved_fl_worker is None:
                with contextlib.suppress(AttributeError):
                    del vllm_fl.worker
            else:
                vllm_fl.worker = saved_fl_worker
            for name, entry in saved.items():
                if entry is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = entry
            # the rollback itself already restored these, but re-assert the
            # pre-call state survived the whole interaction
            assert CudagraphDispatcher.dispatch is dispatch_before


# ---------------------------------------------------------------------------
# RuntimeError wrapping of failed sub-patches (398 et al.)
# ---------------------------------------------------------------------------


class TestPatchFunctionErrorWrapping:
    @pytest.mark.platforms("kunlunxin")
    def test_broken_import_raises_runtime_error_with_cause(self):
        """patch_cudagraph_dispatcher wraps any failure in RuntimeError."""
        ns = _load_patch_module()
        # Poison the vllm.v1.cudagraph_dispatcher module entry so the lazy
        # from-import inside the patch function raises ImportError; the
        # try/except must re-raise it as RuntimeError with the cause kept.
        stub_name = "vllm.v1.cudagraph_dispatcher"
        saved = sys.modules.get(stub_name)
        sys.modules[stub_name] = None  # a None entry makes from-import raise
        try:
            with pytest.raises(RuntimeError, match="Failed to patch") as info:
                ns["patch_cudagraph_dispatcher"](None)
            assert isinstance(info.value.__cause__, ImportError), (
                "the RuntimeError must chain the original ImportError"
            )
        finally:
            if saved is None:
                sys.modules.pop(stub_name, None)
            else:
                sys.modules[stub_name] = saved

    @pytest.mark.platforms("kunlunxin")
    def test_signature_failure_raises_runtime_error_with_cause(self):
        ns = _load_patch_module()
        stub_name = "vllm.v1.cudagraph_dispatcher"
        saved = sys.modules.get(stub_name)

        # import succeeds, but reading CudagraphDispatcher.dispatch explodes
        # with a non-ImportError, exercising the except branch's cause
        # chaining for arbitrary sub-patch failures.
        class _ExplodingDispatch:
            def __get__(self, obj, objtype=None):
                raise TypeError("dispatch descriptor exploded")

        stub = ModuleType(stub_name)

        class BrokenDispatcher:
            dispatch = _ExplodingDispatch()

        stub.CudagraphDispatcher = BrokenDispatcher
        sys.modules[stub_name] = stub
        try:
            with pytest.raises(RuntimeError, match="Failed to patch") as info:
                ns["patch_cudagraph_dispatcher"](None)
            assert isinstance(info.value.__cause__, TypeError), (
                "the RuntimeError must chain the original TypeError"
            )
        finally:
            if saved is None:
                sys.modules.pop(stub_name, None)
            else:
                sys.modules[stub_name] = saved


# ---------------------------------------------------------------------------
# patch_ssm_cache_update fallback to apply_ssm_patch (1157-1171)
# ---------------------------------------------------------------------------


class TestSsmCacheUpdateFallback:
    """The two-stage dispatch of patch_ssm_cache_update.

    The primary path rewrites ``_forward_core``'s source; its first act is
    importing ``_kunlunxin_write_ssm_cache`` from patches/patch_forward_core.
    When that import fails, the function falls through to the second stage
    (``apply_ssm_patch``, patch.py:1157-1162); when the second stage also
    fails, the error is re-raised as RuntimeError (1164-1171).
    """

    FW_NAME = "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_forward_core"

    @pytest.mark.platforms("kunlunxin")
    def test_import_failure_falls_back_to_apply_ssm_patch(self):
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_mod

        ns = _load_patch_module()
        cls = gdn_mod.QwenGatedDeltaNetAttention
        real_core = cls._forward_core

        fallback_calls = []
        fake_fw = ModuleType(self.FW_NAME)
        # importing _kunlunxin_write_ssm_cache from this module raises
        # ImportError (the symbol is absent), routing to the second stage
        fake_fw.apply_ssm_patch = lambda: fallback_calls.append("apply_ssm_patch")

        saved_fw = sys.modules.get(self.FW_NAME)
        sys.modules[self.FW_NAME] = fake_fw
        try:
            ns["patch_ssm_cache_update"]()
            assert fallback_calls == ["apply_ssm_patch"], (
                "the import failure must route execution to apply_ssm_patch"
            )
            # the fallback recording stub installs nothing: the class is
            # untouched (the source-rewrite path never ran)
            assert cls._forward_core is real_core
        finally:
            if saved_fw is None:
                sys.modules.pop(self.FW_NAME, None)
            else:
                sys.modules[self.FW_NAME] = saved_fw
            cls._forward_core = real_core

    @pytest.mark.platforms("kunlunxin")
    def test_fallback_failure_propagates_after_logging(self):
        """A failing apply_ssm_patch propagates its original exception.

        The second stage's except block logs the failure context and
        re-raises with a bare ``raise`` — the caller sees the original
        exception (ValueError here), not a wrapper.
        """
        ns = _load_patch_module()

        def broken_apply_ssm_patch():
            raise ValueError("apply_ssm_patch itself failed")

        fake_fw = ModuleType(self.FW_NAME)
        fake_fw.apply_ssm_patch = broken_apply_ssm_patch

        saved_fw = sys.modules.get(self.FW_NAME)
        sys.modules[self.FW_NAME] = fake_fw
        try:
            with pytest.raises(ValueError, match="apply_ssm_patch itself"):
                ns["patch_ssm_cache_update"]()
        finally:
            if saved_fw is None:
                sys.modules.pop(self.FW_NAME, None)
            else:
                sys.modules[self.FW_NAME] = saved_fw
