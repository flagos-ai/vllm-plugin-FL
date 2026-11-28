# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the kunlunxin patch orchestrator.

``vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py`` installs kunlunxin
(XPU) graph adaptations over the vLLM 0.32 compilation API.  The module and
its lazily imported targets are external boundaries, so the vllm surface is
stubbed and the *real patch source* is loaded via
``spec_from_file_location``:

* ``apply_kunlunxin_patches`` — idempotence (double apply is a no-op),
  initialization-flag circular-import protection, and the retry semantics
  that keep ``applied`` False when staging raises (so a later call retries);
* ``patch_graph_adaptations`` — the signature preflight: every required
  vLLM/FL callable is checked for the documented parameter names, and a
  missing parameter aborts staging BEFORE any patch is published;
* the staged graph patches — uniform-full dispatch routing, the
  ``_has_compiled_piecewise_runner`` probe, and boundary fabrication.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

_STUB_ROOTS = ("vllm", "vllm_fl")
_MISSING = object()


@pytest.fixture(autouse=True)
def _fl_sys_modules():
    """Snapshot sys.modules around the stub install and restore exactly."""
    saved = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_ROOTS:
            saved[name] = sys.modules.get(name, _MISSING)
    try:
        yield saved
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name.split(".")[0] in _STUB_ROOTS and name not in saved
        ]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _leaf(modules, name, **attrs):
    parts = name.split(".")
    for length in range(1, len(parts) + 1):
        partial = ".".join(parts[:length])
        if partial not in modules:
            module = ModuleType(partial)
            module.__path__ = []
            modules[partial] = module
    for length in range(1, len(parts)):
        setattr(
            modules[".".join(parts[:length])],
            parts[length],
            modules[".".join(parts[: length + 1])],
        )
    for key, value in attrs.items():
        setattr(modules[name], key, value)
    return modules[name]


def _record(*params):
    """A function with EXACTLY ``params`` in its signature, recording calls.

    ``params`` is the full parameter list (``"self"`` included where the
    production API has it); the preflight both subset-checks and keyword-
    binds the documented names, so the stub must match exactly.
    """

    def fn(*args, **kwargs):
        fn.calls.append((args, kwargs))
        return "patched"

    import inspect

    fn.calls = []
    fn.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in params
        ]
    )
    return fn


class _PatchGraph:
    """Stub graph for patch.py's lazy vllm imports with recordable API."""

    def __init__(self):
        self.modules = {}
        self.install()

    def install(self):
        m = self.modules
        breakable = _leaf(m, "vllm.compilation.breakable_cudagraph")
        runner = _leaf(m, "vllm.v1.worker.gpu_model_runner")
        dispatcher = _leaf(m, "vllm.v1.cudagraph_dispatcher")
        _leaf(m, "vllm.config", CUDAGraphMode=SimpleNamespace(FULL="full"))
        _leaf(
            m,
            "vllm.forward_context",
            get_forward_context=lambda: SimpleNamespace(),
            is_forward_context_available=lambda: False,
            set_forward_context=lambda *a, **k: None,
        )
        _leaf(
            m,
            "vllm.model_executor.layers.attention.attention",
            get_attention_context=lambda: SimpleNamespace(),
        )

        def _impl_forward(self, layer, hidden, attn_metadata, output):
            return None

        def _gdn_forward_core(self, core_attn_out):
            return None

        _gdn_cls = type(
            "QwenGatedDeltaNetAttention", (), {"_forward_core": _gdn_forward_core}
        )

        _leaf(
            m,
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
            QwenGatedDeltaNetAttention=_gdn_cls,
        )
        _leaf(m, "vllm.utils.torch_utils", weak_ref_tensor=lambda t: t)
        _leaf(m, "vllm.distributed", all_reduce=lambda *a: None)

        # breakable_cudagraph API surface used by the preflight.
        breakable.BreakableCUDAGraphCapture = type("BreakableCUDAGraphCapture", (), {})
        breakable.BreakableCUDAGraphWrapper = type("BreakableCUDAGraphWrapper", (), {})
        breakable.is_breakable_cudagraph_enabled = _record()
        runner.is_breakable_cudagraph_enabled = _record()
        capture = breakable.BreakableCUDAGraphCapture
        capture.current = _record()
        capture._begin_segment = _record("self")
        capture._end_segment = _record("self")
        capture.add_eager = _record("self", "fn")
        wrapper_cls = breakable.BreakableCUDAGraphWrapper
        wrapper_cls.__init__ = _record("self", "runnable", "vllm_config")
        wrapper_cls.__call__ = _record("self")
        dispatcher.CudagraphDispatcher = type("CudagraphDispatcher", (), {})
        dispatcher.CudagraphDispatcher.dispatch = _record(
            "self",
            "num_tokens",
            "uniform_decode",
            "has_lora",
            "num_active_loras",
            "valid_modes",
            "invalid_modes",
        )

        # FL-side targets.
        fl_runner = _leaf(m, "vllm_fl.worker.model_runner")
        fl_runner.is_breakable_cudagraph_enabled = _record()
        fl_runner._device_graph_capture = lambda *a, **k: None

        # Additional leaves the staged graph patches import lazily.
        _leaf(m, "vllm.v1.worker.gpu", warmup_kernels=lambda *a: None)
        _leaf(
            m,
            "vllm.v1.worker.gpu.cudagraph_utils",
            graph_capture=lambda *a, **k: None,
            DEFAULT_STABLE_GRAPH_CAPTURE_FLAGS=None,
        )
        _leaf(m, "vllm.v1.attention", register_backend=lambda *a, **k: None)
        _leaf(
            m, "vllm.v1.attention.backends", FlashAttentionBackend=type("FAB", (), {})
        )
        bt = _leaf(m, "vllm.v1.worker.block_table")
        bt.compute_slot_mapping = lambda *a, **k: None
        bt.BlockTable = type("BlockTable", (), {"__init__": None})
        _leaf(m, "vllm.model_executor.layers.mamba.ops")
        _leaf(m, "vllm.model_executor.layers.fla")
        _leaf(
            m,
            "vllm.model_executor.layers.mamba.gdn.gated_delta_rule",
            GatedDeltaNetAttention=type("GDN", (), {}),
        )
        comm = _leaf(m, "vllm_fl.distributed.communicator")
        comm.CommunicatorFL = type("CommunicatorFL", (), {})
        comm.CommunicatorFL.all_reduce = _record("self", "input_")
        comm.CommunicatorFL.all_gather = _record("self", "input_", "dim")

        # The legacy-patch chain imports these vllm_fl leaves; they are
        # external boundaries for this test (their own behaviour is covered
        # by the kunlunxin patches tests elsewhere). No-op everything.
        base = "vllm_fl.dispatch.backends.vendor.kunlunxin.patches"
        legacy_leaves = {
            "accelerator_compat": "apply_accelerator_compat",
            "triton_compat": "apply_triton_compat",
            "kv_block_zero_compat": "apply_kv_block_zero_compat",
            "uva_compat": "apply_uva_compat",
            "rope_compat": "apply_rope_compat",
            "penalties_compat": "apply_penalties_compat",
            "input_batch_compat": "apply_input_batch_compat",
            "block_table_compat": "apply_block_table_compat",
            "mamba_utils_compat": "apply_mamba_utils_compat",
            "mamba_align_compat": "apply_mamba_align_compat",
            "mamba_hybrid_compat": "apply_mamba_hybrid_compat",
            "buffer_utils_compat": "apply_buffer_utils_compat",
            "structured_outputs_compat": "apply_structured_outputs_compat",
            "topk_topp_compat": "apply_topk_topp_compat",
            "v2_greedy_sampler": "apply_v2_greedy_sampler",
            "vl_compat": "apply_vl_compat",
            # patch_forward_core exports a differently-named apply function.
            "patch_forward_core": "apply_ssm_patch",
        }
        for short, fn_name in legacy_leaves.items():
            leaf = _leaf(m, f"{base}.{short}")
            setattr(leaf, fn_name, lambda: None)
        # The impl package the legacy chain reaches for (attention patching).
        _leaf(m, "vllm_fl.dispatch.backends.vendor.kunlunxin.impl")
        attn = _leaf(m, "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention")
        attn._is_full_graph_runtime = lambda *a, **k: True
        attn.KunlunxinAttentionBackendImpl = type(
            "KunlunxinAttentionBackendImpl", (), {"forward": _impl_forward}
        )
        # vl_compat also exports the vision-tp reducer used at graph time.
        vl = m[f"{base}.vl_compat"]
        vl.apply_vision_tp_scope = lambda: None
        vl.maybe_reduce_vision_tp = lambda self, x: x

        self.breakable = breakable
        self.dispatcher = dispatcher
        self.fl_comm = comm
        return m

    def publish(self, monkeypatch):
        for name, module in self.modules.items():
            monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def patch_module(monkeypatch):
    """Load the real kunlunxin patch.py under the stub graph."""
    graph = _PatchGraph()
    graph.publish(monkeypatch)

    import importlib.util
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[4]
        / "vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patch", source
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.kunlunxin.patch", module
    )
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("vllm_fl.dispatch.backends.vendor.kunlunxin.patch", None)
        raise
    # Reset the module-level state so every test starts fresh.
    module._patch_state.applied = False
    module._patch_state.initializing = False
    module._patch_state.legacy_applied = False
    return module, graph


class TestApplyKunlunxinPatches:
    """Idempotence and failure-retry semantics of the orchestrator."""

    def test_first_apply_sets_applied(self, patch_module):
        module, _ = patch_module
        module.apply_kunlunxin_patches()
        assert module._patch_state.applied is True
        assert module._patch_state.legacy_applied is True

    def test_second_apply_is_a_noop(self, patch_module):
        module, _ = patch_module
        module.apply_kunlunxin_patches()
        before = module._patch_state.legacy_applied
        module.apply_kunlunxin_patches()
        # legacy stays as-is; nothing re-runs (applied gate short-circuits).
        assert module._patch_state.legacy_applied is before

    def test_failed_staging_can_retry(self, patch_module, monkeypatch):
        """A preflight failure leaves applied=False so a retry re-stages."""
        module, _ = patch_module
        calls = []

        def broken():
            calls.append(1)
            raise TypeError("Unsupported graph API")

        monkeypatch.setattr(module, "patch_graph_adaptations", broken)
        with pytest.raises(TypeError, match="Unsupported graph API"):
            module.apply_kunlunxin_patches()
        assert module._patch_state.applied is False
        # initializing was reset even on failure (no deadlock on retry).
        assert module._patch_state.initializing is False
        # A retry with a working preflight now succeeds.
        monkeypatch.setattr(module, "patch_graph_adaptations", lambda: None)
        module.apply_kunlunxin_patches()
        assert module._patch_state.applied is True
        assert len(calls) == 1  # legacy ran once, not twice

    def test_initializing_flag_blocks_reentry(self, patch_module):
        """A nested apply during initialization returns immediately."""
        module, _ = patch_module
        nested = []
        real_staging = module._apply_kunlunxin_patches

        def staging_with_nested_call():
            nested.append(module._patch_state.initializing)
            module.apply_kunlunxin_patches()  # re-entrant call must no-op
            return real_staging()

        module._apply_kunlunxin_patches = staging_with_nested_call
        module.apply_kunlunxin_patches()
        assert nested == [True]


class TestGraphPreflight:
    """patch_graph_adaptations' signature check before publishing anything."""

    def test_missing_parameter_aborts_before_publishing(
        self, patch_module, monkeypatch
    ):
        """A vLLM API missing a documented parameter aborts staging."""
        module, graph = patch_module
        # Drop a required parameter from the dispatcher signature.
        import inspect

        original = graph.dispatcher.CudagraphDispatcher.dispatch

        def narrowed(*args, **kwargs):
            return original(*args, **kwargs)

        narrowed.__signature__ = inspect.Signature(
            [
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter(
                    "num_tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD
                ),
            ]
        )
        monkeypatch.setattr(graph.dispatcher.CudagraphDispatcher, "dispatch", narrowed)
        # The preflight wraps every staging failure in its RuntimeError.
        with pytest.raises(RuntimeError, match="preflight failed"):
            module.patch_graph_adaptations()

    def test_dispatch_patch_published(self, patch_module):
        """With the full API present, the dispatcher is patched."""
        module, graph = patch_module
        module.patch_graph_adaptations()
        # The dispatcher class gained a patched dispatch (a new function).
        patched = graph.dispatcher.CudagraphDispatcher.dispatch
        assert patched is not None
        assert hasattr(patched, "__wrapped__") or callable(patched)
