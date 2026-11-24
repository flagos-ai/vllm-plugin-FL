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

"""G47 coverage: last residual branches in ``ModelRunnerFL`` (814b).

Targets the cluster the earlier waves leave behind: the full ``capture_model``
body (g1b only exercised the ``CUDAGraphMode.NONE`` early return), the
multimodal arms of ``profile_run`` (g1b only did the non-mm sampler/pooler
paths), ``_device_graph_capture``, the async runner-output wrappers, and the
remaining rare branches in ``initialize_attn_backend``,
``_check_and_update_cudagraph_mode`` and ``_compute_cascade_attn_prefix_len``.

CPU-only: accelerator, graph capture, profiler, EP manager and attention
backend collaborators are replaced at the ``vllm_fl.worker.model_runner``
boundary.  No NPU kernel, triton or numba is touched.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_fl.worker.model_runner as mr

try:
    import vllm.distributed.parallel_state as _ps
except Exception:  # pragma: no cover - keep collection safe
    _ps = None


_MISSING = object()


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name, _MISSING)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        if self.saved is _MISSING:
            try:
                delattr(self.module, self.name)
            except AttributeError:
                pass
        else:
            setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


# ==========================================================================
# capture_model
# ==========================================================================
class _Mode:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return id(self)

    def __str__(self):
        return f"CUDAGraphMode.{self.name}"


_NONE_MODE = _Mode("NONE")
_FULL_MODE = _Mode("FULL")


class _Accelerator:
    def __init__(self, mems):
        self.mems = list(mems)
        self.log = []

    def synchronize(self):
        self.log.append("sync")

    def empty_cache(self):
        self.log.append("empty")

    def get_memory_info(self):
        head = self.mems.pop(0) if len(self.mems) > 1 else self.mems[0]
        return (head, 0)


class _Profiler:
    class ProfilerActivity:
        CPU = "cpu"
        CUDA = "cuda"

    @staticmethod
    def profile(**kwargs):
        return contextlib.nullcontext()

    @staticmethod
    def tensorboard_trace_handler(*args, **kwargs):
        return lambda *a, **k: None


def _capture_runner(**over):
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=_FULL_MODE),
        vllm_config=SimpleNamespace(
            profiler_config=SimpleNamespace(
                capture_torch_profiler=False,
                torch_profiler_dir="/tmp/capture",
            )
        ),
        device="cpu",
        device_type="cuda",
        cudagraph_dispatcher=SimpleNamespace(
            get_capture_descs=lambda: [
                ("PIECEWISE", ["a"]),
                ("FULL", ["b"]),
            ]
        ),
        encoder_cudagraph_manager=None,
    )
    runner._maybe = []
    runner._musa = []
    runner._captured = []
    runner._maybe_init_encoder_cudagraph_manager = lambda: runner._maybe.append(1)
    runner._get_musa_graph_capture_context = (
        lambda: runner._musa.append(1) or "MUSA_CTX"
    )
    runner._freeze_gc = lambda: contextlib.nullcontext()
    runner._capture_cudagraphs = (
        lambda *, batch_descriptors, cudagraph_runtime_mode, profiler: (
            runner._captured.append(
                (batch_descriptors, cudagraph_runtime_mode, profiler)
            )
        )
    )
    for key, value in over.items():
        setattr(runner, key, value)
    return runner


def _run_capture(runner, *, local_rank=0, graph_factory=None):
    log = []
    calls = []
    profiler_kwargs = []
    acc = _Accelerator([1000, 400])

    class _P(_Profiler):
        @staticmethod
        def profile(**kwargs):
            profiler_kwargs.append(kwargs)
            return contextlib.nullcontext()

    fake_torch = SimpleNamespace(accelerator=acc, profiler=_P)
    counter = SimpleNamespace(num_gpu_runner_capture_triggers=0)

    if graph_factory is None:

        def graph_factory(device=None, graph_capture_context=None):
            calls.append((device, graph_capture_context))
            return contextlib.nullcontext()

    patches = [
        (
            mr,
            "CUDAGraphMode",
            SimpleNamespace(NONE=_NONE_MODE, FULL=_FULL_MODE),
        ),
        (mr, "compilation_counter", counter),
        (
            mr,
            "current_platform",
            SimpleNamespace(
                device_type=runner.device_type,
                graph_pool_handle=lambda: "POOL",
            ),
        ),
        (
            mr,
            "set_cudagraph_capturing_enabled",
            lambda value: log.append(("enabled", value)),
        ),
        (mr, "lock_workspace", lambda: log.append(("lock",))),
        (mr, "torch", fake_torch),
        (mr, "graph_capture", graph_factory),
    ]
    if _ps is not None:
        patches.append(
            (
                _ps,
                "get_world_group",
                lambda: SimpleNamespace(local_rank=local_rank),
            )
        )
    with _stack(*patches):
        size = mr.ModelRunnerFL.capture_model(runner)
    return SimpleNamespace(
        size=size,
        log=log,
        acc=acc,
        calls=calls,
        counter=counter,
        profiler_kwargs=profiler_kwargs,
    )


def test_capture_model_none_mode_short_circuits():
    runner = _capture_runner()
    runner.compilation_config.cudagraph_mode = _NONE_MODE
    out = _run_capture(runner)
    assert out.size == 0
    assert out.counter.num_gpu_runner_capture_triggers == 0
    assert runner._maybe == []
    assert out.calls == []


def test_capture_model_full_body_profiler_disabled():
    runner = _capture_runner()
    out = _run_capture(runner)

    assert out.size == 600
    assert out.counter.num_gpu_runner_capture_triggers == 1
    assert [entry[0] for entry in out.log] == ["enabled", "enabled", "lock"]
    assert out.log[0] == ("enabled", True)
    assert out.log[1] == ("enabled", False)
    assert out.acc.log.count("empty") == 2
    assert runner._maybe == [1]
    assert [c[1] for c in runner._captured] == ["PIECEWISE", "FULL"]
    assert out.calls == [("cpu", None)]
    assert out.profiler_kwargs == []


def test_capture_model_graph_capture_typeerror_falls_back():
    runner = _capture_runner(device_type="musa")
    calls = []

    def graph_factory(device=None, graph_capture_context=None):
        calls.append((device, graph_capture_context))
        if graph_capture_context is not None:
            raise TypeError("unexpected graph_capture_context")
        return contextlib.nullcontext()

    out = _run_capture(runner, graph_factory=graph_factory)

    assert out.size == 600
    assert runner._musa == [1]
    assert calls[0] == ("cpu", "MUSA_CTX")
    assert calls[1] == ("cpu", None)


def test_capture_model_profiler_enabled_on_rank_zero():
    runner = _capture_runner(device_type="musa")
    runner.vllm_config.profiler_config.capture_torch_profiler = True
    out = _run_capture(runner, local_rank=0)

    assert len(out.profiler_kwargs) == 1
    kwargs = out.profiler_kwargs[0]
    assert kwargs["record_shapes"] is True
    assert kwargs["profile_memory"] is True
    assert kwargs["with_stack"] is True
    assert "on_trace_ready" in kwargs
    assert out.counter.num_gpu_runner_capture_triggers == 1


def test_capture_model_captures_encoder_graphs():
    runner = _capture_runner()
    runner._enc = []
    runner.encoder_cudagraph_manager = SimpleNamespace(
        capture=lambda graph_pool: runner._enc.append(graph_pool)
    )
    out = _run_capture(runner)

    assert out.size == 600
    assert runner._enc == ["POOL"]


# ==========================================================================
# profile_run (multimodal arms)
# ==========================================================================
def _profile_runner(**over):
    runner = SimpleNamespace(
        supports_mm_inputs=False,
        is_pooling_model=False,
        max_num_tokens=8,
        encoder_cache={},
        model_config=SimpleNamespace(multimodal_config=None),
        mm_budget=None,
        model=SimpleNamespace(embed_multimodal=lambda **kwargs: []),
    )
    runner._syncs = []
    runner._dummy_calls = []
    runner._samplers = []
    runner._poolers = []
    runner._sync_device = lambda: runner._syncs.append(1)
    runner._dummy_run = lambda n, **kw: (
        runner._dummy_calls.append((n, kw))
        or (torch.zeros(4, 2), torch.zeros(2, 2))
    )
    runner._dummy_sampler_run = (
        lambda last: runner._samplers.append(last) or "sampler"
    )
    runner._dummy_pooler_run = (
        lambda hidden: runner._poolers.append(hidden) or "pooler"
    )
    for key, value in over.items():
        setattr(runner, key, value)
    return runner


class _MMBudget:
    def __init__(self, budget=10, max_toks=2, modality="image", items=3):
        self._budget = budget
        self.mm_max_toks_per_item = max_toks
        self._modality = modality
        self.mm_max_items_per_batch = {modality: items}

    def get_encoder_budget(self):
        return self._budget

    def get_modality_with_max_tokens(self):
        return self._modality


def _last_rank():
    return SimpleNamespace(is_last_rank=True)


def test_profile_run_skips_mm_profiling_when_requested():
    runner = _profile_runner(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(skip_mm_profiling=True)
        ),
    )
    with _stack((mr, "get_pp_group", _last_rank)):
        mr.ModelRunnerFL.profile_run(runner)

    assert len(runner._dummy_calls) == 1
    assert len(runner._samplers) == 1
    assert runner._syncs == [1]


def test_profile_run_mm_encoder_full_path():
    seen = []
    sanity = []
    runner = _profile_runner(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(skip_mm_profiling=False)
        ),
        mm_budget=_MMBudget(10),
        model=SimpleNamespace(
            embed_multimodal=lambda **kwargs: seen.append(kwargs) or [1, 2, 3]
        ),
    )
    runner._get_mm_dummy_batch = lambda modality, items: {
        "mm": (modality, items)
    }
    with _stack(
        (mr, "get_pp_group", _last_rank),
        (
            mr,
            "sanity_check_mm_encoder_outputs",
            lambda outputs, expected_num_items: sanity.append(
                (len(outputs), expected_num_items)
            ),
        ),
    ):
        mr.ModelRunnerFL.profile_run(runner)

    assert seen == [{"mm": ("image", 3)}]
    assert sanity == [(3, 3)]
    assert runner.encoder_cache == {}
    assert len(runner._dummy_calls) == 1


def test_profile_run_mm_embedding_only_skips_encoder():
    def _boom(**kwargs):
        raise AssertionError("encoder must not run")

    runner = _profile_runner(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(skip_mm_profiling=False)
        ),
        mm_budget=_MMBudget(10, max_toks=0),
        model=SimpleNamespace(embed_multimodal=_boom),
    )
    with _stack((mr, "get_pp_group", _last_rank)):
        mr.ModelRunnerFL.profile_run(runner)

    assert len(runner._dummy_calls) == 1


def test_profile_run_mm_zero_encoder_budget_skips_encoder():
    def _boom(**kwargs):
        raise AssertionError("encoder must not run")

    runner = _profile_runner(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(skip_mm_profiling=False)
        ),
        mm_budget=_MMBudget(0),
        model=SimpleNamespace(embed_multimodal=_boom),
    )
    with _stack((mr, "get_pp_group", _last_rank)):
        mr.ModelRunnerFL.profile_run(runner)

    assert len(runner._dummy_calls) == 1


def test_profile_run_mm_config_none_uses_budget():
    runner = _profile_runner(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(multimodal_config=None),
        mm_budget=_MMBudget(0),
    )
    with _stack((mr, "get_pp_group", _last_rank)):
        mr.ModelRunnerFL.profile_run(runner)

    assert len(runner._dummy_calls) == 1


def test_profile_run_non_last_rank_skips_sampler():
    runner = _profile_runner()
    with _stack(
        (mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False))
    ):
        mr.ModelRunnerFL.profile_run(runner)

    assert runner._samplers == []
    assert runner._poolers == []
    assert runner._syncs == [1]


# ==========================================================================
# _device_graph_capture
# ==========================================================================
class _Stream:
    def __init__(self, name):
        self.name = name
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other)


class _GraphCaptureCtx:
    def __init__(self, stream):
        self.stream = stream


def test_device_graph_capture_builds_context_and_waits():
    created = []
    current = _Stream("current")
    captured = _Stream("captured")

    class _DeviceFn:
        @staticmethod
        def Stream(device=None):
            created.append(device)
            return captured

        @staticmethod
        def current_stream():
            return current

        @staticmethod
        def stream(stream):
            return contextlib.nullcontext()

    with _stack(
        (mr, "current_platform", SimpleNamespace(torch_device_fn=_DeviceFn)),
        (mr, "GraphCaptureContext", _GraphCaptureCtx),
    ):
        with mr._device_graph_capture("cpu") as got:
            assert got.stream is captured

    assert created == ["cpu"]
    assert captured.waits == [current]


def test_device_graph_capture_reuses_matching_stream():
    captured = _Stream("current")

    class _DeviceFn:
        @staticmethod
        def Stream(device=None):
            raise AssertionError("must reuse the supplied context")

        @staticmethod
        def current_stream():
            return captured

        @staticmethod
        def stream(stream):
            return contextlib.nullcontext()

    ctx = _GraphCaptureCtx(captured)
    with _stack(
        (mr, "current_platform", SimpleNamespace(torch_device_fn=_DeviceFn)),
        (mr, "GraphCaptureContext", _GraphCaptureCtx),
    ):
        with mr._device_graph_capture("cpu", ctx) as got:
            assert got is ctx

    assert captured.waits == []


# ==========================================================================
# initialize_attn_backend
# ==========================================================================
class _BackendA:
    @classmethod
    def full_cls_name(cls):
        return "A"


class _BackendFast(_BackendA):
    @classmethod
    def full_cls_name(cls):
        return "Fast"


class _AttentionGroup:
    def __init__(self, backend, layer_names, spec, group_id):
        self.backend = backend
        self.layer_names = layer_names
        self.spec = spec
        self.group_id = group_id


class _UniformSpecs:
    def __init__(self, specs):
        self.kv_cache_specs = specs


def _attn_layer(backend, num_heads=4):
    return SimpleNamespace(
        get_attn_backend=lambda: backend, num_heads=num_heads
    )


def _attn_layers_without_heads(backend):
    return SimpleNamespace(get_attn_backend=lambda: backend)


def test_initialize_attn_backend_multi_group_uniform_and_fast_prefill():
    runner = SimpleNamespace(
        vllm_config="cfg",
        attn_groups=[],
        device="cpu",
        kv_sharing_fast_prefill_eligible_layers={"l2"},
    )
    check = []
    cp = []
    created = []

    runner._check_and_update_cudagraph_mode = (
        lambda backends, groups, is_profiling=False: check.append(
            (len(backends), is_profiling)
        )
    )

    def _create(name, backend):
        created.append((name, backend))
        return _BackendFast

    layers = {
        "l1": _attn_layer(_BackendA),
        "l2": _attn_layer(_BackendA),
    }
    g1 = SimpleNamespace(layer_names=["l1"], kv_cache_spec="spec1")
    g2 = SimpleNamespace(
        layer_names=["l2"], kv_cache_spec=_UniformSpecs({"l2": "s2"})
    )
    with _stack(
        (mr, "get_layers_from_vllm_config", lambda *a: layers),
        (mr, "AttentionGroup", _AttentionGroup),
        (mr, "UniformTypeKVCacheSpecs", _UniformSpecs),
        (mr, "create_fast_prefill_custom_backend", _create),
        (mr, "check_attention_cp_compatibility", lambda cfg: cp.append(cfg)),
    ):
        mr.ModelRunnerFL.initialize_attn_backend(
            runner,
            SimpleNamespace(kv_cache_groups=[g1, g2]),
            is_profiling=True,
        )

    assert len(runner.attn_groups) == 2
    assert runner.attn_groups[0][0].backend is _BackendA
    assert runner.attn_groups[0][0].spec == "spec1"
    assert runner.attn_groups[1][0].backend is _BackendFast
    assert runner.attn_groups[1][0].spec == "s2"
    assert created == [("FastPrefill", _BackendA)]
    assert check == [(2, True)]
    assert cp == ["cfg"]


def test_initialize_attn_backend_non_attention_layers_cluster():
    runner = SimpleNamespace(
        vllm_config="cfg",
        attn_groups=[],
        device="cpu",
        kv_sharing_fast_prefill_eligible_layers=set(),
    )
    runner._check_and_update_cudagraph_mode = (
        lambda backends, groups, is_profiling=False: None
    )
    layers = {
        "m1": _attn_layers_without_heads(_BackendA),
        "m2": _attn_layers_without_heads(_BackendA),
    }
    group = SimpleNamespace(layer_names=["m1", "m2"], kv_cache_spec="mspec")
    with _stack(
        (mr, "get_layers_from_vllm_config", lambda *a: layers),
        (mr, "AttentionGroup", _AttentionGroup),
        (mr, "UniformTypeKVCacheSpecs", _UniformSpecs),
        (mr, "check_attention_cp_compatibility", lambda cfg: None),
    ):
        mr.ModelRunnerFL.initialize_attn_backend(
            runner, SimpleNamespace(kv_cache_groups=[group])
        )

    assert len(runner.attn_groups) == 1
    assert len(runner.attn_groups[0]) == 1
    assert sorted(runner.attn_groups[0][0].layer_names) == ["m1", "m2"]


# ==========================================================================
# _check_and_update_cudagraph_mode
# ==========================================================================
class _CGSupport:
    ALWAYS = SimpleNamespace(value=5)


class _BuilderHigh:
    @staticmethod
    def get_cudagraph_support(vllm_config, kv_cache_spec):
        return SimpleNamespace(value=5)


class _BuilderLow:
    @staticmethod
    def get_cudagraph_support(vllm_config, kv_cache_spec):
        return SimpleNamespace(value=1)


class _BackendHigh:
    @classmethod
    def get_builder_cls(cls):
        return _BuilderHigh


class _BackendLow:
    @classmethod
    def get_builder_cls(cls):
        return _BuilderLow


_BackendHigh.__name__ = "BackendHigh"
_BackendLow.__name__ = "BackendLow"


def _cudagraph_runner(**over):
    runner = SimpleNamespace(
        vllm_config="cfg",
        uniform_decode_query_len=2,
        kv_cache_config="kv",
        max_num_reqs=4,
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=lambda *a, **k: "mode"
        ),
        cudagraph_dispatcher=SimpleNamespace(
            initialize_cudagraph_keys=lambda *a: None
        ),
        speculative_config=None,
    )
    for key, value in over.items():
        setattr(runner, key, value)
    return runner


def test_check_and_update_cudagraph_mode_picks_min_support():
    resolved = []
    keys = []
    runner = _cudagraph_runner()
    runner.compilation_config = SimpleNamespace(
        resolve_cudagraph_mode_and_sizes=lambda *a, **k: (
            resolved.append((a, k)) or "mode"
        )
    )
    runner.cudagraph_dispatcher = SimpleNamespace(
        initialize_cudagraph_keys=lambda mode, qlen: keys.append((mode, qlen))
    )
    groups = [SimpleNamespace(kv_cache_spec="spec")]
    with _stack((mr, "AttentionCGSupport", _CGSupport)):
        mr.ModelRunnerFL._check_and_update_cudagraph_mode(
            runner,
            [{_BackendHigh, _BackendLow}],
            groups,
            is_profiling=True,
        )

    args, kwargs = resolved[0]
    assert args[0].value == 1
    assert args[1] == "BackendLow"
    assert args[2] == 2
    assert kwargs["is_profiling"] is True
    assert keys == [("mode", 2)]


def test_check_and_update_cudagraph_mode_inits_extract_hidden_drafter():
    class _Proposer:
        def __init__(self):
            self.calls = []

        def initialize_cudagraph_keys(self, mode):
            self.calls.append(mode)

    drafter = _Proposer()
    runner = _cudagraph_runner(
        speculative_config=SimpleNamespace(
            use_eagle=lambda: False,
            uses_draft_model=lambda: False,
            uses_extract_hidden_states=lambda: True,
        ),
        drafter=drafter,
    )
    with _stack(
        (mr, "AttentionCGSupport", _CGSupport),
        (mr, "EagleProposer", _Proposer),
        (mr, "DFlashProposer", _Proposer),
        (mr, "DraftModelProposer", _Proposer),
        (mr, "ExtractHiddenStatesProposer", _Proposer),
        (mr, "Gemma4Proposer", _Proposer),
    ):
        mr.ModelRunnerFL._check_and_update_cudagraph_mode(
            runner,
            [{_BackendHigh}],
            [SimpleNamespace(kv_cache_spec="spec")],
        )

    assert drafter.calls == ["mode"]


def test_check_and_update_cudagraph_mode_rejects_wrong_drafter():
    runner = _cudagraph_runner(
        speculative_config=SimpleNamespace(
            use_eagle=lambda: True,
            uses_draft_model=lambda: False,
            uses_extract_hidden_states=lambda: False,
        ),
        drafter=object(),
    )

    class _Proposer:
        def initialize_cudagraph_keys(self, mode):
            raise AssertionError("should not be reached")

    with _stack(
        (mr, "AttentionCGSupport", _CGSupport),
        (mr, "EagleProposer", _Proposer),
        (mr, "DFlashProposer", _Proposer),
        (mr, "DraftModelProposer", _Proposer),
        (mr, "ExtractHiddenStatesProposer", _Proposer),
        (mr, "Gemma4Proposer", _Proposer),
    ):
        with pytest.raises(AssertionError):
            mr.ModelRunnerFL._check_and_update_cudagraph_mode(
                runner,
                [{_BackendHigh}],
                [SimpleNamespace(kv_cache_spec="spec")],
            )


# ==========================================================================
# _compute_cascade_attn_prefix_len
# ==========================================================================
class _SlidingWindowSpec:
    pass


class _AttentionSpec:
    pass


class _ChunkedLocalSpec:
    pass


class _CascadeBuilder:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def use_cascade_attention(self, **kwargs):
        self.kwargs = kwargs
        return self.result


def _cascade_runner():
    return SimpleNamespace(
        num_query_heads=4,
        use_alibi=False,
        num_sms=8,
        dcp_world_size=1,
    )


def _cascade_patches():
    return (
        (mr, "SlidingWindowSpec", _SlidingWindowSpec),
        (mr, "FullAttentionSpec", _AttentionSpec),
        (mr, "ChunkedLocalAttentionSpec", _ChunkedLocalSpec),
        (mr, "AttentionSpec", _AttentionSpec),
    )


def test_cascade_prefix_len_full_attention_sliding_window():
    class _Spec(_AttentionSpec):
        block_size = 16
        num_kv_heads = 2
        sliding_window = 4
        attention_chunk_size = None

    builder = _CascadeBuilder(True)
    with _stack(*_cascade_patches()):
        got = mr.ModelRunnerFL._compute_cascade_attn_prefix_len(
            _cascade_runner(),
            np.array([1, 1]),
            np.array([40, 50]),
            2,
            _Spec(),
            builder,
        )

    assert got == 32
    assert builder.kwargs["use_sliding_window"] is True
    assert builder.kwargs["use_local_attention"] is False


def test_cascade_prefix_len_chunked_local_attention_rejected():
    class _Spec(_ChunkedLocalSpec, _AttentionSpec):
        block_size = 16
        num_kv_heads = 1
        sliding_window = None
        attention_chunk_size = 8

    builder = _CascadeBuilder(False)
    with _stack(*_cascade_patches()):
        got = mr.ModelRunnerFL._compute_cascade_attn_prefix_len(
            _cascade_runner(),
            np.array([1]),
            np.array([8]),
            1,
            _Spec(),
            builder,
        )

    assert got == 0
    assert builder.kwargs["use_local_attention"] is True
    assert builder.kwargs["use_sliding_window"] is False


def test_cascade_prefix_len_zero_blocks_short_circuits():
    builder = _CascadeBuilder(True)
    with _stack(*_cascade_patches()):
        got = mr.ModelRunnerFL._compute_cascade_attn_prefix_len(
            _cascade_runner(),
            np.array([1]),
            np.array([5]),
            0,
            SimpleNamespace(block_size=16),
            builder,
        )

    assert got == 0
    assert builder.kwargs is None


# ==========================================================================
# async runner-output wrappers
# ==========================================================================
class _AsyncStream:
    def wait_stream(self, other):
        return None


class _AsyncEvent:
    def record(self):
        return None

    def synchronize(self):
        return None


class _AsyncPlatform:
    class torch_device_fn:
        Stream = staticmethod(lambda *a, **k: _AsyncStream())
        Event = staticmethod(lambda *a, **k: _AsyncEvent())
        current_stream = staticmethod(lambda *a, **k: _AsyncStream())
        stream = staticmethod(lambda s: contextlib.nullcontext())


class _AsyncOutput(mr.AsyncGPUModelRunnerOutput):
    def get_kv_connector_output(self):
        return None


class _AsyncPoolingOutput(mr.AsyncGPUPoolingModelRunnerOutput):
    def get_kv_connector_output(self):
        return None


def test_async_output_ep_check_without_fault_returns_output():
    class _Manager:
        @staticmethod
        def query_fault():
            return torch.tensor([0], dtype=torch.int32)

    sampled = torch.tensor([[1], [2]], dtype=torch.int64)
    output = SimpleNamespace(
        sampled_token_ids=None,
        logprobs=None,
        req_id_to_index={"a": 0, "b": 1},
    )
    with _stack(
        (mr, "current_platform", _AsyncPlatform),
        (mr, "get_ep_all2all_manager", _Manager),
    ):
        async_out = _AsyncOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=_AsyncStream(),
            vocab_size=10,
            check_ep_fault=True,
        )
        result = async_out.get_output()

    assert result is output
    assert result.sampled_token_ids == [[1], [2]]
    assert result.logprobs is None
    assert async_out._has_fault is not None
    assert async_out._has_fault.item() == 0


def test_async_pooling_output_copies_only_finished_rows():
    raw = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    output = SimpleNamespace(pooler_output=None)
    with _stack((mr, "current_platform", _AsyncPlatform)):
        async_out = _AsyncPoolingOutput(
            model_runner_output=output,
            raw_pooler_output=raw,
            finished_mask=[True, False, True],
            async_output_copy_stream=_AsyncStream(),
        )
        result = async_out.get_output()

    assert result is output
    assert result.pooler_output[1] is None
    assert torch.equal(result.pooler_output[0], raw[0])
    assert torch.equal(result.pooler_output[2], raw[2])
    assert not hasattr(async_out, "_raw_pooler_output")
