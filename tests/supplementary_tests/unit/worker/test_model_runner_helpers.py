# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl/worker/model_runner.py`` helpers.

CPU-only, platform-agnostic checks (stub doubles; no accelerator needed):

* module-level helpers: ``count_nans_per_row``, ``nans_to_dict``,
  ``_get_parameter_for_reload``, ``_accelerator_synchronize``,
  ``_device_graph_capture`` and the ``graph_capture`` module-level selection.
* pure/cached state helpers on ``ModelRunnerFL``: ``_kl_*`` budget guard,
  ``jit_warmup_registry`` property/setter, ``_get_positions``,
  ``_get_or_create_async_output_copy_stream``, ``_make_buffer``,
  ``update_max_model_len``, ``_get_valid_sampled_token_count``,
  ``get_encoder_timing_stats``, ``_is_uniform_decode``.
"""

from __future__ import annotations

import gc
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.supplementary]

MR = pytest.importorskip("vllm_fl.worker.model_runner")


GIB = 1 << 30


# ---------------------------------------------------------------------------
# Module-level free functions
# ---------------------------------------------------------------------------


def test_count_nans_per_row_counts_only_last_dim():
    logits = torch.tensor(
        [[float("nan"), 1.0, 2.0], [0.0, float("nan"), float("nan")],
         [0.0, 1.0, 2.0]]
    )
    counts = MR.count_nans_per_row(logits)
    assert counts.tolist() == [1, 2, 0]
    assert counts.dtype == torch.int32


def test_nans_to_dict_maps_rows_and_defaults_zero():
    counts = [2, 1]
    req_id_to_index = {"a": 0, "b": 1, "c": 5}
    assert MR.nans_to_dict(counts, req_id_to_index) == {
        "a": 2,
        "b": 1,
        "c": 0,
    }


def test_nans_to_dict_accepts_tensor_counts():
    counts = torch.tensor([3, 4], dtype=torch.int32)
    out = MR.nans_to_dict(counts, {"r": 1})
    assert out == {"r": 4}


def test_get_parameter_for_reload_unwraps_lora_base_layer():
    mr = MR

    class FakeBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))

    class FakeLora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = FakeBase()

    base = FakeBase()
    # Direct module: checkpoint names without a base_layer hop.
    param = mr._get_parameter_for_reload(base, "weight")
    assert param is base.weight

    lora = FakeLora()
    # LoRA-wrapped name resolves through base_layer (vllm BaseLayerWithLoRA).
    wrapped = mr._get_parameter_for_reload(lora, "base_layer.weight")
    assert wrapped is lora.base_layer.weight


def test_accelerator_synchronize_dispatches_on_platform(monkeypatch):
    calls = []

    class FakeTorchMusa:
        @staticmethod
        def synchronize():
            calls.append("musa")

    import sys

    monkeypatch.setattr(MR.current_platform, "device_type", "musa")
    monkeypatch.setitem(sys.modules, "torch_musa", FakeTorchMusa)
    MR._accelerator_synchronize()
    assert calls == ["musa"]

    sync = Mock()
    monkeypatch.setattr(MR.current_platform, "device_type", "cuda")
    monkeypatch.setattr(MR.torch.accelerator, "synchronize", sync)
    MR._accelerator_synchronize()
    sync.assert_called_once_with()


def test_device_graph_capture_uses_supplied_context_stream(monkeypatch):
    streams = []
    capture_stream = Mock()
    default_stream = object()
    device_api = NS(
        Stream=Mock(return_value=capture_stream),
        current_stream=Mock(return_value=default_stream),
        stream=lambda s: streams.append(s) or nullcontext(),
    )
    monkeypatch.setattr(MR, "current_platform", NS(torch_device_fn=device_api))

    supplied = NS(stream=capture_stream)
    with MR._device_graph_capture(
        torch.device("cpu"), graph_capture_context=supplied
    ) as ctx:
        assert ctx is supplied
        # The supplied context must be reused, not rebuilt.
    device_api.Stream.assert_not_called()
    device_api.current_stream.assert_called_once_with()
    capture_stream.wait_stream.assert_called_once_with(default_stream)
    assert streams == [capture_stream]


def test_device_graph_capture_builds_context_when_missing(monkeypatch):
    streams = []
    capture_stream = Mock()
    device_api = NS(
        Stream=lambda device: capture_stream,
        current_stream=lambda: object(),
        stream=lambda s: streams.append(s) or nullcontext(),
    )
    monkeypatch.setattr(MR, "current_platform", NS(torch_device_fn=device_api))
    with MR._device_graph_capture(torch.device("cpu")) as ctx:
        assert ctx.stream is capture_stream
    assert streams == [capture_stream]


def test_module_graph_capture_is_the_device_context():
    """The FL module must expose its accelerator-safe capture helper.

    On the FL platform the vendor-conditional block binds ``graph_capture`` to
    the local ``_device_graph_capture`` (flagcx/musa/npu/kunlunxin vendors);
    this asserts the binding rather than upstream's graph_capture.
    """
    assert MR.graph_capture is MR._device_graph_capture


# ---------------------------------------------------------------------------
# _kl_* private-pool budget guard (Kunlunxin)
# ---------------------------------------------------------------------------


def _kl_runner(total=100 * GIB, free=80 * GIB):
    """Runner double exposing the _kl_* methods on a bare object."""
    instance = object.__new__(MR.ModelRunnerFL)
    platform = NS(
        torch_device_fn=NS(
            mem_get_info=Mock(return_value=(free, total)),
            Stream=Mock(),
            Event=Mock(),
            current_stream=Mock(),
            stream=Mock(return_value=nullcontext()),
        )
    )
    MR.ModelRunnerFL._kl_budget_reset(instance)
    return instance, platform


def test_kl_guard_on_reads_current_platform_vendor(monkeypatch):
    monkeypatch.setattr(MR.current_platform, "vendor_name", "kunlunxin")
    instance = object.__new__(MR.ModelRunnerFL)
    assert MR.ModelRunnerFL._kl_guard_on(instance) is True

    monkeypatch.setattr(MR.current_platform, "vendor_name", "nvidia")
    assert MR.ModelRunnerFL._kl_guard_on(instance) is False

    # Missing vendor_name attribute must not raise (getattr default).
    monkeypatch.setattr(
        MR, "current_platform", NS(device_type="cuda"), raising=False
    )
    assert MR.ModelRunnerFL._kl_guard_on(instance) is False


def test_kl_budget_reset_reinitializes_all_state():
    instance, _ = _kl_runner()
    # pre-dirty the state
    instance._kl_captured = {"x": 1}
    instance._kl_used = 123
    instance._kl_budget = 456
    instance._kl_stop = True
    MR.ModelRunnerFL._kl_budget_reset(instance)
    assert instance._kl_captured == {}
    assert instance._kl_used == 0
    assert instance._kl_budget == 0
    assert instance._kl_stop is False


def test_kl_filter_descs_budget_is_min_of_30pct_total_75pct_free(monkeypatch):
    total = 100 * GIB
    # 30% of total = 30 GiB; 75% of free = 15 GiB -> min = 15 GiB.
    free = 20 * GIB
    instance, platform = _kl_runner(total, free)
    monkeypatch.setattr(MR, "current_platform", platform)
    mem_info = platform.torch_device_fn.mem_get_info

    descs = ["d0", "d1", "d2"]
    out = MR.ModelRunnerFL._kl_filter_descs(instance, descs)
    assert out == descs
    assert instance._kl_budget == 15 * GIB
    # mem_get_info is queried exactly once and cached in _kl_budget.
    mem_info.assert_called_once_with()
    MR.ModelRunnerFL._kl_filter_descs(instance, descs)
    mem_info.assert_called_once_with()

    # The other branch: 75% of free (60 GiB) exceeds 30% of total (30 GiB).
    instance2, _ = _kl_runner(total, free=80 * GIB)
    monkeypatch.setattr(
        platform.torch_device_fn, "mem_get_info", Mock(return_value=(80 * GIB, total))
    )
    MR.ModelRunnerFL._kl_filter_descs(instance2, descs)
    assert instance2._kl_budget == 30 * GIB


@pytest.mark.parametrize(
    "used,budget,pre_stopped,expected_stop,expected_result",
    [
        (0, 100, False, False, "same"),       # under budget, no stop
        (99, 100, False, False, "same"),      # just under budget
        (100, 100, False, True, []),          # at budget -> stop
        (150, 100, False, True, []),          # over budget -> stop
        (10, 100, True, True, []),           # pre-stopped flag wins
    ],
)
def test_kl_filter_descs_stop_semantics(
    used, budget, pre_stopped, expected_stop, expected_result
):
    instance, _ = _kl_runner()
    instance._kl_used = used
    instance._kl_budget = budget
    instance._kl_stop = pre_stopped
    descs = ["a", "b"]
    out = MR.ModelRunnerFL._kl_filter_descs(instance, descs)
    if expected_result == "same":
        assert out == descs
    else:
        assert out == []
    assert instance._kl_stop is expected_stop


@pytest.mark.parametrize(
    "mode,num_tokens,sample", [("m0", 8, 100), ("m1", 16, 0), ("m2", 4, -5)]
)
def test_kl_note_accumulates_nonnegative_samples(mode, num_tokens, sample):
    instance, _ = _kl_runner()
    mode_enum = NS(name=mode)
    MR.ModelRunnerFL._kl_note(instance, mode_enum, num_tokens, sample)
    assert instance._kl_used == max(sample, 0)
    assert instance._kl_captured[(mode, num_tokens)] == max(sample, 0)


def test_kl_note_sums_across_calls():
    instance, _ = _kl_runner()
    MR.ModelRunnerFL._kl_note(instance, NS(name="m"), 2, 10)
    MR.ModelRunnerFL._kl_note(instance, NS(name="m"), 4, 20)
    MR.ModelRunnerFL._kl_note(instance, NS(name="m"), 8, -7)
    assert instance._kl_used == 30
    assert instance._kl_captured == {("m", 2): 10, ("m", 4): 20, ("m", 8): 0}


@pytest.mark.parametrize(
    "used,budget,pre_stopped,expected",
    [
        (10, 100, False, False),
        (100, 100, False, True),
        (500, 100, False, True),
        (10, 100, True, True),
        (500, 100, True, True),
    ],
)
def test_kl_over_budget_sets_and_reports_stop(used, budget, pre_stopped, expected):
    instance, _ = _kl_runner()
    instance._kl_used = used
    instance._kl_budget = budget
    instance._kl_stop = pre_stopped
    assert MR.ModelRunnerFL._kl_over_budget(instance) is expected
    assert instance._kl_stop is expected


class _Mode:
    """Enum-like mode double with a name (used as dict key)."""

    def __init__(self, name):
        self.name = name


def _desc(num_tokens):
    return NS(num_tokens=num_tokens)


def test_kl_apply_noop_when_budget_zero_or_not_stopped():
    instance, _ = _kl_runner()
    cc = NS(cudagraph_capture_sizes=[1, 2], max_cudagraph_capture_size=2)
    instance.compilation_config = cc

    # budget == 0 -> untouched.
    MR.ModelRunnerFL._kl_apply(instance, [(_Mode("m"), [_desc(8)])])
    assert cc.cudagraph_capture_sizes == [1, 2]

    # budget set but not stopped -> untouched.
    instance._kl_budget = 10
    instance._kl_stop = False
    MR.ModelRunnerFL._kl_apply(instance, [(_Mode("m"), [_desc(8)])])
    assert cc.cudagraph_capture_sizes == [1, 2]


def test_kl_apply_noop_when_all_modes_empty():
    instance, _ = _kl_runner()
    instance._kl_budget = 10
    instance._kl_stop = True
    cc = NS(cudagraph_capture_sizes=[1], max_cudagraph_capture_size=1)
    instance.compilation_config = cc
    MR.ModelRunnerFL._kl_apply(instance, [(_Mode("m"), [])])
    assert cc.cudagraph_capture_sizes == [1]


def test_kl_apply_keeps_common_captured_prefix_and_shrinks_sizes():
    instance, _ = _kl_runner()
    instance._kl_budget = 10
    instance._kl_stop = True
    instance._kl_used = 3 * GIB
    cc = NS(cudagraph_capture_sizes=[0], max_cudagraph_capture_size=0)
    instance.compilation_config = cc

    m0, m1 = _Mode("PIECEWISE"), _Mode("FULL")
    capture_descs = [
        (m0, [_desc(16), _desc(8), _desc(2)]),
        (m1, [_desc(16), _desc(8)]),
    ]
    # Captured: 2 for both modes; 8 only for m0; 16 never captured.
    instance._kl_captured = {
        (m0.name, 2): 1,
        (m0.name, 8): 1,
        (m1.name, 2): 1,
    }
    MR.ModelRunnerFL._kl_apply(instance, capture_descs)
    # The largest size present in every mode's captured set is 2.
    assert cc.cudagraph_capture_sizes == [2]
    assert cc.max_cudagraph_capture_size == 2


def test_kl_apply_falls_back_to_smallest_size_when_no_common_prefix():
    instance, _ = _kl_runner()
    instance._kl_budget = 10
    instance._kl_stop = True
    cc = NS(cudagraph_capture_sizes=[0], max_cudagraph_capture_size=0)
    instance.compilation_config = cc

    m0, m1 = _Mode("A"), _Mode("B")
    capture_descs = [(m0, [_desc(16)]), (m1, [_desc(8)])]
    # Nothing captured for either mode: no common prefix, keep sizes[:1].
    instance._kl_captured = {}
    MR.ModelRunnerFL._kl_apply(instance, capture_descs)
    # sorted unique sizes = [8, 16]; kept falls back to the first one.
    assert cc.cudagraph_capture_sizes == [8]
    assert cc.max_cudagraph_capture_size == 8


def test_kl_apply_noop_when_kept_equals_sizes():
    instance, _ = _kl_runner()
    instance._kl_budget = 10
    instance._kl_stop = True
    cc = NS(cudagraph_capture_sizes=[99], max_cudagraph_capture_size=99)
    instance.compilation_config = cc

    m0 = _Mode("A")
    capture_descs = [(m0, [_desc(4), _desc(8)])]
    instance._kl_captured = {("A", 4): 1, ("A", 8): 1}
    MR.ModelRunnerFL._kl_apply(instance, capture_descs)
    # every size captured -> kept == sizes -> configuration untouched.
    assert cc.cudagraph_capture_sizes == [99]
    assert cc.max_cudagraph_capture_size == 99


# ---------------------------------------------------------------------------
# Small ModelRunnerFL state helpers
# ---------------------------------------------------------------------------


def test_jit_warmup_registry_property_lazily_caches():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.vllm_config = object()
    assert "jit_warmup_registry" not in instance.__dict__

    registry = MR.ModelRunnerFL.jit_warmup_registry.__get__(instance)
    assert instance.__dict__["_jit_warmup_registry"] is registry

    # Second access returns the cached object without rebuilding.
    again = MR.ModelRunnerFL.jit_warmup_registry.__get__(instance)
    assert again is registry

    # Setter overwrites the slot.
    replacement = object()
    MR.ModelRunnerFL.jit_warmup_registry.__set__(instance, replacement)
    assert instance.__dict__["_jit_warmup_registry"] is replacement
    assert MR.ModelRunnerFL.jit_warmup_registry.__get__(instance) is replacement


def test_update_max_model_len_updates_drafter_len_only_when_unset():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.max_model_len = 128
    instance.speculative_config = None
    MR.ModelRunnerFL.update_max_model_len(instance, 256)
    assert instance.max_model_len == 256

    # Draft config without its own max_model_len inherits the new length.
    instance.speculative_config = NS(
        draft_model_config=NS(max_model_len=None))
    MR.ModelRunnerFL.update_max_model_len(instance, 512)
    assert instance.max_model_len == 512
    assert instance.effective_drafter_max_model_len == 512


def test_update_max_model_len_preserves_drafter_len_when_configured():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.max_model_len = 128
    # An explicit drafter max_model_len is preserved: the assignment only
    # runs on the `is None or draft_config is None` branch.
    instance.speculative_config = NS(
        draft_model_config=NS(max_model_len=64))
    MR.ModelRunnerFL.update_max_model_len(instance, 1024)
    assert instance.max_model_len == 1024
    # effective_drafter_max_model_len was never reassigned by the method.
    assert "effective_drafter_max_model_len" not in instance.__dict__


def test_get_or_create_async_output_copy_stream_caches(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    made = Mock(side_effect=lambda *a, **k: object())
    monkeypatch.setattr(
        MR.current_platform.torch_device_fn, "Stream", made, raising=False
    )
    instance.async_output_copy_stream = None
    first = MR.ModelRunnerFL._get_or_create_async_output_copy_stream(instance)
    assert made.call_count == 1
    again = MR.ModelRunnerFL._get_or_create_async_output_copy_stream(instance)
    assert again is first
    assert made.call_count == 1


def test_get_positions_routes_int_tensor_and_mrope():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.uses_mrope = False
    instance.positions = torch.arange(100)
    assert MR.ModelRunnerFL._get_positions(instance, 10).tolist() == list(range(10))

    idx = torch.tensor([0, 5, 10])
    assert MR.ModelRunnerFL._get_positions(instance, idx).tolist() == [0, 5, 10]

    instance.uses_mrope = True
    instance.mrope_positions = NS(
        gpu=torch.arange(300, dtype=torch.int64).reshape(3, 100))
    assert MR.ModelRunnerFL._get_positions(instance, 4).shape == (3, 4)
    assert MR.ModelRunnerFL._get_positions(
        instance, torch.tensor([9, 1])).shape == (3, 2)


def test_make_buffer_builds_cpu_gpu_buffer_on_cpu(monkeypatch):
    # CpuGpuBuffer binds PIN_MEMORY into a keyword-only default at import
    # time (__kwdefaults__); this CPU/WSL venv has no pinned-memory
    # allocator, so swap the keyword default for the duration of the check.
    from vllm.v1.utils import CpuGpuBuffer

    monkeypatch.setitem(
        CpuGpuBuffer.__init__.__kwdefaults__, "pin_memory", False
    )
    import vllm.v1.utils as v1u

    monkeypatch.setattr(v1u, "PIN_MEMORY", False)
    instance = object.__new__(MR.ModelRunnerFL)
    instance.device = torch.device("cpu")
    buf = MR.ModelRunnerFL._make_buffer(instance, 6, dtype=torch.int32)
    assert buf.cpu.shape == (6,)
    assert buf.gpu.shape == (6,)
    assert buf.np.shape == (6,)
    buf.np[:] = 3
    buf.copy_to_gpu(3)
    assert buf.gpu[:3].tolist() == [3, 3, 3]

    # numpy=False keeps a torch-only buffer (bfloat16-style allocation).
    buf2 = MR.ModelRunnerFL._make_buffer(
        instance, 4, 2, dtype=torch.float32, numpy=False)
    assert not hasattr(buf2, "np")
    assert buf2.cpu.shape == (4, 2)


def test_get_valid_sampled_token_count_requires_event_and_prev_ids():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.valid_sampled_token_count_event = None
    instance.input_batch = NS(prev_sampled_token_ids=None)
    assert MR.ModelRunnerFL._get_valid_sampled_token_count(instance) == []

    event = Mock()
    event.synchronize = Mock()
    instance.valid_sampled_token_count_event = event
    assert MR.ModelRunnerFL._get_valid_sampled_token_count(instance) == []
    event.synchronize.assert_not_called()

    instance.input_batch = NS(prev_sampled_token_ids=torch.ones((3, 1)))
    instance.valid_sampled_token_count_cpu = torch.tensor([2, 4, 6])
    counts = MR.ModelRunnerFL._get_valid_sampled_token_count(instance)
    assert counts == [2, 4, 6]
    event.synchronize.assert_called_once_with()


def test_get_encoder_timing_stats_drains_registry():
    from vllm.v1.worker.utils import EncoderTimingStats

    instance = object.__new__(MR.ModelRunnerFL)
    instance._encoder_timing_lock = __import__("threading").Lock()
    instance.encoder_timing_registry = {}
    assert MR.ModelRunnerFL.get_encoder_timing_stats(instance) == {}

    stats = EncoderTimingStats()
    stats.encoder_forward_secs += 1.25
    stats.num_encoder_calls += 2
    instance.encoder_timing_registry["req-1"] = stats
    out = MR.ModelRunnerFL.get_encoder_timing_stats(instance)
    assert out == {"req-1": {"encoder_forward_secs": 1.25, "num_encoder_calls": 2}}
    # Registry is cleared by the read.
    assert instance.encoder_timing_registry == {}


@pytest.mark.parametrize(
    "max_sched,query_len,tokens,reqs,expected",
    [
        (1, 1, 4, 4, True),
        (1, 1, 4, 3, False),
        (2, 1, 4, 4, False),
        (2, 2, 4, 2, True),
    ],
)
def test_is_uniform_decode_truth_table(max_sched, query_len, tokens, reqs, expected):
    assert (
        MR.ModelRunnerFL._is_uniform_decode(max_sched, query_len, tokens, reqs)
        is expected
    )


def test_is_uniform_decode_force_overrides():
    forced = MR.ModelRunnerFL._is_uniform_decode(9, 9, 9, 9, force_uniform_decode=True)
    assert forced is True
    unforced = MR.ModelRunnerFL._is_uniform_decode(
        9, 9, 9, 9, force_uniform_decode=False
    )
    assert unforced is False


def test_on_request_state_removed_is_a_noop_hook():
    instance = object.__new__(MR.ModelRunnerFL)
    MR.ModelRunnerFL._on_request_state_removed(instance, "req", None)


def test_sync_device_delegates_to_accelerator_synchronize(monkeypatch):
    sync = Mock()
    monkeypatch.setattr(MR.torch.accelerator, "synchronize", sync)
    monkeypatch.setattr(MR.current_platform, "device_type", "cuda")
    instance = object.__new__(MR.ModelRunnerFL)
    MR.ModelRunnerFL._sync_device(instance)
    sync.assert_called_once_with()


def test_get_or_create_encoder_output_from_cache():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.encoder_cache = {"h1": "out-1"}
    assert MR.ModelRunnerFL._get_encoder_output_from_cache(instance, "h1") == "out-1"
    assert MR.ModelRunnerFL._get_encoder_output_from_cache(instance, "missing") is None


def test_process_encoder_cache_scheduler_output_pops_hashes():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.encoder_cache = {"h1": 1, "h2": 2}
    scheduler_output = NS(free_encoder_mm_hashes=["h1"])
    MR.ModelRunnerFL._process_encoder_cache_scheduler_output(
        instance, scheduler_output)
    assert instance.encoder_cache == {"h2": 2}


def test_reset_mm_cache_and_encoder_cache_clear_caches():
    late = Mock()
    instance = object.__new__(MR.ModelRunnerFL)
    instance.mm_budget = Mock()
    instance.late_interaction_runner = late
    instance.encoder_cache = {"h": 1}

    MR.ModelRunnerFL.reset_mm_cache(instance)
    instance.mm_budget.reset_cache.assert_called_once_with()
    late.clear.assert_called_once_with()

    late.clear.reset_mock()
    MR.ModelRunnerFL.reset_encoder_cache(instance)
    assert instance.encoder_cache == {}
    late.clear.assert_called_once_with()


def test_post_kv_cache_wake_up_reinits_fp8_scales(monkeypatch):
    called = []
    monkeypatch.setattr(
        MR.ModelRunnerFL, "init_fp8_kv_scales", lambda self: called.append("fp8")
    )
    instance = object.__new__(MR.ModelRunnerFL)
    MR.ModelRunnerFL.post_kv_cache_wake_up(instance)
    assert called == ["fp8"]


def test_is_all_reqs_chunked_prefill_reads_discard_mask():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.input_batch = NS(num_reqs=3)
    instance.discard_request_mask = NS(np=np.array([True, True, True]))
    assert MR.ModelRunnerFL._is_all_reqs_chunked_prefill(instance) is True

    instance.discard_request_mask = NS(np=np.array([True, False, True]))
    assert MR.ModelRunnerFL._is_all_reqs_chunked_prefill(instance) is False


def test_zero_block_ids_guarded_by_zeroer_presence():
    instance = object.__new__(MR.ModelRunnerFL)
    # Without the zeroer attribute the call is a no-op.
    MR.ModelRunnerFL._zero_block_ids(instance, [1, 2])

    zeroer = Mock()
    instance._kv_block_zeroer = zeroer
    MR.ModelRunnerFL._zero_block_ids(instance, [3, 4])
    zeroer.zero_block_ids.assert_called_once_with([3, 4])


def test_synchronize_input_prep_yields_without_event():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.prepare_inputs_event = None
    with MR.ModelRunnerFL.synchronize_input_prep(instance):
        pass

    event = Mock()
    instance.prepare_inputs_event = event
    with MR.ModelRunnerFL.synchronize_input_prep(instance):
        event.synchronize.assert_called_once_with()
    event.record.assert_called_once_with()


def test_freeze_gc_restores_previous_gc_state():
    # Enable gc, freeze during the context, and verify restoration.
    gc.enable()
    with MR.ModelRunnerFL._freeze_gc():
        pass
    assert gc.isenabled()

    gc.disable()
    with MR.ModelRunnerFL._freeze_gc():
        pass
    assert not gc.isenabled()
    gc.enable()
