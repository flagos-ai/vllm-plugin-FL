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

"""G10-B coverage: batch dispatch, cudagraph capture/iterators, pooling warmup
and kv-cache wiring of ModelRunnerFL (814b).

Targets: _determine_batch_execution_and_padding (incl. DP and metrics paths),
_create_encoder_cudagraph_manager, _maybe_init_encoder_cudagraph_manager,
_get_musa_graph_capture_context, calculate_reorder_batch_threshold,
_attn_group_iterator, _kv_cache_spec_attn_group_iterator,
initialize_kv_cache_tensors, initialize_metadata_builders,
maybe_add_kv_sharing_layers_to_kv_cache_groups, _warmup_and_capture,
_capture_cudagraphs, _dummy_pooler_run_task, _dummy_pooler_run.

The runner is a SimpleNamespace; the accelerator, dispatcher, pooler and
budget collaborators are patched at the module boundary. Pure CPU.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_fl.worker.model_runner as mr


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


class _Ctx:
    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------
# _determine_batch_execution_and_padding
# --------------------------------------------------------------------------
def _determine_runner(**extra):
    base = dict(
        uniform_decode_query_len=1,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        cudagraph_dispatcher=SimpleNamespace(
            dispatch=lambda **kw: (
                mr.CUDAGraphMode.NONE,
                SimpleNamespace(num_tokens=kw["num_tokens"]),
            )
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        ),
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
    )
    base.update(extra)
    return SimpleNamespace(**base)








# --------------------------------------------------------------------------
# EncoderCudaGraph manager helpers
# --------------------------------------------------------------------------
def test_create_encoder_cudagraph_manager_short_circuits_without_flag():
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=False),
        supports_mm_inputs=True,
    )
    assert mr.ModelRunnerFL._create_encoder_cudagraph_manager(runner) is None


def test_create_encoder_cudagraph_manager_short_circuits_without_mm():
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=False,
    )
    assert mr.ModelRunnerFL._create_encoder_cudagraph_manager(runner) is None


def test_maybe_init_encoder_cudagraph_manager_keeps_existing():
    runner = SimpleNamespace(
        encoder_cudagraph_manager="existing",
        _create_encoder_cudagraph_manager=lambda: pytest.fail("must not rebuild"),
    )
    mr.ModelRunnerFL._maybe_init_encoder_cudagraph_manager(runner)
    assert runner.encoder_cudagraph_manager == "existing"


def test_maybe_init_encoder_cudagraph_manager_creates_and_logs():
    calls = []
    runner = SimpleNamespace(
        encoder_cudagraph_manager=None,
        _create_encoder_cudagraph_manager=lambda: "manager",
    )
    with _stack((mr, "logger", SimpleNamespace(info=lambda *a: calls.append(a)))):
        mr.ModelRunnerFL._maybe_init_encoder_cudagraph_manager(runner)
    assert runner.encoder_cudagraph_manager == "manager"
    assert calls


def test_get_musa_graph_capture_context_creates_then_reuses():
    created = []

    def _gcc(stream):
        created.append(stream)
        return "ctx"

    runner = SimpleNamespace(device="cpu")
    platform = SimpleNamespace(
        torch_device_fn=SimpleNamespace(Stream=lambda device: ("stream", device))
    )
    with _stack(
        (mr, "GraphCaptureContext", _gcc),
        (mr, "current_platform", platform),
    ):
        first = mr.ModelRunnerFL._get_musa_graph_capture_context(runner)
        second = mr.ModelRunnerFL._get_musa_graph_capture_context(runner)
    assert first is second == "ctx"
    assert created == [("stream", "cpu")]


# --------------------------------------------------------------------------
# calculate_reorder_batch_threshold
# --------------------------------------------------------------------------
def test_calculate_reorder_batch_threshold_no_groups():
    runner = SimpleNamespace(_attn_group_iterator=lambda: iter(()))
    mr.ModelRunnerFL.calculate_reorder_batch_threshold(runner)
    assert runner.reorder_batch_threshold is None


def test_calculate_reorder_batch_threshold_picks_min():
    def _group(threshold):
        return SimpleNamespace(
            get_metadata_builder=lambda: SimpleNamespace(
                reorder_batch_threshold=threshold
            )
        )

    runner = SimpleNamespace(
        _attn_group_iterator=lambda: iter([_group(None), _group(4), _group(2)])
    )
    mr.ModelRunnerFL.calculate_reorder_batch_threshold(runner)
    assert runner.reorder_batch_threshold == 2


# --------------------------------------------------------------------------
# group iterators / kv cache tensors
# --------------------------------------------------------------------------
def test_attn_group_iterator_flattens_groups():
    runner = SimpleNamespace(attn_groups=[[1], [2, 3]])
    assert list(mr.ModelRunnerFL._attn_group_iterator(runner)) == [1, 2, 3]


def test_kv_cache_spec_attn_group_iterator_empty_without_groups():
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        attn_groups=[[1]],
    )
    assert list(mr.ModelRunnerFL._kv_cache_spec_attn_group_iterator(runner)) == []


def test_kv_cache_spec_attn_group_iterator_yields_groups():
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=["g"]),
        attn_groups=[[1], [2, 3]],
    )
    assert list(mr.ModelRunnerFL._kv_cache_spec_attn_group_iterator(runner)) == [
        1,
        2,
        3,
    ]


def test_initialize_kv_cache_tensors_delegates():
    calls = []
    runner = SimpleNamespace(
        _initialize_kv_cache_tensors_v032=lambda *a: calls.append(a) or "caches"
    )
    out = mr.ModelRunnerFL.initialize_kv_cache_tensors(runner, "cfg", [16], "ctx")
    assert out == "caches"
    assert calls == [("cfg", [16], "ctx")]


# --------------------------------------------------------------------------
# initialize_metadata_builders
# --------------------------------------------------------------------------
def test_initialize_metadata_builders_builds_each_group():
    calls = []
    group = SimpleNamespace(
        create_metadata_builders=lambda *a, **k: calls.append((a, k))
    )
    runner = SimpleNamespace(
        attn_groups=[[group]],
        vllm_config="cfg",
        device="cpu",
        parallel_config=SimpleNamespace(use_ubatching=False, num_ubatches=1),
        calculate_reorder_batch_threshold=lambda: calls.append("threshold"),
        speculative_config=None,
    )
    cfg = SimpleNamespace(kv_cache_groups=["x"])
    mr.ModelRunnerFL.initialize_metadata_builders(runner, cfg, [16])
    assert calls[0][0] == ("cfg", "cpu", 16)
    assert calls[0][1]["num_metadata_builders"] == 1
    assert calls[-1] == "threshold"


def test_initialize_metadata_builders_uses_ubatch_count():
    calls = []
    group = SimpleNamespace(
        create_metadata_builders=lambda *a, **k: calls.append(k)
    )
    runner = SimpleNamespace(
        attn_groups=[[group], [group]],
        vllm_config="cfg",
        device="cpu",
        parallel_config=SimpleNamespace(use_ubatching=True, num_ubatches=2),
        calculate_reorder_batch_threshold=lambda: None,
        speculative_config=None,
    )
    cfg = SimpleNamespace(kv_cache_groups=["x", "y"])
    mr.ModelRunnerFL.initialize_metadata_builders(runner, cfg, [])
    assert calls[0]["num_metadata_builders"] == 2


# --------------------------------------------------------------------------
# maybe_add_kv_sharing_layers_to_kv_cache_groups
# --------------------------------------------------------------------------
def test_maybe_add_kv_sharing_noop_without_sharing():
    runner = SimpleNamespace(shared_kv_cache_layers={})
    with _stack(
        (
            mr,
            "add_kv_sharing_layers_to_kv_cache_groups",
            lambda *a: pytest.fail("must not add"),
        )
    ):
        mr.ModelRunnerFL.maybe_add_kv_sharing_layers_to_kv_cache_groups(
            runner, SimpleNamespace(kv_cache_groups=[])
        )


def test_maybe_add_kv_sharing_adds_layers():
    calls = []
    runner = SimpleNamespace(
        shared_kv_cache_layers={"a": "b"},
        runner_only_attn_layers=["x"],
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
    )
    with _stack(
        (
            mr,
            "add_kv_sharing_layers_to_kv_cache_groups",
            lambda *a: calls.append(a),
        )
    ):
        mr.ModelRunnerFL.maybe_add_kv_sharing_layers_to_kv_cache_groups(
            runner, SimpleNamespace(kv_cache_groups=["g"])
        )
    assert calls == [({"a": "b"}, ["g"], ["x"])]


def test_maybe_add_kv_sharing_marks_fast_prefill_eligible():
    runner = SimpleNamespace(
        shared_kv_cache_layers={"a": "b", "c": "d"},
        runner_only_attn_layers=[],
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=True),
        kv_sharing_fast_prefill_eligible_layers=set(),
        vllm_config="cfg",
    )
    # reversed insertion order is a, c, z; a and c are shared then z breaks.
    layers = {"z": 3, "c": 1, "a": 2}
    with _stack(
        (
            mr,
            "add_kv_sharing_layers_to_kv_cache_groups",
            lambda *a: None,
        ),
        (mr, "get_layers_from_vllm_config", lambda cfg, cls: layers),
    ):
        mr.ModelRunnerFL.maybe_add_kv_sharing_layers_to_kv_cache_groups(
            runner, SimpleNamespace(kv_cache_groups=["g"])
        )
    assert runner.kv_sharing_fast_prefill_eligible_layers == {"a", "c"}


# --------------------------------------------------------------------------
# _warmup_and_capture / _capture_cudagraphs
# --------------------------------------------------------------------------
def _fake_torch(sync_calls, record_names):
    return SimpleNamespace(
        accelerator=SimpleNamespace(synchronize=lambda: sync_calls.append(1)),
        profiler=SimpleNamespace(
            record_function=lambda name: (
                record_names.append(name) or _Ctx()
            )
        ),
    )


def test_warmup_and_capture_runs_warmups_and_capture():
    dummy_calls = []
    sync_calls = []
    record_names = []
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_num_of_warmups=1),
        _dummy_run=lambda *a, **k: dummy_calls.append(k),
    )
    desc = SimpleNamespace(num_tokens=4, uniform=True, num_active_loras=0)
    with _stack((mr, "torch", _fake_torch(sync_calls, record_names))):
        mr.ModelRunnerFL._warmup_and_capture(
            runner, desc, mr.CUDAGraphMode.NONE
        )
    assert len(dummy_calls) == 2
    assert dummy_calls[0]["cudagraph_runtime_mode"] is mr.CUDAGraphMode.NONE
    assert dummy_calls[1]["is_graph_capturing"] is True
    assert sync_calls == [1]
    assert record_names


def test_capture_cudagraphs_empty_descriptors_returns():
    runner = SimpleNamespace()
    with _stack((mr, "torch", _fake_torch([], []))):
        mr.ModelRunnerFL._capture_cudagraphs(
            runner, [], mr.CUDAGraphMode.FULL
        )


def test_capture_cudagraphs_captures_each_descriptor():
    captured = []
    sync_calls = []
    removed = []
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=False),
        _warmup_and_capture=lambda desc, **k: captured.append((desc, k)),
        maybe_remove_all_loras=lambda cfg: removed.append(cfg),
        lora_config="lora",
    )
    desc = SimpleNamespace(num_tokens=8, uniform=True)
    with _stack(
        (mr, "torch", _fake_torch(sync_calls, [])),
        (mr, "is_global_first_rank", lambda: False),
    ):
        mr.ModelRunnerFL._capture_cudagraphs(
            runner, [desc], mr.CUDAGraphMode.FULL
        )
    assert len(captured) == 1
    assert captured[0][1]["cudagraph_runtime_mode"] is mr.CUDAGraphMode.FULL
    assert sync_calls == [1]
    assert removed == ["lora"]


# --------------------------------------------------------------------------
# _dummy_pooler_run_task / _dummy_pooler_run
# --------------------------------------------------------------------------
class _Pooler:
    def __init__(self, calls, result=None, error=None):
        self.calls = calls
        self.result = result
        self.error = error

    def get_pooling_updates(self, task):
        return SimpleNamespace(apply=lambda p: self.calls.append(("apply", task)))

    def __call__(self, **kw):
        self.calls.append(("run", kw))
        if self.error is not None:
            raise self.error
        return self.result


class _PoolingParams:
    def __init__(self, task=None):
        self.task = task
        self.applied = False

    def verify(self, model_config):
        self.model_config = model_config


class _PoolingMetadata:
    def __init__(self, **kw):
        self.kw = kw
        self.cursor = None

    def build_pooling_cursor(self, num_scheduled_tokens_np, seq_lens_cpu, device):
        self.cursor = (num_scheduled_tokens_np, seq_lens_cpu, device)


def _pooler_task_runner(pooler, max_num_seqs=4):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        device="cpu",
        get_model=lambda: SimpleNamespace(pooler=pooler),
        model_config="mc",
    )


def test_dummy_pooler_run_task_builds_and_runs():
    calls = []
    pooler = _Pooler(calls, result=["out"])
    runner = _pooler_task_runner(pooler)
    with _stack(
        (mr, "PoolingParams", _PoolingParams),
        (mr, "PoolingMetadata", _PoolingMetadata),
        (mr, "PoolingStates", lambda: "state"),
    ):
        out = mr.ModelRunnerFL._dummy_pooler_run_task(
            runner, torch.zeros(8, 3), "embed"
        )
    assert out == ["out"]
    assert any(c[0] == "apply" for c in calls)
    assert any(c[0] == "run" for c in calls)


def test_dummy_pooler_run_task_wraps_oom():
    pooler = _Pooler([], error=RuntimeError("CUDA out of memory"))
    runner = _pooler_task_runner(pooler)
    with _stack(
        (mr, "PoolingParams", _PoolingParams),
        (mr, "PoolingMetadata", _PoolingMetadata),
        (mr, "PoolingStates", lambda: "state"),
    ):
        with pytest.raises(RuntimeError, match="warming up pooler"):
            mr.ModelRunnerFL._dummy_pooler_run_task(
                runner, torch.zeros(4, 3), "embed"
            )


def test_dummy_pooler_run_task_reraises_other_runtime_error():
    pooler = _Pooler([], error=RuntimeError("boom"))
    runner = _pooler_task_runner(pooler)
    with _stack(
        (mr, "PoolingParams", _PoolingParams),
        (mr, "PoolingMetadata", _PoolingMetadata),
        (mr, "PoolingStates", lambda: "state"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            mr.ModelRunnerFL._dummy_pooler_run_task(
                runner, torch.zeros(4, 3), "embed"
            )


def test_dummy_pooler_run_encoder_only_returns_empty():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            )
        )
    )
    out = mr.ModelRunnerFL._dummy_pooler_run(runner, torch.zeros(2, 3))
    assert isinstance(out, torch.Tensor)
    assert out.numel() == 0


def test_dummy_pooler_run_raises_without_tasks():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)
        ),
        get_supported_pooling_tasks=lambda: [],
        model_config=SimpleNamespace(model="m"),
    )
    with pytest.raises(RuntimeError, match="does not support"):
        mr.ModelRunnerFL._dummy_pooler_run(runner, torch.zeros(2, 3))


def test_dummy_pooler_run_picks_largest_task():
    tasks = []

    def _run_task(hidden, task):
        tasks.append(task)
        size = 1 if task == "a" else 5
        return [torch.zeros(size)]

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)
        ),
        get_supported_pooling_tasks=lambda: ["a", "b"],
        model_config=SimpleNamespace(model="m"),
        _dummy_pooler_run_task=_run_task,
    )
    out = mr.ModelRunnerFL._dummy_pooler_run(runner, torch.zeros(4, 3))
    assert tasks == ["a", "b", "b"]
    assert out[0].numel() == 5
