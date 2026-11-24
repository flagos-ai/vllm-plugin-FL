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

"""G7-D (batch 2) coverage: Kunlunxin KL guards, PP receive, misc helpers.

Targets (814b): _kl_guard_on 7425-7426, _kl_budget_reset 7428-7432,
_kl_filter_descs 7434-7446, _kl_note 7448-7450, _kl_over_budget 7452-7455,
_kl_apply 7457-7482, _pp_receive_prev_sampled_token_ids_to_input_batch
5079-5109, _get_draft_token_ids_cpu 5154-5167,
_copy_valid_sampled_token_count 5169-5192, update_config 5513-5524,
maybe_randomize_inputs 6023-6070, get_encoder_timing_stats 8167-8180,
timed_encoder_operation 8183-8224, get_supported_generation_tasks 3535-3551,
get_supported_pooling_tasks 3553-3558.

Everything is a SimpleNamespace fake runner; module collaborators are patched at
the ``model_runner`` boundary. No device / network access.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import torch

from vllm_fl.worker import model_runner as mr


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


class _FakeDevFn:
    @contextlib.contextmanager
    def stream(self, _stream):
        yield

    def current_stream(self):
        return SimpleNamespace(wait_stream=lambda _s: None)

    def mem_get_info(self):
        return (2 << 30, 8 << 30)


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


# ---------------------------------------------------------------------------
# Kunlunxin KL guard family
# ---------------------------------------------------------------------------
def test_kl_guard_on_matches_kunlunxin():
    with _stack((mr, "current_platform", SimpleNamespace(vendor_name="kunlunxin"))):
        assert mr.ModelRunnerFL._kl_guard_on(SimpleNamespace()) is True
    with _stack((mr, "current_platform", SimpleNamespace(vendor_name="amd"))):
        assert mr.ModelRunnerFL._kl_guard_on(SimpleNamespace()) is False


def test_kl_budget_reset_clears_state():
    runner = SimpleNamespace(
        _kl_captured={"x": 1}, _kl_used=9, _kl_budget=9, _kl_stop=True
    )
    mr.ModelRunnerFL._kl_budget_reset(runner)
    assert runner._kl_captured == {}
    assert runner._kl_used == 0
    assert runner._kl_budget == 0
    assert runner._kl_stop is False


def test_kl_filter_descs_allocates_budget_then_passes_desc():
    runner = SimpleNamespace(_kl_budget=0, _kl_used=0, _kl_stop=False)
    platform = SimpleNamespace(
        torch_device_fn=SimpleNamespace(mem_get_info=lambda: (2 << 30, 8 << 30))
    )
    descs = ["a", "b"]
    with _stack((mr, "current_platform", platform)):
        out = mr.ModelRunnerFL._kl_filter_descs(runner, descs)
    # min(8GiB*0.30, 2GiB*0.75)
    assert runner._kl_budget == int(min(int(8 << 30) * 0.30, int(2 << 30) * 0.75))
    assert out == descs


def test_kl_filter_descs_stops_when_budget_exhausted():
    runner = SimpleNamespace(_kl_budget=100, _kl_used=100, _kl_stop=False)
    out = mr.ModelRunnerFL._kl_filter_descs(runner, ["a"])
    assert out == []
    assert runner._kl_stop is True


def test_kl_note_accumulates_and_records():
    runner = SimpleNamespace(_kl_captured={}, _kl_used=0)
    mr.ModelRunnerFL._kl_note(runner, SimpleNamespace(name="decode"), 7, 3)
    assert runner._kl_used == 3
    assert runner._kl_captured == {("decode", 7): 3}
    # negative sample clamps to zero
    mr.ModelRunnerFL._kl_note(runner, SimpleNamespace(name="decode"), 7, -5)
    assert runner._kl_used == 3
    assert runner._kl_captured == {("decode", 7): 0}


def test_kl_over_budget_sets_stop():
    runner = SimpleNamespace(_kl_used=5, _kl_budget=5, _kl_stop=False)
    assert mr.ModelRunnerFL._kl_over_budget(runner) is True
    assert runner._kl_stop is True


def test_kl_apply_early_return_without_budget():
    runner = SimpleNamespace(_kl_budget=0, _kl_stop=True, _kl_used=0)
    mr.ModelRunnerFL._kl_apply(runner, [])
    assert not hasattr(runner, "compilation_config")


def test_kl_apply_early_return_when_not_stopped():
    runner = SimpleNamespace(_kl_budget=10, _kl_stop=False, _kl_used=0)
    mr.ModelRunnerFL._kl_apply(runner, [(SimpleNamespace(name="d"), [object()])])
    assert not hasattr(runner, "compilation_config")


def test_kl_apply_shrinks_capture_sizes_to_contiguous_captured_prefix():
    sizes = [SimpleNamespace(num_tokens=500), SimpleNamespace(num_tokens=1000)]
    mode = SimpleNamespace(name="decode")
    runner = SimpleNamespace(
        _kl_budget=100,
        _kl_stop=True,
        _kl_used=42,
        _kl_captured={("decode", 500): 1},
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[500, 1000], max_cudagraph_capture_size=1000
        ),
    )
    mr.ModelRunnerFL._kl_apply(runner, [(mode, sizes)])
    assert runner.compilation_config.cudagraph_capture_sizes == [500]
    assert runner.compilation_config.max_cudagraph_capture_size == 500


def test_kl_apply_keeps_all_sizes_when_all_captured():
    sizes = [SimpleNamespace(num_tokens=500), SimpleNamespace(num_tokens=1000)]
    mode = SimpleNamespace(name="decode")
    runner = SimpleNamespace(
        _kl_budget=100,
        _kl_stop=True,
        _kl_used=42,
        _kl_captured={("decode", 500): 1, ("decode", 1000): 1},
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[500, 1000], max_cudagraph_capture_size=1000
        ),
    )
    mr.ModelRunnerFL._kl_apply(runner, [(mode, sizes)])
    assert runner.compilation_config.cudagraph_capture_sizes == [500, 1000]


# ---------------------------------------------------------------------------
# PP receive + draft ids + valid sampled count
# ---------------------------------------------------------------------------
def _pp_runner(*, chunked):
    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        num_tokens_no_spec=[1, 2],
        is_token_ids=torch.zeros(2, 8, dtype=torch.bool),
        prev_sampled_token_ids=None,
        prev_req_id_to_index=None,
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        device=torch.device("cpu"),
        discard_request_mask=SimpleNamespace(np=np.array([False, True])),
        requests={
            "r0": SimpleNamespace(output_token_ids=[]),
            "r1": SimpleNamespace(output_token_ids=[]),
        },
    )
    runner._is_all_reqs_chunked_prefill = lambda: chunked
    return runner


def test_pp_receive_broadcasts_and_builds_prev_index():
    runner = _pp_runner(chunked=False)
    pp = SimpleNamespace(is_last_rank=False, last_rank=1, device_group=None)
    broadcast = _Recorder()
    with _stack(
        (mr, "get_pp_group", lambda: pp),
        (mr.torch.distributed, "broadcast", broadcast),
    ):
        mr.ModelRunnerFL._pp_receive_prev_sampled_token_ids_to_input_batch(runner)
    assert broadcast.calls, "broadcast should be issued when not chunked prefill"
    assert hasattr(runner, "_pp_recv_work")
    assert runner.input_batch.prev_sampled_token_ids is not None
    assert runner.requests["r0"].output_token_ids == [-1]
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 2
    assert bool(runner.input_batch.is_token_ids[0, 1]) is True
    assert runner.input_batch.prev_req_id_to_index == {"r0": 0}


def test_pp_receive_skips_broadcast_for_chunked_prefill():
    runner = _pp_runner(chunked=True)
    pp = SimpleNamespace(is_last_rank=False, last_rank=1, device_group=None)
    broadcast = _Recorder()
    with _stack(
        (mr, "get_pp_group", lambda: pp),
        (mr.torch.distributed, "broadcast", broadcast),
    ):
        mr.ModelRunnerFL._pp_receive_prev_sampled_token_ids_to_input_batch(runner)
    assert broadcast.calls == []
    assert not hasattr(runner, "_pp_recv_work")


def test_get_draft_token_ids_cpu_list_path_returns_as_is():
    draft = [[1, 2], [3]]
    runner = SimpleNamespace(
        _draft_token_ids=draft, input_batch=SimpleNamespace(req_ids=["a", "b"])
    )
    ids, req_ids = mr.ModelRunnerFL._get_draft_token_ids_cpu(runner)
    assert ids is draft
    assert req_ids == ["a", "b"]


def test_get_draft_token_ids_cpu_tensor_path_syncs_and_slices():
    event = SimpleNamespace(synchronize=lambda: None)
    runner = SimpleNamespace(
        _draft_token_ids=torch.tensor([[1, 2], [3, 4]]),
        _draft_token_req_ids=["a", "b"],
        draft_token_ids_event=event,
        draft_token_ids_cpu=torch.tensor([[1, 2], [3, 4]]),
    )
    ids, req_ids = mr.ModelRunnerFL._get_draft_token_ids_cpu(runner)
    assert ids == [[1, 2], [3, 4]]
    assert req_ids == ["a", "b"]


def test_get_draft_token_ids_cpu_none_req_ids_returns_empty():
    runner = SimpleNamespace(
        _draft_token_ids=torch.tensor([[1]]), _draft_token_req_ids=None
    )
    assert mr.ModelRunnerFL._get_draft_token_ids_cpu(runner) == ([], [])


def test_copy_valid_sampled_token_count_early_return_without_event():
    runner = SimpleNamespace(valid_sampled_token_count_event=None)
    mr.ModelRunnerFL._copy_valid_sampled_token_count(runner, torch.zeros(1), torch.zeros(1))
    assert not hasattr(runner, "input_batch")




# ---------------------------------------------------------------------------
# update_config / maybe_randomize_inputs
# ---------------------------------------------------------------------------
def test_update_config_applies_supported_override():
    runner = SimpleNamespace(load_config={"a": 1}, model_config={"b": 2})
    new_cfg = {"a": 9}
    with _stack((mr, "update_config", lambda cfg, ov: ("updated", ov))):
        mr.ModelRunnerFL.update_config(runner, {"load_config": new_cfg})
    assert runner.load_config == ("updated", new_cfg)
    assert runner.model_config == {"b": 2}


def test_update_config_rejects_unknown_name():
    runner = SimpleNamespace(load_config={}, model_config={})
    try:
        mr.ModelRunnerFL.update_config(runner, {"bogus": {}})
    except ValueError as err:
        assert "bogus" in str(err)
    else:  # pragma: no cover - unexpected
        raise AssertionError("expected ValueError")


def _randomize_runner(parallel_size):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=parallel_size)
        ),
        model_config=SimpleNamespace(get_vocab_size=lambda: 100),
        input_ids=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.long)),
        inputs_embeds=SimpleNamespace(gpu=torch.zeros(4, 3)),
    )


def test_maybe_randomize_inputs_noop_when_disabled():
    runner = _randomize_runner(1)
    ids = torch.ones(4, dtype=torch.long)
    with _stack((mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", False)):
        with mr.ModelRunnerFL.maybe_randomize_inputs(runner, ids, None, False):
            pass
    assert torch.equal(ids, torch.ones(4, dtype=torch.long))


def test_maybe_randomize_inputs_randomizes_then_zeroes_input_ids():
    runner = _randomize_runner(2)
    ids = torch.ones(4, dtype=torch.long)
    with _stack((mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", True)):
        with mr.ModelRunnerFL.maybe_randomize_inputs(runner, ids, None, False):
            assert int(ids.min()) >= 0 and int(ids.max()) < 100
    assert torch.equal(ids, torch.zeros(4, dtype=torch.long))


def test_maybe_randomize_inputs_randomizes_prompt_embeds():
    runner = _randomize_runner(2)
    embeds = torch.ones(4, 3)
    with _stack((mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", True)):
        with mr.ModelRunnerFL.maybe_randomize_inputs(runner, None, embeds, False):
            pass
    assert torch.equal(embeds, torch.zeros(4, 3))


# ---------------------------------------------------------------------------
# encoder timing
# ---------------------------------------------------------------------------
def test_get_encoder_timing_stats_serializes_and_clears():
    registry = {"r0": SimpleNamespace(to_dict=lambda: {"s": 1.0, "n": 2})}
    runner = SimpleNamespace(
        _encoder_timing_lock=contextlib.nullcontext(),
        encoder_timing_registry=registry,
    )
    out = mr.ModelRunnerFL.get_encoder_timing_stats(runner)
    assert out == {"r0": {"s": 1.0, "n": 2}}
    assert registry == {}


def test_timed_encoder_operation_disabled_is_passthrough():
    runner = SimpleNamespace()
    with mr.ModelRunnerFL.timed_encoder_operation(runner, False, [("r", 0)], 0, 1):
        pass
    assert not hasattr(runner, "encoder_timing_registry")


def test_timed_encoder_operation_accumulates_per_request():
    registry = {}
    runner = SimpleNamespace(
        _encoder_timing_lock=contextlib.nullcontext(),
        encoder_timing_registry=registry,
    )
    fake_accel = SimpleNamespace(synchronize=lambda: None)
    with _stack((mr.torch, "accelerator", fake_accel)):
        with mr.ModelRunnerFL.timed_encoder_operation(
            runner, True, [("r0", {}), ("r0", {})], 0, 2
        ):
            pass
    assert set(registry) == {"r0"}
    assert registry["r0"].num_encoder_calls == 1


# ---------------------------------------------------------------------------
# supported tasks (cheap)
# ---------------------------------------------------------------------------
def test_get_supported_generation_tasks_generate_and_realtime():
    runner = SimpleNamespace(get_model=lambda: "m")
    with _stack(
        (mr, "is_text_generation_model", lambda m: True),
        (mr, "supports_transcription", lambda m: False),
        (mr, "supports_realtime", lambda m: True),
    ):
        tasks = mr.ModelRunnerFL.get_supported_generation_tasks(runner)
    assert tasks == ["generate", "realtime"]


def test_get_supported_generation_tasks_transcription_only():
    model = SimpleNamespace(supports_transcription_only=True)
    runner = SimpleNamespace(get_model=lambda: model)
    with _stack(
        (mr, "is_text_generation_model", lambda m: False),
        (mr, "supports_transcription", lambda m: True),
        (mr, "supports_realtime", lambda m: False),
    ):
        tasks = mr.ModelRunnerFL.get_supported_generation_tasks(runner)
    assert tasks == ["transcription"]


def test_get_supported_pooling_tasks_empty_when_not_pooling():
    runner = SimpleNamespace(get_model=lambda: "m")
    with _stack((mr, "is_pooling_model", lambda m: False)):
        assert mr.ModelRunnerFL.get_supported_pooling_tasks(runner) == []


def test_get_supported_pooling_tasks_delegates_to_pooler():
    model = SimpleNamespace(pooler=SimpleNamespace(get_supported_tasks=lambda: ["embed"]))
    runner = SimpleNamespace(get_model=lambda: model)
    with _stack((mr, "is_pooling_model", lambda m: True)):
        assert mr.ModelRunnerFL.get_supported_pooling_tasks(runner) == ["embed"]
