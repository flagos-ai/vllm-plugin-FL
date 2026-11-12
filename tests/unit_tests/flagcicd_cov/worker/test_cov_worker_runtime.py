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

"""Coverage tests for vllm_fl/worker/worker.py runtime paths.

``execute_model`` (forward/no-forward, pipeline-middle-rank send),
``execute_dummy_batch``, ``annotate_profile``, and the model-runner
passthroughs (LoRA, model, tasks, caches, config, sampling, health) are
driven here through unbound WorkerFL method calls on SimpleNamespace
workers — the same pattern the dev tests (tests/unit_tests/worker/
test_worker.py) use. Per CONTRACT R2 the vllm engine objects, distributed
groups, and module-level collaborators are inputs replaced with
recorders; nothing on WorkerFL itself is patched to select a branch.
"""

from . import _imports  # noqa: F401  (installs local import shims first)

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput

import vllm_fl.worker.worker as worker_module
from vllm_fl.worker.worker import WorkerFL


class _Patch:
    """Deterministic set/restore of a module attribute (R2 input boundary)."""

    def __init__(self, module, name, value):
        self.module = module
        self.name = name
        self.value = value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _no_sp_vllm_config(pp_size=1, backend="mp"):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            distributed_executor_backend=backend,
        ),
    )


def _forward_worker(runner, pp_size=1):
    """Worker namespace with annotate_profile bound like a real instance."""
    worker = SimpleNamespace(
        vllm_config=_no_sp_vllm_config(pp_size),
        model_runner=runner,
        profiler=None,
    )
    worker.annotate_profile = lambda sched: WorkerFL.annotate_profile(worker, sched)
    return worker


class _RecordingRunner:
    """Model runner recording execute_model and _dummy_run calls."""

    def __init__(self, result=None):
        self.exec_calls = []
        self.dummy_runs = []
        self.result = result

    def execute_model(self, scheduler_output, intermediate_tensors):
        self.exec_calls.append((scheduler_output, intermediate_tensors))
        return self.result

    def _dummy_run(self, num_tokens, **kwargs):
        self.dummy_runs.append((num_tokens, kwargs))
        return ("hidden", "last_hidden")


def test_execute_model_without_forward_skips_model_runner():
    """Zero scheduled tokens short-circuit straight through the runner."""
    runner = _RecordingRunner()
    worker = _forward_worker(runner)
    sched = SimpleNamespace(total_num_scheduled_tokens=0)

    result = WorkerFL.execute_model(worker, sched)

    assert result is None
    # The runner was invoked with the scheduler output and no intermediates.
    assert runner.exec_calls == [(sched, None)]


def test_execute_model_returns_model_runner_output_on_last_rank():
    """A ModelRunnerOutput from the runner is returned unchanged."""
    mro = ModelRunnerOutput(
        req_ids=["r1"], req_id_to_index={"r1": 0}, sampled_token_ids=[[1, 2]]
    )
    runner = _RecordingRunner(result=mro)
    worker = _forward_worker(runner)
    sched = SimpleNamespace(total_num_scheduled_tokens=5)

    with _Patch(
        worker_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    ):
        result = WorkerFL.execute_model(worker, sched)

    assert result is mro
    assert runner.exec_calls == [(sched, None)]


def test_execute_model_middle_rank_sends_intermediate_tensors():
    """A middle pipeline rank receives, runs, then sends its tensors."""
    inter = IntermediateTensors({"a": torch.zeros(2)})
    runner = _RecordingRunner(result=inter)
    worker = _forward_worker(runner, pp_size=2)
    sched = SimpleNamespace(total_num_scheduled_tokens=5)
    sent = []
    received = {}

    class _FakePP:
        is_first_rank = False
        is_last_rank = False

        def recv_tensor_dict(self, all_gather_group=None, all_gather_tensors=None):
            received["recv"] = (all_gather_group is tp_group_placeholder,
                                dict(all_gather_tensors))
            return {"hidden": torch.zeros(2)}

        def send_tensor_dict(self, tensors, all_gather_group=None,
                             all_gather_tensors=None):
            sent.append((tensors, all_gather_tensors))

    tp_group_placeholder = object()

    class _FakeTP:
        pass

    with _Patch(worker_module, "get_pp_group", lambda: _FakePP()), _Patch(
        worker_module, "get_tp_group", lambda: tp_group_placeholder
    ):
        result = WorkerFL.execute_model(worker, sched)

    assert result is None
    # The recv used the tp group and the (empty) sp gather map.
    assert received["recv"] == (True, {})
    assert len(sent) == 1
    # The runner received the rewrapped IntermediateTensors.
    assert isinstance(runner.exec_calls[0][1], IntermediateTensors)


def test_execute_dummy_batch_runs_uniform_decode():
    """execute_dummy_batch runs the uniform-decode dummy path."""
    runner = _RecordingRunner()
    runner.uniform_decode_query_len = 3
    WorkerFL.execute_dummy_batch(SimpleNamespace(model_runner=runner))
    assert runner.dummy_runs == [(3, {"uniform_decode": True})]


def test_passthroughs_delegate_to_model_runner():
    """LoRA, model, task, cache, config, and stats calls all forward."""
    calls = {}

    def record(name):
        def _record(*args, **kwargs):
            calls[name] = (args, kwargs)
            return f"ret-{name}"

        return _record

    runner = SimpleNamespace(
        add_lora=record("add_lora"),
        remove_lora=record("remove_lora"),
        list_loras=lambda: {1, 2},
        pin_lora=record("pin_lora"),
        take_draft_token_ids=lambda: "draft-ids",
        get_model=lambda: "MODEL",
        get_supported_tasks=lambda: ("generate",),
        reset_mm_cache=lambda: calls.update(mm=True),
        reset_encoder_cache=lambda: calls.update(enc=True),
        get_encoder_timing_stats=lambda: {"enc": 1.0},
        update_config=lambda o: calls.update(upd=o),
        reload_weights=lambda: calls.update(rw=True),
        save_tensorized_model=lambda **k: calls.update(tz=k),
        initialize_kv_cache=lambda cfg: calls.update(ikv=cfg),
        sample_tokens=lambda g: ("sampled", g),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=SimpleNamespace(kv_transfer_config=None),
    )

    assert WorkerFL.add_lora(worker, "req") == "ret-add_lora"
    assert WorkerFL.remove_lora(worker, 1) == "ret-remove_lora"
    assert WorkerFL.list_loras(worker) == {1, 2}
    assert WorkerFL.pin_lora(worker, 1) == "ret-pin_lora"
    assert WorkerFL.take_draft_token_ids(worker) == "draft-ids"
    assert WorkerFL.get_model(worker) == "MODEL"
    assert WorkerFL.get_supported_tasks(worker) == ("generate",)
    assert WorkerFL.get_encoder_timing_stats(worker) == {"enc": 1.0}

    WorkerFL.reset_mm_cache(worker)
    WorkerFL.reset_encoder_cache(worker)
    assert calls["mm"] is True and calls["enc"] is True

    WorkerFL.update_config(worker, {"a": 1})
    assert calls["upd"] == {"a": 1}
    WorkerFL.reload_weights(worker)
    assert calls["rw"] is True
    WorkerFL.save_tensorized_model(worker, tensorizer_config="TC")
    assert calls["tz"] == {"tensorizer_config": "TC"}
    WorkerFL.initialize_from_config(worker, "KVCC")
    assert calls["ikv"] == "KVCC"

    assert WorkerFL.sample_tokens(worker, "g") == ("sampled", "g")
    assert WorkerFL.check_health(SimpleNamespace()) is None


def test_annotate_profile_returns_nullcontext_without_profiler():
    """No profiler means the scheduling context is a no-op nullcontext."""
    worker = SimpleNamespace(profiler=None)
    sched = SimpleNamespace(
        scheduled_new_reqs=[1], scheduled_cached_reqs=SimpleNamespace(req_ids=[])
    )
    assert isinstance(WorkerFL.annotate_profile(worker, sched), nullcontext)


def test_annotate_profile_steps_and_labels_new_and_cached_counts():
    """With a profiler the step counter and new/cached label are exact."""
    annotations = []

    class _Profiler:
        def step(self):
            annotations.append("step")

        def annotate_context_manager(self, name):
            annotations.append(name)
            return nullcontext()

    worker = SimpleNamespace(profiler=_Profiler())
    sched = SimpleNamespace(
        scheduled_new_reqs=[1, 2, 3],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "b"]),
    )
    WorkerFL.annotate_profile(worker, sched)
    assert annotations == ["step", "execute_new_3_cached_2"]

