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

"""G10-E coverage: draft-weight, profiling, execution and weight-transfer
methods of WorkerFL (814b).

Targets: supports_draft_weight_updates 1172-1183, _set_draft_weight_update_target
1185-1203, get_supported_tasks 1205-1206, get_encoder_timing_stats 1213-1215,
annotate_profile 1217-1328, sample_tokens 1330-1335, execute_model 1337-1398,
take_draft_token_ids 1400-1401, profile 1403-1465, execute_dummy_batch
1467-1469, lora/health/save 1471-1507, weight-transfer 1509-1620,
shutdown 1622-1648, elastic_ep_execute 1650-1651.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.worker import worker as wmod


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
# draft weight updates
# --------------------------------------------------------------------------
def test_supports_draft_weight_updates_false_without_engine():
    worker = SimpleNamespace(
        weight_transfer_engine=None,
        speculative_config=None,
        model_runner=SimpleNamespace(),
    )
    assert wmod.WorkerFL.supports_draft_weight_updates(worker) is False


def test_supports_draft_weight_updates_false_without_draft_model():
    worker = SimpleNamespace(
        weight_transfer_engine=SimpleNamespace(supports_draft_weight_update=True),
        speculative_config=SimpleNamespace(draft_model_config="dm"),
        model_runner=SimpleNamespace(get_draft_model=lambda: None),
    )
    assert wmod.WorkerFL.supports_draft_weight_updates(worker) is False


def test_supports_draft_weight_updates_true():
    worker = SimpleNamespace(
        weight_transfer_engine=SimpleNamespace(supports_draft_weight_update=True),
        speculative_config=SimpleNamespace(draft_model_config="dm"),
        model_runner=SimpleNamespace(get_draft_model=lambda: "draft"),
    )
    assert wmod.WorkerFL.supports_draft_weight_updates(worker) is True


def test_set_draft_weight_update_target_errors():
    with pytest.raises(RuntimeError, match="no draft model"):
        wmod.WorkerFL._set_draft_weight_update_target(
            SimpleNamespace(
                weight_transfer_engine="e", get_draft_model=lambda: None
            )
        )
    with pytest.raises(RuntimeError, match="no draft model config"):
        wmod.WorkerFL._set_draft_weight_update_target(
            SimpleNamespace(
                weight_transfer_engine="e",
                get_draft_model=lambda: "draft",
                speculative_config=None,
            )
        )


def test_set_draft_weight_update_target_sets_engine_target():
    calls = []
    worker = SimpleNamespace(
        weight_transfer_engine=SimpleNamespace(
            set_weight_update_target=lambda m, cfg: calls.append((m, cfg))
        ),
        get_draft_model=lambda: "draft",
        speculative_config=SimpleNamespace(draft_model_config="dm"),
    )
    wmod.WorkerFL._set_draft_weight_update_target(worker)
    assert calls == [("draft", "dm")]


# --------------------------------------------------------------------------
# simple delegates
# --------------------------------------------------------------------------
def test_get_supported_tasks_and_timing_stats_delegate():
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            get_supported_tasks=lambda: ("generate",),
            get_encoder_timing_stats=lambda: {"enc": {}},
        )
    )
    assert wmod.WorkerFL.get_supported_tasks(worker) == ("generate",)
    assert wmod.WorkerFL.get_encoder_timing_stats(worker) == {"enc": {}}


# --------------------------------------------------------------------------
# annotate_profile
# --------------------------------------------------------------------------
def _iter_details():
    return SimpleNamespace(
        num_ctx_requests=1,
        num_ctx_tokens=2,
        num_generation_requests=3,
        num_generation_tokens=4,
    )


def test_annotate_profile_without_profiler_returns_nullcontext():
    worker = SimpleNamespace(profiler=None)
    out = wmod.WorkerFL.annotate_profile(worker, SimpleNamespace())
    assert isinstance(out, contextlib.nullcontext)


def test_annotate_profile_not_running_returns_nullcontext():
    worker = SimpleNamespace(
        profiler=SimpleNamespace(step=lambda: None, is_running=False)
    )
    with _stack(
        (wmod, "compute_iteration_details", lambda so: _iter_details())
    ):
        out = wmod.WorkerFL.annotate_profile(worker, SimpleNamespace())
    assert isinstance(out, contextlib.nullcontext)


def test_annotate_profile_simple_annotation():
    captured = []
    worker = SimpleNamespace(
        profiler=SimpleNamespace(
            step=lambda: None,
            is_running=True,
            annotate_context_manager=lambda ann: captured.append(ann) or "cm",
        ),
        vllm_config=SimpleNamespace(
            profiler_config=SimpleNamespace(detailed_trace_annotation=False)
        ),
    )
    with _stack(
        (wmod, "compute_iteration_details", lambda so: _iter_details())
    ):
        out = wmod.WorkerFL.annotate_profile(worker, SimpleNamespace())
    assert out == "cm"
    assert captured[0] == "execute_context_1(2)_generation_3(4)"


def test_annotate_profile_detailed_annotation_builds_roofline():
    captured = []

    class _CachedReqs:
        req_ids = ["r0", "r1"]
        num_computed_tokens = [5, 6]

        def is_context_phase(self, req_id):
            return req_id == "r0"

    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="r0", num_computed_tokens=5)
        ],
        scheduled_cached_reqs=_CachedReqs(),
        num_scheduled_tokens={"r0": 2, "r1": 1},
    )
    worker = SimpleNamespace(
        profiler=SimpleNamespace(
            step=lambda: None,
            is_running=True,
            annotate_context_manager=lambda ann: captured.append(ann) or "cm",
        ),
        vllm_config=SimpleNamespace(
            profiler_config=SimpleNamespace(detailed_trace_annotation=True)
        ),
    )
    with _stack(
        (wmod, "compute_iteration_details", lambda so: _iter_details())
    ):
        out = wmod.WorkerFL.annotate_profile(worker, scheduler_output)
    assert out == "cm"
    assert captured[0].startswith("execute_3_context_1")


# --------------------------------------------------------------------------
# sample_tokens / execute_model / take_draft_token_ids
# --------------------------------------------------------------------------
def test_sample_tokens_delegates():
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(sample_tokens=lambda g: ("out", g))
    )
    assert wmod.WorkerFL.sample_tokens(worker, "grammar") == ("out", "grammar")


def test_execute_model_returns_runner_output():
    calls = []
    worker = SimpleNamespace(
        _pp_send_work=[],
        annotate_profile=lambda so: _Ctx(),
        model_runner=SimpleNamespace(
            execute_model=lambda so, it: calls.append((so, it)) or None
        ),
        use_v2_model_runner=False,
    )
    sched = SimpleNamespace(total_num_scheduled_tokens=0)
    with _stack((wmod, "get_pp_group", lambda: SimpleNamespace())):
        out = wmod.WorkerFL.execute_model(worker, sched)
    assert out is None
    assert calls == [(sched, None)]


def test_execute_model_v2_pooling_runs_pool():
    pooled = []
    worker = SimpleNamespace(
        _pp_send_work=[],
        annotate_profile=lambda so: _Ctx(),
        model_runner=SimpleNamespace(
            execute_model=lambda so, it: None,
            is_pooling_model=True,
            pool=lambda: pooled.append(1),
        ),
        use_v2_model_runner=True,
    )
    sched = SimpleNamespace(total_num_scheduled_tokens=0)
    with _stack((wmod, "get_pp_group", lambda: SimpleNamespace())):
        out = wmod.WorkerFL.execute_model(worker, sched)
    assert out is None
    assert pooled == [1]


def test_execute_model_pipeline_sends_intermediate_tensors():
    post = []
    received = []
    pp = SimpleNamespace(
        is_first_rank=False,
        is_last_rank=False,
        irecv_tensor_dict=lambda **kw: (
            {"h": torch.zeros(2)},
            None,
            [lambda: post.append(1)],
        ),
        isend_tensor_dict=lambda *a, **k: received.append((a, k)) or ["send"],
    )
    worker = SimpleNamespace(
        _pp_send_work=[],
        annotate_profile=lambda so: _Ctx(),
        model_runner=SimpleNamespace(
            execute_model=lambda so, it: wmod.IntermediateTensors(
                {"h": torch.zeros(2)}
            )
        ),
        use_v2_model_runner=False,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                distributed_executor_backend="mp"
            )
        ),
    )
    sched = SimpleNamespace(total_num_scheduled_tokens=2)
    with _stack(
        (wmod, "get_pp_group", lambda: pp),
        (wmod, "get_tp_group", lambda: "tp"),
    ):
        out = wmod.WorkerFL.execute_model(worker, sched)
    assert out is None
    assert worker._pp_send_work == ["send"]
    assert received


def test_execute_model_waits_for_previous_pp_sends():
    class _H:
        def __init__(self):
            self.waits = 0

        def wait(self):
            self.waits += 1

    handle = _H()
    worker = SimpleNamespace(
        _pp_send_work=[handle],
        annotate_profile=lambda so: _Ctx(),
        model_runner=SimpleNamespace(execute_model=lambda so, it: None),
        use_v2_model_runner=False,
    )
    sched = SimpleNamespace(total_num_scheduled_tokens=0)
    with _stack((wmod, "get_pp_group", lambda: SimpleNamespace())):
        wmod.WorkerFL.execute_model(worker, sched)
    assert handle.waits == 1
    assert worker._pp_send_work == []


def test_take_draft_token_ids_delegates():
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(take_draft_token_ids=lambda: "draft")
    )
    assert wmod.WorkerFL.take_draft_token_ids(worker) == "draft"


# --------------------------------------------------------------------------
# profile / execute_dummy_batch
# --------------------------------------------------------------------------
def test_profile_requires_config():
    worker = SimpleNamespace(profiler_config=None)
    with pytest.raises(RuntimeError, match="Profiling is not enabled"):
        wmod.WorkerFL.profile(worker)


def test_profile_stop_without_start_warns():
    warnings = []
    worker = SimpleNamespace(
        profiler_config=SimpleNamespace(profiler="torch"), profiler=None
    )
    with _stack(
        (wmod, "logger", SimpleNamespace(warning=lambda *a: warnings.append(a)))
    ):
        wmod.WorkerFL.profile(worker, is_start=False)
    assert warnings


def test_profile_stop_runs_profiler():
    calls = []
    worker = SimpleNamespace(
        profiler_config=SimpleNamespace(profiler="torch"),
        profiler=SimpleNamespace(stop=lambda: calls.append("stop")),
    )
    wmod.WorkerFL.profile(worker, is_start=False)
    assert calls == ["stop"]


def test_profile_stop_proton_clears_profiler():
    worker = SimpleNamespace(
        profiler_config=SimpleNamespace(profiler="proton"),
        profiler=SimpleNamespace(stop=lambda: None),
    )
    wmod.WorkerFL.profile(worker, is_start=False)
    assert worker.profiler is None


def test_execute_dummy_batch_uses_uniform_decode_len():
    calls = []
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            uniform_decode_query_len=4,
            _dummy_run=lambda *a, **k: calls.append((a, k)),
        )
    )
    wmod.WorkerFL.execute_dummy_batch(worker)
    assert calls[0][0] == (4,)
    assert calls[0][1]["uniform_decode"] is True


# --------------------------------------------------------------------------
# lora / health / save
# --------------------------------------------------------------------------
def test_lora_delegates():
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            add_lora=lambda r: ("add", r),
            remove_lora=lambda i: ("remove", i),
            list_loras=lambda: {1},
            pin_lora=lambda i: ("pin", i),
        )
    )
    assert wmod.WorkerFL.add_lora(worker, "req") == ("add", "req")
    assert wmod.WorkerFL.remove_lora(worker, 1) == ("remove", 1)
    assert wmod.WorkerFL.list_loras(worker) == {1}
    assert wmod.WorkerFL.pin_lora(worker, 1) == ("pin", 1)


def test_check_health_returns_none():
    assert wmod.WorkerFL.check_health(SimpleNamespace()) is None


def test_save_tensorized_model_delegates():
    calls = []
    worker = SimpleNamespace(
        get_model=lambda: "model",
        model_config="mc",
    )
    with _stack(
        (
            wmod,
            "TensorizerLoader",
            SimpleNamespace(
                save_model=lambda *a, **k: calls.append((a, k))
            ),
        )
    ):
        wmod.WorkerFL.save_tensorized_model(worker, "tcfg")
    assert calls[0][1]["tensorizer_config"] == "tcfg"
    assert calls[0][1]["model_config"] == "mc"


# --------------------------------------------------------------------------
# weight transfer lifecycle
# --------------------------------------------------------------------------
def test_check_weight_transfer_engine_raises_without_engine():
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        wmod.WorkerFL._check_weight_transfer_engine(
            SimpleNamespace(weight_transfer_engine=None)
        )


def test_init_weight_transfer_engine_parses_and_inits():
    calls = []
    engine = SimpleNamespace(
        parse_init_info=lambda info: ("typed", info),
        init_transfer_engine=lambda typed: calls.append(typed),
    )
    worker = SimpleNamespace(
        weight_transfer_engine=engine,
        _check_weight_transfer_engine=lambda: None,
    )
    wmod.WorkerFL.init_weight_transfer_engine(worker, {"x": 1})
    assert calls == [("typed", {"x": 1})]


def test_start_weight_update_wrappers_delegate():
    calls = []
    worker = SimpleNamespace(
        vllm_config="cfg",
        _start_weight_update=lambda is_draft=False: calls.append(is_draft),
    )
    with _stack((wmod, "set_current_vllm_config", lambda cfg: _Ctx())):
        wmod.WorkerFL.start_weight_update(worker)
        wmod.WorkerFL.start_draft_weight_update(worker)
    assert calls == [False, True]


def test_start_weight_update_rejects_unsupported_draft():
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=SimpleNamespace(
            supports_draft_weight_update=False
        ),
        _weight_update_active=False,
    )
    with pytest.raises(RuntimeError, match="does not support"):
        wmod.WorkerFL._start_weight_update(worker, is_draft=True)


def test_start_weight_update_rejects_active_session():
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=SimpleNamespace(),
        _weight_update_active=True,
    )
    with pytest.raises(RuntimeError, match="already active"):
        wmod.WorkerFL._start_weight_update(worker)


def test_start_weight_update_success_and_draft_target():
    calls = []

    def _engine(draft_support=True):
        return SimpleNamespace(
            supports_draft_weight_update=draft_support,
            start_weight_update=lambda: calls.append("start"),
            reset_weight_update_target=lambda: calls.append("reset"),
        )

    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=_engine(),
        _weight_update_active=False,
    )
    wmod.WorkerFL._start_weight_update(worker)
    assert calls == ["start"]
    assert worker._weight_update_active is True
    assert worker._weight_update_is_draft is False

    calls.clear()
    worker2 = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=_engine(),
        _weight_update_active=False,
        _set_draft_weight_update_target=lambda: calls.append("target"),
    )
    wmod.WorkerFL._start_weight_update(worker2, is_draft=True)
    assert calls == ["target", "start"]
    assert worker2._weight_update_is_draft is True


def test_start_weight_update_resets_on_failure():
    calls = []

    def _boom():
        raise RuntimeError("fail")

    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=SimpleNamespace(
            supports_draft_weight_update=True,
            start_weight_update=_boom,
            reset_weight_update_target=lambda: calls.append("reset"),
        ),
        _weight_update_active=False,
    )
    with pytest.raises(RuntimeError, match="fail"):
        wmod.WorkerFL._start_weight_update(worker)
    assert calls == ["reset"]
    assert worker._weight_update_active is False


def test_update_weights_requires_active_session():
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=SimpleNamespace(),
        _weight_update_active=False,
        vllm_config="cfg",
    )
    with pytest.raises(RuntimeError, match="start_weight_update"):
        wmod.WorkerFL.update_weights(worker, {})


def test_update_weights_success_and_failure():
    calls = []
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=SimpleNamespace(
            update_weights=lambda info: calls.append(("update", info)),
            reset_weight_update_target=lambda: calls.append("reset"),
        ),
        _weight_update_active=True,
        vllm_config="cfg",
    )
    with _stack((wmod, "set_current_vllm_config", lambda cfg: _Ctx())):
        wmod.WorkerFL.update_weights(worker, {"w": 1})
    assert calls == [("update", {"w": 1})]

    class _Boom:
        def update_weights(self, info):
            raise RuntimeError("boom")

        def reset_weight_update_target(self):
            calls.append("reset-fail")

    worker2 = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=_Boom(),
        _weight_update_active=True,
        vllm_config="cfg",
    )
    with _stack((wmod, "set_current_vllm_config", lambda cfg: _Ctx())):
        with pytest.raises(RuntimeError, match="boom"):
            wmod.WorkerFL.update_weights(worker2, {})
    assert worker2._weight_update_active is False
    assert calls[-1] == "reset-fail"


def test_finish_weight_update_requires_active_session():
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=SimpleNamespace(),
        _weight_update_active=False,
        vllm_config="cfg",
    )
    with pytest.raises(RuntimeError, match="without a matching"):
        wmod.WorkerFL.finish_weight_update(worker)


def test_finish_weight_update_resets_lora_for_weights():
    calls = []
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=SimpleNamespace(
            finish_weight_update=lambda: calls.append("finish"),
            reset_weight_update_target=lambda: calls.append("reset"),
        ),
        _weight_update_active=True,
        _weight_update_is_draft=False,
        model_runner=SimpleNamespace(
            reset_lora_state=lambda: calls.append("lora")
        ),
        vllm_config="cfg",
    )
    with _stack((wmod, "set_current_vllm_config", lambda cfg: _Ctx())):
        wmod.WorkerFL.finish_weight_update(worker)
    assert calls == ["finish", "reset", "lora"]
    assert worker._weight_update_active is False


def test_finish_weight_update_draft_skips_lora_reset():
    calls = []
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=SimpleNamespace(
            finish_weight_update=lambda: calls.append("finish"),
            reset_weight_update_target=lambda: calls.append("reset"),
        ),
        _weight_update_active=True,
        _weight_update_is_draft=True,
        model_runner=SimpleNamespace(
            reset_lora_state=lambda: pytest.fail("draft must not reset lora")
        ),
        vllm_config="cfg",
    )
    with _stack((wmod, "set_current_vllm_config", lambda cfg: _Ctx())):
        wmod.WorkerFL.finish_weight_update(worker)
    assert calls == ["finish", "reset"]


# --------------------------------------------------------------------------
# shutdown / elastic
# --------------------------------------------------------------------------
def test_shutdown_releases_resources():
    calls = []
    worker = SimpleNamespace(
        profiler=SimpleNamespace(shutdown=lambda: calls.append("profiler")),
        weight_transfer_engine=SimpleNamespace(
            shutdown=lambda: calls.append("wt")
        ),
        model_runner=SimpleNamespace(
            shutdown=lambda: calls.append("runner")
        ),
    )
    with _stack(
        (wmod, "gc", SimpleNamespace(unfreeze=lambda: calls.append("gc"))),
        (
            wmod,
            "ensure_kv_transfer_shutdown",
            lambda: calls.append("kv"),
        ),
        (
            wmod,
            "ensure_ec_transfer_shutdown",
            lambda: calls.append("ec"),
        ),
        (
            wmod,
            "current_platform",
            SimpleNamespace(is_cuda_alike=lambda: False),
        ),
    ):
        wmod.WorkerFL.shutdown(worker)
    assert calls == ["gc", "kv", "ec", "profiler", "wt", "runner"]


def test_shutdown_skips_absent_components():
    worker = SimpleNamespace(profiler=None, weight_transfer_engine=None)
    with _stack(
        (wmod, "gc", SimpleNamespace(unfreeze=lambda: None)),
        (wmod, "ensure_kv_transfer_shutdown", None),
        (wmod, "ensure_ec_transfer_shutdown", None),
        (
            wmod,
            "current_platform",
            SimpleNamespace(is_cuda_alike=lambda: False),
        ),
    ):
        wmod.WorkerFL.shutdown(worker)


def test_elastic_ep_execute_delegates():
    worker = SimpleNamespace(
        elastic_ep_executor=SimpleNamespace(
            execute=lambda method, *a, **k: (method, a, k)
        )
    )
    assert wmod.WorkerFL.elastic_ep_execute(worker, "run", 1, x=2) == (
        "run",
        (1,),
        {"x": 2},
    )
