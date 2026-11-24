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

"""G10-A coverage: small helper / accessor methods of ModelRunnerFL (814b).

Targets: get_model / get_draft_model / get_supported_tasks /
sync_and_gather_intermediate_tensors / eplb_step / setup_eplb_from_mapping /
_prepare_mm_inputs / _sample / synchronize_input_prep / _model_forward /
_register_layerwise_nvtx_hooks / _get_encoder_output_from_cache /
_make_buffer / jit_warmup_registry / _get_mamba_bufs / _init_kv_zero_meta /
_pp_broadcast_prev_sampled_token_ids / take_draft_token_ids /
_copy_draft_token_ids_to_cpu / _get_valid_sampled_token_count.

The runner is a SimpleNamespace; module level collaborators are patched at the
``vllm_fl.worker.model_runner`` boundary. Pure CPU.
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


def _ctx(value=None):
    class _C:
        def __enter__(self):
            return value

        def __exit__(self, *exc):
            return False

    return _C()


# --------------------------------------------------------------------------
# get_model / get_draft_model
# --------------------------------------------------------------------------
def test_get_model_raises_before_init():
    with pytest.raises(ValueError):
        mr.ModelRunnerFL.get_model(SimpleNamespace())


def test_get_model_returns_plain_module():
    model = object()
    assert mr.ModelRunnerFL.get_model(SimpleNamespace(model=model)) is model


def test_get_model_unwraps_cudagraph_wrapper():
    class _Wrap:
        def __init__(self, raw):
            self._raw = raw

        def unwrap(self):
            return self._raw

    raw = object()
    with _stack(
        (mr, "CUDAGraphWrapper", _Wrap),
        (mr, "GraphWrapper", _Wrap),
        (mr, "UBatchWrapper", _Wrap),
        (mr, "BreakableCUDAGraphWrapper", _Wrap),
    ):
        assert mr.ModelRunnerFL.get_model(SimpleNamespace(model=_Wrap(raw))) is raw


def test_get_draft_model_none_without_drafter():
    assert mr.ModelRunnerFL.get_draft_model(SimpleNamespace()) is None


def test_get_draft_model_none_when_drafter_has_no_model():
    assert (
        mr.ModelRunnerFL.get_draft_model(
            SimpleNamespace(drafter=SimpleNamespace(model=None))
        )
        is None
    )


def test_get_draft_model_unwraps_wrapper():
    class _Wrap:
        def __init__(self, raw):
            self._raw = raw

        def unwrap(self):
            return self._raw

    raw = object()
    with _stack(
        (mr, "CUDAGraphWrapper", _Wrap),
        (mr, "GraphWrapper", _Wrap),
        (mr, "UBatchWrapper", _Wrap),
        (mr, "BreakableCUDAGraphWrapper", _Wrap),
    ):
        runner = SimpleNamespace(drafter=SimpleNamespace(model=_Wrap(raw)))
        assert mr.ModelRunnerFL.get_draft_model(runner) is raw


# --------------------------------------------------------------------------
# get_supported_tasks / get_supported_pooling_tasks
# --------------------------------------------------------------------------
def test_get_supported_tasks_generate_runner():
    runner = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate"),
        get_supported_generation_tasks=lambda: ["generate", "realtime"],
        get_supported_pooling_tasks=lambda: pytest.fail("pooling must not run"),
    )
    assert mr.ModelRunnerFL.get_supported_tasks(runner) == (
        "generate",
        "realtime",
    )


def test_get_supported_tasks_pooling_runner():
    runner = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="pooling"),
        get_supported_generation_tasks=lambda: pytest.fail("gen must not run"),
        get_supported_pooling_tasks=lambda: ["embed"],
    )
    assert mr.ModelRunnerFL.get_supported_tasks(runner) == ("embed",)


def test_get_supported_pooling_tasks_not_a_pooling_model():
    runner = SimpleNamespace(get_model=lambda: object())
    with _stack((mr, "is_pooling_model", lambda m: False)):
        assert mr.ModelRunnerFL.get_supported_pooling_tasks(runner) == []


def test_get_supported_pooling_tasks_delegates_to_pooler():
    model = SimpleNamespace(
        pooler=SimpleNamespace(get_supported_tasks=lambda: ("embed",))
    )
    runner = SimpleNamespace(get_model=lambda: model)
    with _stack((mr, "is_pooling_model", lambda m: True)):
        assert mr.ModelRunnerFL.get_supported_pooling_tasks(runner) == ["embed"]


# --------------------------------------------------------------------------
# sync_and_gather_intermediate_tensors / eplb
# --------------------------------------------------------------------------
def test_sync_and_gather_copies_and_slices():
    runner = SimpleNamespace(intermediate_tensors={"h": torch.zeros(4, 2)})
    incoming = {"h": torch.ones(4, 2)}
    out = mr.ModelRunnerFL.sync_and_gather_intermediate_tensors(
        runner, 2, incoming, True
    )
    assert runner.intermediate_tensors["h"][:2].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert out.tensors["h"].shape[0] == 2


def test_sync_and_gather_without_sync_self():
    runner = SimpleNamespace(intermediate_tensors={"h": torch.arange(8).reshape(4, 2)})
    out = mr.ModelRunnerFL.sync_and_gather_intermediate_tensors(
        runner, 3, None, False
    )
    assert out.tensors["h"].shape[0] == 3


def test_eplb_step_early_return_disabled():
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=False, eplb_config=None),
        eep_eplb_suppressed=False,
    )
    assert mr.ModelRunnerFL.eplb_step(runner) is None


def test_eplb_step_early_return_suppressed():
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=True, eplb_config=None),
        eep_eplb_suppressed=True,
    )
    assert mr.ModelRunnerFL.eplb_step(runner) is None


def test_eplb_step_runs_state():
    calls = []
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_eplb=True,
            eplb_config=SimpleNamespace(log_balancedness=True),
        ),
        eep_eplb_suppressed=False,
        eplb_state=SimpleNamespace(step=lambda *a, **k: calls.append((a, k))),
        _moe_model=object(),
    )
    mr.ModelRunnerFL.eplb_step(runner, True, False)
    assert calls == [((True, False), {"log_stats": True})]


def test_setup_eplb_from_mapping_builds_state():
    calls = []

    def _from_mapping(**kw):
        calls.append(kw)
        return "state"

    runner = SimpleNamespace(
        _moe_model="moe",
        model_config="mc",
        device="cpu",
        parallel_config="pc",
    )
    with _stack(
        (mr, "EplbState", SimpleNamespace(from_mapping=_from_mapping))
    ):
        mr.ModelRunnerFL.setup_eplb_from_mapping(runner, "expand", 7)
    assert runner.eplb_state == "state"
    assert calls[0]["num_valid_physical_experts"] == 7
    assert calls[0]["expanded_physical_to_logical"] == "expand"


# --------------------------------------------------------------------------
# _prepare_mm_inputs / _model_forward
# --------------------------------------------------------------------------
def test_prepare_mm_inputs_raw_tokens_branch():
    runner = SimpleNamespace(
        model=SimpleNamespace(requires_raw_input_tokens=True),
        input_ids=SimpleNamespace(gpu=torch.arange(5)),
        inputs_embeds=SimpleNamespace(gpu=torch.zeros(5, 3)),
    )
    ids, embeds = mr.ModelRunnerFL._prepare_mm_inputs(runner, 3)
    assert ids.tolist() == [0, 1, 2]
    assert embeds.shape == (3, 3)


def test_prepare_mm_inputs_embeds_only_branch():
    runner = SimpleNamespace(
        model=SimpleNamespace(requires_raw_input_tokens=False),
        input_ids=SimpleNamespace(gpu=torch.arange(5)),
        inputs_embeds=SimpleNamespace(gpu=torch.zeros(5, 3)),
    )
    ids, embeds = mr.ModelRunnerFL._prepare_mm_inputs(runner, 2)
    assert ids is None
    assert embeds.shape == (2, 3)


def test_model_forward_calls_model():
    calls = []
    runner = SimpleNamespace(model=lambda **kw: calls.append(kw) or "out")
    result = mr.ModelRunnerFL._model_forward(
        runner, input_ids="ids", positions="pos"
    )
    assert result == "out"
    assert calls[0]["input_ids"] == "ids"
    assert calls[0]["positions"] == "pos"


# --------------------------------------------------------------------------
# _sample
# --------------------------------------------------------------------------
def _sample_input_batch(**extra):
    base = dict(
        sampling_metadata=SimpleNamespace(generators={}),
        update_async_output_token_ids=lambda: None,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def test_sample_without_spec_decode_calls_sampler():
    calls = []
    runner = SimpleNamespace(
        input_batch=_sample_input_batch(),
        sampler=lambda **kw: calls.append(kw) or "sampled",
    )
    out = mr.ModelRunnerFL._sample(runner, "logits", None)
    assert out == "sampled"
    assert calls[0]["logits"] == "logits"


def test_sample_records_cpu_generator_state():
    gen = SimpleNamespace(
        device=SimpleNamespace(type="cpu"), get_state=lambda: "cpu-state"
    )
    runner = SimpleNamespace(
        input_batch=_sample_input_batch(
            sampling_metadata=SimpleNamespace(generators={3: gen})
        ),
        sampler=lambda **kw: "sampled",
    )
    mr.ModelRunnerFL._sample(runner, "logits", None)
    assert runner._cpu_generator_states == {3: "cpu-state"}


def test_sample_with_spec_decode_calls_rejection_sampler():
    calls = []
    runner = SimpleNamespace(
        input_batch=_sample_input_batch(),
        use_async_scheduling=False,
        _get_spec_decode_draft_probs=lambda meta: ("probs",),
        rejection_sampler=lambda *a: calls.append(a) or "rejected",
    )
    out = mr.ModelRunnerFL._sample(runner, "logits", "meta")
    assert out == "rejected"
    assert calls[0][0] == "meta"


def test_sample_async_updates_spec_token_ids():
    calls = []
    runner = SimpleNamespace(
        input_batch=_sample_input_batch(
            update_async_spec_token_ids=lambda ids: calls.append(("spec", ids))
        ),
        use_async_scheduling=True,
        _draft_token_req_ids=["r0"],
        _get_draft_token_ids_cpu=lambda: ([[1]], ["r0"]),
        _get_spec_decode_draft_probs=lambda meta: "probs",
        rejection_sampler=lambda *a: "rejected",
    )
    assert mr.ModelRunnerFL._sample(runner, "logits", "meta") == "rejected"
    assert calls == [("spec", [[1]])]


# --------------------------------------------------------------------------
# synchronize_input_prep
# --------------------------------------------------------------------------
def test_synchronize_input_prep_without_event_is_passthrough():
    runner = SimpleNamespace(prepare_inputs_event=None)
    with mr.ModelRunnerFL.synchronize_input_prep(runner):
        pass


def test_synchronize_input_prep_synchronizes_and_records():
    calls = []
    runner = SimpleNamespace(
        prepare_inputs_event=SimpleNamespace(
            synchronize=lambda: calls.append("sync"),
            record=lambda: calls.append("record"),
        )
    )
    with mr.ModelRunnerFL.synchronize_input_prep(runner):
        calls.append("body")
    assert calls == ["sync", "body", "record"]


# --------------------------------------------------------------------------
# _register_layerwise_nvtx_hooks
# --------------------------------------------------------------------------
def _nvtx_runner(enabled, *, tracing_registered=False, mode=None):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(
                enable_layerwise_nvtx_tracing=enabled
            ),
            compilation_config=SimpleNamespace(mode=mode),
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE),
        layerwise_nvtx_hooks_registered=tracing_registered,
        model=SimpleNamespace(__class__=type("M", (), {})),
    )


def test_register_layerwise_nvtx_hooks_disabled_is_noop():
    calls = []
    runner = _nvtx_runner(False)
    with _stack(
        (mr, "logger", SimpleNamespace(debug_once=lambda *a: calls.append(a)))
    ):
        mr.ModelRunnerFL._register_layerwise_nvtx_hooks(runner)
    assert calls == []
    assert runner.layerwise_nvtx_hooks_registered is False


def test_register_layerwise_nvtx_hooks_stock_compile_skips():
    calls = []
    runner = _nvtx_runner(
        True, mode=mr.CompilationMode.STOCK_TORCH_COMPILE
    )
    with _stack(
        (mr, "logger", SimpleNamespace(debug_once=lambda *a: calls.append(a)))
    ):
        mr.ModelRunnerFL._register_layerwise_nvtx_hooks(runner)
    assert runner.layerwise_nvtx_hooks_registered is False
    assert calls  # debug_once fired


def test_register_layerwise_nvtx_hooks_registers_pyt_hooks():
    registered = []

    class _Pyt:
        def register_hooks(self, model, name):
            registered.append((model, name))

    runner = _nvtx_runner(True, mode="not-stock")
    with _stack((mr, "PytHooks", _Pyt)):
        mr.ModelRunnerFL._register_layerwise_nvtx_hooks(runner)
    assert runner.layerwise_nvtx_hooks_registered is True
    assert len(registered) == 1


def test_register_layerwise_nvtx_hooks_already_registered_is_noop():
    runner = _nvtx_runner(True, tracing_registered=True, mode="not-stock")
    with _stack((mr, "PytHooks", lambda: pytest.fail("must not build"))):
        mr.ModelRunnerFL._register_layerwise_nvtx_hooks(runner)


# --------------------------------------------------------------------------
# _get_encoder_output_from_cache / _make_buffer
# --------------------------------------------------------------------------
def test_get_encoder_output_from_cache_hit_and_miss():
    runner = SimpleNamespace(encoder_cache={"h": "cached"})
    assert mr.ModelRunnerFL._get_encoder_output_from_cache(runner, "h") == "cached"
    assert mr.ModelRunnerFL._get_encoder_output_from_cache(runner, "x") is None


def test_make_buffer_builds_cpu_gpu_buffer():
    calls = []

    def _buf(*size, **kw):
        calls.append((size, kw))
        return "buffer"

    runner = SimpleNamespace(device="cpu")
    with _stack((mr, "CpuGpuBuffer", _buf)):
        out = mr.ModelRunnerFL._make_buffer(runner, 4, dtype=torch.int32)
    assert out == "buffer"
    assert calls[0][0] == (4,)
    assert calls[0][1]["with_numpy"] is True


# --------------------------------------------------------------------------
# jit_warmup_registry property
# --------------------------------------------------------------------------
def test_jit_warmup_registry_reuses_and_sets():
    runner = SimpleNamespace(vllm_config="cfg", _jit_warmup_registry="reg")
    assert mr.ModelRunnerFL.jit_warmup_registry.fget(runner) == "reg"
    new = object()
    mr.ModelRunnerFL.jit_warmup_registry.fset(runner, new)
    assert mr.ModelRunnerFL.jit_warmup_registry.fget(runner) is new


# --------------------------------------------------------------------------
# _get_mamba_bufs
# --------------------------------------------------------------------------
def test_get_mamba_bufs_builds_and_caches():
    calls = []

    def _create(**kw):
        calls.append(kw)
        return "bufs"

    fake = SimpleNamespace(MambaBuffers=SimpleNamespace(create=_create))
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        _mamba_bufs=None,
        max_num_reqs=8,
        kv_cache_config="kvc",
        _get_mamba_state_copy_funcs=lambda: "funcs",
        _make_buffer=lambda *a, **k: None,
        device="cpu",
        speculative_config=None,
        model_config=SimpleNamespace(is_hybrid=False),
    )
    with _stack((mr, "mamba_utils", fake)):
        assert mr.ModelRunnerFL._get_mamba_bufs(runner) == "bufs"
        assert mr.ModelRunnerFL._get_mamba_bufs(runner) == "bufs"
    assert len(calls) == 1
    assert calls[0]["copy_funcs"] == "funcs"
    assert calls[0]["with_postprocess_align"] is False


def test_get_mamba_bufs_requires_align_mode():
    runner = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none"))
    with pytest.raises(AssertionError):
        mr.ModelRunnerFL._get_mamba_bufs(runner)


# --------------------------------------------------------------------------
# _init_kv_zero_meta
# --------------------------------------------------------------------------
def test_init_kv_zero_meta_builds_zeroer():
    calls = []

    def _zeroer(*a, **k):
        calls.append((a, k))
        return "zeroer"

    runner = SimpleNamespace(
        _kv_cache_spec_attn_group_iterator=lambda: ["g"],
        _kernel_block_sizes=[16],
        compilation_config=SimpleNamespace(static_forward_context="ctx"),
        kv_cache_config=SimpleNamespace(num_blocks=100),
        runner_only_attn_layers=["layer"],
        device="cpu",
    )
    with _stack((mr, "KVBlockZeroer", _zeroer)):
        mr.ModelRunnerFL._init_kv_zero_meta(runner)
    assert runner._kv_block_zeroer == "zeroer"
    assert calls[0][1]["num_blocks"] == 100
    assert calls[0][1]["kernel_block_sizes"] == [16]


# --------------------------------------------------------------------------
# _pp_broadcast_prev_sampled_token_ids
# --------------------------------------------------------------------------
def test_pp_broadcast_broadcasts_when_not_chunked():
    calls = []
    pp = SimpleNamespace(is_last_rank=True, rank=0, device_group="grp")
    runner = SimpleNamespace(_is_all_reqs_chunked_prefill=lambda: False)
    with _stack(
        (mr, "get_pp_group", lambda: pp),
        (mr, "torch", SimpleNamespace(
            distributed=SimpleNamespace(
                broadcast=lambda *a, **k: calls.append((a, k))
            )
        )),
    ):
        mr.ModelRunnerFL._pp_broadcast_prev_sampled_token_ids(
            runner, torch.zeros(2, 1, dtype=torch.int32)
        )
    assert len(calls) == 1


def test_pp_broadcast_skips_for_chunked_prefill():
    pp = SimpleNamespace(is_last_rank=True, rank=0, device_group="grp")
    runner = SimpleNamespace(_is_all_reqs_chunked_prefill=lambda: True)
    with _stack(
        (mr, "get_pp_group", lambda: pp),
        (mr, "torch", SimpleNamespace(
            distributed=SimpleNamespace(
                broadcast=lambda *a, **k: pytest.fail("must not broadcast")
            )
        )),
    ):
        mr.ModelRunnerFL._pp_broadcast_prev_sampled_token_ids(
            runner, torch.zeros(2, 1, dtype=torch.int32)
        )


# --------------------------------------------------------------------------
# take_draft_token_ids
# --------------------------------------------------------------------------
def test_take_draft_token_ids_none_without_spec():
    runner = SimpleNamespace(num_spec_tokens=0, _draft_token_req_ids=["r"])
    assert mr.ModelRunnerFL.take_draft_token_ids(runner) is None


def test_take_draft_token_ids_none_without_req_ids():
    runner = SimpleNamespace(num_spec_tokens=2, _draft_token_req_ids=[])
    assert mr.ModelRunnerFL.take_draft_token_ids(runner) is None


def test_take_draft_token_ids_builds_payload():
    runner = SimpleNamespace(
        num_spec_tokens=2,
        _draft_token_req_ids=["r0"],
        _get_draft_token_ids_cpu=lambda: ([[1, 2]], ["r0"]),
    )
    with _stack((mr, "DraftTokenIds", lambda a, b: ("draft", a, b))):
        assert mr.ModelRunnerFL.take_draft_token_ids(runner) == (
            "draft",
            ["r0"],
            [[1, 2]],
        )


# --------------------------------------------------------------------------
# _copy_draft_token_ids_to_cpu / _get_valid_sampled_token_count
# --------------------------------------------------------------------------
class _Stream:
    def __init__(self):
        self.waited = None

    def wait_stream(self, other):
        self.waited = other

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Event:
    def __init__(self):
        self.records = 0

    def record(self):
        self.records += 1


def _copy_runner(**extra):
    base = dict(
        _draft_token_ids=torch.arange(6).reshape(2, 3),
        prev_num_spec_tokens=0,
        use_async_scheduling=False,
        input_batch=SimpleNamespace(
            req_ids=["a", "b"],
            sampling_metadata=SimpleNamespace(output_token_ids=[]),
        ),
        draft_token_ids_event=_Event(),
        draft_token_ids_copy_stream=_Stream(),
        draft_token_ids_cpu=torch.zeros(2, 3, dtype=torch.int64),
        _draft_token_req_ids=None,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def _fake_platform():
    return SimpleNamespace(
        torch_device_fn=SimpleNamespace(
            current_stream=lambda: "default",
            stream=lambda s: s,
        )
    )


def test_copy_draft_token_ids_to_cpu_copies_tensor():
    runner = _copy_runner()
    sched = SimpleNamespace(has_structured_output_requests=False)
    with _stack((mr, "current_platform", _fake_platform())):
        mr.ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, sched)
    assert runner.prev_num_spec_tokens == 3
    assert runner._draft_token_req_ids == ["a", "b"]
    assert runner.draft_token_ids_cpu.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert runner.draft_token_ids_event.records == 1


def test_copy_draft_token_ids_to_cpu_zeros_only():
    runner = _copy_runner(draft_token_ids_cpu=torch.ones(2, 3, dtype=torch.int64))
    sched = SimpleNamespace(has_structured_output_requests=False)
    with _stack((mr, "current_platform", _fake_platform())):
        mr.ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, sched, zeros_only=True)
    assert runner.draft_token_ids_cpu.tolist() == [[0, 0, 0], [0, 0, 0]]


def test_copy_draft_token_ids_to_cpu_async_skips_without_need():
    runner = _copy_runner(use_async_scheduling=True)
    sched = SimpleNamespace(has_structured_output_requests=False)
    mr.ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, sched)
    assert runner._draft_token_req_ids is None
    assert runner.draft_token_ids_event.records == 0


def test_copy_draft_token_ids_to_cpu_non_tensor_returns():
    runner = _copy_runner(_draft_token_ids=None)
    sched = SimpleNamespace(has_structured_output_requests=False)
    mr.ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, sched)
    assert runner._draft_token_req_ids == ["a", "b"]


def test_get_valid_sampled_token_count_empty_without_event():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(prev_sampled_token_ids=torch.zeros(2)),
        valid_sampled_token_count_event=None,
    )
    assert mr.ModelRunnerFL._get_valid_sampled_token_count(runner) == []


def test_get_valid_sampled_token_count_reads_cpu_slice():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(prev_sampled_token_ids=torch.zeros(2, 1)),
        valid_sampled_token_count_event=SimpleNamespace(
            synchronize=lambda: None
        ),
        valid_sampled_token_count_cpu=torch.tensor([3, 4, 5]),
    )
    assert mr.ModelRunnerFL._get_valid_sampled_token_count(runner) == [3, 4]
