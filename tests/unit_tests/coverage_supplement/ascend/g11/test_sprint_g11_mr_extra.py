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

"""G11 coverage: remaining small ModelRunnerFL helpers (814b, CPU only).

Targets module helpers, async output wrappers, pooler copy, multimodal dummy
batching, attention-backend / KV-cache wiring and misc lifecycle helpers that
were still uncovered after g1-g10. The runner is a SimpleNamespace; module level
collaborators are patched at the ``vllm_fl.worker.model_runner`` boundary.
Pure CPU, no triton/NPU calls.
"""

from __future__ import annotations

import contextlib
import sys
import types
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


class _Buf:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_count_nans_per_row_counts_each_row():
    logits = torch.tensor([[0.0, float("nan")], [float("nan"), float("nan")]])
    counts = mr.count_nans_per_row(logits)
    assert counts.tolist() == [1, 2]
    assert counts.dtype == torch.int32


def test_nans_to_dict_pads_missing_rows_with_zero():
    assert mr.nans_to_dict([3], {"a": 0, "b": 5}) == {"a": 3, "b": 0}


def test_accelerator_synchronize_musa(monkeypatch):
    calls = []
    fake = types.ModuleType("torch_musa")
    fake.synchronize = lambda: calls.append("musa")
    monkeypatch.setitem(sys.modules, "torch_musa", fake)
    monkeypatch.setattr(
        mr, "current_platform", SimpleNamespace(device_type="musa")
    )
    mr._accelerator_synchronize()
    assert calls == ["musa"]


def test_accelerator_synchronize_default(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mr, "current_platform", SimpleNamespace(device_type="cuda")
    )
    monkeypatch.setattr(
        mr.torch.accelerator, "synchronize", lambda: calls.append("acc")
    )
    mr._accelerator_synchronize()
    assert calls == ["acc"]


def test_get_parameter_for_reload_plain_module():
    plain = SimpleNamespace(get_parameter=lambda name: ("p", name))
    model = SimpleNamespace(get_submodule=lambda name: plain)
    assert mr._get_parameter_for_reload(model, "block.attn.weight") == (
        "p",
        "weight",
    )


def test_get_parameter_for_reload_unwraps_lora_layer():
    class _Base:
        pass

    plain = SimpleNamespace(get_parameter=lambda name: ("p", name))
    layer = _Base()
    layer.base_layer = plain
    model = SimpleNamespace(get_submodule=lambda name: layer)
    with _stack((mr, "BaseLayerWithLoRA", _Base)):
        assert mr._get_parameter_for_reload(model, "layer.qkv") == ("p", "qkv")


def test_get_positions_int_without_mrope():
    runner = SimpleNamespace(
        uses_mrope=False,
        positions=np.arange(10),
        mrope_positions=SimpleNamespace(gpu=np.arange(20).reshape(2, 10)),
    )
    assert mr.ModelRunnerFL._get_positions(runner, 3).tolist() == [0, 1, 2]


def test_get_positions_int_with_mrope():
    runner = SimpleNamespace(
        uses_mrope=True,
        positions=np.arange(10),
        mrope_positions=SimpleNamespace(gpu=np.arange(20).reshape(2, 10)),
    )
    out = mr.ModelRunnerFL._get_positions(runner, 3)
    assert out.tolist() == [[0, 1, 2], [10, 11, 12]]


def test_get_positions_index_array_with_mrope():
    runner = SimpleNamespace(
        uses_mrope=True,
        positions=np.arange(10),
        mrope_positions=SimpleNamespace(gpu=np.arange(20).reshape(2, 10)),
    )
    idx = np.array([1, 3])
    out = mr.ModelRunnerFL._get_positions(runner, idx)
    assert out.tolist() == [[1, 3], [11, 13]]


def test_make_buffer_forwards_arguments():
    runner = SimpleNamespace(device=torch.device("cpu"))
    with _stack((mr, "CpuGpuBuffer", _Buf)):
        buf = mr.ModelRunnerFL._make_buffer(
            runner, 2, 3, dtype=torch.float16, numpy=False
        )
    assert buf.args == (2, 3)
    assert buf.kwargs["dtype"] is torch.float16
    assert buf.kwargs["device"] is runner.device
    assert buf.kwargs["with_numpy"] is False


def test_post_kv_cache_wake_up_delegates():
    calls = []
    runner = SimpleNamespace(init_fp8_kv_scales=lambda: calls.append("init"))
    mr.ModelRunnerFL.post_kv_cache_wake_up(runner)
    assert calls == ["init"]


def test_get_cumsum_and_arange_builds_offsets():
    runner = SimpleNamespace(arange_np=np.arange(20))
    out = np.zeros(10, dtype=np.int32)
    cu = mr.ModelRunnerFL._get_cumsum_and_arange(
        runner, np.array([2, 5, 3]), out
    )
    assert cu.tolist() == [2, 7, 10]
    assert out[:10].tolist() == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]


def test_get_mamba_state_copy_funcs_legacy_tuple():
    runner = SimpleNamespace(
        kv_cache_config="cfg",
        model=SimpleNamespace(get_mamba_state_copy_func=lambda: "fn"),
    )
    fake = SimpleNamespace(
        get_mamba_groups=lambda cfg: [SimpleNamespace(mamba_type="m1")],
        validate_mamba_state_copy_funcs=lambda groups, funcs: None,
    )
    with _stack((mr, "mamba_utils", fake)):
        out = mr.ModelRunnerFL._get_mamba_state_copy_funcs(runner)
    assert out == {"m1": "fn"}
    assert runner._mamba_state_copy_funcs == {"m1": "fn"}


def test_copy_pooler_output_tensor_all_finished():
    raw = torch.tensor([[1, 2], [3, 4]])
    out = mr._copy_pooler_output_to_cpu(raw, [True, True])
    assert [t.tolist() for t in out] == [[1, 2], [3, 4]]


def test_copy_pooler_output_tensor_none_finished():
    raw = torch.tensor([[1, 2], [3, 4]])
    assert mr._copy_pooler_output_to_cpu(raw, [False, False]) == [None, None]


def test_copy_pooler_output_tensor_partial_finished():
    raw = torch.arange(6).reshape(3, 2)
    out = mr._copy_pooler_output_to_cpu(raw, [True, False, True])
    assert out[0].tolist() == [0, 1]
    assert out[1] is None
    assert out[2].tolist() == [4, 5]


def test_copy_pooler_output_tensor_batch_mismatch():
    with pytest.raises(ValueError):
        mr._copy_pooler_output_to_cpu(torch.zeros(2), [True])


def test_copy_pooler_output_list_filters_unfinished_and_none():
    raw = [torch.tensor([1]), None, torch.tensor([3])]
    out = mr._copy_pooler_output_to_cpu(raw, [True, True, False])
    assert out[0].tolist() == [1]
    assert out[1] is None and out[2] is None


def test_copy_pooler_output_list_mismatch():
    with pytest.raises(ValueError):
        mr._copy_pooler_output_to_cpu([torch.zeros(1)], [True, True])


def _async_output(
    sampled_cpu,
    invalid,
    *,
    logprobs=None,
    logprobs_cpu=None,
    num_nans_cpu=None,
    has_fault=None,
    req_id_to_index=None,
):
    out = object.__new__(mr.AsyncGPUModelRunnerOutput)
    out.sampled_token_ids_cpu = sampled_cpu
    out.async_copy_ready_event = SimpleNamespace(synchronize=lambda: None)
    out._invalid_req_indices = invalid
    out._logprobs_tensors = logprobs
    out._sampled_token_ids = "device-ids"
    out._logprobs_tensors_cpu = logprobs_cpu
    out._num_nans = None
    out._num_nans_cpu = num_nans_cpu
    out._has_fault = has_fault
    out.vocab_size = 16
    out._model_runner_output = SimpleNamespace(
        sampled_token_ids=None,
        logprobs=None,
        num_nans_in_logits=None,
        req_id_to_index=req_id_to_index or {},
    )
    return out


def test_async_output_get_output_single_token_clears_invalid():
    out = _async_output(torch.tensor([[1], [2]]), [1])
    res = mr.AsyncGPUModelRunnerOutput.get_output(out)
    assert res.sampled_token_ids == [[1], []]
    assert res.logprobs is None


def test_async_output_get_output_single_token_with_logprobs():
    out = _async_output(
        torch.tensor([[1]]),
        [],
        logprobs_cpu=SimpleNamespace(tolists=lambda: [["lp"]]),
    )
    res = mr.AsyncGPUModelRunnerOutput.get_output(out)
    assert res.logprobs == [["lp"]]


def test_async_output_get_output_multi_token_uses_rejection_sampler():
    sampler = SimpleNamespace(
        parse_output=staticmethod(lambda *a, **k: ([[7]], [[0.5]]))
    )
    out = _async_output(
        torch.tensor([[1, 2]]),
        [0],
        num_nans_cpu=torch.tensor([3]),
        req_id_to_index={"a": 0},
    )
    with _stack(
        (mr, "RejectionSampler", sampler),
        (mr, "envs", SimpleNamespace(VLLM_RAISE_ON_LOGIT_NANS=False)),
    ):
        res = mr.AsyncGPUModelRunnerOutput.get_output(out)
    assert res.sampled_token_ids == [[7]]
    assert res.num_nans_in_logits == {"a": 3}


def test_async_output_get_output_raises_on_nan_when_configured():
    out = _async_output(
        torch.tensor([[1]]), [], num_nans_cpu=torch.tensor([1])
    )

    def _raise(counts):
        raise ValueError("nan")

    with _stack(
        (mr, "envs", SimpleNamespace(VLLM_RAISE_ON_LOGIT_NANS=True)),
        (mr, "raise_if_nan_logits", _raise),
    ):
        with pytest.raises(ValueError, match="nan"):
            mr.AsyncGPUModelRunnerOutput.get_output(out)


def test_async_output_get_output_raises_on_ep_fault():
    mask = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [0, 1]))
    out = _async_output(
        torch.tensor([[1]]),
        [],
        has_fault=SimpleNamespace(item=lambda: True),
    )
    with _stack(
        (mr, "envs", SimpleNamespace(VLLM_RAISE_ON_LOGIT_NANS=False)),
        (
            mr,
            "get_ep_all2all_manager",
            lambda: SimpleNamespace(query_active_mask=lambda: mask),
        ),
    ):
        with pytest.raises(RuntimeError, match="Fault detected"):
            mr.AsyncGPUModelRunnerOutput.get_output(out)


def test_async_pooling_output_get_output_returns_runner_output():
    out = object.__new__(mr.AsyncGPUPoolingModelRunnerOutput)
    out.async_copy_ready_event = SimpleNamespace(synchronize=lambda: None)
    out._raw_pooler_output = "raw"
    out._model_runner_output = "result"
    assert mr.AsyncGPUPoolingModelRunnerOutput.get_output(out) == "result"


def test_create_encoder_cudagraph_manager_disabled_returns_none():
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=False),
        supports_mm_inputs=True,
    )
    assert mr.ModelRunnerFL._create_encoder_cudagraph_manager(runner) is None


def test_create_encoder_cudagraph_manager_unsupported_model(monkeypatch):
    interfaces = types.ModuleType("vllm.model_executor.models.interfaces")
    interfaces.SupportsEncoderCudaGraph = object
    interfaces.supports_encoder_cudagraph = lambda model: False
    enc_mod = types.ModuleType("vllm.v1.worker.encoder_cudagraph")
    enc_mod.EncoderCudaGraphManager = object
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.interfaces", interfaces
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker.encoder_cudagraph", enc_mod
    )
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=True,
        get_model=lambda: object(),
    )
    assert mr.ModelRunnerFL._create_encoder_cudagraph_manager(runner) is None


def test_create_encoder_cudagraph_manager_builds_manager(monkeypatch):
    class _ECM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    interfaces = types.ModuleType("vllm.model_executor.models.interfaces")
    interfaces.SupportsEncoderCudaGraph = object
    interfaces.supports_encoder_cudagraph = lambda model: True
    enc_mod = types.ModuleType("vllm.v1.worker.encoder_cudagraph")
    enc_mod.EncoderCudaGraphManager = _ECM
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.interfaces", interfaces
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker.encoder_cudagraph", enc_mod
    )
    raw = object()
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=True,
        get_model=lambda: raw,
        vllm_config="cfg",
        device="cpu",
        dtype="dtype",
    )
    out = mr.ModelRunnerFL._create_encoder_cudagraph_manager(runner)
    assert isinstance(out, _ECM)
    assert out.kwargs["model"] is raw
    assert out.kwargs["vllm_config"] == "cfg"


def test_maybe_init_encoder_cudagraph_manager_creates_when_missing():
    runner = SimpleNamespace(
        encoder_cudagraph_manager=None,
        _create_encoder_cudagraph_manager=lambda: "mgr",
    )
    mr.ModelRunnerFL._maybe_init_encoder_cudagraph_manager(runner)
    assert runner.encoder_cudagraph_manager == "mgr"


def test_maybe_init_encoder_cudagraph_manager_keeps_existing():
    def _boom():
        raise AssertionError("should not be called")

    runner = SimpleNamespace(
        encoder_cudagraph_manager="keep",
        _create_encoder_cudagraph_manager=_boom,
    )
    mr.ModelRunnerFL._maybe_init_encoder_cudagraph_manager(runner)
    assert runner.encoder_cudagraph_manager == "keep"


def test_maybe_randomize_inputs_disabled_yields_without_changes():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        )
    )
    ids = torch.ones(3, dtype=torch.long)
    with _stack(
        (mr, "envs", SimpleNamespace(VLLM_RANDOMIZE_DP_DUMMY_INPUTS=False))
    ):
        with mr.ModelRunnerFL.maybe_randomize_inputs(
            runner, ids, None, False
        ):
            pass
    assert torch.equal(ids, torch.ones(3, dtype=torch.long))


def test_maybe_randomize_inputs_randomizes_and_restores_ids():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        ),
        input_ids=SimpleNamespace(gpu=torch.zeros(5, dtype=torch.long)),
        model_config=SimpleNamespace(get_vocab_size=lambda: 10),
    )
    ids = torch.ones(3, dtype=torch.long)
    with _stack(
        (mr, "envs", SimpleNamespace(VLLM_RANDOMIZE_DP_DUMMY_INPUTS=False))
    ):
        with mr.ModelRunnerFL.maybe_randomize_inputs(runner, ids, None, True):
            pass
    assert ids.tolist() == [0, 0, 0]


def test_maybe_randomize_inputs_randomizes_and_restores_embeds():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        ),
        inputs_embeds=SimpleNamespace(gpu=torch.zeros(4, 3)),
    )
    embeds = torch.ones(2, 3)
    with _stack(
        (mr, "envs", SimpleNamespace(VLLM_RANDOMIZE_DP_DUMMY_INPUTS=False))
    ):
        with mr.ModelRunnerFL.maybe_randomize_inputs(
            runner, None, embeds, True
        ):
            pass
    assert embeds.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]




def test_get_mm_dummy_batch_falls_back_to_registry():
    budget = SimpleNamespace(cache={})
    registry = SimpleNamespace(
        get_dummy_mm_inputs=lambda *a, **k: {"mm_kwargs": {"image": ["item"]}}
    )
    runner = SimpleNamespace(
        mm_budget=budget,
        mm_registry=registry,
        model_config="cfg",
        device=torch.device("cpu"),
    )
    with _stack(
        (
            mr,
            "group_and_batch_mm_kwargs",
            lambda inputs, device, pin_memory: [("m", 0, {"items": len(inputs)})],
        )
    ):
        out = mr.ModelRunnerFL._get_mm_dummy_batch(runner, "image", 3)
    assert out == {"items": 3}


def test_calculate_reorder_batch_threshold_empty_groups():
    runner = SimpleNamespace(_attn_group_iterator=lambda: iter([]))
    mr.ModelRunnerFL.calculate_reorder_batch_threshold(runner)
    assert runner.reorder_batch_threshold is None


def test_calculate_reorder_batch_threshold_picks_min_non_none():
    groups = [
        SimpleNamespace(
            get_metadata_builder=lambda: SimpleNamespace(
                reorder_batch_threshold=value
            )
        )
        for value in (None, 6, 2)
    ]
    runner = SimpleNamespace(_attn_group_iterator=lambda: iter(groups))
    mr.ModelRunnerFL.calculate_reorder_batch_threshold(runner)
    assert runner.reorder_batch_threshold == 2


class _CG:
    ALWAYS = SimpleNamespace(value=5)


class _BackendA:
    @classmethod
    def get_builder_cls(cls):
        return _Builder

    @classmethod
    def full_cls_name(cls):
        return "A"


class _Builder:
    @staticmethod
    def get_cudagraph_support(vllm_config, kv_cache_spec):
        return SimpleNamespace(value=1)


def test_check_and_update_cudagraph_mode_resolves_and_inits():
    resolved = []
    keys = []
    runner = SimpleNamespace(
        vllm_config="cfg",
        uniform_decode_query_len=2,
        kv_cache_config="kv",
        max_num_reqs=4,
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=lambda *a, **k: resolved.append(
                (a, k)
            )
            or "mode"
        ),
        cudagraph_dispatcher=SimpleNamespace(
            initialize_cudagraph_keys=lambda mode, qlen: keys.append(
                (mode, qlen)
            )
        ),
        speculative_config=None,
    )
    groups = [SimpleNamespace(kv_cache_spec="spec")]
    with _stack((mr, "AttentionCGSupport", _CG)):
        mr.ModelRunnerFL._check_and_update_cudagraph_mode(
            runner, [{_BackendA}], groups
        )
    assert resolved and resolved[0][1]["is_profiling"] is False
    assert keys == [("mode", 2)]


def test_check_and_update_cudagraph_mode_inits_drafter():
    class _P:
        def __init__(self):
            self.calls = []

        def initialize_cudagraph_keys(self, mode):
            self.calls.append(mode)

    runner = SimpleNamespace(
        vllm_config="cfg",
        uniform_decode_query_len=1,
        kv_cache_config="kv",
        max_num_reqs=1,
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=lambda *a, **k: "mode"
        ),
        cudagraph_dispatcher=SimpleNamespace(
            initialize_cudagraph_keys=lambda *a: None
        ),
        speculative_config=SimpleNamespace(
            use_eagle=lambda: True,
            uses_draft_model=lambda: False,
            uses_extract_hidden_states=lambda: False,
        ),
        drafter=_P(),
    )
    with _stack(
        (mr, "AttentionCGSupport", _CG),
        (mr, "EagleProposer", _P),
        (mr, "DFlashProposer", _P),
        (mr, "DraftModelProposer", _P),
        (mr, "ExtractHiddenStatesProposer", _P),
        (mr, "Gemma4Proposer", _P),
    ):
        mr.ModelRunnerFL._check_and_update_cudagraph_mode(runner, [set()], [])
    assert runner.drafter.calls == ["mode"]


class _AttnSpec:
    pass


def test_get_kv_cache_spec_ec_producer_returns_empty():
    runner = SimpleNamespace()
    with _stack(
        (mr, "has_ec_transfer", lambda: True),
        (mr, "get_ec_transfer", lambda: SimpleNamespace(is_consumer=False)),
    ):
        assert mr.ModelRunnerFL.get_kv_cache_spec(runner) == {}


def test_get_kv_cache_spec_skips_shared_and_collects_specs():
    class _Attn:
        def __init__(self, target, spec):
            self.kv_sharing_target_layer_name = target
            self._spec = spec

        def get_kv_cache_spec(self, cfg):
            return self._spec

        def get_attn_backend(self):
            return SimpleNamespace(customize_spec=lambda spec: "customized")

    layers = {
        "shared": _Attn("target", None),
        "plain": _Attn(None, _AttnSpec()),
        "none": _Attn(None, None),
    }
    runner = SimpleNamespace(
        vllm_config="cfg",
        shared_kv_cache_layers={},
        use_ascend_hybrid_cache=False,
        cache_config="cache",
    )
    with _stack(
        (mr, "has_ec_transfer", lambda: False),
        (mr, "Attention", _Attn),
        (mr, "AttentionSpec", _AttnSpec),
        (mr, "get_layers_from_vllm_config", lambda *a: layers),
        (mr, "set_current_vllm_config", lambda cfg: _ctx()),
    ):
        out = mr.ModelRunnerFL.get_kv_cache_spec(runner)
    assert out == {"plain": "customized"}
    assert runner.shared_kv_cache_layers == {"shared": "target"}


def test_get_kv_cache_spec_pads_hybrid_specs(monkeypatch):
    hybrid = types.ModuleType("vllm_fl.kv_cache.ascend.hybrid")
    hybrid.pad_attention_specs = lambda specs, size: "padded"
    monkeypatch.setitem(
        sys.modules, "vllm_fl.kv_cache", types.ModuleType("vllm_fl.kv_cache")
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.kv_cache.ascend",
        types.ModuleType("vllm_fl.kv_cache.ascend"),
    )
    monkeypatch.setitem(
        sys.modules, "vllm_fl.kv_cache.ascend.hybrid", hybrid
    )
    runner = SimpleNamespace(
        vllm_config="cfg",
        shared_kv_cache_layers={},
        use_ascend_hybrid_cache=True,
        cache_config=SimpleNamespace(mamba_page_size_padded=128),
    )
    with _stack(
        (mr, "has_ec_transfer", lambda: False),
        (mr, "get_layers_from_vllm_config", lambda *a: {}),
    ):
        assert mr.ModelRunnerFL.get_kv_cache_spec(runner) == "padded"


def test_may_add_encoder_only_layers_no_encoder_layers():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16)
        ),
        kv_cache_dtype="dtype",
        runner_only_attn_layers=set(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    with _stack(
        (mr, "get_layers_from_vllm_config", lambda *a: {}),
        (mr, "AttentionType", SimpleNamespace(ENCODER_ONLY="enc")),
    ):
        mr.ModelRunnerFL.may_add_encoder_only_layers_to_kv_cache_config(runner)
    assert runner.kv_cache_config.kv_cache_groups == []


def test_may_add_encoder_only_layers_appends_group():
    class _Spec:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Group:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    layer = SimpleNamespace(
        attn_type="enc", num_kv_heads=2, head_size=8
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16)
        ),
        kv_cache_dtype="dtype",
        runner_only_attn_layers=set(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    with _stack(
        (mr, "get_layers_from_vllm_config", lambda *a: {"l1": layer}),
        (mr, "AttentionType", SimpleNamespace(ENCODER_ONLY="enc")),
        (mr, "EncoderOnlyAttentionSpec", _Spec),
        (mr, "KVCacheGroupSpec", _Group),
    ):
        mr.ModelRunnerFL.may_add_encoder_only_layers_to_kv_cache_config(runner)
    assert "l1" in runner.runner_only_attn_layers
    assert len(runner.kv_cache_config.kv_cache_groups) == 1
    assert runner.kv_cache_config.kv_cache_groups[0].kwargs["layer_names"] == [
        "l1"
    ]


class _AG:
    def __init__(self, backend, layer_names, spec, group_id):
        self.backend = backend
        self.layer_names = layer_names
        self.spec = spec
        self.group_id = group_id


def _attn_layer(backend, num_heads=4):
    return SimpleNamespace(
        get_attn_backend=lambda: backend, num_heads=num_heads
    )


def test_initialize_attn_backend_groups_layers_by_backend():
    runner = SimpleNamespace(
        vllm_config="cfg",
        attn_groups=[],
        device="cpu",
        kv_sharing_fast_prefill_eligible_layers=set(),
        _check_and_update_cudagraph_mode=lambda backends, groups, is_profiling=False: None,
    )
    group_spec = SimpleNamespace(
        layer_names=["l1", "l2"], kv_cache_spec="spec"
    )
    layers = {
        "l1": _attn_layer(_BackendA),
        "l2": _attn_layer(_BackendA),
    }
    with _stack(
        (mr, "get_layers_from_vllm_config", lambda *a: layers),
        (mr, "AttentionGroup", _AG),
        (mr, "UniformTypeKVCacheSpecs", _AttnSpec),
        (mr, "check_attention_cp_compatibility", lambda cfg: None),
    ):
        mr.ModelRunnerFL.initialize_attn_backend(
            runner, SimpleNamespace(kv_cache_groups=[group_spec])
        )
    assert len(runner.attn_groups) == 1
    groups = runner.attn_groups[0]
    assert len(groups) == 1 and groups[0].layer_names == ["l1", "l2"]


def test_initialize_attn_backend_uniform_spec_per_layer():
    runner = SimpleNamespace(
        vllm_config="cfg",
        attn_groups=[],
        device="cpu",
        kv_sharing_fast_prefill_eligible_layers=set(),
        _check_and_update_cudagraph_mode=lambda backends, groups, is_profiling=False: None,
    )

    class _UT:
        def __init__(self, specs):
            self.kv_cache_specs = specs

    group_spec = SimpleNamespace(
        layer_names=["l1", "l2"],
        kv_cache_spec=_UT({"l1": "s1", "l2": "s2"}),
    )
    layers = {
        "l1": _attn_layer(_BackendA),
        "l2": _attn_layer(_BackendA),
    }
    with _stack(
        (mr, "get_layers_from_vllm_config", lambda *a: layers),
        (mr, "AttentionGroup", _AG),
        (mr, "UniformTypeKVCacheSpecs", _UT),
        (mr, "check_attention_cp_compatibility", lambda cfg: None),
    ):
        mr.ModelRunnerFL.initialize_attn_backend(
            runner, SimpleNamespace(kv_cache_groups=[group_spec])
        )
    specs = sorted(g.spec for g in runner.attn_groups[0])
    assert specs == ["s1", "s2"]


def test_initialize_attn_backend_fast_prefill_eligible_layer():
    created = []

    class _FastBackend(_BackendA):
        pass

    def _create(name, backend):
        created.append((name, backend))
        return _FastBackend

    runner = SimpleNamespace(
        vllm_config="cfg",
        attn_groups=[],
        device="cpu",
        kv_sharing_fast_prefill_eligible_layers={"l1"},
        _check_and_update_cudagraph_mode=lambda backends, groups, is_profiling=False: None,
    )
    group_spec = SimpleNamespace(layer_names=["l1"], kv_cache_spec="spec")
    with _stack(
        (
            mr,
            "get_layers_from_vllm_config",
            lambda *a: {"l1": _attn_layer(_BackendA)},
        ),
        (mr, "AttentionGroup", _AG),
        (mr, "UniformTypeKVCacheSpecs", _AttnSpec),
        (mr, "create_fast_prefill_custom_backend", _create),
        (mr, "check_attention_cp_compatibility", lambda cfg: None),
    ):
        mr.ModelRunnerFL.initialize_attn_backend(
            runner, SimpleNamespace(kv_cache_groups=[group_spec])
        )
    assert created == [("FastPrefill", _BackendA)]
    assert runner.attn_groups[0][0].backend is _FastBackend


def test_initialize_metadata_builders_single_builder():
    calls = []
    group = SimpleNamespace(
        create_metadata_builders=lambda *a, **k: calls.append((a, k))
    )
    runner = SimpleNamespace(
        vllm_config="cfg",
        device="cpu",
        attn_groups=[[group]],
        parallel_config=SimpleNamespace(use_ubatching=False),
        speculative_config=None,
        calculate_reorder_batch_threshold=lambda: calls.append("threshold"),
    )
    kv = SimpleNamespace(kv_cache_groups=[object()])
    mr.ModelRunnerFL.initialize_metadata_builders(runner, kv, [8])
    assert calls[0][1]["num_metadata_builders"] == 1
    assert calls[0][0][2] == 8
    assert calls[-1] == "threshold"


def test_initialize_metadata_builders_ubatching_count_and_missing_kernel():
    calls = []
    group = SimpleNamespace(
        create_metadata_builders=lambda *a, **k: calls.append((a, k))
    )
    runner = SimpleNamespace(
        vllm_config="cfg",
        device="cpu",
        attn_groups=[[group]],
        parallel_config=SimpleNamespace(
            use_ubatching=True, num_ubatches=4
        ),
        speculative_config=None,
        calculate_reorder_batch_threshold=lambda: None,
    )
    kv = SimpleNamespace(kv_cache_groups=[object()])
    mr.ModelRunnerFL.initialize_metadata_builders(runner, kv, [])
    assert calls[0][1]["num_metadata_builders"] == 4
    assert calls[0][0][2] is None


def test_initialize_metadata_builders_initializes_drafter():
    class _P:
        def __init__(self):
            self.calls = []

        def initialize_attn_backend(self, kv, kernel):
            self.calls.append((kv, kernel))

    drafter = _P()
    group = SimpleNamespace(create_metadata_builders=lambda *a, **k: None)
    runner = SimpleNamespace(
        vllm_config="cfg",
        device="cpu",
        attn_groups=[[group]],
        parallel_config=SimpleNamespace(use_ubatching=False),
        speculative_config=SimpleNamespace(
            use_eagle=lambda: True, uses_draft_model=lambda: False
        ),
        drafter=drafter,
        calculate_reorder_batch_threshold=lambda: None,
    )
    kv = SimpleNamespace(kv_cache_groups=[object()])
    with _stack(
        (mr, "EagleProposer", _P),
        (mr, "DFlashProposer", _P),
        (mr, "DraftModelProposer", _P),
        (mr, "Gemma4Proposer", _P),
    ):
        mr.ModelRunnerFL.initialize_metadata_builders(runner, kv, [4])
    assert drafter.calls == [(kv, [4])]


class _IB:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _reinit_runner(**overrides):
    spec = SimpleNamespace(
        kind="attn",
        block_size=16,
        max_num_blocks_per_req=lambda cfg, ml: 10,
    )
    base = dict(
        max_model_len=128,
        max_encoder_len=64,
        max_num_reqs=4,
        max_num_tokens=512,
        device="cpu",
        num_spec_tokens=0,
        is_pooling_model=False,
        vllm_config=SimpleNamespace(reasoning_config=None),
        model_config=SimpleNamespace(get_vocab_size=lambda: 100),
        cache_config=SimpleNamespace(use_replayssm=False),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        input_batch=SimpleNamespace(
            logitsprocs="lp", logitsprocs_need_output_token_ids=False
        ),
        jit_warmup_registry=SimpleNamespace(activate=lambda: _ctx()),
        _init_block_sizes=[16],
        _init_kernel_block_sizes=[8],
        _init_max_num_blocks=[10],
        _init_slot_mapping_modes=["tok"],
    )
    base.update(overrides)
    runner = SimpleNamespace(**base)
    kv = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    return runner, kv


def _reinit_patches():
    return (
        (mr, "get_kv_cache_spec_kind", lambda spec: spec.kind),
        (
            mr,
            "KVCacheSpecKind",
            SimpleNamespace(ENCODER_ONLY_ATTENTION="enc", MAMBA="mamba"),
        ),
        (
            mr,
            "SlotMappingMode",
            SimpleNamespace(NONE="none", TOKEN_TO_KV_SLOT="tok"),
        ),
        (mr, "InputBatch", _IB),
    )


def test_maybe_reinitialize_input_batch_no_change_keeps_batch():
    runner, kv = _reinit_runner()
    with _stack(*_reinit_patches()):
        mr.ModelRunnerFL.may_reinitialize_input_batch(runner, kv, [8])
    assert not isinstance(runner.input_batch, _IB)
    assert runner._init_block_sizes == [16]


def test_maybe_reinitialize_input_batch_geometry_change_rebuilds():
    runner, kv = _reinit_runner(_init_block_sizes=[4])
    with _stack(*_reinit_patches()):
        mr.ModelRunnerFL.may_reinitialize_input_batch(runner, kv, [8])
    assert isinstance(runner.input_batch, _IB)
    assert runner.input_batch.kwargs["block_sizes"] == [16]
    assert runner.input_batch.kwargs["max_num_blocks_per_req"] == [10]


def test_initialize_kv_cache_tensors_delegates_to_v032():
    runner = SimpleNamespace(
        _initialize_kv_cache_tensors_v032=lambda cfg, kbs, ctx: (
            "ok",
            cfg,
            kbs,
            ctx,
        )
    )
    out = mr.ModelRunnerFL.initialize_kv_cache_tensors(
        runner, "cfg", [8], "ctx"
    )
    assert out == ("ok", "cfg", [8], "ctx")


def test_init_kv_zero_meta_builds_zeroer():
    captured = {}

    class _KZ:
        def __init__(self, device, **kwargs):
            captured["device"] = device
            captured.update(kwargs)

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _kv_cache_spec_attn_group_iterator=lambda: iter([]),
        _kernel_block_sizes=[8],
        compilation_config=SimpleNamespace(static_forward_context="sfc"),
        kv_cache_config=SimpleNamespace(num_blocks=32),
        runner_only_attn_layers=["l1"],
    )
    with _stack((mr, "KVBlockZeroer", _KZ)):
        mr.ModelRunnerFL._init_kv_zero_meta(runner)
    assert isinstance(runner._kv_block_zeroer, _KZ)
    assert captured["num_blocks"] == 32
    assert captured["kernel_block_sizes"] == [8]


def test_synchronize_input_prep_no_event_yields():
    runner = SimpleNamespace(prepare_inputs_event=None)
    ran = []
    with mr.ModelRunnerFL.synchronize_input_prep(runner):
        ran.append("body")
    assert ran == ["body"]


def test_synchronize_input_prep_records_event():
    calls = []
    event = SimpleNamespace(
        synchronize=lambda: calls.append("sync"),
        record=lambda: calls.append("record"),
    )
    runner = SimpleNamespace(prepare_inputs_event=event)
    with mr.ModelRunnerFL.synchronize_input_prep(runner):
        calls.append("body")
    assert calls == ["sync", "body", "record"]


def test_model_forward_forwards_all_arguments():
    captured = {}

    def _model(**kwargs):
        captured.update(kwargs)
        return "out"

    runner = SimpleNamespace(model=_model)
    out = mr.ModelRunnerFL._model_forward(
        runner,
        input_ids="ids",
        positions="pos",
        intermediate_tensors="inter",
        inputs_embeds="emb",
        extra="x",
    )
    assert out == "out"
    assert captured == {
        "input_ids": "ids",
        "positions": "pos",
        "intermediate_tensors": "inter",
        "inputs_embeds": "emb",
        "extra": "x",
    }


def test_is_uniform_decode_true():
    assert mr.ModelRunnerFL._is_uniform_decode(2, 2, 4, 2) is True


def test_is_uniform_decode_false_on_token_mismatch():
    assert mr.ModelRunnerFL._is_uniform_decode(2, 2, 5, 2) is False


def test_is_uniform_decode_forced_true():
    assert mr.ModelRunnerFL._is_uniform_decode(1, 2, 9, 3, True) is True


def test_is_uniform_decode_forced_false():
    assert mr.ModelRunnerFL._is_uniform_decode(2, 2, 4, 2, False) is False
