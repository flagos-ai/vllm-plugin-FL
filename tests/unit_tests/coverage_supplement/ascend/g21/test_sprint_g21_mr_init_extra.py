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

"""G21 coverage: ModelRunnerFL.__init__ remaining (814b L584-1112).

Complements G13. G13 already drives the default wiring, ascend hybrid/graph,
EP fault (dp>1 moe + single dp), DCP multi-rank, external-launcher PP
broadcast, encoder-decoder, async scheduling, cudagraph sizes, M-RoPE,
KV-sharing fast prefill, MM attr/fallback, and every speculative drafter arm.
This file targets the still-uncovered assembly branches: calculate_kv_scales
present, EP short-circuits, non-external PP broadcast, MM fallback/attr with
supports=False, drafter effective-length fallbacks, ngram-gpu buffer absence,
pooling/prompt-embed/raw-input/alibi/prefix-lm/cascade flags, logits-processor
custom tuple + need-output-token-ids, default block size, encoder-decoder
InputBatch sizing, buffer shapes, dcp_local_seq_lens, ascend-off graph
runtime, and the lazy-state / attribute initialisation details.
"""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace

import torch

from vllm_fl.worker import model_runner as mr


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


_MISSING = object()


@contextlib.contextmanager
def _fake_modules(mapping):
    saved = {name: sys.modules.get(name, _MISSING) for name in mapping}
    for name, mod in mapping.items():
        sys.modules[name] = mod
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


_TORCH_ZEROS = torch.zeros
_TORCH_EMPTY = torch.empty


def _zeros_no_pin(*args, **kwargs):
    kwargs.pop("pin_memory", None)
    return _TORCH_ZEROS(*args, **kwargs)


def _empty_no_pin(*args, **kwargs):
    kwargs.pop("pin_memory", None)
    return _TORCH_EMPTY(*args, **kwargs)


class _FakeEvent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _spec(
    method="none",
    *,
    draft_model=False,
    ngram_gpu=False,
    gemma4=False,
    step3p5=False,
    dflash=False,
    eagle=False,
    num_spec=2,
    draft_model_config=None,
):
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=num_spec,
        draft_model_config=draft_model_config,
        uses_draft_model=lambda: draft_model,
        use_ngram_gpu=lambda: ngram_gpu,
        use_gemma4_mtp=lambda: gemma4,
        use_step3p5_mtp=lambda: step3p5,
        use_dflash=lambda: dflash,
        use_eagle=lambda: eagle,
    )


def _config(
    model=None,
    cache=None,
    parallel=None,
    scheduler=None,
    compilation=None,
    spec=None,
    drop_mm=False,
):
    model_cfg = SimpleNamespace(
        dtype=torch.float16,
        is_encoder_decoder=False,
        disable_cascade_attn=False,
        is_mm_prefix_lm=False,
        is_multimodal_raw_input_only_model=False,
        enable_prompt_embeds=False,
        runner_type="generate",
        max_model_len=128,
        is_moe=False,
        uses_alibi=False,
        uses_mrope=False,
        logprobs_mode="raw_logprobs",
        use_fp64_gumbel=False,
        logits_processors=None,
        supports_multimodal_inputs=False,
        get_num_attention_heads=lambda pc: 8,
        get_inputs_embeds_size=lambda: 0,
        get_vocab_size=lambda: 100,
    )
    cache_cfg = SimpleNamespace(
        cache_dtype="auto",
        block_size=16,
        use_replayssm=False,
        kv_sharing_fast_prefill=False,
    )
    parallel_cfg = SimpleNamespace(
        data_parallel_size=1,
        decode_context_parallel_size=1,
        distributed_executor_backend="mp",
        cp_kv_cache_interleave_size=1,
    )
    sched_cfg = SimpleNamespace(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_num_encoder_input_tokens=0,
        async_scheduling=False,
    )
    comp_cfg = SimpleNamespace(
        cudagraph_capture_sizes=None,
        cudagraph_mode=mr.CUDAGraphMode.NONE,
    )
    for target, over in (
        (model_cfg, model),
        (cache_cfg, cache),
        (parallel_cfg, parallel),
        (sched_cfg, scheduler),
        (comp_cfg, compilation),
    ):
        if over:
            for key, val in over.items():
                setattr(target, key, val)
    if drop_mm:
        del model_cfg.supports_multimodal_inputs
    return SimpleNamespace(
        model_config=model_cfg,
        cache_config=cache_cfg,
        offload_config=SimpleNamespace(),
        compilation_config=comp_cfg,
        lora_config=None,
        load_config=None,
        parallel_config=parallel_cfg,
        scheduler_config=sched_cfg,
        speculative_config=spec,
        observability_config=None,
        reasoning_config=None,
    )


def _run(config, *, vendor="cuda", mm_supports=False, extra=()):
    record = {}
    streams = []

    class _Stream:
        def __init__(self, *a, **k):
            streams.append(self)

    class _Event:
        def __init__(self, *a, **k):
            streams.append(("event", k))

    class _Sampler:
        def __init__(self, logprobs_mode=None, use_fp64_gumbel=False):
            record["sampler"] = (logprobs_mode, use_fp64_gumbel)

    class _LateInteraction:
        pass

    class _InputBatch:
        def __init__(self, *a, **k):
            record["input_batch"] = k

    class _Dispatcher:
        def __init__(self, cfg):
            record["dispatcher"] = cfg

    class _Budget:
        def __init__(self, cfg, registry):
            record["budget"] = (cfg, registry)

    class _Reject:
        def __init__(self, sampler, spec, device):
            record["rejection_sampler"] = (sampler, spec, device)

    def _build_logitsprocs(*a, **k):
        record["logprocs_args"] = a
        record["logprocs_kwargs"] = k
        return ("lp",)

    def _drafter_cls(name):
        class _D:
            def __init__(self, *a, **k):
                self.args = a
                self.kwargs = k
                self.eagle3_use_aux_hidden_state = True
                record.setdefault("drafters", []).append(name)

        _D.__name__ = name
        return _D

    def _fake_make_buffer(*size, dtype=None, numpy=True):
        record.setdefault("buffers", []).append((size, dtype, numpy))
        return SimpleNamespace(size=size, dtype=dtype, numpy=numpy)

    def _set_num_sms():
        record["num_sms"] = 1

    fake_registry = SimpleNamespace(
        supports_multimodal_inputs=lambda mc: mm_supports,
    )
    fake_platform = SimpleNamespace(
        vendor_name=vendor,
        torch_device_fn=SimpleNamespace(Stream=_Stream, Event=_Event),
    )

    base = [
        (mr, "current_platform", fake_platform),
        (mr, "kv_cache_dtype_str_to_dtype", lambda dtype, mc: torch.float16),
        (mr, "Sampler", _Sampler),
        (mr, "LateInteractionRunner", _LateInteraction),
        (mr, "InputBatch", _InputBatch),
        (mr, "build_logitsprocs", _build_logitsprocs),
        (mr, "PIN_MEMORY", False),
        (mr, "CudagraphDispatcher", _Dispatcher),
        (mr, "create_offloader", lambda cfg: "offloader"),
        (mr, "set_offloader", lambda o: record.__setitem__("offloader", o)),
        (mr, "get_pp_group", lambda: SimpleNamespace(ranks=[0], is_last_rank=True)),
        (mr, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=1)),
        (
            mr,
            "get_ep_all2all_manager",
            lambda: SimpleNamespace(support_fault_tolerance=True),
        ),
        (mr, "RejectionSampler", _Reject),
        (mr, "create_custom_proposer", lambda cfg: "custom-proposer"),
        (mr, "MULTIMODAL_REGISTRY", fake_registry),
        (mr, "MultiModalBudget", _Budget),
        (mr.torch, "zeros", _zeros_no_pin),
        (mr.torch, "empty", _empty_no_pin),
        (mr.torch, "Event", _FakeEvent),
    ]
    for name in (
        "DraftModelProposer",
        "NgramProposerGPU",
        "Gemma4Proposer",
        "Step3p5MTPProposer",
        "DFlashProposer",
        "SuffixDecodingProposer",
        "EagleProposer",
        "MedusaProposer",
        "ExtractHiddenStatesProposer",
    ):
        if hasattr(mr, name):
            base.append((mr, name, _drafter_cls(name)))

    runner = SimpleNamespace()
    runner._make_buffer = _fake_make_buffer
    runner._init_device_properties = _set_num_sms
    with _stack(*(base + list(extra))):
        mr.ModelRunnerFL.__init__(runner, config, torch.device("cpu"))
    record["streams"] = streams
    return runner, record


def _has_buffer(record, *args):
    return any(tuple(b[0]) == tuple(args) for b in record.get("buffers", []))


def _buffer_entry(record, *args):
    for b in record.get("buffers", []):
        if tuple(b[0]) == tuple(args):
            return b
    return None


# --------------------------------------------------------------------------
# scalar / short-circuit branches
# --------------------------------------------------------------------------
def test_init_calculate_kv_scales_present_true():
    runner, _ = _run(_config(cache={"calculate_kv_scales": True}))
    assert runner.calculate_kv_scales is True


def test_init_dcp_rank_zero_when_world_size_one():
    runner, _ = _run(_config(parallel={"decode_context_parallel_size": 1}))
    assert runner.dcp_world_size == 1
    assert runner.dcp_rank == 0


def test_init_ep_fault_skips_manager_when_not_moe():
    def _boom():
        raise AssertionError("get_ep_all2all_manager must be short-circuited")

    cfg = _config(model={"is_moe": False}, parallel={"data_parallel_size": 4})
    runner, _ = _run(cfg, extra=[(mr, "get_ep_all2all_manager", _boom)])
    assert runner.check_ep_fault is False


def test_init_ep_fault_false_when_no_tolerance():
    cfg = _config(model={"is_moe": True}, parallel={"data_parallel_size": 2})
    runner, _ = _run(
        cfg,
        extra=[
            (
                mr,
                "get_ep_all2all_manager",
                lambda: SimpleNamespace(support_fault_tolerance=False),
            )
        ],
    )
    assert runner.check_ep_fault is False


def test_init_broadcast_pp_false_when_backend_not_external():
    cfg = _config(parallel={"distributed_executor_backend": "mp"})
    runner, _ = _run(
        cfg,
        extra=[
            (
                mr,
                "get_pp_group",
                lambda: SimpleNamespace(ranks=[0, 1], is_last_rank=True),
            )
        ],
    )
    assert runner.broadcast_pp_output is False


# --------------------------------------------------------------------------
# multimodal support resolution
# --------------------------------------------------------------------------
def test_init_mm_fallback_supports_false_no_budget():
    runner, record = _run(_config(drop_mm=True), mm_supports=False)
    assert runner.supports_mm_inputs is False
    assert runner.mm_budget is None
    assert "budget" not in record


def test_init_mm_attr_supports_false_no_budget():
    runner, record = _run(
        _config(model={"supports_multimodal_inputs": False}), mm_supports=True
    )
    assert runner.supports_mm_inputs is False
    assert runner.mm_budget is None
    assert "budget" not in record


# --------------------------------------------------------------------------
# drafter effective length fallbacks / ngram-gpu absence
# --------------------------------------------------------------------------
def test_init_drafter_len_falls_back_when_draft_config_none():
    runner, _ = _run(_config(spec=_spec(method="custom_class", num_spec=2)))
    assert runner.effective_drafter_max_model_len == runner.max_model_len


def test_init_drafter_len_falls_back_when_draft_len_none():
    draft = SimpleNamespace(max_model_len=None)
    runner, _ = _run(
        _config(
            spec=_spec(
                method="custom_class", num_spec=2, draft_model_config=draft
            )
        )
    )
    assert runner.effective_drafter_max_model_len == runner.max_model_len


def test_init_ngram_gpu_buffers_absent_for_other_spec():
    runner, _ = _run(_config(spec=_spec(method="custom_class", num_spec=2)))
    assert runner._num_valid_draft_tokens_cpu is None
    assert runner._num_valid_draft_tokens_event is None
    assert runner._num_valid_draft_tokens_copy_stream is None
    assert not hasattr(runner, "num_tokens_no_spec_gpu")


def test_init_async_spec_decode_false_without_spec_tokens():
    cfg = _config(scheduler={"async_scheduling": True}, spec=None)
    runner, _ = _run(cfg)
    assert runner.use_async_scheduling is True
    assert runner.use_async_spec_decode is False
    assert runner.valid_sampled_token_count_cpu is None
    assert runner.valid_sampled_token_count_event is None


def test_init_no_spec_leaves_drafter_and_buffers_unset():
    runner, _ = _run(_config())
    assert not hasattr(runner, "drafter")
    assert not hasattr(runner, "rejection_sampler")
    assert not hasattr(runner, "effective_drafter_max_model_len")
    assert runner.draft_token_ids_cpu is None
    assert runner.draft_token_ids_event is None
    assert runner.num_accepted_tokens_event is None


# --------------------------------------------------------------------------
# flag / model-config branches
# --------------------------------------------------------------------------
def test_init_pooling_model_flags():
    runner, record = _run(_config(model={"runner_type": "pooling"}))
    assert runner.is_pooling_model is True
    assert record["input_batch"]["is_pooling_model"] is True
    assert record["logprocs_args"][3] is True


def test_init_prompt_embeds_and_raw_input_flags():
    runner, _ = _run(
        _config(
            model={
                "enable_prompt_embeds": True,
                "is_multimodal_raw_input_only_model": True,
            }
        )
    )
    assert runner.enable_prompt_embeds is True
    assert runner.is_multimodal_raw_input_only_model is True


def test_init_alibi_prefix_lm_and_cascade_disabled():
    runner, _ = _run(
        _config(
            model={
                "uses_alibi": True,
                "is_mm_prefix_lm": True,
                "disable_cascade_attn": True,
            }
        )
    )
    assert runner.use_alibi is True
    assert runner.is_mm_prefix_lm is True
    assert runner.cascade_attn_enabled is False


# --------------------------------------------------------------------------
# logits processors
# --------------------------------------------------------------------------
def test_init_logits_processors_enable_output_token_ids():
    runner, record = _run(
        _config(model={"logits_processors": ["lp_a", "lp_b"]})
    )
    assert record["logprocs_args"][4] == ("lp_a", "lp_b")
    assert record["input_batch"]["logitsprocs_need_output_token_ids"] is True


def test_init_logits_processors_none_disables_output_token_ids():
    _, record = _run(_config())
    assert record["logprocs_args"][4] == ()
    assert record["input_batch"]["logitsprocs_need_output_token_ids"] is False


# --------------------------------------------------------------------------
# block size / encoder-decoder sizing / buffers
# --------------------------------------------------------------------------
def test_init_block_size_default_when_none():
    _, record = _run(_config(cache={"block_size": None}))
    default = mr.CacheConfig.DEFAULT_BLOCK_SIZE
    assert record["input_batch"]["block_sizes"] == [default]


def test_init_encoder_decoder_input_batch_max_len_uses_encoder_len():
    cfg = _config(
        model={"is_encoder_decoder": True},
        scheduler={"max_num_encoder_input_tokens": 300},
    )
    _, record = _run(cfg)
    assert record["input_batch"]["max_model_len"] == 300


def test_init_inputs_embeds_buffer_shape_and_numpy_false():
    runner, record = _run(_config(model={"get_inputs_embeds_size": lambda: 32}))
    assert runner.inputs_embeds_size == 32
    entry = _buffer_entry(record, 64, 32)
    assert entry is not None
    assert entry[1] is torch.float16
    assert entry[2] is False


def test_init_query_buffers_allocated():
    runner, record = _run(_config())
    assert _has_buffer(record, 5)
    assert runner.query_pos is not None
    assert any(
        b[0] == (64,) and b[1] is torch.int64 for b in record["buffers"]
    )


def test_init_dcp_local_seq_lens_allocated_when_multi_rank():
    runner, _ = _run(_config(parallel={"decode_context_parallel_size": 2}))
    assert runner.dcp_local_seq_lens is not None


def test_init_mrope_positions_absent_when_disabled():
    runner, _ = _run(_config(model={"uses_mrope": False}))
    assert runner.uses_mrope is False
    assert not hasattr(runner, "mrope_positions")


def test_init_kv_sharing_indices_none_when_disabled():
    runner, _ = _run(_config(cache={"kv_sharing_fast_prefill": False}))
    assert runner.kv_sharing_fast_prefill_logits_indices is None


# --------------------------------------------------------------------------
# ascend-off / arithmetic / lazy state
# --------------------------------------------------------------------------
def test_init_ascend_off_leaves_graph_runtime_none():
    runner, _ = _run(_config(), vendor="cuda")
    assert runner.use_ascend_attention is False
    assert runner.use_ascend_hybrid_cache is False
    assert runner.graph_runtime is None
    assert runner._ascend_attn_state is None


def test_init_uniform_decode_query_len_tracks_spec_tokens():
    runner, _ = _run(_config(spec=_spec(method="custom_class", num_spec=4)))
    assert runner.num_spec_tokens == 4
    assert runner.uniform_decode_query_len == 5


def test_init_block_size_bookkeeping():
    runner, _ = _run(_config())
    assert runner._init_block_sizes == [16]
    assert runner._init_kernel_block_sizes == [16]
    assert runner._init_max_num_blocks == [8]
    assert runner._init_slot_mapping_modes == [mr.SlotMappingMode.TOKEN_TO_KV_SLOT]


def test_init_lazy_state_attribute_defaults():
    runner, _ = _run(_config())
    assert runner.eplb_state is None
    assert runner._moe_model is None
    assert runner.eep_eplb_suppressed is False
    assert runner.routed_experts_initialized is False
    assert runner.is_multimodal_pruning_enabled is False
    assert runner.requires_sequential_video_encoding is False
    assert runner.kv_caches == []
    assert runner.cross_layers_kv_cache is None
    assert runner.cross_layers_attn_backend is None
    assert runner.attn_groups == []
    assert runner.encoder_cache == {}
    assert runner.encoder_cudagraph_manager is None
    assert runner.intermediate_tensors is None
    assert runner._draft_token_ids is None
    assert runner._draft_probs is None
    assert runner._draft_prob_req_ids is None
    assert runner._draft_token_req_ids is None
    assert runner.shared_kv_cache_layers == {}
    assert runner.kv_sharing_fast_prefill_eligible_layers == set()
    assert runner.runner_only_attn_layers == set()
    assert runner.reorder_batch_threshold is None
    assert runner.encoder_timing_registry == {}
    assert runner.mamba_state_idx == {}
    assert runner._mamba_bufs is None
    assert runner.layerwise_nvtx_hooks_registered is False
    assert runner.execute_model_state is None
    assert runner.kv_connector_output is None
    assert runner.requests == {}
    assert runner.num_prompt_logprobs == {}
    assert runner._pp_recv_work is None
    assert runner.transfer_event is not None
