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

"""G13 coverage: ModelRunnerFL.__init__ (814b L584-1112).

The whole constructor is driven end-to-end against a SimpleNamespace fake
``self``: every heavy module-level collaborator (current_platform, Sampler,
InputBatch, CudagraphDispatcher, offloader, drafters, RejectionSampler,
MULTIMODAL_REGISTRY, ...) is patched at the ``model_runner`` module boundary,
and ``_make_buffer`` / ``_init_device_properties`` are replaced with recording
fakes so no accelerator is required. Branches covered: default wiring, ascend
attention + graph runtime, EP fault, DCP, external-launcher PP broadcast,
encoder-decoder, async scheduling, cudagraph capture sizes, M-RoPE,
KV-sharing fast prefill, MM registry fallback + budget, and all speculative
drafter dispatch arms (incl. unknown-method ValueError).
"""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace

import torch

from vllm_fl.worker import model_runner as mr


# --------------------------------------------------------------------------
# patch helpers
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# config builder
# --------------------------------------------------------------------------
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
    cfg = SimpleNamespace(
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
    return cfg


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


# --------------------------------------------------------------------------
# end-to-end runner
# --------------------------------------------------------------------------
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
        (mr, "build_logitsprocs", lambda *a, **k: ("lp",)),
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
        # strip hardcoded pin_memory=True so CPU-only torch works
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


# --------------------------------------------------------------------------
# default path
# --------------------------------------------------------------------------
def test_init_default_path_wires_core_attributes():
    runner, record = _run(_config())
    assert runner.max_model_len == 128
    assert runner.max_num_tokens == 64
    assert runner.max_num_reqs == 4
    assert runner.num_query_heads == 8
    assert runner.kv_cache_dtype == torch.float16
    assert runner.is_pooling_model is False
    assert runner.dtype == torch.float16
    assert runner.cascade_attn_enabled is True
    assert runner.calculate_kv_scales is False
    assert runner.use_ascend_attention is False
    assert runner.use_async_scheduling is False
    assert runner.num_spec_tokens == 0
    assert runner.use_async_spec_decode is False
    assert runner.cudagraph_batch_sizes == []
    assert record["sampler"] == ("raw_logprobs", False)
    assert record["offloader"] == "offloader"
    assert record["num_sms"] == 1
    assert runner.mm_budget is None
    assert runner.input_batch is not None


def test_init_sampler_and_input_batch_receive_expected_kwargs():
    _, record = _run(_config())
    kwargs = record["input_batch"]
    assert kwargs["max_num_reqs"] == 4
    assert kwargs["max_num_batched_tokens"] == 64
    assert kwargs["vocab_size"] == 100
    assert kwargs["block_sizes"] == [16]
    assert kwargs["is_pooling_model"] is False


def test_init_offloader_installed_before_use():
    _, record = _run(_config())
    assert record["offloader"] == "offloader"


# --------------------------------------------------------------------------
# ascend attention
# --------------------------------------------------------------------------
def _append_ascend_modules(record):
    hybrid = types.ModuleType("vllm_fl.kv_cache.ascend.hybrid")
    hybrid.uses_dense_hybrid_cache = lambda mc: record.setdefault("dense", True) or True
    gr = types.ModuleType("vllm_fl.compilation.graph_runtime")

    class GraphRuntimeController:
        def __init__(self, vllm_config=None, full_graph_wrapper_type=None):
            record["graph_runtime"] = (vllm_config, full_graph_wrapper_type)

    gr.GraphRuntimeController = GraphRuntimeController
    return {
        "vllm_fl.kv_cache.ascend.hybrid": hybrid,
        "vllm_fl.compilation.graph_runtime": gr,
    }




def test_init_ascend_dense_hybrid_cache_false():
    mods = {
        "vllm_fl.kv_cache.ascend.hybrid": types.ModuleType(
            "vllm_fl.kv_cache.ascend.hybrid"
        ),
        "vllm_fl.compilation.graph_runtime": types.ModuleType(
            "vllm_fl.compilation.graph_runtime"
        ),
    }
    mods["vllm_fl.kv_cache.ascend.hybrid"].uses_dense_hybrid_cache = lambda mc: False

    class GRC:
        def __init__(self, **k):
            pass

    mods["vllm_fl.compilation.graph_runtime"].GraphRuntimeController = GRC
    with _fake_modules(mods):
        runner, _ = _run(_config(), vendor="ascend")
    assert runner.use_ascend_hybrid_cache is False


# --------------------------------------------------------------------------
# EP fault / DCP / PP broadcast
# --------------------------------------------------------------------------
def test_init_ep_fault_tolerance_enabled():
    cfg = _config(model={"is_moe": True}, parallel={"data_parallel_size": 2})
    runner, _ = _run(
        cfg,
        extra=[(mr, "get_ep_all2all_manager", lambda: SimpleNamespace(
            support_fault_tolerance=True))],
    )
    assert runner.check_ep_fault is True


def test_init_ep_fault_disabled_when_single_dp():
    cfg = _config(model={"is_moe": True}, parallel={"data_parallel_size": 1})
    runner, _ = _run(cfg)
    assert runner.check_ep_fault is False


def test_init_dcp_multiple_ranks():
    cfg = _config(parallel={"decode_context_parallel_size": 2})
    runner, _ = _run(cfg)
    assert runner.dcp_world_size == 2
    assert runner.dcp_rank == 1


def test_init_broadcast_pp_output_external_launcher():
    cfg = _config(parallel={"distributed_executor_backend": "external_launcher"})
    runner, _ = _run(
        cfg,
        extra=[(mr, "get_pp_group", lambda: SimpleNamespace(
            ranks=[0, 1], is_last_rank=True))],
    )
    assert runner.broadcast_pp_output is True


def test_init_broadcast_pp_output_single_rank():
    cfg = _config(parallel={"distributed_executor_backend": "external_launcher"})
    runner, _ = _run(cfg)  # base ranks == [0]
    assert runner.broadcast_pp_output is False


def test_init_encoder_decoder_sets_max_encoder_len():
    cfg = _config(
        model={"is_encoder_decoder": True},
        scheduler={"max_num_encoder_input_tokens": 32},
    )
    runner, _ = _run(cfg)
    assert runner.max_encoder_len == 32


def test_init_decoder_only_sets_zero_encoder_len():
    runner, _ = _run(_config())
    assert runner.max_encoder_len == 0


# --------------------------------------------------------------------------
# async scheduling / cudagraph sizes / mrope / kv sharing
# --------------------------------------------------------------------------
def test_init_async_scheduling_creates_stream_and_event():
    cfg = _config(scheduler={"async_scheduling": True})
    runner, record = _run(cfg)
    assert runner.use_async_scheduling is True
    assert runner.async_output_copy_stream is not None
    assert runner.prepare_inputs_event is not None
    assert record["streams"]


def test_init_cudagraph_capture_sizes_sorted():
    cfg = _config(
        compilation={"cudagraph_capture_sizes": [4, 1, 3], "cudagraph_mode": "full"}
    )
    runner, _ = _run(cfg)
    assert runner.cudagraph_batch_sizes == [1, 3, 4]


def test_init_cudagraph_sizes_empty_when_mode_none():
    cfg = _config(compilation={"cudagraph_capture_sizes": [4, 2]})
    runner, _ = _run(cfg)
    assert runner.cudagraph_batch_sizes == []


def test_init_uses_mrope_allocates_positions():
    runner, record = _run(_config(model={"uses_mrope": True}))
    assert runner.uses_mrope is True
    assert runner.mrope_positions is not None


def test_init_kv_sharing_fast_prefill_allocates_indices():
    cfg = _config(cache={"kv_sharing_fast_prefill": True})
    runner, _ = _run(cfg)
    assert runner.kv_sharing_fast_prefill_logits_indices is not None


# --------------------------------------------------------------------------
# multimodal registry
# --------------------------------------------------------------------------
def test_init_mm_attr_path_uses_model_config():
    runner, record = _run(
        _config(model={"supports_multimodal_inputs": True}), mm_supports=True
    )
    assert runner.supports_mm_inputs is True
    assert "budget" in record


def test_init_mm_registry_fallback_and_budget():
    cfg = _config(drop_mm=True)
    runner, record = _run(cfg, mm_supports=True)
    assert runner.supports_mm_inputs is True
    assert record["budget"][1] is runner.mm_registry


# --------------------------------------------------------------------------
# speculative decoding dispatch arms
# --------------------------------------------------------------------------
def test_init_spec_custom_class():
    runner, record = _run(_config(spec=_spec(method="custom_class")))
    assert runner.drafter == "custom-proposer"
    assert "rejection_sampler" in record


def test_init_spec_ngram():
    mod = types.ModuleType("vllm.v1.spec_decode.ngram_proposer")

    class NgramProposer:
        def __init__(self, cfg):
            self.cfg = cfg

    mod.NgramProposer = NgramProposer
    with _fake_modules({"vllm.v1.spec_decode.ngram_proposer": mod}):
        runner, _ = _run(_config(spec=_spec(method="ngram")))
    assert isinstance(runner.drafter, NgramProposer)


def test_init_spec_draft_model_and_effective_len():
    draft = SimpleNamespace(max_model_len=77)
    runner, record = _run(
        _config(spec=_spec(method="draft_model", draft_model=True,
                           draft_model_config=draft))
    )
    assert "DraftModelProposer" in record["drafters"]
    assert runner.effective_drafter_max_model_len == 77


def test_init_spec_ngram_gpu_allocates_buffers():
    runner, record = _run(_config(spec=_spec(method="ngram_gpu", ngram_gpu=True)))
    assert "NgramProposerGPU" in record["drafters"]
    assert runner.num_tokens_no_spec_gpu is not None
    assert runner.token_ids_gpu_tensor is not None
    assert runner._num_valid_draft_tokens_cpu is not None
    assert runner._num_valid_draft_tokens_event is not None


def test_init_spec_gemma4_mtp():
    _, record = _run(_config(spec=_spec(method="gemma4", gemma4=True)))
    assert "Gemma4Proposer" in record["drafters"]


def test_init_spec_step3p5_mtp():
    _, record = _run(_config(spec=_spec(method="step3p5", step3p5=True)))
    assert "Step3p5MTPProposer" in record["drafters"]


def test_init_spec_dflash_sets_aux_outputs():
    runner, record = _run(_config(spec=_spec(method="dflash", dflash=True)))
    assert "DFlashProposer" in record["drafters"]
    assert runner.use_aux_hidden_state_outputs is True


def test_init_spec_suffix():
    _, record = _run(_config(spec=_spec(method="suffix")))
    assert "SuffixDecodingProposer" in record["drafters"]


def test_init_spec_eagle3_uses_aux_from_drafter():
    runner, record = _run(_config(spec=_spec(method="eagle3", eagle=True)))
    assert "EagleProposer" in record["drafters"]
    assert runner.use_aux_hidden_state_outputs is True


def test_init_spec_medusa():
    _, record = _run(_config(spec=_spec(method="medusa")))
    assert "MedusaProposer" in record["drafters"]


def test_init_spec_extract_hidden_states_aux():
    runner, record = _run(_config(spec=_spec(method="extract_hidden_states")))
    assert "ExtractHiddenStatesProposer" in record["drafters"]
    assert runner.use_aux_hidden_state_outputs is True


def test_init_spec_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown speculative decoding method"):
        _run(_config(spec=_spec(method="bogus")))




def test_init_spec_not_last_rank_skips_drafter():
    spec = _spec(method="ngram_gpu", ngram_gpu=True, num_spec=1)
    runner, record = _run(
        _config(spec=spec),
        extra=[(mr, "get_pp_group", lambda: SimpleNamespace(
            ranks=[0], is_last_rank=False))],
    )
    assert "drafters" not in record
    assert not hasattr(runner, "drafter")
    assert runner.num_spec_tokens == 1


