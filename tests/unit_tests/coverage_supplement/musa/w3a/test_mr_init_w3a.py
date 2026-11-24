"""Coverage tests for ``ModelRunnerFL.__init__`` (model_runner.py lines ~576-1113).

The module is not imported. ``__init__`` is lifted out of the class with ``ast``
(stripping the ``JitWarmupRegistry.capture`` decorator) and executed with a
hand-built namespace of light stubs.  It is compiled under the real source path
so coverage lands on ``vllm_fl/worker/model_runner.py``.

Every branch is reached with ``object``-free lightweight dummy runners; no GPU,
weights or real vLLM config objects are involved.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


def _source_path() -> Path:
    spec = importlib.util.find_spec("vllm_fl")
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parent / "worker" / "model_runner.py"
    return Path.cwd() / "vllm_fl" / "worker" / "model_runner.py"


_SRC = _source_path()


def _load_init(namespace):
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunnerFL"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    # Drop @JitWarmupRegistry.capture: irrelevant to the body, avoids registry use.
    fn.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, fn], type_ignores=[])
    )
    exec(compile(module, str(_SRC), "exec"), namespace)
    return namespace["__init__"]


class DummyRunner:
    """Stands in for ``object.__new__(ModelRunnerFL)``."""

    def __init__(self):
        self.buffers = {}
        self.init_device_properties_called = False

    def _make_buffer(self, *size, dtype=None, numpy=True, **kwargs):
        buf = SimpleNamespace(size=size, dtype=dtype, numpy=numpy)
        self.buffers[str(size)] = buf
        return buf

    def _init_device_properties(self):
        self.init_device_properties_called = True


class _StubClass:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _model_config(**overrides):
    mc = SimpleNamespace(
        dtype="float16",
        is_moe=False,
        runner_type="generate",
        enable_prompt_embeds=False,
        is_multimodal_raw_input_only_model=False,
        max_model_len=2048,
        uses_alibi=False,
        disable_cascade_attn=False,
        is_mm_prefix_lm=False,
        uses_mrope=False,
        supports_multimodal_inputs=False,
        is_encoder_decoder=False,
        logprobs_mode="raw_logprobs",
        use_fp64_gumbel=False,
        logits_processors=None,
        get_num_attention_heads=lambda parallel_config: 8,
        get_inputs_embeds_size=lambda: 16,
        get_vocab_size=lambda: 100,
    )
    mc.__dict__.update(overrides)
    return mc


class _CacheConfig:
    DEFAULT_BLOCK_SIZE = 16


_SPEC_FLAGS = (
    "uses_draft_model",
    "use_ngram_gpu",
    "use_gemma4_mtp",
    "use_step3p5_mtp",
    "use_dflash",
    "use_eagle",
)


def _spec(method="medusa", *, num=2, draft=None, **flags):
    spec = SimpleNamespace(
        method=method, num_speculative_tokens=num, draft_model_config=draft
    )
    for flag in _SPEC_FLAGS:
        setattr(spec, flag, (lambda v: (lambda *a, **k: v))(flags.get(flag, False)))
    return spec


def _make_namespace():
    calls = SimpleNamespace(offloader=[], proposals=[])
    device_fn = SimpleNamespace(
        Stream=lambda *a, **k: SimpleNamespace(kind="stream", kwargs=k),
        Event=lambda *a, **k: SimpleNamespace(kind="event", kwargs=k),
    )
    ns = {
        "torch": MagicMock(),
        "np": np,
        "threading": __import__("threading"),
        "current_platform": SimpleNamespace(
            vendor_name="musa", torch_device_fn=device_fn
        ),
        "CacheConfig": _CacheConfig,
        "cdiv": lambda a, b: -(-a // b),
        "SlotMappingMode": SimpleNamespace(TOKEN_TO_KV_SLOT=0),
        "PIN_MEMORY": False,
        "CUDAGraphMode": SimpleNamespace(NONE="NONE"),
        "MULTIMODAL_REGISTRY": SimpleNamespace(
            supports_multimodal_inputs=lambda mc: True
        ),
        "Sampler": _StubClass,
        "LateInteractionRunner": lambda *a, **k: SimpleNamespace(kind="late"),
        "InputBatch": lambda *a, **k: SimpleNamespace(kind="input_batch", kwargs=k),
        "build_logitsprocs": lambda *a, **k: ["procs"],
        "CudagraphDispatcher": _StubClass,
        "MultiModalBudget": _StubClass,
        "set_offloader": lambda off: calls.offloader.append(off),
        "create_offloader": lambda cfg: SimpleNamespace(cfg=cfg, kind="offloader"),
        "get_ep_all2all_manager": lambda: SimpleNamespace(
            support_fault_tolerance=True,
            query_fault=lambda: None,
            query_active_mask=lambda: None,
        ),
        "kv_cache_dtype_str_to_dtype": lambda cache_dtype, mc: "float16",
        "get_dcp_group": lambda: SimpleNamespace(rank_in_group=3),
        "get_pp_group": lambda: SimpleNamespace(is_last_rank=True, ranks=[0, 1]),
        "create_custom_proposer": lambda cfg: SimpleNamespace(
            kind="custom", cfg=cfg
        ),
        "NgramProposer": _StubClass,
        "NgramProposerGPU": _StubClass,
        "SuffixDecodingProposer": _StubClass,
        "EagleProposer": _StubClass,
        "DFlashProposer": _StubClass,
        "DraftModelProposer": _StubClass,
        "MedusaProposer": _StubClass,
        "ExtractHiddenStatesProposer": _StubClass,
        "Gemma4Proposer": _StubClass,
        "Step3p5MTPProposer": _StubClass,
        "RejectionSampler": _StubClass,
        "GraphWrapper": object,
        "_calls": calls,
    }
    ns["__init__"] = _load_init(ns)
    return ns


def _vllm_config(
    model=None,
    cache=None,
    scheduler=None,
    parallel=None,
    compilation=None,
    speculative=None,
):
    model = model if model is not None else _model_config()
    cache = (
        cache
        if cache is not None
        else SimpleNamespace(
            cache_dtype="auto",
            calculate_kv_scales=False,
            block_size=16,
            use_replayssm=False,
            kv_sharing_fast_prefill=False,
        )
    )
    scheduler = (
        scheduler
        if scheduler is not None
        else SimpleNamespace(
            max_num_batched_tokens=8192,
            max_num_seqs=8,
            max_num_encoder_input_tokens=1024,
            async_scheduling=False,
        )
    )
    parallel = (
        parallel
        if parallel is not None
        else SimpleNamespace(
            data_parallel_size=1,
            decode_context_parallel_size=1,
            distributed_executor_backend="mp",
            cp_kv_cache_interleave_size=16,
        )
    )
    compilation = (
        compilation
        if compilation is not None
        else SimpleNamespace(
            cudagraph_capture_sizes=None,
            cudagraph_mode="NONE",
            static_forward_context={},
        )
    )
    return SimpleNamespace(
        model_config=model,
        cache_config=cache,
        offload_config=SimpleNamespace(),
        compilation_config=compilation,
        lora_config=SimpleNamespace(),
        load_config=SimpleNamespace(),
        parallel_config=parallel,
        scheduler_config=scheduler,
        speculative_config=speculative,
        observability_config=SimpleNamespace(),
        reasoning_config=SimpleNamespace(),
    )


def _run(ns, vllm_config=None):
    runner = DummyRunner()
    ns["__init__"](runner, vllm_config or _vllm_config(), "cpu-device")
    return runner


# --------------------------------------------------------------------------- #
def test_init_minimal_defaults():
    ns = _make_namespace()
    r = _run(ns)
    assert r.vllm_config is not None
    assert r.device == "cpu-device"
    assert r.dtype == "float16"
    assert r.is_pooling_model is False
    assert r.enable_prompt_embeds is False
    assert r.is_multimodal_raw_input_only_model is False
    assert r.is_multimodal_pruning_enabled is False
    assert r.requires_sequential_video_encoding is False
    assert r.routed_experts_initialized is False
    assert r.max_model_len == 2048
    assert r.calculate_kv_scales is False
    assert r.dcp_world_size == 1
    assert r.dcp_rank == 0
    assert r.max_num_tokens == 8192
    assert r.max_num_reqs == 8
    assert r.broadcast_pp_output is False
    assert r.check_ep_fault is False
    assert r.num_query_heads == 8
    assert r.inputs_embeds_size == 16
    assert r.use_alibi is False
    assert r.cascade_attn_enabled is True
    assert r.is_mm_prefix_lm is False
    assert r.uses_mrope is False
    assert r.supports_mm_inputs is False
    assert r.max_encoder_len == 0
    assert r.use_async_scheduling is False
    assert r._pp_recv_work is None
    assert r.num_spec_tokens == 0
    assert r.prev_num_spec_tokens == 0
    assert r.valid_sampled_token_count_gpu is None
    assert r.use_async_spec_decode is False
    assert r.requests == {}
    assert r.num_prompt_logprobs == {}
    assert r.async_output_copy_stream is None
    assert r.prepare_inputs_event is None
    assert r.cudagraph_batch_sizes == []
    assert r.init_device_properties_called is True
    assert r.encoder_timing_registry == {}
    assert r._init_block_sizes == [16]
    assert r._init_kernel_block_sizes == [16]
    assert r._init_max_num_blocks == [128]
    assert r._init_slot_mapping_modes == [0]
    assert r.use_ascend_attention is False
    assert r.use_ascend_hybrid_cache is False
    assert r.graph_runtime is None
    assert r._ascend_attn_state is None
    assert r.uniform_decode_query_len == 1
    assert r.kv_sharing_fast_prefill_logits_indices is None
    assert r.mm_budget is None
    assert r.reorder_batch_threshold is None
    assert r._draft_token_ids is None
    assert r.execute_model_state is None
    assert r.kv_connector_output is None
    assert r.mamba_state_idx == {}
    assert r._mamba_bufs is None
    assert r.layerwise_nvtx_hooks_registered is False
    assert r.sampler is not None
    assert r.cudagraph_dispatcher is not None
    assert len(ns["_calls"].offloader) == 1


def test_init_pooling_model_and_prompt_embeds():
    ns = _make_namespace()
    mc = _model_config(runner_type="pooling", enable_prompt_embeds=True)
    r = _run(ns, _vllm_config(model=mc))
    assert r.is_pooling_model is True
    assert r.enable_prompt_embeds is True


def test_init_ascend_branch(monkeypatch):
    ns = _make_namespace()
    ns["current_platform"].vendor_name = "ascend"
    created = []

    class FakeController:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.kv_cache.ascend.hybrid",
        SimpleNamespace(uses_dense_hybrid_cache=lambda mc: True),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.compilation.graph_runtime",
        SimpleNamespace(GraphRuntimeController=FakeController),
    )
    r = _run(ns)
    assert r.use_ascend_attention is True
    assert r.use_ascend_hybrid_cache is True
    assert isinstance(r.graph_runtime, FakeController)
    assert created[0]["full_graph_wrapper_type"] is object


def test_init_ep_fault_when_dp_and_moe():
    ns = _make_namespace()
    mc = _model_config(is_moe=True)
    parallel = SimpleNamespace(
        data_parallel_size=2,
        decode_context_parallel_size=1,
        distributed_executor_backend="mp",
        cp_kv_cache_interleave_size=16,
    )
    r = _run(ns, _vllm_config(model=mc, parallel=parallel))
    assert r.check_ep_fault is True


def test_init_dcp_gt_one():
    ns = _make_namespace()
    parallel = SimpleNamespace(
        data_parallel_size=1,
        decode_context_parallel_size=4,
        distributed_executor_backend="mp",
        cp_kv_cache_interleave_size=16,
    )
    r = _run(ns, _vllm_config(parallel=parallel))
    assert r.dcp_world_size == 4
    assert r.dcp_rank == 3
    assert "dcp_local_seq_lens" in r.__dict__


def test_init_broadcast_pp_output():
    ns = _make_namespace()
    parallel = SimpleNamespace(
        data_parallel_size=1,
        decode_context_parallel_size=1,
        distributed_executor_backend="external_launcher",
        cp_kv_cache_interleave_size=16,
    )
    ns["get_pp_group"] = lambda: SimpleNamespace(is_last_rank=True, ranks=[0, 1])
    r = _run(ns, _vllm_config(parallel=parallel))
    assert r.broadcast_pp_output is True


def test_init_supports_mm_fallback_to_registry():
    ns = _make_namespace()
    mc = _model_config()
    del mc.supports_multimodal_inputs
    r = _run(ns, _vllm_config(model=mc))
    assert r.supports_mm_inputs is True
    assert r.mm_budget is not None


def test_init_encoder_decoder_sets_encoder_len():
    ns = _make_namespace()
    mc = _model_config(is_encoder_decoder=True)
    r = _run(ns, _vllm_config(model=mc))
    assert r.max_encoder_len == 1024


def test_init_block_size_defaults_when_none():
    ns = _make_namespace()
    cache = SimpleNamespace(
        cache_dtype="auto",
        calculate_kv_scales=False,
        block_size=None,
        use_replayssm=False,
        kv_sharing_fast_prefill=False,
    )
    r = _run(ns, _vllm_config(cache=cache))
    assert r._init_block_sizes == [16]
    assert r._init_max_num_blocks == [128]


def test_init_block_size_custom():
    ns = _make_namespace()
    cache = SimpleNamespace(
        cache_dtype="auto",
        calculate_kv_scales=False,
        block_size=32,
        use_replayssm=False,
        kv_sharing_fast_prefill=False,
    )
    r = _run(ns, _vllm_config(cache=cache))
    assert r._init_block_sizes == [32]
    assert r._init_max_num_blocks == [64]


def test_init_async_scheduling_allocates_streams():
    ns = _make_namespace()
    scheduler = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=8,
        max_num_encoder_input_tokens=1024,
        async_scheduling=True,
    )
    r = _run(ns, _vllm_config(scheduler=scheduler))
    assert r.use_async_scheduling is True
    assert r.async_output_copy_stream is not None
    assert r.prepare_inputs_event is not None


def test_init_cudagraph_batch_sizes_sorted():
    ns = _make_namespace()
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=[8, 2, 4],
        cudagraph_mode="FULL",
        static_forward_context={},
    )
    r = _run(ns, _vllm_config(compilation=compilation))
    assert r.cudagraph_batch_sizes == [2, 4, 8]


def test_init_cudagraph_ignored_when_mode_none():
    ns = _make_namespace()
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=[8, 2],
        cudagraph_mode="NONE",
        static_forward_context={},
    )
    r = _run(ns, _vllm_config(compilation=compilation))
    assert r.cudagraph_batch_sizes == []


def test_init_uses_mrope_allocates_positions():
    ns = _make_namespace()
    mc = _model_config(uses_mrope=True)
    r = _run(ns, _vllm_config(model=mc))
    assert r.uses_mrope is True
    assert "mrope_positions" in r.__dict__


def test_init_kv_sharing_fast_prefill_allocates_indices():
    ns = _make_namespace()
    cache = SimpleNamespace(
        cache_dtype="auto",
        calculate_kv_scales=False,
        block_size=16,
        use_replayssm=False,
        kv_sharing_fast_prefill=True,
    )
    r = _run(ns, _vllm_config(cache=cache))
    assert r.kv_sharing_fast_prefill_logits_indices is not None


def test_init_spec_not_last_rank_skips_drafter():
    ns = _make_namespace()
    ns["get_pp_group"] = lambda: SimpleNamespace(is_last_rank=False, ranks=[0])
    r = _run(ns, _vllm_config(speculative=_spec("medusa", num=3)))
    assert not hasattr(r, "drafter")
    assert r.num_spec_tokens == 3
    assert r.prev_num_spec_tokens == 3
    assert r.effective_drafter_max_model_len == 2048
    assert r.use_async_spec_decode is False


def test_init_spec_custom_class():
    ns = _make_namespace()
    r = _run(ns, _vllm_config(speculative=_spec("custom_class")))
    assert r.drafter.kind == "custom"
    assert r.rejection_sampler is not None


def test_init_spec_medusa():
    ns = _make_namespace()
    r = _run(ns, _vllm_config(speculative=_spec("medusa")))
    assert isinstance(r.drafter, _StubClass)
    assert r.drafter.kwargs["vllm_config"] is not None


def test_init_spec_extract_hidden_states_enables_aux():
    ns = _make_namespace()
    r = _run(ns, _vllm_config(speculative=_spec("extract_hidden_states")))
    assert r.use_aux_hidden_state_outputs is True


def test_init_spec_eagle3_uses_aux_state():
    ns = _make_namespace()

    class Eagle(_StubClass):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.eagle3_use_aux_hidden_state = True

    ns["EagleProposer"] = Eagle
    r = _run(ns, _vllm_config(speculative=_spec("eagle3", use_eagle=True)))
    assert r.use_aux_hidden_state_outputs is True


def test_init_spec_unknown_method_raises():
    ns = _make_namespace()
    with pytest.raises(ValueError, match="Unknown speculative decoding method"):
        _run(ns, _vllm_config(speculative=_spec("bogus")))


def test_init_spec_draft_config_max_len():
    ns = _make_namespace()
    spec = _spec("medusa", draft=SimpleNamespace(max_model_len=4096))
    r = _run(ns, _vllm_config(speculative=spec))
    assert r.effective_drafter_max_model_len == 4096


def test_init_spec_ngram_gpu_allocates_buffers():
    ns = _make_namespace()
    ns["get_pp_group"] = lambda: SimpleNamespace(is_last_rank=False, ranks=[0])
    r = _run(ns, _vllm_config(speculative=_spec("medusa", use_ngram_gpu=True)))
    assert r._num_valid_draft_tokens_cpu is not None
    assert r._num_valid_draft_tokens_event is not None
    assert r._num_valid_draft_tokens_copy_stream is not None


def test_init_spec_async_scheduling_allocates_count_buffers():
    ns = _make_namespace()
    scheduler = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=8,
        max_num_encoder_input_tokens=1024,
        async_scheduling=True,
    )
    r = _run(
        ns, _vllm_config(scheduler=scheduler, speculative=_spec("medusa", num=2))
    )
    assert r.use_async_spec_decode is True
    assert r.valid_sampled_token_count_event is not None
    assert r.valid_sampled_token_count_copy_stream is not None
    assert r.valid_sampled_token_count_cpu is not None
    assert r.draft_token_ids_event is not None
    assert r.draft_token_ids_copy_stream is not None
    assert r.num_accepted_tokens_event is not None
    assert r.draft_token_ids_cpu is not None
