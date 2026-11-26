# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for the ModelRunnerFL execution-method family.

Targets the largest remaining uncovered regions of
``vllm_fl/worker/model_runner.py``: ``execute_model``, ``sample_tokens``,
``_bookkeeping_sync``, ``_preprocess``, ``_pool``,
``_get_prompt_logprobs_dict``, ``_get_slot_mappings``,
``_copy_draft_token_ids_to_cpu``,
``_pp_receive_prev_sampled_token_ids_to_input_batch``,
``_allow_microbatching``, ``_get_nans_in_logits`` and
``_get_spec_decode_draft_probs``.

The module is imported through the same CUDA-ness proxy fixture the
constructor sweep uses (see ``test_model_runner_deep_paths.py``), and the
runners are real ``ModelRunnerFL`` instances built via the raw
``__init__.__wrapped__`` entry point.  Only well-identified GPU boundaries
are replaced:

* the pipeline-parallel group lookup and the platform stream/event factory
  (CPU-only torch builds have neither);
* the sampler / rejection-sampler call itself (``vllm._C`` is not compiled
  on CPU hosts, so the real kernels cannot run);
* the model forward object (``self.model``), which is a plain recording
  stand-in returning real CPU hidden-state tensors;
* the torch.distributed broadcast in the PP async receive.

Everything else -- ``_update_states``, ``_prepare_inputs``, block tables,
slot mappings, attention metadata plumbing, bookkeeping, the grammar
bitmask application and all tensor math -- runs the production code on real
CPU tensors, and every numeric expectation is verified against an
independent plain-torch oracle.
"""

import contextlib
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

try:  # pragma: no cover - dual-environment guard
    import torch

    import vllm  # noqa: F401

    HAS_VLLM = True
except ImportError:  # pragma: no cover - Windows machines without vllm
    HAS_VLLM = False

pytestmark = [
    pytest.mark.skipif(not HAS_VLLM, reason="requires real vllm"),
]

CPU_DEVICE = "cpu"


class _FakeCudaPlatform:
    """Delegate to the real platform; fake only CUDA-ness for import checks."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    @property
    def device_type(self):
        return "cuda"

    @property
    def dispatch_key(self):
        return "CUDA"

    @property
    def vendor_name(self):
        return None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


@pytest.fixture
def model_runner_module(monkeypatch):
    """Import vllm_fl.worker.model_runner with CUDA-ness faked at import."""
    pytest.importorskip("vllm", reason="requires real vllm")
    platforms = import_module("vllm.platforms")
    real = platforms.current_platform
    monkeypatch.setattr(platforms, "current_platform", _FakeCudaPlatform(real))
    module = import_module("vllm_fl.worker.model_runner")
    yield module
    monkeypatch.setattr(platforms, "current_platform", real)


class _FakeEvent:
    """Minimal stand-in for a torch device Event (CPU torch cannot record)."""

    def __init__(self, blocking=False):
        self.blocking = blocking
        self.record_count = 0
        self.synchronized = False

    def record(self):
        self.record_count += 1

    def synchronize(self):
        self.synchronized = True


class _FakeStream:
    """Minimal stand-in for a torch device Stream."""

    def __init__(self):
        self.waited = []
        self.wait_event_count = 0

    def wait_stream(self, other):
        self.waited.append(other)

    def wait_event(self, event):
        self.wait_event_count += 1


@pytest.fixture
def patched_module(model_runner_module, monkeypatch):
    """Patch the GPU/distributed boundaries on the module under test.

    Compared with the deep-paths fixture, this fake platform also provides
    ``current_stream``, ``stream`` and ``is_cuda_alike``, which the
    execution methods (``_copy_draft_token_ids_to_cpu``, ``_pool``) require.
    """
    pp_group = SimpleNamespace(
        is_last_rank=True,
        is_first_rank=True,
        ranks=[0],
        world_size=1,
        rank=0,
        last_rank=0,
        device_group=None,
    )
    fake_platform = SimpleNamespace(
        torch_device_fn=SimpleNamespace(
            Stream=_FakeStream,
            Event=_FakeEvent,
            # _copy_draft_token_ids_to_cpu and _copy_valid_sampled_token_count
            # fetch the default stream from the platform's device_fn.
            current_stream=lambda: _FakeStream(),
            # ... and enter torch_device_fn.stream(...).
            stream=lambda stream: contextlib.nullcontext(),
        ),
        current_stream=lambda: _FakeStream(),
        is_cuda_alike=lambda: False,
        # _kl_guard_on reads vendor_name in _prepare_inputs.
        vendor_name=None,
    )
    monkeypatch.setattr(model_runner_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(model_runner_module, "current_platform", fake_platform)
    monkeypatch.setattr(model_runner_module, "num_compute_units", lambda device_id: 108)
    return model_runner_module


@pytest.fixture
def pp_group(monkeypatch, patched_module):
    """Expose (and return) a fresh fake pipeline group for per-test overrides."""
    group = SimpleNamespace(
        is_last_rank=True,
        is_first_rank=True,
        ranks=[0],
        world_size=1,
        rank=0,
        last_rank=0,
        device_group=None,
    )
    monkeypatch.setattr(patched_module, "get_pp_group", lambda: group)
    return group


# ---------------------------------------------------------------------------
# CPU-safe slot-mapping kernel (pure-torch oracle for the triton kernel).
# ---------------------------------------------------------------------------


class _CpuSlotMappingKernel:
    """Pure-torch stand-in for the triton slot-mapping kernel boundary."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __call__(
        self,
        num_reqs,
        num_tokens,
        max_num_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table_stride,
        slot_mapping,
        kv_cache_block_size,
        total_cp_world_size,
        total_cp_rank,
        cp_kv_cache_interleave_size,
    ):
        pad_id = -1
        for i in range(num_tokens, max_num_tokens):
            slot_mapping[i] = pad_id
        flat_block_table = block_table.reshape(-1)
        for req_idx in range(num_reqs):
            start = int(query_start_loc[req_idx])
            end = int(query_start_loc[req_idx + 1])
            for i in range(start, end):
                pos = int(positions[i])
                vbi = pos // kv_cache_block_size
                vbo = pos - vbi * kv_cache_block_size
                bn = int(flat_block_table[req_idx * block_table_stride + vbi])
                slot_mapping[i] = bn * kv_cache_block_size + vbo


@pytest.fixture
def cpu_slot_mapping_kernel(monkeypatch):
    """Replace the triton slot-mapping kernel with a pure-torch oracle."""
    block_table_module = import_module("vllm.v1.worker.block_table")
    real = block_table_module._COMPUTE_SLOT_MAPPING_KERNEL
    proxy = _CpuSlotMappingKernel(real)
    monkeypatch.setattr(block_table_module, "_COMPUTE_SLOT_MAPPING_KERNEL", proxy)
    return proxy


# ---------------------------------------------------------------------------
# Config namespaces (extended from the deep-paths sweep).
# ---------------------------------------------------------------------------


def _make_model_config(**overrides):
    """A ModelConfig-shaped namespace with the fields the runner reads."""
    config = SimpleNamespace(
        model="facebook/opt-125m",
        hf_text_config=SimpleNamespace(sliding_window=None),
        hf_config=SimpleNamespace(model_type="opt"),
        dtype=torch.float32,
        runner_type="generate",
        enable_prompt_embeds=False,
        is_multimodal_raw_input_only_model=False,
        is_moe=False,
        max_model_len=64,
        uses_alibi=False,
        disable_cascade_attn=True,
        is_mm_prefix_lm=False,
        uses_mrope=False,
        supports_multimodal_inputs=False,
        is_encoder_decoder=False,
        is_hybrid=False,
        logprobs_mode="processed_logprobs",
        use_fp64_gumbel=False,
        logits_processors=None,
        multimodal_config=None,
        quantization=None,
        get_num_attention_heads=lambda parallel_config: 8,
        get_inputs_embeds_size=lambda: 4,
        get_vocab_size=lambda: 32,
        rswa_window=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_spec_config(method, num_speculative_tokens=2, **overrides):
    """A SpeculativeConfig-shaped namespace with all dispatch predicates."""
    spec = SimpleNamespace(
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        uses_draft_model=lambda: False,
        use_ngram_gpu=lambda: False,
        use_gemma4_mtp=lambda: False,
        use_step3p5_mtp=lambda: False,
        use_dflash=lambda: False,
        use_eagle=lambda: False,
        uses_extract_hidden_states=lambda: False,
        disable_padded_drafter_batch=False,
        rejection_sample_method="rejection_sampler",
        draft_sample_method="greedy",
        draft_model_config=None,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def _make_vllm_config(**overrides):
    """Assemble the vllm_config namespace the runner reads."""
    config_mod = import_module("vllm.config")
    model_config = overrides.pop("model_config", None) or _make_model_config()
    speculative = overrides.pop("speculative_config", None)
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            block_size=16,
            calculate_kv_scales=False,
            mamba_cache_mode=None,
            use_replayssm=False,
            kv_sharing_fast_prefill=False,
        ),
        offload_config=config_mod.OffloadConfig(),
        compilation_config=SimpleNamespace(
            cudagraph_mode=config_mod.CUDAGraphMode.NONE,
            cudagraph_capture_sizes=None,
            max_cudagraph_capture_size=None,
            mode=config_mod.CompilationMode.NONE,
            static_forward_context={},
            fast_moe_cold_start=False,
            static_all_moe_layers=[],
        ),
        lora_config=None,
        load_config=SimpleNamespace(load_format="dummy", use_tqdm_on_load=False),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            decode_context_parallel_size=1,
            distributed_executor_backend="mp",
            cp_kv_cache_interleave_size=1,
            use_ubatching=False,
            num_ubatches=1,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
            is_moe_model=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_num_encoder_input_tokens=0,
            async_scheduling=False,
        ),
        speculative_config=speculative,
        # cudagraph_metrics / enable_layerwise_nvtx_tracing are read by
        # _determine_batch_execution_and_padding on every execute_model.
        observability_config=SimpleNamespace(
            report_metrics=False,
            cudagraph_metrics=False,
            enable_layerwise_nvtx_tracing=False,
        ),
        reasoning_config=None,
        num_speculative_tokens=(
            speculative.num_speculative_tokens if speculative else 0
        ),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _build_runner(module, vllm_config):
    """Instantiate ModelRunnerFL with the raw (unwrapped) constructor."""
    ModelRunnerFL = module.ModelRunnerFL
    runner = object.__new__(ModelRunnerFL)
    ModelRunnerFL.__init__.__wrapped__(runner, vllm_config, torch.device(CPU_DEVICE))
    # torch.Event objects exist on CPU builds but cannot be recorded; the
    # runner's sync/async copy events sit exactly on that GPU boundary.
    runner.transfer_event = _FakeEvent()
    if runner.num_accepted_tokens_event is not None:
        runner.num_accepted_tokens_event = _FakeEvent()
    if runner.draft_token_ids_event is not None:
        runner.draft_token_ids_event = _FakeEvent()
    if runner.valid_sampled_token_count_event is not None:
        runner.valid_sampled_token_count_event = _FakeEvent()
    return runner


# ---------------------------------------------------------------------------
# Recording model + sampler standing in for the GPU forward/sampling edge.
# ---------------------------------------------------------------------------


class _RecordingModel:
    """A model stand-in whose forward/compute_logits run real CPU math.

    The forward maps each token id to a deterministic hidden-state row so
    downstream gathers and logits indexing can be verified against a
    plain-torch oracle, and records every keyword it was called with so
    tests can assert the exact arguments the production code handed across
    the model boundary.
    """

    def __init__(self, vocab_size=32, hidden=8):
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.calls = []
        self.logits_calls = []
        # Deterministic embedding: row i is one-hot(i) tiled over the hidden
        # dimension, so embed[i, j] == float(i).
        self.embed = torch.arange(vocab_size, dtype=torch.float32)
        self.embed = self.embed[:, None].expand(vocab_size, hidden).clone()
        self.logits_head = torch.arange(1, vocab_size + 1, dtype=torch.float32)

    def embed_input_ids(self, input_ids, **kwargs):
        """Real-torch embedding lookup used by the prompt-embeds path."""
        return self.embed[input_ids.long()]

    def __call__(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self.calls.append(
            SimpleNamespace(
                input_ids=None if input_ids is None else input_ids.clone(),
                positions=None if positions is None else positions.clone(),
                inputs_embeds=(
                    None if inputs_embeds is None else inputs_embeds.clone()
                ),
                intermediate_tensors=intermediate_tensors,
                kwargs=dict(kwargs),
            )
        )
        if inputs_embeds is not None:
            base = inputs_embeds.clone()
        else:
            assert input_ids is not None
            base = self.embed[input_ids.long()]
        # Positions fold in so gather indices are observable downstream.
        if positions is not None:
            base = base * (positions.to(torch.float32).unsqueeze(-1) + 1.0)
        return base

    def compute_logits(self, hidden_states):
        self.logits_calls.append(hidden_states)
        # Row-sum projection to vocab: fully reproducible in the oracle.
        return hidden_states.sum(dim=-1, keepdim=True) * self.logits_head

    def named_parameters(self):
        return iter([])


class _RecordingSampler:
    """Sampler stand-in: greedy argmax over the real logits on CPU.

    The production ``Sampler.forward`` requires ``torch.ops._C`` kernels
    that are not compiled on CPU-only hosts, so this stand-in performs the
    same greedy-argmax semantics with plain torch and records its inputs.
    """

    def __init__(self, logprobs_mode="processed_logprobs"):
        self.logprobs_mode = logprobs_mode
        self.calls = []

    def __call__(self, logits, sampling_metadata, **kwargs):
        from vllm.v1.outputs import SamplerOutput

        self.calls.append(
            SimpleNamespace(
                logits=None if logits is None else logits.clone(),
                sampling_metadata=sampling_metadata,
                kwargs=dict(kwargs),
            )
        )
        assert logits is not None
        sampled = logits.argmax(dim=-1)
        # [num_reqs, 1] shaped, matching the production sampler contract.
        token_ids = sampled.reshape(-1, 1).to(torch.int64)
        return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

    @staticmethod
    def compute_logprobs(logits):
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(logprobs, num_logprobs, token_ids):
        """Mirrors the production 4-tuple unpack in model_runner L5979."""
        assert token_ids.dtype == torch.int64
        topk_lp, topk_idx = torch.topk(logprobs, num_logprobs, dim=-1)
        token_ids = token_ids.unsqueeze(-1)
        token_lp = logprobs.gather(-1, token_ids)
        ranks = (logprobs > token_lp).sum(dim=-1).to(torch.int32)
        indices = torch.cat((token_ids, topk_idx), dim=1).to(torch.int32)
        lps = torch.cat((token_lp, topk_lp), dim=1)
        # The production code unpacks 4 values (token_ids, logprobs, ranks, _).
        return indices, lps, ranks, None


class _RecordingRejectionSampler:
    """Rejection-sampler stand-in accepting every draft token.

    Implements the production contract (``[num_reqs, spec+1]`` output with
    rejected slots as -1) with a deterministic policy, using only plain
    torch ops so results can be oracle-checked.
    """

    def __init__(self, policy="accept_all"):
        self.policy = policy
        self.calls = []

    def __call__(self, metadata, draft_probs, logits, sampling_metadata):
        from vllm.v1.outputs import SamplerOutput

        self.calls.append(
            SimpleNamespace(
                metadata=metadata,
                draft_probs=draft_probs,
                logits=None if logits is None else logits.clone(),
                sampling_metadata=sampling_metadata,
            )
        )
        assert logits is not None
        num_reqs = logits.shape[0]
        num_draft = max(metadata.num_draft_tokens) if metadata.num_draft_tokens else 0
        # Greedy bonus token per request.
        sampled = logits.argmax(dim=-1)
        out = torch.full((num_reqs, num_draft + 1), -1, dtype=torch.int64)
        out[:, 0] = sampled[:num_reqs]
        if self.policy == "accept_all":
            draft_ids = getattr(metadata, "draft_token_ids", None)
            if torch.is_tensor(draft_ids):
                out[:, 1 : num_draft + 1] = draft_ids[:num_reqs, :num_draft]
        return SamplerOutput(sampled_token_ids=out, logprobs_tensors=None)


class _FakeSuffixProposer:
    """A SuffixDecodingProposer CLASS stand-in.

    The real suffix proposer lazy-imports ``arctic_inference`` in its
    constructor, which is absent on CPU-only venvs, so every spec-configured
    runner must patch the module-level class before construction.  This must
    be a real class (not a lambda): ``propose_draft_token_ids`` runs
    ``assert isinstance(self.drafter, SuffixDecodingProposer)`` and
    ``isinstance`` against a function raises TypeError.
    """

    instances = []

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.proposals = []
        self.kv_cache_gid = 0
        _FakeSuffixProposer.instances.append(self)

    def propose(self, num_tokens, input_batch, sampled, slot_mappings=None):
        self.proposals.append((num_tokens, input_batch, sampled, slot_mappings))
        # Deterministic drafts: one row per sampled request.
        return [[10 + 2 * i, 11 + 2 * i] for i in range(len(sampled))]

    def dummy_run(self, num_tokens=1):
        self.proposals.append(("dummy_run", num_tokens))

    @staticmethod
    def reset():
        _FakeSuffixProposer.instances = []


# ---------------------------------------------------------------------------
# Real InputBatch / SchedulerOutput builders.
# ---------------------------------------------------------------------------


def _make_input_batch(patched_module, *, num_groups=1, is_pooling_model=False):
    """Build a real InputBatch sized for these tests."""
    InputBatch = import_module("vllm.v1.worker.gpu_input_batch").InputBatch
    return InputBatch(
        max_num_reqs=8,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device(CPU_DEVICE),
        vocab_size=32,
        block_sizes=[16] * num_groups,
        max_num_blocks_per_req=[4] * num_groups,
        is_pooling_model=is_pooling_model,
    )


def _make_req_state(patched_module, req_id, prompt, block, **overrides):
    """Build a real CachedRequestState with greedy sampling params."""
    SamplingParams = import_module("vllm.sampling_params").SamplingParams
    SamplingType = import_module("vllm.sampling_params").SamplingType
    CachedRequestState = import_module(
        "vllm.v1.worker.gpu_input_batch"
    ).CachedRequestState

    params = SamplingParams()
    params.sampling_type = SamplingType.GREEDY
    state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=params,
        generator=None,
        block_ids=(list(block),),
        num_computed_tokens=0,
        output_token_ids=[],
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _pooling_req_state(patched_module, req_id, prompt, block, task="embed"):
    """Build a real CachedRequestState for a pooling request."""
    PoolingParams = import_module("vllm.pooling_params").PoolingParams
    CachedRequestState = import_module(
        "vllm.v1.worker.gpu_input_batch"
    ).CachedRequestState
    params = PoolingParams(task=task)
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=None,
        pooling_params=params,
        generator=None,
        block_ids=(list(block),),
        num_computed_tokens=0,
        output_token_ids=[],
    )


def _scheduler_output(
    patched_module,
    *,
    num_scheduled_tokens=None,
    total_num_scheduled_tokens=0,
    spec_tokens=None,
    **field_overrides,
):
    """Build a real SchedulerOutput dataclass with the given contents."""
    CachedRequestData = import_module("vllm.v1.core.sched.output").CachedRequestData
    SchedulerOutput = import_module("vllm.v1.core.sched.output").SchedulerOutput
    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens=dict(num_scheduled_tokens or {}),
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=dict(spec_tokens or {}),
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    for key, value in field_overrides.items():
        setattr(output, key, value)
    return output


# ---------------------------------------------------------------------------
# Attention-group wiring for execute_model: recording metadata builders.
# ---------------------------------------------------------------------------


class _RecordingBuilder:
    """A minimal AttentionMetadataBuilder stand-in recording build calls."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.builds = []
        self.capture_builds = []
        self.updates = []

    supports_update_block_table = False

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.builds.append((common_prefix_len, common_attn_metadata, kwargs))
        return {"cm": common_attn_metadata, "prefix_len": common_prefix_len}

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self.capture_builds.append(common_attn_metadata)
        return {"cm": common_attn_metadata, "capture": True}

    def update_block_table(self, metadata, blk_table, slot_mapping):
        self.updates.append((metadata, blk_table, slot_mapping))
        return {"cm": metadata["cm"], "updated": True}


class _FullAttnGroupBackend:
    """Attention backend class standing in for the real backend objects.

    ``forward_includes_kv_cache_update`` is the only attribute the
    production ``execute_model`` loop reads from the group backend.
    """

    forward_includes_kv_cache_update = True

    @classmethod
    def full_cls_name(cls):
        return "tests.FakeFullAttnBackend"


def _full_spec():
    """A real FullAttentionSpec sized for the CPU test batch."""
    FullAttentionSpec = import_module("vllm.v1.kv_cache_interface").FullAttentionSpec
    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
    )


def _wire_attn_groups(patched_module, runner, layer_names=("model.layers.0",)):
    """Attach a real KVCacheConfig + AttentionGroup graph to the runner.

    Returns the list of recording builders so tests can assert what the
    production code passed down to each attention group.
    """
    AttentionGroup = import_module("vllm.v1.worker.utils").AttentionGroup
    kv_interface = import_module("vllm.v1.kv_cache_interface")

    layer_names = list(layer_names)
    builders = [
        _RecordingBuilder(_full_spec(), [name], None, runner.device)
        for name in layer_names
    ]
    group = AttentionGroup(
        backend=_FullAttnGroupBackend,
        layer_names=layer_names,
        kv_cache_spec=_full_spec(),
        kv_cache_group_id=0,
    )
    group.metadata_builders = builders
    runner.attn_groups = [[group]]
    runner.kv_cache_config = kv_interface.KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            kv_interface.KVCacheGroupSpec(
                layer_names=layer_names, kv_cache_spec=_full_spec()
            )
        ],
    )
    return builders


def _prepare_execution_runner(
    patched_module,
    *,
    prompts=(("r1", (5, 6, 7)), ("r2", (8, 9))),
    spec_config=None,
    model_config=None,
    async_sched=False,
    suffix_proposer_cls=None,
    **runner_overrides,
):
    """Build a real runner wired for a full execute_model/sample_tokens cycle.

    Returns ``(runner, scheduler_output, model)``.  The scheduler output
    schedules every prompt token as a fresh prefill of a new request (via
    ``scheduled_new_reqs``), so ``_update_states`` runs its real
    new-request registration path before the forward.

    ``suffix_proposer_cls`` patches the module-level
    ``SuffixDecodingProposer`` class before the runner is constructed (the
    real one imports ``arctic_inference`` lazily, unavailable on CPU venvs);
    the constructed instance is returned as the fourth element of
    ``runner``'s ``drafter`` attribute and may be inspected by tests.
    """
    NewRequestData = import_module("vllm.v1.core.sched.output").NewRequestData
    SamplingParams = import_module("vllm.sampling_params").SamplingParams
    SamplingType = import_module("vllm.sampling_params").SamplingType

    original_suffix_cls = None
    if suffix_proposer_cls is not None:
        assert spec_config is not None and spec_config.method == "suffix"
        original_suffix_cls = patched_module.SuffixDecodingProposer
        patched_module.SuffixDecodingProposer = suffix_proposer_cls
    try:
        config = _make_vllm_config(
            speculative_config=spec_config,
            model_config=model_config or _make_model_config(),
        )
        config.scheduler_config.async_scheduling = async_sched
        runner = _build_runner(patched_module, config)
    finally:
        if original_suffix_cls is not None:
            patched_module.SuffixDecodingProposer = original_suffix_cls

    new_reqs = []
    scheduled = {}
    for req_id, prompt in prompts:
        params = SamplingParams()
        params.sampling_type = SamplingType.GREEDY
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=list(prompt),
                mm_features=[],
                sampling_params=params,
                pooling_params=None,
                block_ids=([9],),
                num_computed_tokens=0,
                lora_request=None,
            )
        )
        scheduled[req_id] = len(prompt)

    total = sum(scheduled.values())
    sched = _scheduler_output(
        patched_module,
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=total,
    )
    sched.scheduled_new_reqs = new_reqs

    model = _RecordingModel()
    runner.model = model
    runner.sampler = _RecordingSampler(logprobs_mode=config.model_config.logprobs_mode)
    runner.rejection_sampler = _RecordingRejectionSampler()
    # _may_reorder_batch (inside _update_states) reads kv_cache_config even
    # on zero-token steps; attach an empty-group default. Tests driving a
    # real forward overwrite it via _wire_attn_groups.
    kv_interface = import_module("vllm.v1.kv_cache_interface")
    runner.kv_cache_config = kv_interface.KVCacheConfig(
        num_blocks=8, kv_cache_tensors=[], kv_cache_groups=[]
    )
    runner.attn_groups = []

    for key, value in runner_overrides.items():
        setattr(runner, key, value)
    return runner, sched, model


# ---------------------------------------------------------------------------
# execute_model: early exits and state-machine guards.
# ---------------------------------------------------------------------------


class TestExecuteModelEarlyExits:
    """execute_model guard branches reachable without a GPU forward."""

    def test_state_error_when_execute_model_state_pending(self, patched_module):
        """A second execute_model before sample_tokens raises RuntimeError."""
        runner, sched, _ = _prepare_execution_runner(patched_module)
        runner.execute_model_state = SimpleNamespace(marker="pending")
        with pytest.raises(RuntimeError, match="sample_tokens"):
            patched_module.ModelRunnerFL.execute_model(runner, sched)

    def test_zero_tokens_returns_empty_output(self, patched_module):
        """A zero-token scheduler output short-circuits to the shared empty
        ModelRunnerOutput sentinel without touching the model."""
        runner, sched, model = _prepare_execution_runner(patched_module)
        empty = _scheduler_output(patched_module)
        result = patched_module.ModelRunnerFL.execute_model(runner, empty)
        outputs_mod = import_module("vllm.v1.outputs")
        assert result is outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT
        assert model.calls == []
        assert runner.execute_model_state is None

    def test_zero_tokens_with_dummy_run_for_external_launcher(
        self, patched_module, monkeypatch
    ):
        """external_launcher + DP>1 drives a dummy run before the empty
        return; the boundary call receives num_tokens=1."""
        runner, sched, _ = _prepare_execution_runner(patched_module)
        runner.parallel_config.distributed_executor_backend = "external_launcher"
        runner.parallel_config.data_parallel_size = 2
        dummy_calls = []
        monkeypatch.setattr(
            runner, "_dummy_run", lambda num_tokens: dummy_calls.append(num_tokens)
        )
        empty = _scheduler_output(patched_module)
        result = patched_module.ModelRunnerFL.execute_model(runner, empty)
        outputs_mod = import_module("vllm.v1.outputs")
        assert result is outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT
        assert dummy_calls == [1]

    def test_zero_tokens_kv_connector_no_forward(self, patched_module, monkeypatch):
        """With a KV transfer group configured the zero-token branch defers
        to kv_connector_no_forward and returns its output."""
        runner, sched, _ = _prepare_execution_runner(patched_module)
        monkeypatch.setattr(patched_module, "has_kv_transfer_group", lambda: True)
        fake_group = SimpleNamespace(handle_preemptions=lambda md: None)
        monkeypatch.setattr(patched_module, "get_kv_transfer_group", lambda: fake_group)
        sentinel = object()
        monkeypatch.setattr(
            runner,
            "kv_connector_no_forward",
            lambda scheduler_output, vllm_config: sentinel,
        )
        empty = _scheduler_output(patched_module)
        # The kv-transfer branch asserts the connector metadata is present.
        empty.kv_connector_metadata = SimpleNamespace(marker="kv-md")
        assert patched_module.ModelRunnerFL.execute_model(runner, empty) is sentinel

    def test_kv_transfer_preemption_handled_before_scheduling(
        self, patched_module, monkeypatch
    ):
        """A configured KV transfer group receives handle_preemptions with the
        scheduler output's connector metadata before any scheduling work."""
        runner, sched, model = _prepare_execution_runner(patched_module)
        metadata = SimpleNamespace(marker="kv-metadata")
        handled = []
        fake_group = SimpleNamespace(handle_preemptions=lambda md: handled.append(md))
        monkeypatch.setattr(patched_module, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(patched_module, "get_kv_transfer_group", lambda: fake_group)
        sentinel = object()
        monkeypatch.setattr(
            runner,
            "kv_connector_no_forward",
            lambda scheduler_output, vllm_config: sentinel,
        )
        empty = _scheduler_output(patched_module)
        empty.kv_connector_metadata = metadata
        result = patched_module.ModelRunnerFL.execute_model(runner, empty)
        # The group saw the connector metadata before anything else ran.
        assert handled == [metadata]
        # The zero-token step then returns through the connector-only path
        # without any forward.
        assert result is sentinel
        assert model.calls == []


# ---------------------------------------------------------------------------
# execute_model: the full happy path on real CPU tensors.
# ---------------------------------------------------------------------------


class _PoolingModel:
    """A callable pooling-model stand-in exposing the pooler boundary.

    ``_update_states`` consults ``model.pooler.get_pooling_updates`` when a
    pooling request is registered, and ``_pool`` invokes the pooler through
    ``model.pooler(...)``; ``_model_forward`` still needs a callable model,
    so this class keeps a deterministic recording forward.
    """

    def __init__(self, pooler):
        self.pooler = pooler
        self.calls = []

    def __call__(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        self.calls.append(
            SimpleNamespace(
                input_ids=None if input_ids is None else input_ids.clone(),
                positions=None if positions is None else positions.clone(),
                kwargs=dict(kwargs),
            )
        )
        assert input_ids is not None
        # Deterministic CPU forward: row i is (token * (pos + 1)) floats.
        base = input_ids.to(torch.float32).unsqueeze(-1)
        if positions is not None:
            base = base * (positions.to(torch.float32).unsqueeze(-1) + 1.0)
        return base


class TestExecuteModelHappyPath:
    """The complete prefill execute_model path on CPU tensors."""

    def test_full_prefill_cycle_returns_none_and_stages_state(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """A two-request prefill step drives the real _update_states /
        _prepare_inputs / slot-mapping / attention-metadata / _preprocess
        pipeline, calls the model with the real gathered input ids and
        positions, and stages an ExecuteModelState instead of returning
        output."""
        runner, sched, model = _prepare_execution_runner(
            patched_module,
            prompts=(("r1", (5, 6, 7)), ("r2", (8, 9, 10, 11))),
        )
        builders = _wire_attn_groups(patched_module, runner)

        result = patched_module.ModelRunnerFL.execute_model(runner, sched)
        assert result is None
        state = runner.execute_model_state
        assert state is not None

        # The requests were registered by the real _update_states path.
        assert runner.input_batch.num_reqs == 2
        assert set(runner.requests) == {"r1", "r2"}

        # The model boundary received the exact scheduled token ids and
        # positions, gathered through the real _prepare_inputs path.
        call = model.calls[0]
        expected_ids = torch.tensor([5, 6, 7, 8, 9, 10, 11], dtype=torch.int32)
        assert torch.equal(call.input_ids, expected_ids)
        expected_pos = torch.tensor([0, 1, 2, 0, 1, 2, 3], dtype=torch.int64)
        assert torch.equal(call.positions, expected_pos)

        # Independent oracle for the hidden states: embed[token] scaled by
        # (position + 1) per row.
        base = model.embed[expected_ids.long()]
        oracle = base * (expected_pos.to(torch.float32).unsqueeze(-1) + 1.0)
        assert torch.allclose(state.hidden_states, oracle)

        # compute_logits got exactly the final-token rows per request and
        # applied the row-sum projection.
        assert state.logits is not None
        assert state.logits.shape == (2, model.vocab_size)
        oracle_logits = oracle[[2, 6]].sum(dim=-1, keepdim=True) * model.logits_head
        assert torch.allclose(state.logits, oracle_logits)

        # sample_hidden_states matches hidden_states[logits_indices].
        assert torch.equal(state.sample_hidden_states, oracle[[2, 6]])

        # The attention builder boundary received the real
        # CommonAttentionMetadata with the token count and per-request
        # query bounds.
        prefix_len, cm, kwargs = builders[0].builds[0]
        assert cm.num_actual_tokens == 7
        assert cm.query_start_loc.tolist() == [0, 3, 7]

    def test_second_execute_raises_until_sample(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """The staged state enforces the two-phase protocol."""
        runner, sched, _ = _prepare_execution_runner(
            patched_module, prompts=(("r1", (5, 6, 7)),)
        )
        _wire_attn_groups(patched_module, runner)
        patched_module.ModelRunnerFL.execute_model(runner, sched)
        assert runner.execute_model_state is not None
        with pytest.raises(RuntimeError):
            patched_module.ModelRunnerFL.execute_model(runner, sched)

    def test_pooling_model_returns_pooler_output(
        self, patched_module, cpu_slot_mapping_kernel, monkeypatch
    ):
        """A pooling runner registers the new request through the real
        pooling branch of _update_states (pooling_params asserted, pooler
        updates applied), then routes the forward output through _pool on
        the CPU path (rows copied to host, device synchronized)."""
        PoolingParams = import_module("vllm.pooling_params").PoolingParams
        NewRequestData = import_module("vllm.v1.core.sched.output").NewRequestData

        runner, sched, model = _prepare_execution_runner(
            patched_module,
            prompts=(("r1", (5, 6, 7)),),
            model_config=_make_model_config(runner_type="pooling"),
        )
        _wire_attn_groups(patched_module, runner)

        # A pooling request: the scheduler sends pooling_params, and the
        # runner's model exposes the pooler boundary _update_states consults.
        pooling_params = PoolingParams(task="embed")
        applied = []

        class _FakePooler:
            def get_pooling_updates(self, task):
                update = SimpleNamespace(
                    apply=lambda params: applied.append((task, params))
                )
                return update

            def __call__(self, hidden_states, pooling_metadata):
                cursor = pooling_metadata.pooling_cursor
                start = int(cursor.first_token_indices_gpu[0])
                end = int(cursor.last_token_indices_gpu[0]) + 1
                return [hidden_states[start:end].mean(dim=0).clone()]

        sched.scheduled_new_reqs = [
            NewRequestData(
                req_id="r1",
                prompt_token_ids=[5, 6, 7],
                mm_features=[],
                sampling_params=None,
                pooling_params=pooling_params,
                block_ids=([9],),
                num_computed_tokens=0,
                lora_request=None,
            )
        ]
        pooler = _FakePooler()
        runner.model = _PoolingModel(pooler)
        # The forward boundary must still be callable for _model_forward.

        syncs = []
        monkeypatch.setattr(runner, "_sync_device", lambda: syncs.append(1))

        result = patched_module.ModelRunnerFL.execute_model(runner, sched)
        # The real _update_states pooling branch applied the pooler updates.
        assert applied == [("embed", pooling_params)]
        assert runner.requests["r1"].pooling_params is pooling_params
        # _pool is called instead of staging execute_model_state.
        assert runner.execute_model_state is None
        assert result is not None
        assert result.req_ids == ["r1"]
        # Oracle: the pooled row is the mean of the three scheduled hidden
        # states, each of which is (token * (pos + 1)) from _PoolingModel.
        rows = [torch.tensor([float(t * (p + 1))]) for t, p in zip((5, 6, 7), range(3))]
        oracle = torch.stack(rows).mean(dim=0)
        assert torch.allclose(result.pooler_output[0], oracle)
        # The forward boundary saw the real gathered input ids.
        call = runner.model.calls[0]
        assert call.input_ids.tolist() == [5, 6, 7]
        assert call.positions.tolist() == [0, 1, 2]
        assert syncs == [1]


# ---------------------------------------------------------------------------
# sample_tokens.
# ---------------------------------------------------------------------------


class TestSampleTokens:
    """sample_tokens over a staged ExecuteModelState on CPU tensors."""

    @staticmethod
    def _run_cycle(
        patched_module,
        monkeypatch,
        *,
        prompts,
        spec_config=None,
        suffix_proposer_cls=None,
        **runner_overrides,
    ):
        """Run execute_model then sample_tokens; return (runner, output).

        ``suffix_proposer_cls`` stays patched across the whole cycle: the
        production ``propose_draft_token_ids`` runs an isinstance check
        against the module-level ``SuffixDecodingProposer`` at call time.
        """
        if suffix_proposer_cls is not None:
            monkeypatch.setattr(
                patched_module, "SuffixDecodingProposer", suffix_proposer_cls
            )
        runner, sched, model = _prepare_execution_runner(
            patched_module,
            prompts=prompts,
            spec_config=spec_config,
            suffix_proposer_cls=suffix_proposer_cls,
            **runner_overrides,
        )
        _wire_attn_groups(patched_module, runner)
        assert patched_module.ModelRunnerFL.execute_model(runner, sched) is None
        syncs = []
        monkeypatch.setattr(runner, "_sync_device", lambda: syncs.append(1))
        output = patched_module.ModelRunnerFL.sample_tokens(runner, None)
        return runner, sched, model, output

    def test_no_state_returns_kv_conn_only(self, patched_module):
        """With no staged state, sample_tokens returns the passthrough
        ModelRunnerOutput built from the cached kv_connector_output.

        ``with_kv_conn_output_only`` consults ``is_empty()`` on the cached
        connector output, so a non-empty fake must honor that contract.
        """
        runner, sched, _ = _prepare_execution_runner(patched_module)
        outputs_mod = import_module("vllm.v1.outputs")
        # An empty connector output maps to the shared empty sentinel.
        runner.kv_connector_output = SimpleNamespace(is_empty=lambda: True)
        output = patched_module.ModelRunnerFL.sample_tokens(runner, None)
        assert output is outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT
        assert runner.kv_connector_output is None

        # A non-empty connector output is carried through a copy of the
        # empty sentinel.
        marker = SimpleNamespace(is_empty=lambda: False)
        runner.kv_connector_output = marker
        output = patched_module.ModelRunnerFL.sample_tokens(runner, None)
        assert output is not outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT
        assert output.kv_connector_output is marker
        assert output.req_ids == []
        assert runner.kv_connector_output is None

    def test_sync_cycle_samples_and_updates_batch(
        self, patched_module, monkeypatch, cpu_slot_mapping_kernel
    ):
        """A full execute->sample cycle on one request: the greedy argmax of
        the real logits lands in the output and the request's cached state
        advances by exactly that token."""
        runner, sched, model, output = self._run_cycle(
            patched_module, monkeypatch, prompts=(("r1", (5, 6, 7)),)
        )
        assert output.req_ids == ["r1"]
        assert len(output.sampled_token_ids) == 1
        # Oracle: greedy pick from the sampled logits row. The model records
        # the hidden rows handed to compute_logits; the sampler scores are
        # those rows projected by the model head.
        assert runner.execute_model_state is None
        sampled_hidden = model.logits_calls[-1][0]
        logits_row = sampled_hidden.sum() * model.logits_head
        expected_token = logits_row.argmax().item()
        assert output.sampled_token_ids[0] == [expected_token]
        # The request state and token cache advanced.
        assert runner.requests["r1"].output_token_ids == [expected_token]
        assert runner.input_batch.num_tokens_no_spec[0] == 4
        # The state machine was reset for the next step.
        assert runner.execute_model_state is None

    def test_spec_decode_cycle_proposes_drafts(
        self, patched_module, monkeypatch, cpu_slot_mapping_kernel
    ):
        """With a speculative config (suffix method), sample_tokens runs the
        drafter after bookkeeping and stages the draft token ids."""
        spec = _make_spec_config("suffix", num_speculative_tokens=2)
        _FakeSuffixProposer.reset()
        runner, sched, model, output = self._run_cycle(
            patched_module,
            monkeypatch,
            prompts=(("r1", (5, 6, 7)), ("r2", (8, 9))),
            spec_config=spec,
            suffix_proposer_cls=_FakeSuffixProposer,
        )
        drafter = runner.drafter
        assert isinstance(drafter, _FakeSuffixProposer)
        assert drafter.proposals, "drafter boundary must be called"
        # Production passes scheduler_output.num_spec_tokens_to_schedule,
        # which defaults to 0 on a SchedulerOutput with no spec scheduling.
        num_tok, input_batch, sampled, slots = drafter.proposals[0]
        assert num_tok == sched.num_spec_tokens_to_schedule == 0
        # The drafter saw the bookkept (CPU-list) sampled tokens.
        assert isinstance(sampled, list) and len(sampled) == 2
        assert input_batch is runner.input_batch
        # Draft ids staged on the runner for the next step.
        assert runner._draft_token_ids is not None
        # The fake drafter returns 2 draft tokens per request.
        assert runner._draft_token_ids == [[10, 11], [12, 13]]
        # _copy_draft_token_ids_to_cpu snapshotted the request ids.
        assert runner._draft_token_req_ids == ["r1", "r2"]

    def test_spec_decode_zero_drafts_when_input_does_not_fit(
        self, patched_module, monkeypatch, cpu_slot_mapping_kernel
    ):
        """When the input does not fit in the drafter (a seq len beyond the
        drafter's effective max length), drafts are zeroed and the drafter
        never runs."""
        spec = _make_spec_config("suffix", num_speculative_tokens=2)
        _FakeSuffixProposer.reset()
        runner, sched, model, output = self._run_cycle(
            patched_module,
            monkeypatch,
            prompts=(("r1", (5, 6, 7)), ("r2", (8, 9))),
            spec_config=spec,
            suffix_proposer_cls=_FakeSuffixProposer,
            effective_drafter_max_model_len=3,
        )
        # The drafter never ran: drafts were zeroed instead.
        assert runner.drafter.proposals == []
        assert runner._draft_token_ids is not None
        # The zeroed drafts staged a zero row per request (an int32
        # expanded view of shape [num_reqs, num_spec_tokens]).
        drafts = runner._draft_token_ids
        assert torch.as_tensor(drafts).shape == (2, spec.num_speculative_tokens)
        assert torch.as_tensor(drafts).dtype == torch.int32
        assert torch.as_tensor(drafts).abs().sum().item() == 0
        # The zeroed branch snapshots request ids with zeros_only=True.
        assert runner._draft_token_req_ids == ["r1", "r2"]


# ---------------------------------------------------------------------------
# _get_slot_mappings.
# ---------------------------------------------------------------------------


class TestGetSlotMappings:
    """_get_slot_mappings across padding, encoder-only, and ubatch branches."""

    @staticmethod
    def _runner_with(patched_module, batch, specs, layer_name_lists):
        runner = _build_runner(patched_module, _make_vllm_config())
        AttentionGroup = import_module("vllm.v1.worker.utils").AttentionGroup
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        groups = []
        kv_groups = []
        for gid, (spec, layer_names) in enumerate(zip(specs, layer_name_lists)):
            group = AttentionGroup(
                backend=_FullAttnGroupBackend,
                layer_names=list(layer_names),
                kv_cache_spec=spec,
                kv_cache_group_id=gid,
            )
            group.metadata_builders = [
                _RecordingBuilder(spec, list(layer_names), None, runner.device)
            ]
            groups.append([group])
            kv_groups.append(
                kv_interface.KVCacheGroupSpec(
                    layer_names=list(layer_names), kv_cache_spec=spec
                )
            )
        runner.attn_groups = groups
        runner.kv_cache_config = kv_interface.KVCacheConfig(
            num_blocks=8, kv_cache_tensors=[], kv_cache_groups=kv_groups
        )
        runner.input_batch = batch
        return runner

    def test_no_kv_cache_config_returns_none_pair(self, patched_module):
        """Without a kv_cache_config the slot mapping pair is (None, None)."""
        runner = _build_runner(patched_module, _make_vllm_config())
        gid_map, layer_map = patched_module.ModelRunnerFL._get_slot_mappings(
            runner, num_tokens_padded=4, num_reqs_padded=1, num_tokens_unpadded=3
        )
        assert gid_map is None and layer_map is None

    def test_slot_mapping_values_match_block_table_oracle(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """The per-gid slot mapping equals block_id * block_size + offset for
        each scheduled token, with -1 filling the padded tail."""
        batch = _make_input_batch(patched_module)
        state = _make_req_state(patched_module, "r1", (5, 6, 7, 9), (3,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner = self._runner_with(
            patched_module, batch, [_full_spec()], [["model.layers.0"]]
        )
        # The block table's slot mapping is only committed by the production
        # compute_slot_mapping call (a triton kernel boundary in production),
        # so run it through the CPU oracle kernel first. add_request writes
        # only the CPU copy; commit_block_table publishes it to the GPU view
        # the kernel reads.
        # (query_start_loc is sized max_num_seqs + 1 = 5.)
        runner.query_start_loc.cpu[:] = torch.tensor([0, 4, 4, 4, 4])
        runner.query_start_loc.copy_to_gpu()
        runner.positions[:4] = torch.arange(4, dtype=torch.int64)
        batch.block_table.commit_block_table(1)
        batch.block_table.compute_slot_mapping(
            1, runner.query_start_loc.gpu[:2], runner.positions[:4]
        )
        gid_map, layer_map = patched_module.ModelRunnerFL._get_slot_mappings(
            runner,
            num_tokens_padded=6,
            num_reqs_padded=1,
            num_tokens_unpadded=4,
        )
        slot = gid_map[0]
        assert slot.shape == (6,)
        # Oracle: block 3 with block size 16 hosts positions 0..3.
        expected = [3 * 16 + i for i in range(4)] + [-1, -1]
        assert slot.tolist() == expected
        # The per-layer map aliases the same tensor.
        assert layer_map["model.layers.0"] is slot

    def test_encoder_only_spec_zeros_slot_mapping(self, patched_module):
        """An EncoderOnlyAttentionSpec group gets an all-zero slot mapping
        regardless of the block table."""
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        enc_spec = kv_interface.EncoderOnlyAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )
        batch = _make_input_batch(patched_module)
        state = _make_req_state(patched_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner = self._runner_with(patched_module, batch, [enc_spec], [["enc.0"]])
        gid_map, layer_map = patched_module.ModelRunnerFL._get_slot_mappings(
            runner, num_tokens_padded=4, num_reqs_padded=1, num_tokens_unpadded=3
        )
        assert gid_map[0].tolist() == [0, 0, 0, -1]
        assert layer_map["enc.0"] is gid_map[0]

    def test_ubatch_slices_produce_per_slice_dicts(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """With ubatch slices the per-layer mapping becomes a list of dicts,
        one per ubatch, each slicing the shared slot tensors."""
        batch = _make_input_batch(patched_module)
        state = _make_req_state(patched_module, "r1", (5, 6, 7, 9), (3,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner = self._runner_with(
            patched_module, batch, [_full_spec()], [["model.layers.0"]]
        )
        runner.query_start_loc.cpu[:] = torch.tensor([0, 4, 4, 4, 4])
        runner.query_start_loc.copy_to_gpu()
        runner.positions[:4] = torch.arange(4, dtype=torch.int64)
        batch.block_table.commit_block_table(1)
        batch.block_table.compute_slot_mapping(
            1, runner.query_start_loc.gpu[:2], runner.positions[:4]
        )
        ubatch = SimpleNamespace(
            req_slice=slice(0, 1), token_slice=slice(0, 2), block_slice=slice(0, 2)
        )
        gid_map, layer_list = patched_module.ModelRunnerFL._get_slot_mappings(
            runner,
            num_tokens_padded=4,
            num_reqs_padded=1,
            num_tokens_unpadded=4,
            ubatch_slices=[ubatch],
        )
        assert isinstance(layer_list, list) and len(layer_list) == 1
        sliced = layer_list[0]["model.layers.0"]
        assert sliced.shape == (2,)
        assert sliced.tolist() == gid_map[0][:2].tolist()


# ---------------------------------------------------------------------------
# _allow_microbatching.
# ---------------------------------------------------------------------------


class TestAllowMicrobatching:
    """_allow_microbatching's reader/writer prefix-split veto."""

    def _runner(self, patched_module, batch, use_ubatching=True, threshold=None):
        runner = _build_runner(patched_module, _make_vllm_config())
        runner.input_batch = batch
        runner.parallel_config = SimpleNamespace(use_ubatching=use_ubatching)
        runner.reorder_batch_threshold = threshold
        return runner

    def test_disabled_ubatching_always_allows(self, patched_module):
        """Without ubatching the veto never fires."""
        batch = _make_input_batch(patched_module)
        runner = self._runner(patched_module, batch, use_ubatching=False)
        assert (
            patched_module.ModelRunnerFL._allow_microbatching(
                runner, 2, np.array([4, 4], dtype=np.int32)
            )
            is True
        )

    def test_single_request_always_allows(self, patched_module):
        """num_reqs < 2 short-circuits to True."""
        batch = _make_input_batch(patched_module)
        runner = self._runner(patched_module, batch)
        assert (
            patched_module.ModelRunnerFL._allow_microbatching(
                runner, 1, np.array([8], dtype=np.int32)
            )
            is True
        )

    def test_decode_readers_allow(self, patched_module):
        """All-decode batches (query len 1) have no readers to protect."""
        batch = _make_input_batch(patched_module)
        for req_id, prompt in (("r1", (1, 2, 3)), ("r2", (4, 5, 6))):
            batch.add_request(_make_req_state(patched_module, req_id, prompt, (1,)))
        batch.refresh_metadata()
        runner = self._runner(patched_module, batch)
        # num_computed_tokens > 0 but query len 1 -> no readers.
        batch.num_computed_tokens_cpu[:2] = [3, 3]
        assert (
            patched_module.ModelRunnerFL._allow_microbatching(
                runner, 2, np.array([1, 1], dtype=np.int32)
            )
            is True
        )

    def test_prefill_after_prefill_without_shared_blocks_allows(self, patched_module):
        """Two prefills writing disjoint block ranges do not trip the veto."""
        batch = _make_input_batch(patched_module)
        state1 = _make_req_state(patched_module, "r1", tuple(range(8)), (1,))
        state2 = _make_req_state(patched_module, "r2", tuple(range(8)), (5,))
        batch.add_request(state1)
        batch.add_request(state2)
        batch.refresh_metadata()
        batch.num_computed_tokens_cpu[:2] = [0, 0]
        runner = self._runner(patched_module, batch)
        assert (
            patched_module.ModelRunnerFL._allow_microbatching(
                runner, 2, np.array([8, 8], dtype=np.int32)
            )
            is True
        )

    def test_reader_of_blocks_written_by_other_request_vetoes(self, patched_module):
        """A reader whose already-computed prefix includes a block another
        request in this batch is only writing now must not be split.

        Trace of the scenario (block size 16): r1 has 16 tokens computed
        (whole blocks: [0]) and schedules 8 more; r2 prefills 40 tokens
        over blocks (7, 8, 0), so block 0 -- inside r1's computed prefix --
        is written by this step.  The written set {7, 8, 0} intersects
        r1's whole-block prefix [0], so the veto fires.
        """
        batch = _make_input_batch(patched_module)
        # Reader r1: 24-token prompt on block 0, 16 tokens already computed.
        state1 = _make_req_state(patched_module, "r1", tuple(range(24)), (0,))
        # Writer r2: 40-token prefill spanning blocks 7, 8 and 0.
        state2 = _make_req_state(patched_module, "r2", tuple(range(40)), (7, 8, 0))
        batch.add_request(state1)
        batch.add_request(state2)
        batch.refresh_metadata()
        batch.num_computed_tokens_cpu[:2] = [16, 0]
        runner = self._runner(patched_module, batch)
        result = patched_module.ModelRunnerFL._allow_microbatching(
            runner, 2, np.array([8, 40], dtype=np.int32)
        )
        assert result is False


# ---------------------------------------------------------------------------
# _preprocess.
# ---------------------------------------------------------------------------


class TestPreprocess:
    """_preprocess branches: text-only, prompt-embeds, position padding."""

    @staticmethod
    def _text_runner(
        patched_module, cpu_slot_mapping_kernel, num_scheduled=3, **config_overrides
    ):
        config = _make_vllm_config(**config_overrides)
        runner = _build_runner(patched_module, config)
        batch = _make_input_batch(patched_module)
        state = _make_req_state(patched_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner.requests = {"r1": state}
        runner.input_batch = batch
        sched = _scheduler_output(
            patched_module,
            num_scheduled_tokens={"r1": num_scheduled},
            total_num_scheduled_tokens=num_scheduled,
        )
        # Pre-fill the persistent buffers through the production helper.
        patched_module.ModelRunnerFL._prepare_inputs(
            runner, sched, np.array([num_scheduled], dtype=np.int32)
        )
        return runner, sched

    def test_text_only_uses_input_ids(self, patched_module, cpu_slot_mapping_kernel):
        """A text-only model passes token ids through and zeroes positions
        in the padded tail."""
        runner, sched = self._text_runner(
            patched_module, cpu_slot_mapping_kernel, num_scheduled=3
        )
        (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_out,
        ) = patched_module.ModelRunnerFL._preprocess(runner, sched, num_input_tokens=5)
        assert input_ids is not None and inputs_embeds is None
        assert input_ids.tolist()[:3] == [5, 6, 7]
        assert positions.shape == (5,)
        assert positions[3:].tolist() == [0, 0]
        assert intermediate_tensors is None
        assert model_kwargs == {}
        assert ec_out is None

    def test_prompt_embeds_embeds_token_positions(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """With prompt embeds enabled, token-id rows are embedded through the
        model and merged into inputs_embeds; the returned input_ids is
        None."""
        runner, sched = self._text_runner(
            patched_module,
            cpu_slot_mapping_kernel,
            num_scheduled=3,
            model_config=_make_model_config(enable_prompt_embeds=True),
        )
        # The inputs_embeds buffer is sized by get_inputs_embeds_size() (=4),
        # so the model's embedding width must match for the row writes.
        model = _RecordingModel(hidden=4)
        runner.model = model
        # Mark which scheduled positions carry token ids.
        runner.is_token_ids.np[:3] = True
        (
            input_ids,
            inputs_embeds,
            positions,
            _it,
            model_kwargs,
            _ec,
        ) = patched_module.ModelRunnerFL._preprocess(runner, sched, num_input_tokens=3)
        assert input_ids is None
        assert inputs_embeds is not None
        assert inputs_embeds.shape == (3, 4)
        # Oracle: embed_input_ids returns embed[token] rows, written into
        # the persistent inputs_embeds buffer at the token-id positions.
        expected = model.embed[torch.tensor([5, 6, 7])]
        assert torch.allclose(inputs_embeds, expected)
        assert model_kwargs == {}

    def test_positions_tail_zeroed_when_padded(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """When padding grows the token count, the position tail is
        zeroed in place."""
        runner, sched = self._text_runner(
            patched_module, cpu_slot_mapping_kernel, num_scheduled=2
        )
        # Only poison the tail; the _prepare_inputs-written [0, 1] must
        # survive for the scheduled positions.
        runner.positions[2:] = torch.full_like(runner.positions[2:], 7)
        _, _, positions, _, _, _ = patched_module.ModelRunnerFL._preprocess(
            runner, sched, num_input_tokens=4
        )
        assert positions[:2].tolist() == [0, 1]
        assert positions[2:].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# _pool.
# ---------------------------------------------------------------------------


class TestPool:
    """_pool on real CPU tensors with a recording pooler boundary."""

    @staticmethod
    def _pool_runner(patched_module, prompts=(("r1", (5, 6, 7)), ("r2", (8, 9)))):
        config = _make_vllm_config(
            model_config=_make_model_config(runner_type="pooling")
        )
        runner = _build_runner(patched_module, config)
        batch = _make_input_batch(patched_module, is_pooling_model=True)
        requests = {}
        for req_id, prompt in prompts:
            state = _pooling_req_state(patched_module, req_id, prompt, (9,))
            requests[req_id] = state
            batch.add_request(state)
        batch.refresh_metadata()
        runner.requests = requests
        runner.input_batch = batch

        class _FakePooler:
            def __init__(self):
                self.calls = []

            def __call__(self, hidden_states, pooling_metadata):
                self.calls.append((hidden_states, pooling_metadata))
                # One row per request: the mean of its scheduled rows,
                # derived from the cursor's per-request token bounds.
                cursor = pooling_metadata.pooling_cursor
                outputs = []
                params = pooling_metadata.pooling_params
                for i in range(len(params)):
                    start = int(cursor.first_token_indices_gpu[i])
                    end = int(cursor.last_token_indices_gpu[i]) + 1
                    outputs.append(hidden_states[start:end].mean(dim=0).clone())
                return outputs

        pooler = _FakePooler()
        runner.model = SimpleNamespace(pooler=pooler)
        return runner, batch, pooler

    def test_pool_returns_copied_rows_on_cpu(self, patched_module, monkeypatch):
        """On a CPU platform _pool copies finished rows to host and syncs,
        returning a ModelRunnerOutput whose pooler_output rows match the
        per-request mean-of-hidden-states oracle."""
        runner, batch, pooler = self._pool_runner(patched_module)
        syncs = []
        monkeypatch.setattr(runner, "_sync_device", lambda: syncs.append(1))
        # Sequences complete at their prompt ends: the finished mask must
        # mark every request, and the cursor bounds must match the schedule.
        runner.optimistic_seq_lens_cpu[:2] = torch.tensor([3, 2], dtype=torch.int32)
        # query_start_loc is sized max_num_seqs + 1 = 5.
        runner.query_start_loc.cpu[:] = torch.tensor([0, 3, 5, 5, 5])
        runner.query_start_loc.copy_to_gpu()

        hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        result = patched_module.ModelRunnerFL._pool(
            runner,
            hidden_states=hidden,
            num_scheduled_tokens=5,
            num_scheduled_tokens_np=np.array([3, 2], dtype=np.int32),
            kv_connector_output=None,
        )
        assert pooler.calls, "pooler boundary must be called"
        ph, pm = pooler.calls[0]
        assert ph.shape == (5, 2)
        # Oracle: mean over each request's scheduled rows.
        oracle_r1 = hidden[:3].mean(dim=0)
        oracle_r2 = hidden[3:].mean(dim=0)
        assert torch.allclose(result.pooler_output[0], oracle_r1)
        assert torch.allclose(result.pooler_output[1], oracle_r2)
        assert result.req_ids == ["r1", "r2"]
        assert syncs == [1]
        assert result.kv_connector_output is None

    def test_pool_unfinished_requests_return_none_rows(
        self, patched_module, monkeypatch
    ):
        """Requests whose sequence is still shorter than the prompt get
        None pooler rows (nothing copied for them)."""
        runner, batch, pooler = self._pool_runner(patched_module)
        # Optimistic lens shorter than prompts -> nothing finished.
        runner.optimistic_seq_lens_cpu[:2] = torch.tensor([1, 1], dtype=torch.int32)
        monkeypatch.setattr(runner, "_sync_device", lambda: None)
        runner.query_start_loc.cpu[:] = torch.tensor([0, 3, 5, 5, 5])
        runner.query_start_loc.copy_to_gpu()
        hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        result = patched_module.ModelRunnerFL._pool(
            runner,
            hidden_states=hidden,
            num_scheduled_tokens=5,
            num_scheduled_tokens_np=np.array([3, 2], dtype=np.int32),
            kv_connector_output=None,
        )
        assert result.pooler_output == [None, None]

    def test_pool_raw_output_none_returns_all_none(self, patched_module, monkeypatch):
        """A None pooler output short-circuits to all-None rows."""
        runner, batch, pooler = self._pool_runner(patched_module)
        pooler.__call__ = lambda hidden_states, pooling_metadata: None
        monkeypatch.setattr(runner, "_sync_device", lambda: None)
        runner.query_start_loc.cpu[:] = torch.tensor([0, 3, 5, 5, 5])
        runner.query_start_loc.copy_to_gpu()
        hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        result = patched_module.ModelRunnerFL._pool(
            runner,
            hidden_states=hidden,
            num_scheduled_tokens=5,
            num_scheduled_tokens_np=np.array([3, 2], dtype=np.int32),
            kv_connector_output=None,
        )
        assert result.pooler_output == [None, None]


# ---------------------------------------------------------------------------
# _get_prompt_logprobs_dict.
# ---------------------------------------------------------------------------


class TestGetPromptLogprobsDict:
    """_get_prompt_logprobs_dict on real CPU tensors end to end."""

    def _runner(self, patched_module, *, num_prompt_logprobs=1):
        config = _make_vllm_config()
        runner = _build_runner(patched_module, config)
        batch = _make_input_batch(patched_module)
        state = _make_req_state(
            patched_module, "r1", (5, 6, 7, 9), (9,), num_computed_tokens=0
        )
        batch.add_request(state)
        batch.refresh_metadata()
        runner.requests = {"r1": state}
        runner.num_prompt_logprobs = {"r1": num_prompt_logprobs}
        runner.input_batch = batch
        runner.sampler = _RecordingSampler()
        runner.model = _RecordingModel()
        return runner, state

    def test_empty_dict_short_circuits(self, patched_module):
        """No prompt logprob requests return an empty dict without touching
        the model."""
        runner, state = self._runner(patched_module)
        runner.num_prompt_logprobs = {}
        hidden = torch.randn(4, 8)
        result = patched_module.ModelRunnerFL._get_prompt_logprobs_dict(
            runner, hidden, {"r1": 4}
        )
        assert result == {}
        assert runner.model.logits_calls == []

    def test_full_prompt_chunk_fills_tensors_and_completes(
        self, patched_module, monkeypatch
    ):
        """A final prompt chunk computes per-position top-k logprobs with the
        real sampler stand-in, writes them into the LogprobsTensors, moves
        the request out of num_prompt_logprobs, and returns its tensors."""
        runner, state = self._runner(patched_module)
        syncs = []
        monkeypatch.setattr(runner, "_sync_device", lambda: syncs.append(1))
        # Point the batch row so query_start_loc[offset] resolves.
        runner.query_start_loc.cpu[:] = torch.tensor([0, 4, 4, 4, 4])
        runner.query_start_loc.copy_to_gpu()
        hidden = torch.randn(4, 8)

        result = patched_module.ModelRunnerFL._get_prompt_logprobs_dict(
            runner, hidden, {"r1": 4}
        )
        assert set(result) == {"r1"}
        tensors = result["r1"]
        # 3 logprob positions for a 4-token prompt.
        assert tensors.logprob_token_ids.shape == (3, 2)
        assert tensors.logprobs.shape == (3, 2)
        assert tensors.selected_token_ranks.shape == (3,)
        # Oracle: the model boundary saw this chunk's hidden rows; the
        # sampler scores are the projection of those rows by the model head.
        prompt_hidden = runner.model.logits_calls[0]
        assert prompt_hidden.shape == (3, 8)
        logits = prompt_hidden.sum(dim=-1, keepdim=True) * runner.model.logits_head
        oracle_scores = logits.to(torch.float32)
        lp = oracle_scores.log_softmax(dim=-1)
        topk_lp, topk_idx = torch.topk(lp, 1, dim=-1)
        tgt = torch.tensor([6, 7, 9], dtype=torch.int64).unsqueeze(-1)
        tgt_lp = lp.gather(-1, tgt)
        expected_ids = torch.cat((tgt, topk_idx), dim=1).to(torch.int32)
        expected_lp = torch.cat((tgt_lp, topk_lp), dim=1)
        assert torch.equal(tensors.logprob_token_ids, expected_ids)
        assert torch.allclose(tensors.logprobs, expected_lp)
        # The request is retired from the pending dict.
        assert "r1" not in runner.num_prompt_logprobs
        assert state.in_progress_prompt_logprobs_cpu is None
        assert syncs == [1]

    def test_partial_chunk_defers_completion(self, patched_module):
        """A mid-prompt chunk fills only its slice and stays pending."""
        runner, state = self._runner(patched_module)
        state.num_computed_tokens = 0
        runner.query_start_loc.cpu[:] = torch.tensor([0, 2, 2, 2, 2])
        runner.query_start_loc.copy_to_gpu()
        hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        result = patched_module.ModelRunnerFL._get_prompt_logprobs_dict(
            runner, hidden, {"r1": 2}
        )
        assert result == {}
        assert "r1" in runner.num_prompt_logprobs
        stored = state.in_progress_prompt_logprobs_cpu
        assert stored is not None
        # The full-prompt-sized buffers were allocated (4-token prompt ->
        # 3 logprob positions), pending the remaining chunks.
        assert stored.logprob_token_ids.shape == (3, 2)
        assert stored.logprobs.shape == (3, 2)
        assert stored.selected_token_ranks.shape == (3,)
        # Oracle: rows 0..1 hold this chunk's scores, computed by the same
        # projection + log_softmax the sampler stand-in applies.
        prompt_hidden = runner.model.logits_calls[0]
        assert prompt_hidden.shape == (2, 8)
        logits = prompt_hidden.sum(dim=-1, keepdim=True) * runner.model.logits_head
        lp = logits.to(torch.float32).log_softmax(dim=-1)
        topk_lp, topk_idx = torch.topk(lp, 1, dim=-1)
        tgt = torch.tensor([6, 7], dtype=torch.int64).unsqueeze(-1)
        tgt_lp = lp.gather(-1, tgt)
        expected_ids = torch.cat((tgt, topk_idx), dim=1).to(torch.int32)
        expected_lp = torch.cat((tgt_lp, topk_lp), dim=1)
        expected_ranks = (lp > tgt_lp).sum(dim=-1).to(torch.int32)
        assert torch.equal(stored.logprob_token_ids[:2], expected_ids)
        assert torch.allclose(stored.logprobs[:2], expected_lp)
        assert torch.equal(stored.selected_token_ranks[:2], expected_ranks)
        # Row 2 was not written by this chunk: it still differs from the
        # chunk-computed scores (empty_cpu allocates uninitialized memory,
        # so the exact contents are unspecified, but they cannot equal a
        # freshly computed row for this chunk's tokens).
        assert not torch.allclose(stored.logprob_token_ids[2], expected_ids[1])

    def test_preempted_request_skipped(self, patched_module):
        """A request absent from num_scheduled_tokens is skipped (preempted
        in the prefill stage)."""
        runner, state = self._runner(patched_module)
        hidden = torch.randn(4, 8)
        result = patched_module.ModelRunnerFL._get_prompt_logprobs_dict(
            runner, hidden, {}
        )
        assert result == {}
        assert "r1" in runner.num_prompt_logprobs

    def test_prompt_embeds_request_skipped(self, patched_module):
        """A request with prompt_token_ids=None (prompt embeds) is skipped."""
        runner, state = self._runner(patched_module)
        state.prompt_token_ids = None
        hidden = torch.randn(4, 8)
        result = patched_module.ModelRunnerFL._get_prompt_logprobs_dict(
            runner, hidden, {"r1": 4}
        )
        assert result == {}
        assert "r1" in runner.num_prompt_logprobs


# ---------------------------------------------------------------------------
# _bookkeeping_sync and _get_nans_in_logits.
# ---------------------------------------------------------------------------


class TestBookkeepingSync:
    """_bookkeeping_sync over real InputBatch state on CPU."""

    @staticmethod
    def _runner(patched_module, prompts=(("r1", (5, 6, 7)),), async_sched=False):
        config = _make_vllm_config()
        config.scheduler_config.async_scheduling = async_sched
        runner = _build_runner(patched_module, config)
        batch = _make_input_batch(patched_module)
        requests = {}
        for req_id, prompt in prompts:
            state = _make_req_state(patched_module, req_id, prompt, (9,))
            requests[req_id] = state
            batch.add_request(state)
        batch.refresh_metadata()
        runner.requests = requests
        runner.input_batch = batch
        return runner

    @staticmethod
    def _sampler_output(sampled):
        from vllm.v1.outputs import SamplerOutput

        return SamplerOutput(
            sampled_token_ids=torch.tensor(sampled, dtype=torch.int64),
            logprobs_tensors=None,
        )

    def test_sync_greedy_tokens_cached_into_batch(self, patched_module):
        """Sampled tokens are written into token_ids_cpu / is_token_ids /
        num_tokens_no_spec and extended into the request's output list."""
        runner = self._runner(patched_module)
        sched = _scheduler_output(
            patched_module,
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        sampler_output = self._sampler_output([[11]])
        result = patched_module.ModelRunnerFL._bookkeeping_sync(
            runner, sched, sampler_output, None, torch.randn(3, 8), 3
        )
        (
            nans_dict,
            nans_dev,
            logprobs_lists,
            valid_ids,
            prompt_dict,
            req_ids,
            req_index,
            invalid,
        ) = result
        assert nans_dict == {} and nans_dev is None
        assert valid_ids == [[11]]
        assert req_ids == ["r1"]
        assert invalid == []
        # Batch state advanced by exactly one sampled token.
        assert runner.input_batch.num_tokens_no_spec[0] == 4
        assert runner.input_batch.token_ids_cpu[0, 3] == 11
        assert runner.input_batch.is_token_ids[0, 3]
        assert runner.requests["r1"].output_token_ids == [11]
        assert prompt_dict == {}

    def test_discarded_request_tokens_cleared(self, patched_module):
        """A request flagged in discard_request_mask has its sampled tokens
        dropped from the output list (chunked prefill)."""
        runner = self._runner(
            patched_module, prompts=(("r1", (5, 6, 7)), ("r2", (8, 9)))
        )
        sched = _scheduler_output(
            patched_module,
            num_scheduled_tokens={"r1": 3, "r2": 2},
            total_num_scheduled_tokens=5,
        )
        runner.discard_request_mask.np[0] = True
        sampler_output = self._sampler_output([[11], [21]])
        result = patched_module.ModelRunnerFL._bookkeeping_sync(
            runner, sched, sampler_output, None, torch.randn(5, 8), 5
        )
        valid_ids = result[3]
        assert valid_ids == [[], [21]]
        assert runner.requests["r1"].output_token_ids == []
        assert runner.requests["r2"].output_token_ids == [21]

    def test_spec_decode_tokens_parsed_via_rejection_parse(self, patched_module):
        """Multi-column sampled ids (spec decode) route through
        RejectionSampler.parse_output: -1 slots are dropped per row."""
        runner = self._runner(
            patched_module, prompts=(("r1", (5, 6, 7)), ("r2", (8, 9)))
        )
        sched = _scheduler_output(
            patched_module,
            num_scheduled_tokens={"r1": 3, "r2": 2},
            total_num_scheduled_tokens=5,
        )
        sampled = [[1, 2, 3], [4, -1, -1]]
        sampler_output = self._sampler_output(sampled)
        result = patched_module.ModelRunnerFL._bookkeeping_sync(
            runner, sched, sampler_output, None, torch.randn(5, 8), 5
        )
        valid_ids = result[3]
        assert valid_ids == [[1, 2, 3], [4]]
        assert runner.requests["r1"].output_token_ids == [1, 2, 3]
        assert runner.requests["r2"].output_token_ids == [4]
        assert runner.input_batch.num_tokens_no_spec.tolist()[:2] == [6, 3]

    def test_async_path_caches_prev_sampled_on_gpu(self, patched_module):
        """Async scheduling keeps tokens on device: prev_sampled_token_ids is
        set, invalid rows are -1 placeholders, and per-row -1 entries extend
        the cached output lists."""
        runner = self._runner(
            patched_module,
            prompts=(("r1", (5, 6, 7)), ("r2", (8, 9))),
            async_sched=True,
        )
        sched = _scheduler_output(
            patched_module,
            num_scheduled_tokens={"r1": 3, "r2": 2},
            total_num_scheduled_tokens=5,
        )
        runner.discard_request_mask.np[0] = True
        sampler_output = self._sampler_output([[11], [21]])
        result = patched_module.ModelRunnerFL._bookkeeping_sync(
            runner, sched, sampler_output, None, torch.randn(5, 8), 5
        )
        (
            _nans_dict,
            _nans_dev,
            logprobs_lists,
            valid_ids,
            _prompt_dict,
            req_ids,
            req_index,
            invalid,
        ) = result
        assert valid_ids == []
        assert invalid == [0]
        assert runner.input_batch.prev_sampled_token_ids is not None
        assert torch.equal(
            runner.input_batch.prev_sampled_token_ids,
            torch.tensor([[11], [21]], dtype=torch.int64),
        )
        # The non-discarded request got a placeholder appended.
        assert runner.requests["r2"].output_token_ids == [-1]
        assert runner.requests["r1"].output_token_ids == []

    def test_cpu_generator_state_restored_for_discarded_request(self, patched_module):
        """A discarded request with a CPU generator has its RNG state rolled
        back via set_state (chunked prefill must not advance the stream)."""
        runner = self._runner(patched_module)
        sched = _scheduler_output(
            patched_module,
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1234)
        saved_state = gen.get_state()
        runner.input_batch.generators[0] = gen
        # The production code reads the saved states recorded by _sample.
        runner._cpu_generator_states = {0: saved_state.clone()}
        # Advance the generator so a rollback is observable.
        gen.manual_seed(99)
        runner.discard_request_mask.np[0] = True
        sampler_output = self._sampler_output([[11]])
        patched_module.ModelRunnerFL._bookkeeping_sync(
            runner, sched, sampler_output, None, torch.randn(3, 8), 3
        )
        assert torch.equal(gen.get_state(), saved_state)

    def test_nans_in_logits_dict_reported(self, patched_module, monkeypatch):
        """With VLLM_COMPUTE_NANS_IN_LOGITS set, per-request NaN counts are
        reported through the real count_nans_per_row helper."""
        runner = self._runner(
            patched_module, prompts=(("r1", (5, 6, 7)), ("r2", (8, 9)))
        )
        sched = _scheduler_output(
            patched_module,
            num_scheduled_tokens={"r1": 3, "r2": 2},
            total_num_scheduled_tokens=5,
        )
        monkeypatch.setattr(
            import_module("vllm.envs"), "VLLM_COMPUTE_NANS_IN_LOGITS", True
        )
        logits = torch.zeros(2, 4)
        logits[0, 1] = float("nan")
        logits[0, 3] = float("nan")
        sampler_output = self._sampler_output([[1], [2]])
        result = patched_module.ModelRunnerFL._bookkeeping_sync(
            runner, sched, sampler_output, logits, torch.randn(5, 8), 5
        )
        nans_dict = result[0]
        assert nans_dict == {"r1": 2, "r2": 0}

    def test_get_nans_in_logits_missing_rows_report_zero(
        self, patched_module, monkeypatch
    ):
        """_get_nans_in_logits maps row counts onto request ids; requests
        whose index points beyond the counted rows are reported as 0
        (``nans_to_dict`` treats rows and requests as not one-to-one)."""
        runner = self._runner(patched_module)
        # req_id_to_index references a row beyond the logits rows.
        runner.input_batch.req_id_to_index = {"r1": 5}
        logits = torch.zeros(1, 4)
        monkeypatch.setattr(
            import_module("vllm.envs"), "VLLM_RAISE_ON_LOGIT_NANS", False
        )
        result = patched_module.ModelRunnerFL._get_nans_in_logits(runner, logits)
        assert result == {"r1": 0}

    def test_get_nans_in_logits_none_logits_returns_zeroed(
        self, patched_module, monkeypatch
    ):
        """A None logits tensor short-circuits the NaN count to an empty
        list, so every request reports 0."""
        runner = self._runner(
            patched_module, prompts=(("r1", (5, 6, 7)), ("r2", (8, 9)))
        )
        monkeypatch.setattr(
            import_module("vllm.envs"), "VLLM_RAISE_ON_LOGIT_NANS", False
        )
        result = patched_module.ModelRunnerFL._get_nans_in_logits(runner, None)
        assert result == {"r1": 0, "r2": 0}


# ---------------------------------------------------------------------------
# _pp_receive_prev_sampled_token_ids_to_input_batch.
# ---------------------------------------------------------------------------


class TestPpReceivePrevSampledTokenIds:
    """The PP async receive path driven to the torch.distributed boundary."""

    def _runner(self, patched_module, *, chunked_prefill=False):
        config = _make_vllm_config()
        runner = _build_runner(patched_module, config)
        batch = _make_input_batch(patched_module)
        requests = {}
        for req_id, prompt in (("r1", (5, 6, 7)), ("r2", (8, 9))):
            state = _make_req_state(patched_module, req_id, prompt, (9,))
            requests[req_id] = state
            batch.add_request(state)
        batch.refresh_metadata()
        runner.requests = requests
        runner.input_batch = batch
        if chunked_prefill:
            runner.discard_request_mask.np[:2] = True
        return runner

    def test_receive_appends_placeholder_and_advances_batch(
        self, patched_module, pp_group, monkeypatch
    ):
        """The receive path allocates the recv buffer, skips the actual
        broadcast for chunked-prefill batches, and leaves every request
        excluded from the prev map (all requests are marked for discard,
        so none advances a placeholder token)."""
        runner = self._runner(patched_module, chunked_prefill=True)
        # Not the last rank: the receive path only runs on earlier stages.
        pp_group.is_last_rank = False
        broadcasts = []

        def _fake_broadcast(tensor, src, group, async_op=False):
            broadcasts.append((tuple(tensor.shape), src, async_op))
            return None

        monkeypatch.setattr(torch.distributed, "broadcast", _fake_broadcast)
        patched_module.ModelRunnerFL._pp_receive_prev_sampled_token_ids_to_input_batch(
            runner
        )
        # Chunked prefill skips the broadcast entirely.
        assert broadcasts == []
        prev = runner.input_batch.prev_sampled_token_ids
        assert prev is not None and prev.shape == (2, 1)
        # Both requests are discarded (chunked prefill): no placeholders
        # appended and no rows advance.
        assert runner.requests["r1"].output_token_ids == []
        assert runner.requests["r2"].output_token_ids == []
        assert runner.input_batch.num_tokens_no_spec.tolist()[:2] == [3, 2]
        assert not runner.input_batch.is_token_ids[0, 3]
        assert runner.input_batch.prev_req_id_to_index == {}

    def test_receive_broadcasts_from_last_rank(
        self, patched_module, pp_group, monkeypatch
    ):
        """Without chunked prefill the broadcast fires with async_op=True
        from the last PP rank (the GPU/distributed edge)."""
        runner = self._runner(patched_module)
        pp_group.is_last_rank = False
        broadcasts = []

        def _fake_broadcast(tensor, src, group, async_op=False):
            broadcasts.append((tuple(tensor.shape), src, async_op))
            return "work-handle"

        monkeypatch.setattr(torch.distributed, "broadcast", _fake_broadcast)
        patched_module.ModelRunnerFL._pp_receive_prev_sampled_token_ids_to_input_batch(
            runner
        )
        assert broadcasts == [((2, 1), 0, True)]
        assert runner._pp_recv_work == "work-handle"
        assert runner.input_batch.prev_req_id_to_index == {"r1": 0, "r2": 1}

    def test_discarded_requests_excluded_from_prev_map(
        self, patched_module, pp_group, monkeypatch
    ):
        """Requests flagged for discard are excluded from the prev map and
        do not get the placeholder append."""
        runner = self._runner(patched_module)
        pp_group.is_last_rank = False
        runner.discard_request_mask.np[0] = True
        monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **k: None)
        patched_module.ModelRunnerFL._pp_receive_prev_sampled_token_ids_to_input_batch(
            runner
        )
        assert runner.input_batch.prev_req_id_to_index == {"r2": 1}
        assert runner.requests["r1"].output_token_ids == []
        assert runner.requests["r2"].output_token_ids == [-1]


# ---------------------------------------------------------------------------
# _copy_draft_token_ids_to_cpu.
# ---------------------------------------------------------------------------


class TestCopyDraftTokenIdsToCpu:
    """The async draft-token D2H copy driven on the fake stream boundary."""

    def _runner(self, patched_module, spec_tokens=2, async_sched=False):
        # The real SuffixDecodingProposer lazy-imports arctic_inference in
        # its constructor (absent on CPU venvs), so patch the module-level
        # class before the runner is built.
        original_cls = patched_module.SuffixDecodingProposer
        patched_module.SuffixDecodingProposer = _FakeSuffixProposer
        try:
            config = _make_vllm_config(
                speculative_config=_make_spec_config(
                    "suffix", num_speculative_tokens=spec_tokens
                )
            )
            config.scheduler_config.async_scheduling = async_sched
            runner = _build_runner(patched_module, config)
        finally:
            patched_module.SuffixDecodingProposer = original_cls
        batch = _make_input_batch(patched_module)
        state = _make_req_state(patched_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner.requests = {"r1": state}
        runner.input_batch = batch
        return runner

    def test_tensor_copy_records_event_and_req_ids(self, patched_module):
        """A tensor-valued _draft_token_ids is copied into the CPU buffer on
        the dedicated stream, the event is recorded, and the request ids are
        snapshotted."""
        runner = self._runner(patched_module)
        drafts = torch.tensor([[7, 8]], dtype=torch.int64)
        runner._draft_token_ids = drafts
        sched = _scheduler_output(patched_module)
        patched_module.ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, sched)
        assert runner._draft_token_req_ids == ["r1"]
        assert runner.draft_token_ids_cpu[:1, :2].tolist() == [[7, 8]]
        assert runner.draft_token_ids_event.record_count == 1
        # The copy stream waited on the default stream first.
        assert runner.draft_token_ids_copy_stream.waited

    def test_zeros_only_zeroes_cpu_buffer(self, patched_module):
        """zeros_only skips the copy and zeroes the CPU buffer instead."""
        runner = self._runner(patched_module)
        runner.draft_token_ids_cpu[:1, :2] = 5
        runner._draft_token_ids = torch.tensor([[7, 8]], dtype=torch.int64)
        sched = _scheduler_output(patched_module)
        patched_module.ModelRunnerFL._copy_draft_token_ids_to_cpu(
            runner, sched, zeros_only=True
        )
        assert runner.draft_token_ids_cpu[:1, :2].tolist() == [[0, 0]]
        assert runner.draft_token_ids_event.record_count == 1

    def test_async_scheduling_skips_without_structured_output(self, patched_module):
        """Under async scheduling with no structured output requests and no
        output-token history, the copy is skipped entirely."""
        runner = self._runner(patched_module, async_sched=True)
        runner._draft_token_ids = torch.tensor([[7, 8]], dtype=torch.int64)
        sched = _scheduler_output(patched_module)
        sched.has_structured_output_requests = False
        # sampling_metadata.output_token_ids empty.
        patched_module.ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, sched)
        assert runner._draft_token_req_ids is None
        assert runner.draft_token_ids_event.record_count == 0

    def test_list_valued_draft_ids_records_req_ids_only(self, patched_module):
        """A list-valued _draft_token_ids (CPU drafter) snapshots request
        ids but skips the tensor copy machinery."""
        runner = self._runner(patched_module)
        runner._draft_token_ids = [[1, 2]]
        sched = _scheduler_output(patched_module)
        patched_module.ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, sched)
        assert runner._draft_token_req_ids == ["r1"]
        assert runner.draft_token_ids_event.record_count == 0


# ---------------------------------------------------------------------------
# _get_spec_decode_draft_probs.
# ---------------------------------------------------------------------------


class TestGetSpecDecodeDraftProbabilities:
    """Cached draft-probability alignment across request reordering."""

    def _runner(self, patched_module, req_ids, spec_md):
        config = _make_vllm_config()
        runner = _build_runner(patched_module, config)
        batch = _make_input_batch(patched_module)
        for req_id in req_ids:
            batch.add_request(_make_req_state(patched_module, req_id, (5, 6, 7), (9,)))
        batch.refresh_metadata()
        runner.input_batch = batch
        runner._draft_probs = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
        runner._draft_prob_req_ids = ["ra", "rb"]
        return runner

    def test_rows_aligned_by_req_id(self, patched_module):
        """Rows are re-assembled in batch order, truncated to each request's
        num_draft_tokens."""
        runner = self._runner(
            patched_module,
            ["rb", "ra"],
            SimpleNamespace(num_draft_tokens=[3, 2]),
        )
        result = patched_module.ModelRunnerFL._get_spec_decode_draft_probs(
            runner, SimpleNamespace(num_draft_tokens=[3, 2])
        )
        assert result is not None
        # rb row (arange 4..7) first truncated to 3, then ra row to 2.
        expected = torch.cat(
            [
                torch.tensor([4.0, 5.0, 6.0]),
                torch.tensor([0.0, 1.0]),
            ]
        )
        assert torch.equal(result, expected)

    def test_missing_req_falls_back_to_none(self, patched_module):
        """A request with no cached probability row forces the legacy
        (None) rejection behavior."""
        runner = self._runner(
            patched_module,
            ["rx"],
            SimpleNamespace(num_draft_tokens=[2]),
        )
        result = patched_module.ModelRunnerFL._get_spec_decode_draft_probs(
            runner, SimpleNamespace(num_draft_tokens=[2])
        )
        assert result is None

    def test_zero_draft_tokens_returns_none(self, patched_module):
        """Requests that scheduled no drafts produce no concatenated rows."""
        runner = self._runner(
            patched_module, ["ra"], SimpleNamespace(num_draft_tokens=[0])
        )
        result = patched_module.ModelRunnerFL._get_spec_decode_draft_probs(
            runner, SimpleNamespace(num_draft_tokens=[0])
        )
        assert result is None
