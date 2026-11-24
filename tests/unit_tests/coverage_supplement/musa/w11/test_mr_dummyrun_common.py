"""Shared light doubles for the _dummy_run / sampler / pooler coverage tests.

No real model, sampler, pooler, attention backend, graph capture or device
work happens here.  Every double returns fixed-shape CPU tensors so the
methods under test can be driven through their branch logic.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm_fl.worker import model_runner as mr

MR = mr.ModelRunnerFL
CUDAGraphMode = mr.CUDAGraphMode


def new_runner(**attrs):
    runner = object.__new__(MR)
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


class FakeBuffer:
    """Minimal stand-in for vllm's CpuGpuBuffer."""

    def __init__(self, size, dtype=np.int32, value=0):
        self.np = np.full(size, value, dtype=dtype)
        self.dtype = dtype
        self.copy_count = 0

    def copy_to_gpu(self):
        self.copy_count += 1
        return self


class LightModel:
    """Callable model double returning a fixed-shape hidden-state tensor."""

    requires_raw_input_tokens = False

    def __init__(
        self,
        num_tokens=8,
        hidden=4,
        vocab=8,
        aux_hidden_state=False,
        num_prompt_logprobs=None,
    ):
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.vocab = vocab
        self.aux_hidden_state = aux_hidden_state
        self.calls = []
        self.intermediate_shapes = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        out = torch.zeros(self.num_tokens, self.hidden, dtype=torch.float32)
        if self.aux_hidden_state:
            return (out, None)
        return out

    def compute_logits(self, hidden_states):
        return torch.zeros(hidden_states.shape[0], self.vocab, dtype=torch.float32)

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        self.intermediate_shapes.append((batch_size, dtype, device))
        return {"h": torch.zeros(batch_size, 2, dtype=dtype, device=device)}


class LightSampler:
    """Sampler double; records each call and optionally fails."""

    def __init__(self, logprobs_mode="raw_logits", error=None):
        self.logprobs_mode = logprobs_mode
        self.error = error
        self.calls = []

    def __call__(self, logits, sampling_metadata):
        self.calls.append(SimpleNamespace(logits=logits, metadata=sampling_metadata))
        if self.error is not None:
            raise self.error
        return torch.zeros(logits.shape[0], dtype=torch.long)


class LightPooler:
    """Pooler double returning per-task fixed shapes; optionally fails."""

    def __init__(self, task_sizes=None, error=None, include_none=False):
        self.task_sizes = dict(task_sizes or {})
        self.error = error
        self.include_none = include_none
        self.calls = []
        self.applied = []

    def get_pooling_updates(self, task):
        def apply(pooling_params):
            self.applied.append((task, pooling_params))

        return SimpleNamespace(apply=apply)

    def __call__(self, hidden_states, pooling_metadata):
        self.calls.append(SimpleNamespace(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        ))
        if self.error is not None:
            raise self.error
        task = pooling_metadata.pooling_params[0].task
        size = self.task_sizes.get(task, 1)
        out = [torch.zeros(size, dtype=torch.float32)]
        if self.include_none:
            out.append(None)
        return out


class FakePoolingParams:
    def __init__(self, task):
        self.task = task
        self.extra_kwargs = None
        self.verified_with = None

    def verify(self, model_config):
        self.verified_with = model_config


class FakePoolingStates:
    pass


class FakePoolingMetadata:
    def __init__(
        self,
        prompt_lens,
        prompt_token_ids,
        prompt_token_ids_cpu,
        pooling_params,
        pooling_states,
    ):
        self.prompt_lens = prompt_lens
        self.prompt_token_ids = prompt_token_ids
        self.prompt_token_ids_cpu = prompt_token_ids_cpu
        self.pooling_params = pooling_params
        self.pooling_states = pooling_states
        self.cursor = None

    def build_pooling_cursor(self, num_scheduled_tokens, seq_lens_cpu, device):
        self.cursor = SimpleNamespace(
            num_scheduled_tokens=num_scheduled_tokens,
            seq_lens_cpu=seq_lens_cpu,
            device=device,
        )


def _fake_forward_context(store):
    @contextmanager
    def set_forward_context(attn_metadata=None, vllm_config=None, **kwargs):
        store.append(
            {
                "attn_metadata": attn_metadata,
                "vllm_config": vllm_config,
                **kwargs,
            }
        )
        yield

    return set_forward_context


def install_module_stubs(monkeypatch, runner):
    """Patch module-level helpers that would otherwise touch vLLM internals.

    The replacements read state off the shared state dict so an individual
    test can tune behaviour after the runner was built.
    """

    monkeypatch.setattr(
        mr, "set_forward_context", _fake_forward_context(runner.forwards)
    )
    monkeypatch.setattr(
        mr, "get_dcp_dummy_context_len", lambda *a, **k: _CURRENT["dcp_len"]
    )
    monkeypatch.setattr(
        mr, "prepare_dcp_dummy_context_metadata", MagicMock()
    )
    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=_CURRENT["pp_first_rank"]),
    )
    monkeypatch.setattr(
        mr,
        "maybe_create_ubatch_slices",
        lambda *a, **k: _CURRENT["ubatch_slices"],
    )


_CURRENT = {}


def set_dummy_state(**kwargs):
    _CURRENT.update(kwargs)


def build_dummy_run_runner(
    *,
    num_tokens=8,
    max_num_tokens=32,
    max_num_seqs=4,
    uniform_decode_query_len=2,
    mode=CUDAGraphMode.NONE,
    batch_num_tokens=None,
    batch_num_reqs=None,
    dcp_len=0,
    ubatch_slices=(None, None),
    pp_first_rank=True,
    slot_mappings=(None, None),
    model=None,
    vocab=8,
):
    batch_tokens = num_tokens if batch_num_tokens is None else batch_num_tokens
    if model is None:
        model = LightModel(num_tokens=batch_tokens, vocab=vocab)

    model_config = SimpleNamespace(
        multimodal_config=None,
        is_encoder_decoder=False,
        dtype=torch.float32,
        use_mla=False,
        get_vocab_size=lambda: vocab,
        model="dummy",
    )
    parallel_config = SimpleNamespace(
        cp_kv_cache_interleave_size=1,
        data_parallel_size=1,
        num_ubatches=2,
        enable_eplb=False,
        eplb_config=SimpleNamespace(log_balancedness=False),
    )
    vllm_config = SimpleNamespace(
        model_config=model_config, parallel_config=parallel_config
    )

    runner = new_runner(
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_config=model_config,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        max_num_tokens=max_num_tokens,
        uniform_decode_query_len=uniform_decode_query_len,
        dcp_world_size=1,
        compilation_config=SimpleNamespace(cudagraph_specialize_lora=False),
        model=model,
        input_ids=SimpleNamespace(
            gpu=torch.zeros(max_num_tokens, dtype=torch.long)
        ),
        inputs_embeds=SimpleNamespace(
            gpu=torch.zeros(max_num_tokens, 4, dtype=torch.float32)
        ),
        positions=torch.zeros(max_num_tokens, dtype=torch.long),
        mrope_positions=SimpleNamespace(
            gpu=torch.zeros(3, max_num_tokens, dtype=torch.float32)
        ),
        query_pos=FakeBuffer(max_num_tokens + 2, dtype=np.int64),
        query_start_loc=FakeBuffer(max_num_seqs + 2, dtype=np.int32),
        arange_np=np.arange(max_num_tokens + max_num_seqs + 8, dtype=np.int64),
        seq_lens=torch.zeros(max_num_seqs, dtype=torch.int32),
        optimistic_seq_lens_cpu=torch.zeros(max_num_seqs, dtype=torch.int32),
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(commit_block_table=MagicMock())
        ),
        prepare_inputs_event=None,
        intermediate_tensors=None,
        lora_config=None,
        requests={},
        kv_connector_metadata=None,
        kv_cache_config=None,
        layerwise_nvtx_hooks_registered=False,
        eep_eplb_suppressed=False,
        supports_mm_inputs=False,
        enable_prompt_embeds=False,
        uses_mrope=False,
        use_aux_hidden_state_outputs=False,
        use_ascend_attention=False,
        speculative_config=None,
        forwards=[],
        determine_calls=[],
        attention_calls=[],
        mm_calls=[],
        model_kwargs_calls=[],
    )

    runner._mode = mode
    runner._batch_num_tokens = batch_tokens
    runner._batch_num_reqs = batch_num_reqs
    runner._num_tokens_across_dp = None
    runner._should_ubatch = False
    runner._slot_mappings = slot_mappings
    runner._attn_metadata = None

    def determine(**kwargs):
        runner.determine_calls.append(kwargs)
        return (
            runner._mode,
            SimpleNamespace(
                num_tokens=runner._batch_num_tokens,
                num_reqs=runner._batch_num_reqs,
            ),
            runner._should_ubatch,
            runner._num_tokens_across_dp,
            None,
        )

    def get_slot_mappings(**kwargs):
        runner.slot_mapping_calls = kwargs
        return runner._slot_mappings

    def build_attention_metadata(**kwargs):
        runner.attention_calls.append(kwargs)
        return runner._attn_metadata, None

    def prepare_mm_inputs(n):
        runner.mm_calls.append(n)
        return None, runner.inputs_embeds.gpu[:n]

    def dummy_mm_kwargs(n):
        runner.model_kwargs_calls.append(n)
        return {"mm_arg": n}

    runner._determine_batch_execution_and_padding = determine
    runner._get_slot_mappings = get_slot_mappings
    runner._build_attention_metadata = build_attention_metadata
    runner._prepare_mm_inputs = prepare_mm_inputs
    runner._dummy_mm_kwargs = dummy_mm_kwargs
    runner._init_model_kwargs = MagicMock(return_value={})
    runner.maybe_dummy_run_with_lora = lambda *a, **k: nullcontext()
    runner._register_layerwise_nvtx_hooks = MagicMock()
    runner.eplb_step = MagicMock()
    runner._pad_query_start_loc_for_fia = MagicMock(return_value=batch_num_reqs or max_num_seqs)

    set_dummy_state(
        dcp_len=dcp_len,
        pp_first_rank=pp_first_rank,
        ubatch_slices=ubatch_slices,
    )
    runner._ubatch_slices = ubatch_slices
    return runner
