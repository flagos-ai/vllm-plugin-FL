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

"""Coverage tests for the dummy-run family of ModelRunnerFL
(vllm_fl/worker/model_runner.py): ``_dummy_run`` (mm-encoder-only
early return and the recorded forward pipeline for the default,
uniform-decode and mixed batches), ``_dummy_sampler_run`` (a real
Sampler over a real CPU logits computation) and
``_dummy_pooler_run_task`` (real PoolingParams / PoolingMetadata with a
recording pooler).

The batch-execution / slot-mapping / attention-metadata collaborators
are recorders (CONTRACT R2); ``synchronize_input_prep`` /
``maybe_dummy_run_with_lora`` (null with lora_config None) /
``set_forward_context`` are bound for real. ``PIN_MEMORY`` in
``vllm.v1.pool.metadata`` is patched to False — a device probe boundary;
on the CI CUDA image pinned allocation succeeds.
"""

import os

# The FlashInfer sampler probe imports the flashinfer package when
# VLLM_USE_FLASHINFER_SAMPLER is set; the package is absent on this dev
# host (present on the CI image, where this default is harmless because
# the probe then resolves for real). Set before the vllm imports below.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from types import SimpleNamespace

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker.lora_model_runner_mixin import (
    LoRAModelRunnerMixin,
)

import vllm_fl.worker.model_runner as model_runner_module
from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


class _Patch:
    """Deterministic set/restore of a module attribute (R2 input boundary)."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _forward_model(calls):
    class _Model:
        def __call__(
            inner_self,
            *,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        ):
            calls.append(("forward", input_ids, positions, kwargs))
            # hidden states: one row per token, plus the identity for
            # the sampled rows (num_reqs rows).
            n = input_ids.shape[0]
            return torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)

    return _Model()


def _dummy_runner(num_tokens=8, max_num_seqs=4, **extra):
    calls = SimpleNamespace(forward=[], camd=[], nvtx=0, eplb=[])
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=None,
                is_encoder_decoder=False,
                use_mla=False,
                dtype=torch.float32,
            ),
            parallel_config=SimpleNamespace(
                num_ubatches=1,
                data_parallel_size=1,
                use_sequence_parallel_moe=False,
            ),
            compilation_config=SimpleNamespace(
                static_forward_context={},
                fast_moe_cold_start=False,
                cudagraph_specialize_lora=False,
            ),
            observability_config=SimpleNamespace(
                enable_layerwise_nvtx_tracing=False,
            ),
        ),
        model_config=SimpleNamespace(
            multimodal_config=None,
            is_encoder_decoder=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        max_num_tokens=num_tokens,
        max_num_reqs=4,
        uniform_decode_query_len=4,
        device=torch.device("cpu"),
        supports_mm_inputs=False,
        enable_prompt_embeds=False,
        uses_mrope=False,
        uses_xdrope_dim=0,
        use_aux_hidden_state_outputs=False,
        use_ascend_attention=False,
        speculative_config=None,
        lora_config=None,
        common_attention_metadata_graph=None,
        input_ids=SimpleNamespace(
            gpu=torch.arange(num_tokens, dtype=torch.long),
        ),
        positions=torch.arange(num_tokens, dtype=torch.long),
        calls=calls,
    )
    runner.model = _forward_model(calls.forward)
    runner.calls = calls
    # Real bound helpers.
    runner._register_layerwise_nvtx_hooks = (
        ModelRunnerFL._register_layerwise_nvtx_hooks.__get__(
            runner,
            ModelRunnerFL,
        )
    )
    runner.synchronize_input_prep = ModelRunnerFL.synchronize_input_prep.__get__(
        SimpleNamespace(prepare_inputs_event=None),
        ModelRunnerFL,
    )
    runner.maybe_dummy_run_with_lora = (
        LoRAModelRunnerMixin.maybe_dummy_run_with_lora.__get__(
            runner,
            LoRAModelRunnerMixin,
        )
    )
    runner.maybe_setup_dummy_loras = (
        LoRAModelRunnerMixin.maybe_setup_dummy_loras.__get__(
            runner,
            LoRAModelRunnerMixin,
        )
    )
    # Real DP-input randomization gate: env var off => a null context.
    runner.maybe_randomize_inputs = ModelRunnerFL.maybe_randomize_inputs.__get__(
        runner,
        ModelRunnerFL,
    )
    # Real cumsum/arange helper: pure numpy over a shared arange buffer.
    runner.arange_np = np.arange(4096)
    runner._get_cumsum_and_arange = ModelRunnerFL._get_cumsum_and_arange.__get__(
        runner,
        ModelRunnerFL,
    )
    runner.maybe_select_dummy_loras = (
        LoRAModelRunnerMixin.maybe_select_dummy_loras.__get__(
            runner,
            LoRAModelRunnerMixin,
        )
    )
    # Recording collaborators.
    runner._determine_batch_execution_and_padding = lambda **kw: (
        CUDAGraphMode.NONE,
        SimpleNamespace(num_tokens=num_tokens, num_reqs=2),
        False,
        None,
        None,
    )
    runner._get_slot_mappings = lambda **kw: (None, None)
    runner._init_model_kwargs = lambda: {}
    runner.model_state = SimpleNamespace(
        prepare_dummy_inputs=lambda num_reqs, num_reqs_padded, nst: {},
    )
    runner.eplb_step = lambda **kw: calls.eplb.append(kw)
    runner._run_common_attention_metadata = lambda *a, **kw: calls.camd.append((a, kw))
    runner._build_attention_metadata = lambda **kw: ("attn-md", None)
    for key, value in extra.items():
        setattr(runner, key, value)
    return runner


def test_dummy_run_mm_encoder_only_skips_lm_execution():
    """A multimodal encoder-only config returns empty tensors without
    touching the model."""
    runner = _dummy_runner()
    runner.vllm_config.model_config.multimodal_config = SimpleNamespace(
        mm_encoder_only=True,
    )
    runner.model_config.multimodal_config = (
        runner.vllm_config.model_config.multimodal_config
    )

    hidden, last = ModelRunnerFL._dummy_run(runner, 8)

    torch.testing.assert_close(hidden, torch.tensor([]))
    torch.testing.assert_close(last, torch.tensor([]))
    assert runner.calls.forward == []


def test_dummy_run_default_batch_runs_recorded_pipeline():
    """The default (non-uniform) batch splits num_tokens across up to
    max_num_seqs requests and runs the full recorded pipeline."""
    runner = _dummy_runner(num_tokens=8, max_num_seqs=4)
    runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(
            commit_block_table=lambda n: None,
        ),
    )
    runner.optimistic_seq_lens_cpu = torch.zeros(16, dtype=torch.int)
    runner.seq_lens = SimpleNamespace(
        copy_=lambda src, non_blocking: None,
    )
    runner.query_pos = SimpleNamespace(np=np.zeros(16))
    runner.query_start_loc = SimpleNamespace(
        np=np.zeros(17),
        copy_to_gpu=lambda: None,
    )
    runner.intermediate_tensors = None

    with _Patch(
        model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    ):
        hidden, last = ModelRunnerFL._dummy_run(
            runner,
            8,
            force_attention=True,
        )

    # One forward with the padded input ids and positions.
    assert len(runner.calls.forward) == 1
    _, input_ids, positions, kwargs = runner.calls.forward[0]
    assert input_ids.shape[0] == 8
    assert positions.shape[0] == 8
    # Attention metadata builder was invoked (force_attention=True).
    assert len(runner.calls.camd) == 1
    # The EPLB step ran as a dummy step.
    assert runner.calls.eplb == [{"is_dummy": True, "is_profile": False}]
    # The returned hidden states are the model output and its last
    # token rows: 8 tokens over 4 requests ([2, 2, 2, 2]) sample rows
    # [1, 3, 5, 7].
    torch.testing.assert_close(
        hidden, torch.arange(32, dtype=torch.float32).reshape(8, 4)
    )
    assert last.shape == (4, 4)
    torch.testing.assert_close(last, hidden[[1, 3, 5, 7]])


def test_dummy_run_uniform_decode_batch_uses_query_len():
    """A uniform decode batch uses uniform_decode_query_len as the per-
    request query length."""
    runner = _dummy_runner(num_tokens=8, max_num_seqs=4)
    runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(
            commit_block_table=lambda n: None,
        ),
    )
    runner.optimistic_seq_lens_cpu = torch.zeros(16, dtype=torch.int)
    runner.seq_lens = SimpleNamespace(
        copy_=lambda src, non_blocking: None,
    )
    runner.query_pos = SimpleNamespace(np=np.zeros(16))
    runner.query_start_loc = SimpleNamespace(
        np=np.zeros(17),
        copy_to_gpu=lambda: None,
    )
    runner.intermediate_tensors = None
    runner._determine_batch_execution_and_padding = lambda **kw: (
        CUDAGraphMode.NONE,
        SimpleNamespace(num_tokens=8, num_reqs=2),
        False,
        None,
        None,
    )
    seen = {}
    runner._run_common_attention_metadata = lambda *a, **kw: seen.setdefault(
        "camd", []
    ).append(kw)

    with _Patch(
        model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    ):
        ModelRunnerFL._dummy_run(
            runner,
            8,
            uniform_decode=True,
            force_attention=True,
        )

    assert len(runner.calls.forward) == 1
    # seq_lens filled with the uniform query length (4).
    assert runner.optimistic_seq_lens_cpu[:2].tolist() == [4, 4]
    assert seen["camd"][0]["capture"] is False


def test_dummy_sampler_run_produces_sampler_output():
    """The sampler warm-up runs a real Sampler over real CPU logits."""
    from vllm.v1.sample.sampler import Sampler

    hidden_states = torch.rand(3, 4)

    class _Model:
        def compute_logits(self, sample_hidden_states):
            # Row r -> one-hot over an 8-vocab.
            return torch.eye(8, dtype=torch.float32)[: sample_hidden_states.shape[0]]

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None),
        ),
        model_config=SimpleNamespace(multimodal_config=None),
        model=_Model(),
        sampler=Sampler(logprobs_mode="raw_logprobs"),
        device=torch.device("cpu"),
        speculative_config=None,
    )

    output = ModelRunnerFL._dummy_sampler_run(runner, hidden_states)

    # A real sampler output: sampled token ids for the 3 requests.
    assert output.sampled_token_ids.shape == (3, 1)


def test_dummy_sampler_run_mm_encoder_only_skips():
    """MM encoder-only models skip the sampler warm-up entirely."""
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True),
            ),
        ),
    )

    output = ModelRunnerFL._dummy_sampler_run(runner, torch.rand(2, 2))

    torch.testing.assert_close(output, torch.tensor([]))


def test_dummy_pooler_run_task_builds_metadata_and_calls_pooler():
    """The pooler warm-up builds real PoolingParams / PoolingMetadata
    and runs the model pooler over the given hidden states."""
    import vllm.v1.pool.metadata as pool_metadata_module
    from vllm.v1.pool.metadata import PoolingMetadata

    calls = SimpleNamespace(pooler=[])

    class _Pooler:
        def __call__(inner_self, *, hidden_states, pooling_metadata):
            calls.pooler.append((hidden_states, pooling_metadata))
            return [torch.zeros(1) for _ in pooling_metadata.prompt_lens]

    applied = []
    pooler = _Pooler()
    pooler.get_pooling_updates = lambda task: SimpleNamespace(
        apply=lambda params: applied.append((task, params)),
    )
    model = SimpleNamespace(pooler=pooler)
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(
            pooler_config=None,
            is_multimodal_model=False,
        ),
        device=torch.device("cpu"),
        get_model=lambda: model,
    )

    with _Patch(pool_metadata_module, "PIN_MEMORY", False):
        output = ModelRunnerFL._dummy_pooler_run_task(
            runner,
            torch.rand(6, 2),
            task="embed",
        )

    # The pooler ran once over all 6 rows with real metadata.
    assert len(calls.pooler) == 1
    hidden_states, metadata = calls.pooler[0]
    assert hidden_states.shape[0] == 6
    assert isinstance(metadata, PoolingMetadata)
    # 6 tokens over at most 4 requests: floor + remainder on the last.
    assert metadata.prompt_lens.tolist() == [1, 1, 1, 3]
    # The pooler returned its recorded outputs.
    assert len(output) == 4
