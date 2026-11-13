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

"""Coverage tests for ModelRunnerFL._dummy_run.

The dummy run is the warmup/capture primitive: it synthesizes a schedule
(mixed, uniform-decode, or linear split), asks the CUDA-graph dispatcher for
a batch descriptor, optionally builds attention metadata, runs the fake
model, forwards the run to the drafter, and returns the last hidden states
per request. Every collaborator is a recorder; the return value is checked
against an independently computed cumsum-of-scheduled-tokens reference.
"""

from contextlib import ExitStack, contextmanager, nullcontext  # noqa: I001  (shim first)
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
from vllm.config import CUDAGraphMode

from vllm_fl.worker.model_runner import ModelRunnerFL


class _Patch:
    """Deterministic set/restore of a module attribute (input boundary)."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


class _Recorder:
    def __init__(self, result=None, result_factory=None):
        self.calls = []
        self.result = result
        self.result_factory = result_factory

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.result_factory is not None:
            return self.result_factory(*args, **kwargs)
        return self.result


class _Buffer:
    """CpuGpuBuffer stand-in with a numpy view and a Gpu copy recorder."""

    def __init__(self, size, dtype):
        self.np = np.zeros(size, dtype=dtype)
        self.copies = 0

    def copy_to_gpu(self, *args):
        self.copies += 1

    def copy_(self, *args, **kwargs):
        self.copied = (args, kwargs)


class _FakeProposer:
    def __init__(self):
        self.calls = []

    def dummy_run(self, *args, **kwargs):
        self.calls.append((args, kwargs))


HIDDEN = torch.arange(32, dtype=torch.float32).reshape(16, 2)


def _runner(
    *,
    mode=CUDAGraphMode.NONE,
    padded=5,
    should_ubatch=False,
    across_dp=None,
    **overrides,
):
    def _make_descriptor(**kwargs):
        return (
            mode,
            SimpleNamespace(
                num_tokens=padded,
                num_reqs=kwargs["num_reqs"],
                uniform=bool(kwargs.get("force_uniform_decode")),
                num_active_loras=kwargs.get("force_num_active_loras") or 0,
            ),
            should_ubatch,
            across_dp,
            None,
        )

    determine = _Recorder(result_factory=_make_descriptor)

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None),
            parallel_config=SimpleNamespace(num_ubatches=1),
        ),
        uniform_decode_query_len=1,
        max_num_tokens=16,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        _determine_batch_execution_and_padding=determine,
        _get_slot_mappings=_Recorder(result=(None, None)),
        synchronize_input_prep=lambda: nullcontext(),
        common_attention_metadata_graph=None,
        use_ascend_attention=False,
        maybe_dummy_run_with_lora=lambda *a: nullcontext(),
        lora_config=None,
        _init_model_kwargs=_Recorder(result={}),
        supports_mm_inputs=False,
        _prepare_mm_inputs=_Recorder(result=(torch.zeros(4, 2), torch.zeros(4, 2))),
        _dummy_mm_kwargs=_Recorder(result={"mm_kw": torch.zeros(1)}),
        enable_prompt_embeds=False,
        input_ids=SimpleNamespace(gpu=torch.zeros(16, dtype=torch.int32)),
        inputs_embeds=SimpleNamespace(gpu=torch.zeros(16, 2)),
        model_state=SimpleNamespace(
            prepare_dummy_inputs=_Recorder(result={"state_kw": torch.zeros(1)})
        ),
        model_config=SimpleNamespace(is_encoder_decoder=False, dtype=torch.float16),
        uses_mrope=False,
        uses_xdrope_dim=0,
        positions=torch.zeros(16, dtype=torch.int64),
        intermediate_tensors=None,
        use_aux_hidden_state_outputs=False,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_specialize_lora=False),
        _register_layerwise_nvtx_hooks=_Recorder(),
        eplb_step=_Recorder(),
        device=torch.device("cpu"),
        model=_Recorder(result=HIDDEN),
        maybe_randomize_inputs=lambda *a: nullcontext(),
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner, determine


@contextmanager
def _patches(status_calls, pp_is_first=True):
    @contextmanager
    def _set_forward_context(*args, **kwargs):
        status_calls.append((args, kwargs))
        yield

    with ExitStack() as stack:
        stack.enter_context(
            _Patch(
                mr_module,
                "get_pp_group",
                lambda: SimpleNamespace(is_first_rank=pp_is_first),
            )
        )
        stack.enter_context(
            _Patch(mr_module, "set_forward_context", _set_forward_context)
        )
        stack.enter_context(
            _Patch(mr_module, "maybe_create_ubatch_slices", lambda *a: (None, None))
        )
        yield


def _prepare_attention_buffers(runner, *, max_reqs=8, cumsum=(1, 2, 3, 5)):
    runner.optimistic_seq_lens_cpu = torch.zeros(max_reqs, dtype=torch.int64)
    runner.seq_lens = _Buffer(max_reqs, np.int64)
    runner.query_pos = SimpleNamespace(np=np.zeros(max_reqs, dtype=np.int32))
    runner.query_start_loc = _Buffer(max_reqs + 1, np.int32)
    runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(commit_block_table=_Recorder())
    )
    runner._get_cumsum_and_arange = _Recorder(result=np.array(cumsum, dtype=np.int32))
    runner._run_common_attention_metadata = _Recorder()
    runner._build_attention_metadata = _Recorder(result=({}, None))


def test_dummy_run_splits_linear_schedule_and_returns_last_hidden_states():
    """A plain dummy batch splits tokens evenly and indexes cumsum-1 rows."""
    runner, determine = _runner(padded=5)
    forward = []
    with _patches(forward):
        hidden, logits = ModelRunnerFL._dummy_run(runner, 5)

    det = determine.calls[0][1]
    np.testing.assert_array_equal(
        det["num_scheduled_tokens_np"], np.array([1, 1, 1, 2], dtype=np.int32)
    )
    assert det["num_tokens"] == 5
    assert det["num_reqs"] == 4
    assert det["max_num_scheduled_tokens"] == 5
    assert det["force_eager"] is False
    assert det["force_uniform_decode"] is False
    assert det["force_num_active_loras"] == 0
    # Four requests: cumsum([1,1,1,2])-1 = [0,1,2,4].
    torch.testing.assert_close(logits, HIDDEN[torch.tensor([0, 1, 2, 4])])
    assert hidden is HIDDEN
    model_kwargs = runner.model.calls[0][1]
    assert model_kwargs["positions"].shape == (5,)
    assert model_kwargs["state_kw"].tolist() == [0.0]
    assert model_kwargs["input_ids"].tolist() == [0, 0, 0, 0, 0]
    assert runner.eplb_step.calls[0][1] == {"is_dummy": True, "is_profile": False}
    assert len(runner._register_layerwise_nvtx_hooks.calls) == 1
    # Slot mappings were requested with the descriptor's padded size.
    slot_kwargs = runner._get_slot_mappings.calls[0][1]
    assert slot_kwargs["num_tokens_padded"] == 5
    assert slot_kwargs["num_tokens_unpadded"] == 5


def test_dummy_run_uniform_decode_uses_query_len_per_request():
    """Uniform decode batches have max_query_len tokens per request."""
    runner, determine = _runner(padded=6)
    runner.uniform_decode_query_len = 2
    with _patches([]):
        _, logits = ModelRunnerFL._dummy_run(runner, 6, uniform_decode=True)

    det = determine.calls[0][1]
    np.testing.assert_array_equal(
        det["num_scheduled_tokens_np"], np.array([2, 2, 2], dtype=np.int32)
    )
    assert det["max_num_scheduled_tokens"] == 2
    assert det["force_uniform_decode"] is True
    # cumsum([2,2,2])-1 = [1,3,5].
    torch.testing.assert_close(logits, HIDDEN[torch.tensor([1, 3, 5])])


def test_dummy_run_mixed_batch_puts_prefill_last():
    """create_mixed_batch emits one-token decode rows then one prefill row."""
    runner, determine = _runner(padded=5)
    with _patches([]):
        ModelRunnerFL._dummy_run(runner, 5, create_mixed_batch=True)

    det = determine.calls[0][1]
    np.testing.assert_array_equal(
        det["num_scheduled_tokens_np"], np.array([1, 1, 3], dtype=np.int32)
    )
    assert det["max_num_scheduled_tokens"] == 3
    assert det["force_uniform_decode"] is False


def test_dummy_run_mm_encoder_only_skips_the_language_model():
    """mm_encoder_only configs return empty tensors without touching the model."""
    runner, _ = _runner(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            ),
            parallel_config=SimpleNamespace(num_ubatches=1),
        )
    )
    hidden, logits = ModelRunnerFL._dummy_run(runner, 4)

    assert hidden.numel() == 0 and logits.numel() == 0
    assert runner.model.calls == []


def test_dummy_run_rejects_token_count_above_limit():
    """num_tokens beyond max_num_tokens trips the hard assertion."""
    runner, _ = _runner()
    with pytest.raises(AssertionError):
        ModelRunnerFL._dummy_run(runner, 17)


def test_dummy_run_rejects_mode_mismatch():
    """An explicit runtime mode must match the dispatcher's decision."""
    runner, _ = _runner(mode=CUDAGraphMode.NONE)
    with pytest.raises(AssertionError, match="Cudagraph runtime mode mismatch"):
        ModelRunnerFL._dummy_run(
            runner, 4, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
        )


def test_dummy_run_builds_attention_metadata_with_profile_seq_lens():
    """force_attention initializes the shared metadata buffers and builders."""
    runner, _ = _runner(padded=5)
    _prepare_attention_buffers(runner)
    forward = []
    with _patches(forward):
        ModelRunnerFL._dummy_run(runner, 5, force_attention=True, profile_seq_lens=7)

    # seq_lens mirrors the profiling length on live rows and zero elsewhere.
    assert runner.optimistic_seq_lens_cpu[:4].tolist() == [7, 7, 7, 7]
    assert runner.optimistic_seq_lens_cpu[4:].eq(0).all()
    assert runner.seq_lens.copied[1]["non_blocking"] is True
    # query_start_loc = [0, cumsum..., last].
    assert runner.query_start_loc.np[0] == 0
    assert runner.query_start_loc.np[1:5].tolist() == [1, 2, 3, 5]
    assert (runner.query_start_loc.np[5:] == 5).all()
    assert runner.query_start_loc.copies == 1
    commit = runner.input_batch.block_table.commit_block_table.calls[0][0]
    assert commit[0] == 4
    common_call = runner._run_common_attention_metadata.calls[0]
    assert common_call[0][0] == 4
    assert common_call[0][1] == CUDAGraphMode.NONE
    assert common_call[1] == {"capture": False}
    meta = runner._build_attention_metadata.calls[0][1]
    assert meta["num_tokens"] == 5
    assert meta["num_tokens_padded"] is None
    assert meta["num_reqs"] == 4
    assert meta["max_query_len"] == 5
    assert meta["for_cudagraph_capture"] is False
    assert meta["use_spec_decode"] is False
    assert meta["ascend_attn_state"] is None
    assert meta["block_table_rows_are_current"] is False
    assert forward[0][1]["num_tokens"] == 5


def test_dummy_run_mixed_batch_uses_short_sequence_lengths():
    """Mixed batches stage shorter seq_lens to speed up FI warmup."""
    runner, _ = _runner(padded=5)
    _prepare_attention_buffers(runner, cumsum=(1, 2, 5))
    with _patches([]):
        ModelRunnerFL._dummy_run(
            runner, 5, force_attention=True, create_mixed_batch=True
        )

    assert runner.optimistic_seq_lens_cpu[:3].tolist() == [1, 1, 4]


def test_dummy_run_prompt_embeds_branch_omits_input_ids():
    """Prompt-embed mode passes the embed buffer and no token ids."""
    runner, _ = _runner(enable_prompt_embeds=True, padded=4)
    with _patches([]):
        ModelRunnerFL._dummy_run(runner, 4)

    kwargs = runner.model.calls[0][1]
    assert kwargs["input_ids"] is None
    torch.testing.assert_close(kwargs["inputs_embeds"], torch.zeros(4, 2))


def test_dummy_run_multimodal_branch_merges_mm_kwargs():
    """Multimodal dummies prepare dummy MM inputs and merge their kwargs."""
    runner, _ = _runner(supports_mm_inputs=True, padded=4)
    with _patches([]):
        ModelRunnerFL._dummy_run(runner, 4)

    assert runner._prepare_mm_inputs.calls[0][0][0] == 4
    assert runner._dummy_mm_kwargs.calls[0][0][0] == 4
    kwargs = runner.model.calls[0][1]
    assert kwargs["mm_kw"].tolist() == [0.0]
    assert kwargs["state_kw"].tolist() == [0.0]


def test_dummy_run_non_first_pp_rank_gathers_intermediate_tensors():
    """Non-first PP ranks construct/collect intermediate tensors for the model."""
    runner, _ = _runner(padded=4)
    runner.model = _Recorder(result=HIDDEN)
    runner.model.make_empty_intermediate_tensors = _Recorder(result="inter")
    runner.sync_and_gather_intermediate_tensors = _Recorder(result="gathered")
    with _patches([], pp_is_first=False):
        ModelRunnerFL._dummy_run(runner, 4)

    assert runner.sync_and_gather_intermediate_tensors.calls[0][0] == (4, None, False)
    assert runner.model.calls[0][1]["intermediate_tensors"] == "gathered"


def test_dummy_run_adjusts_padded_size_for_a_single_ubatch():
    """Ubatching rewrites num_tokens_padded (and the DP count) before forward."""
    across = torch.tensor([6, 6])
    runner, _ = _runner(padded=6, should_ubatch=True, across_dp=across)
    padded_slice = SimpleNamespace(num_tokens=3)
    forward = []
    with (
        _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True)),
        _Patch(mr_module, "set_forward_context", _ctx(forward)),
        _Patch(
            mr_module,
            "maybe_create_ubatch_slices",
            lambda *a: ([SimpleNamespace(num_tokens=6)], [padded_slice]),
        ),
    ):
        ModelRunnerFL._dummy_run(runner, 4)

    assert forward[0][1]["num_tokens"] == 3
    assert across.tolist() == [3, 3]
    # Input buffers were sliced from the descriptor's padded size (6) before
    # the single-ubatch adjustment; only the forward token count shrinks.
    assert runner.model.calls[0][1]["input_ids"].shape == (6,)


def _ctx(calls):
    @contextmanager
    def _set_forward_context(*args, **kwargs):
        calls.append((args, kwargs))
        yield

    return _set_forward_context


def _spec(*, eagle=True, draft=False, extract=False, enforce_eager=False):
    return SimpleNamespace(
        use_eagle=lambda: eagle,
        uses_draft_model=lambda: draft,
        uses_extract_hidden_states=lambda: extract,
        enforce_eager=enforce_eager,
    )


def _run_spec_case(
    *, spec, specialize=False, capturing=True, mode=CUDAGraphMode.PIECEWISE
):
    runner, _ = _runner(mode=mode)
    drafter = _FakeProposer()
    runner.speculative_config = spec
    runner.drafter = drafter
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_lora=specialize)
    with (
        _Patch(mr_module, "EagleProposer", _FakeProposer),
        _Patch(mr_module, "DFlashProposer", _FakeProposer),
        _Patch(mr_module, "DraftModelProposer", _FakeProposer),
        _Patch(mr_module, "ExtractHiddenStatesProposer", _FakeProposer),
        _Patch(mr_module, "Gemma4Proposer", _FakeProposer),
        _patches([]),
    ):
        ModelRunnerFL._dummy_run(
            runner,
            4,
            cudagraph_runtime_mode=mode,
            is_graph_capturing=capturing,
            num_active_loras=2,
        )
    return drafter.calls[0]


def test_dummy_run_forwards_graph_capture_flag_to_drafter():
    """Capturing a PIECEWISE graph lets the drafter use CUDA graphs."""
    args, kwargs = _run_spec_case(spec=_spec(eagle=True))
    assert args[0] == 4
    assert kwargs["use_cudagraphs"] is True
    assert kwargs["is_graph_capturing"] is True
    assert kwargs["slot_mappings"] is None


def test_dummy_run_enforce_eager_disables_drafter_cudagraphs():
    """enforce_eager on the speculative config forces eager drafter runs."""
    _, kwargs = _run_spec_case(spec=_spec(draft=True, enforce_eager=True))
    assert kwargs["use_cudagraphs"] is False


def test_dummy_run_lora_specialization_disables_drafter_cudagraphs():
    """Specialized LoRA graphs are disabled for the drafter while capturing."""
    _, kwargs = _run_spec_case(spec=_spec(extract=True), specialize=True)
    assert kwargs["use_cudagraphs"] is False


def test_dummy_run_skips_eplb_when_requested():
    """skip_eplb suppresses the dummy EPLB state update."""
    runner, _ = _runner()
    with _patches([]):
        ModelRunnerFL._dummy_run(runner, 4, skip_eplb=True)
    assert runner.eplb_step.calls == []
