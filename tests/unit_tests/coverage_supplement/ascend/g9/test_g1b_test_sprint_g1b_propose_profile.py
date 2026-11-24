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

"""G1b sprint coverage: proposal dispatch, prompt logprobs, dummy/profile
and the execute/sample tail of ModelRunnerFL (814b line).

Targets: ``propose_draft_token_ids`` (ngram / custom_class / suffix /
extract_hidden_states guard), ``_get_spec_decode_draft_probs``,
``_get_prompt_logprobs_dict``, ``_dummy_run`` / ``_dummy_sampler_run``
(encoder-only early returns), ``profile_run`` and
``_cleanup_profiling_kv_cache``, ``sample_tokens`` (no-forward passthrough
and the sync pipeline) and ``_input_fits_in_drafter``.

Collaborators are namespaces / recorders; ``get_pp_group``, ``advance_io_step``
and ``torch.accelerator`` are patched at their boundaries.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.outputs as v1_outputs
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput
from vllm.v1.spec_decode.extract_hidden_states import (
    ExtractHiddenStatesProposer,
)
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

import vllm_fl.worker.model_runner as model_runner_module
from vllm_fl.worker.model_runner import ExecuteModelState, ModelRunnerFL


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


@pytest.fixture(autouse=True)
def _cpu_logprob_buffers(monkeypatch):
    """Allocate prompt-logprob CPU buffers without a pinned device."""
    if hasattr(v1_outputs, "PIN_MEMORY"):
        monkeypatch.setattr(v1_outputs, "PIN_MEMORY", False)


# ------------------------------------------------------- proposer dispatch


class _NgramProposer(NgramProposer):
    def __init__(self, calls, result):
        self._calls, self._result = calls, result

    def propose(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        return self._result


class _SuffixProposer(SuffixDecodingProposer):
    def __init__(self, calls, result):
        self._calls, self._result = calls, result

    def propose(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        return self._result


class _ExtractProposer(ExtractHiddenStatesProposer):
    def __init__(self):
        pass


def _propose_runner(spec_config, drafter, **extra):
    # The 814b dispatch chain probes use_ngram_gpu() before the suffix /
    # extract-hidden-state branches, so every lightly-built spec config needs it.
    if not hasattr(spec_config, "use_ngram_gpu"):
        spec_config.use_ngram_gpu = lambda: False
    runner = SimpleNamespace(
        speculative_config=spec_config,
        drafter=drafter,
        input_batch=SimpleNamespace(
            num_tokens_no_spec=[4, 2],
            token_ids_cpu=[[1, 2, 3, 4], [5, 6]],
            req_ids=["r0", "r1"],
        ),
        _draft_probs="stale",
        _draft_prob_req_ids="stale",
        use_aux_hidden_state_outputs=False,
        requests="reqs",
        discard_request_mask=SimpleNamespace(gpu="mask"),
    )
    for key, value in extra.items():
        setattr(runner, key, value)
    return runner


def _propose(runner, *, sampled_token_ids, spec_decode_metadata=None,
             slot_mappings=None, aux_hidden_states=None):
    return ModelRunnerFL.propose_draft_token_ids(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=6,
            num_spec_tokens_to_schedule=3,
        ),
        sampled_token_ids=sampled_token_ids,
        sampling_metadata="sampling-md",
        hidden_states=torch.zeros(6, 2),
        sample_hidden_states=torch.zeros(2, 2),
        aux_hidden_states=aux_hidden_states,
        spec_decode_metadata=spec_decode_metadata,
        common_attn_metadata="attn-md",
        slot_mappings=slot_mappings,
    )


def test_propose_ngram_routes_cpu_token_state():
    calls = []
    drafter = _NgramProposer(calls, result=[[1, 2], [3]])
    slots = {"layer.0": torch.zeros(6, dtype=torch.int64)}
    runner = _propose_runner(
        SimpleNamespace(method="ngram"), drafter
    )

    result = _propose(
        runner,
        sampled_token_ids=[[7], [8]],
        slot_mappings=slots,
    )

    assert result == [[1, 2], [3]]
    assert calls == [
        (
            (3, [[7], [8]], [4, 2], [[1, 2, 3, 4], [5, 6]]),
            {"slot_mappings": slots},
        )
    ]
    assert runner._draft_probs is None
    assert runner._draft_prob_req_ids is None


def test_propose_custom_class_routes_without_spec_count():
    calls = []
    drafter = SimpleNamespace(
        propose=lambda *a, **k: calls.append((a, k)) or [[9]],
    )
    runner = _propose_runner(
        SimpleNamespace(method="custom_class"), drafter
    )

    result = _propose(runner, sampled_token_ids=[[7]])

    assert result == [[9]]
    assert calls == [
        (
            ([[7]], [4, 2], [[1, 2, 3, 4], [5, 6]]),
            {"slot_mappings": None},
        )
    ]


def test_propose_suffix_routes_input_batch():
    calls = []
    drafter = _SuffixProposer(calls, result=[[4]])
    runner = _propose_runner(SimpleNamespace(method="suffix"), drafter)

    result = _propose(runner, sampled_token_ids=[[7], [8]])

    assert result == [[4]]
    assert calls == [
        ((3, runner.input_batch, [[7], [8]]), {"slot_mappings": None})
    ]


def test_propose_extract_hidden_states_requires_aux_states():
    import pytest

    cfg = SimpleNamespace(
        method=None,
        use_ngram_gpu=lambda: False,
        uses_extract_hidden_states=lambda: True,
    )
    runner = _propose_runner(cfg, _ExtractProposer())

    with pytest.raises(ValueError, match="aux_hidden_states"):
        _propose(
            runner,
            sampled_token_ids=torch.zeros(2, 1, dtype=torch.int32),
        )


def test_spec_decode_draft_probs_none_and_missing_rows(caplog):
    import logging

    runner = SimpleNamespace(
        _draft_probs=None,
        _draft_prob_req_ids=None,
        input_batch=SimpleNamespace(req_ids=["r0"]),
    )
    meta = SimpleNamespace(num_draft_tokens=[1])
    assert ModelRunnerFL._get_spec_decode_draft_probs(runner, meta) is None

    runner._draft_probs = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    runner._draft_prob_req_ids = ["other"]
    with caplog.at_level(
        logging.WARNING, logger="vllm_fl.worker.model_runner"
    ):
        assert ModelRunnerFL._get_spec_decode_draft_probs(runner, meta) is None
    assert any("Missing cached draft" in r.getMessage() for r in caplog.records)


def test_spec_decode_draft_probs_gathers_draft_rows():
    probs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    runner = SimpleNamespace(
        _draft_probs=probs,
        _draft_prob_req_ids=["r0", "r1"],
        input_batch=SimpleNamespace(req_ids=["r0", "r1"]),
    )
    meta = SimpleNamespace(num_draft_tokens=[2, 1])

    result = ModelRunnerFL._get_spec_decode_draft_probs(runner, meta)

    # r0 contributes rows [0, :2], r1 contributes row [1, :1]; the 814b helper
    # concatenates the slices into one flat (3,) tensor rather than nesting.
    assert result.shape == (3,)
    assert result.tolist() == [0.0, 1.0, 4.0]
    assert result.is_contiguous()


# --------------------------------------------------------- prompt logprobs


class _Request:
    def __init__(self, prompt_token_ids, num_computed_tokens=0):
        self.prompt_token_ids = prompt_token_ids
        self.num_computed_tokens = num_computed_tokens
        self.in_progress_prompt_logprobs_cpu = None


class _PromptRunner:
    def __init__(self, requests, req_id_to_index, query_start_loc):
        self.num_prompt_logprobs = {}
        self.requests = requests
        self.input_batch = SimpleNamespace(req_id_to_index=req_id_to_index)
        self.query_start_loc = SimpleNamespace(np=query_start_loc)
        self.device = torch.device("cpu")
        self.model_config = SimpleNamespace(logprobs_mode="raw_logprobs")
        self.calls = SimpleNamespace(compute_logits=[], logprobs=[], gather=[], sync=0)

        class _Model:
            def compute_logits(inner_self, hidden):
                self.calls.compute_logits.append(hidden)
                n = hidden.shape[0]
                return torch.eye(16)[:n]

        class _Sampler:
            def compute_logprobs(inner_self, logits):
                self.calls.logprobs.append(logits)
                return torch.log_softmax(logits.float(), dim=-1)

            def gather_logprobs(inner_self, logprobs, k, tgt):
                self.calls.gather.append((logprobs, k, tgt))
                n = logprobs.shape[0]
                token_ids = torch.arange(
                    n * (k + 1), dtype=torch.int32
                ).reshape(n, k + 1)
                gathered = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                out = torch.zeros(n, k + 1)
                out[:, 0] = gathered
                ranks = torch.zeros(n, dtype=torch.int32)
                return token_ids, out, ranks, None

        self.model = _Model()
        self.sampler = _Sampler()

    def _sync_device(self):
        self.calls.sync += 1


def test_prompt_logprobs_empty_when_unused():
    runner = _PromptRunner({}, {}, np.zeros(1))
    assert ModelRunnerFL._get_prompt_logprobs_dict(
        runner, torch.zeros(4, 8), {"r0": 4}
    ) == {}
    assert runner.calls.compute_logits == []


def test_prompt_logprobs_final_chunk_computes_and_returns():
    prompt = [5, 6, 7, 8, 9]
    request = _Request(prompt, num_computed_tokens=1)
    runner = _PromptRunner(
        {"r0": request},
        {"r0": 0},
        np.array([0, 4], dtype=np.int64),
    )
    runner.num_prompt_logprobs = {"r0": 1}
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    result = ModelRunnerFL._get_prompt_logprobs_dict(
        runner, hidden, {"r0": 4}
    )

    assert set(result) == {"r0"}
    assert len(runner.calls.compute_logits) == 1
    torch.testing.assert_close(runner.calls.compute_logits[0], hidden[:3])
    _, _, tgt = runner.calls.gather[0]
    assert tgt.tolist() == prompt[2:5]
    assert result["r0"].logprob_token_ids.shape[0] == 4
    assert runner.num_prompt_logprobs == {}
    assert request.in_progress_prompt_logprobs_cpu is None
    assert runner.calls.sync == 1


# ------------------------------------------------- dummy / profile / sample


def test_dummy_run_mm_encoder_only_returns_empty():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            )
        ),
    )
    hidden, last = ModelRunnerFL._dummy_run(runner, 8)
    torch.testing.assert_close(hidden, torch.tensor([]))
    torch.testing.assert_close(last, torch.tensor([]))


def test_dummy_sampler_run_mm_encoder_only_returns_empty():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            )
        ),
    )
    out = ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(2, 2))
    assert out.numel() == 0
    assert out.dtype == torch.float32


def test_profile_run_non_mm_sampler_warmup():
    calls = SimpleNamespace(dummy=[], sampler=[], sync=[], pooler=[])
    runner = SimpleNamespace(
        supports_mm_inputs=False,
        is_pooling_model=False,
        max_num_tokens=8,
        encoder_cache={"tmp_0": object()},
    )
    runner._dummy_run = lambda n, **kw: (
        calls.dummy.append((n, kw))
        or (torch.zeros(4, 2), torch.zeros(2, 2))
    )
    runner._dummy_sampler_run = lambda last: (
        calls.sampler.append(last) or "sampler"
    )
    runner._dummy_pooler_run = lambda hidden: (
        calls.pooler.append(hidden) or "pooler"
    )
    runner._sync_device = lambda: calls.sync.append(True)

    with _stack(
        (model_runner_module, "get_pp_group",
         lambda: SimpleNamespace(is_last_rank=True)),
    ):
        ModelRunnerFL.profile_run(runner)

    assert calls.dummy == [(8, {"is_profile": True})]
    assert calls.pooler == []
    assert len(calls.sampler) == 1
    assert calls.sampler[0].shape == (2, 2)
    assert calls.sync == [True]
    assert runner.encoder_cache == {}


def test_profile_run_pooling_model_uses_pooler():
    calls = SimpleNamespace(dummy=[], sampler=[], sync=[], pooler=[])
    runner = SimpleNamespace(
        supports_mm_inputs=False,
        is_pooling_model=True,
        max_num_tokens=8,
        encoder_cache={},
    )
    runner._dummy_run = lambda n, **kw: (
        torch.zeros(4, 2), torch.zeros(2, 2)
    )
    runner._dummy_sampler_run = lambda last: calls.sampler.append(last)
    runner._dummy_pooler_run = lambda hidden: (
        calls.pooler.append(hidden) or "pooler"
    )
    runner._sync_device = lambda: calls.sync.append(True)

    with _stack(
        (model_runner_module, "get_pp_group",
         lambda: SimpleNamespace(is_last_rank=True)),
    ):
        ModelRunnerFL.profile_run(runner)

    assert len(calls.pooler) == 1
    assert calls.sampler == []


def test_cleanup_profiling_kv_cache_resets_state():
    layer_t = SimpleNamespace(kv_cache=torch.zeros(2, 2))
    layer_plain = SimpleNamespace()
    layer_q = SimpleNamespace(
        kv_cache=[],
        impl=SimpleNamespace(_k_scale_cache=1.0, _v_scale_cache=2.0),
    )
    static_forward_context = {
        "l0": layer_t,
        "l1": layer_plain,
        "l2": layer_q,
    }
    runner = SimpleNamespace(
        kv_caches=[torch.zeros(1), torch.zeros(1)],
        cross_layers_kv_cache=object(),
        cross_layers_attn_backend=object(),
        attn_groups=[[object()]],
        kv_cache_config=SimpleNamespace(num_blocks=4),
        cache_config=SimpleNamespace(num_gpu_blocks=16),
        compilation_config=SimpleNamespace(
            static_forward_context=static_forward_context
        ),
    )

    with _stack(
        (torch.accelerator, "synchronize", lambda: None),
        (torch.accelerator, "empty_cache", lambda: None),
    ):
        ModelRunnerFL._cleanup_profiling_kv_cache(runner)

    assert runner.kv_caches == []
    assert runner.cross_layers_kv_cache is None
    assert runner.cross_layers_attn_backend is None
    assert runner.attn_groups == []
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cache_config.num_gpu_blocks is None
    assert layer_t.kv_cache.numel() == 0
    assert layer_q.kv_cache == []
    assert layer_q.impl._k_scale_cache is None
    assert layer_q.impl._v_scale_cache is None
    assert not hasattr(layer_plain, "kv_cache")


def test_sample_tokens_without_forward_passthrough():
    runner = SimpleNamespace(
        execute_model_state=None,
        kv_connector_output=None,
        use_async_scheduling=False,
    )
    result = ModelRunnerFL.sample_tokens(runner, grammar_output=None)
    assert isinstance(result, ModelRunnerOutput)
    assert runner.kv_connector_output is None


def test_sample_tokens_sync_pipeline_builds_output():
    calls = SimpleNamespace(
        sample=[], post=[], bookkeeping=[], eplb=[], finalize=[]
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=5,
        num_scheduled_tokens={"r0": 3, "r1": 2},
    )
    logits = torch.zeros(2, 4)
    hidden = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    sample_hidden = hidden[[2, 4]]
    state = ExecuteModelState(
        scheduler_output=scheduler_output,
        logits=logits,
        spec_decode_metadata=None,
        spec_decode_common_attn_metadata=None,
        hidden_states=hidden,
        sample_hidden_states=sample_hidden,
        aux_hidden_states=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        slot_mappings=None,
    )

    class _SamplerOutput:
        sampled_token_ids = torch.tensor([[10], [20]], dtype=torch.int32)
        logprobs_tensors = None

    def _bookkeeping(sched, sampler_out, rec_logits, rec_hidden, total):
        calls.bookkeeping.append((sched, rec_logits, rec_hidden, total))
        return (
            {"r0": 0},   # num_nans_in_logits
            2,           # num_nans_device
            {"r0": None},  # logprobs_lists
            [[10], [20]],  # valid_sampled_token_ids
            {},          # prompt_logprobs_dict
            ["r0", "r1"],  # req_ids
            {"r0": 0, "r1": 1},  # req_id_to_index
            [1],         # invalid_req_indices
        )

    runner = SimpleNamespace(
        execute_model_state=state,
        kv_connector_output=None,
        use_async_scheduling=False,
        _sample=lambda l, m: calls.sample.append(l) or _SamplerOutput(),
        _update_states_after_model_execute=(
            lambda sampled, sched: calls.post.append((sampled, sched))
        ),
        _bookkeeping_sync=_bookkeeping,
        eplb_step=lambda: calls.eplb.append(True),
        finalize_kv_connector=lambda: calls.finalize.append(True),
        speculative_config=None,
        supports_mm_inputs=False,
        _draft_token_ids=object(),
        _draft_probs=object(),
        _draft_prob_req_ids=object(),
        _draft_token_req_ids=object(),
        valid_sampled_token_count_gpu=object(),
        input_batch=SimpleNamespace(prev_sampled_token_ids=object()),
    )

    with _stack(
        (model_runner_module, "advance_io_step", lambda: calls.finalize.append("io")),
    ):
        output = ModelRunnerFL.sample_tokens(runner, grammar_output=None)

    assert calls.sample == [logits]
    assert calls.post == [
        (_SamplerOutput.sampled_token_ids, scheduler_output)
    ]
    assert calls.bookkeeping == [(scheduler_output, logits, hidden, 5)]
    assert calls.eplb == [True]
    # No speculative config: deferred KV-connector finalize is skipped, but
    # the io-dump step advance still runs.
    assert calls.finalize == ["io"]
    assert runner._draft_token_ids is None
    assert runner._draft_probs is None
    assert runner.valid_sampled_token_count_gpu is None
    assert runner.input_batch.prev_sampled_token_ids is None
    assert runner.execute_model_state is None
    assert output.req_ids == ["r0", "r1"]
    assert output.req_id_to_index == {"r0": 0, "r1": 1}
    assert output.sampled_token_ids == [[10], [20]]
    assert output.logprobs == {"r0": None}
    assert output.prompt_logprobs_dict == {}
    assert output.num_nans_in_logits == {"r0": 0}


def test_input_fits_in_drafter_bounds():
    assert ModelRunnerFL._input_fits_in_drafter(
        SimpleNamespace(speculative_config=None), None
    ) is False

    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
        num_spec_tokens=2,
        effective_drafter_max_model_len=10,
    )
    # DFlash adds one bonus query token: 8 + 3 > 10, 7 + 3 <= 10.
    assert ModelRunnerFL._input_fits_in_drafter(
        runner, SimpleNamespace(max_seq_len=8)
    ) is False
    assert ModelRunnerFL._input_fits_in_drafter(
        runner, SimpleNamespace(max_seq_len=7)
    ) is True
