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

"""G44 coverage: the branches the g7/g15/g17/g23/g28/g34/g37 waves left
uncovered in ``ModelRunnerFL``.

Primary target is ``_dummy_run`` (814b L6111-6555), which no earlier wave
touched: it is exercised end-to-end on CPU with every collaborator faked at
the module/instance boundary (``_determine_batch_execution_and_padding``,
``_get_slot_mappings``, ``_build_attention_metadata``, the model forward,
``set_forward_context``, ubatch/DCP helpers, ...).  A few remaining branch
edges in ``_prepare_inputs`` (L2136-2484) and ``_build_attention_metadata``
(L2486-2828) are also covered, explicitly avoiding the scenarios already
owned by the earlier suites.

All fakes mirror the production contracts used by the earlier passing waves
(int64 for query_pos / req_indices, the ``_get_cumsum_and_arange`` write-out
behaviour, ...).  No NPU kernel, attention backend, triton or numba is
touched.
"""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import pytest
import torch

from vllm_fl.worker import model_runner as mr


_MISSING = object()


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        if isinstance(self.module, dict):
            self.saved = self.module.get(self.name, _MISSING)
            self.module[self.name] = self.value
        else:
            self.saved = getattr(self.module, self.name, _MISSING)
            setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        if isinstance(self.module, dict):
            if self.saved is _MISSING:
                self.module.pop(self.name, None)
            else:
                self.module[self.name] = self.saved
        else:
            if self.saved is _MISSING:
                delattr(self.module, self.name)
            else:
                setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


class _Buf:
    def __init__(self, np_arr, gpu=None, cpu=None):
        np_arr = np.asarray(np_arr)
        self.np = np_arr
        if gpu is None:
            gpu = torch.from_numpy(np_arr.copy())
        if cpu is None:
            cpu = torch.from_numpy(np_arr.copy())
        self.gpu = gpu
        self.cpu = cpu

    def copy_to_gpu(self, n=None):
        src = torch.as_tensor(self.np)
        if n is None:
            self.gpu.copy_(src)
        else:
            self.gpu[:n].copy_(src[:n])


# ==========================================================================
# CUDAGraphMode shim (kept independent of the real vLLM enum so the branch
# comparisons in _dummy_run are fully deterministic).
# ==========================================================================
class _Mode:
    def __init__(self, name, valid=True):
        self.name = name
        self._valid = valid

    def is_valid_runtime_mode(self) -> bool:
        return self._valid

    def __eq__(self, other):
        return isinstance(other, _Mode) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"CUDAGraphMode.{self.name}"


_MODES = types.SimpleNamespace(
    NONE=_Mode("NONE"),
    PIECEWISE=_Mode("PIECEWISE"),
    FULL=_Mode("FULL"),
)

_ABSENT = type("_AbsentProposer", (), {})
_PROPOSER_NAMES = (
    "EagleProposer",
    "DFlashProposer",
    "DraftModelProposer",
    "ExtractHiddenStatesProposer",
    "Gemma4Proposer",
)


def _proposer_patches(mapping):
    return tuple(
        (mr, name, mapping.get(name, _ABSENT)) for name in _PROPOSER_NAMES
    )


# ==========================================================================
# _dummy_run fakes
# ==========================================================================
class _Model:
    def __init__(self, use_aux=False):
        self.use_aux = use_aux
        self.calls = []
        self.empty_calls = []
        self.requires_raw_input_tokens = False

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        base = kwargs.get("input_ids")
        if base is None:
            base = kwargs.get("inputs_embeds")
        n = int(base.shape[0])
        hidden = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
        if self.use_aux:
            return hidden, torch.zeros_like(hidden)
        return hidden

    def make_empty_intermediate_tensors(self, **kwargs):
        self.empty_calls.append(kwargs)
        return types.SimpleNamespace(marker="inter")


class _Spec:
    def __init__(self, eagle=True, draft=False, extract=False,
                 enforce_eager=False, method="eagle"):
        self._eagle = eagle
        self._draft = draft
        self._extract = extract
        self.enforce_eager = enforce_eager
        self.method = method

    def use_eagle(self):
        return self._eagle

    def uses_draft_model(self):
        return self._draft

    def uses_extract_hidden_states(self):
        return self._extract


class _Eagle:
    def __init__(self, kv_cache_gid=0):
        self.kv_cache_gid = kv_cache_gid
        self.calls = []

    def dummy_run(self, num_tokens, **kwargs):
        self.calls.append((num_tokens, kwargs))


def _dummy_runner(
    num_tokens=4,
    *,
    max_num_tokens=128,
    max_num_seqs=8,
    uniform_decode_query_len=1,
    mode=None,
    batch_num_tokens=None,
    batch_num_reqs=None,
    should_ubatch=False,
    ubatch_slices=None,
    ubatch_slices_padded=None,
    num_tokens_across_dp=None,
    dcp_dummy_len=0,
    dcp_world_size=1,
    mm_encoder_only=False,
    has_kv=True,
    supports_mm_inputs=False,
    enable_prompt_embeds=False,
    uses_mrope=False,
    use_aux=False,
    use_ascend_attention=False,
    pp_first_rank=True,
    lora_config=None,
    speculative_config=None,
    drafter=None,
    cudagraph_specialize_lora=False,
    intermediate_tensors=None,
    is_encoder_decoder=False,
    slot_tensor=None,
):
    mode = _MODES.NONE if mode is None else mode
    if batch_num_tokens is None:
        batch_num_tokens = num_tokens

    model_config = types.SimpleNamespace(
        multimodal_config=(
            types.SimpleNamespace(mm_encoder_only=True)
            if mm_encoder_only
            else None
        ),
        is_encoder_decoder=is_encoder_decoder,
        dtype=torch.float32,
        use_mla=False,
    )
    vllm_config = types.SimpleNamespace(
        model_config=model_config,
        parallel_config=types.SimpleNamespace(num_ubatches=1),
    )

    model = _Model(use_aux=use_aux)
    runner = types.SimpleNamespace(
        _pp_first_rank=pp_first_rank,
        _dcp_dummy_len=dcp_dummy_len,
        _dcp_calls=[],
        _det_calls=[],
        _slot_calls=[],
        _attn_calls=[],
        _fwd_calls=[],
        _nvtx=[],
        _eplb=[],
        _commit_calls=[],
        _pad_calls=[],
        _ubatch_slices=ubatch_slices,
        _ubatch_slices_padded=ubatch_slices_padded,
        vllm_config=vllm_config,
        model_config=model_config,
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=types.SimpleNamespace(cp_kv_cache_interleave_size=1),
        dcp_world_size=dcp_world_size,
        max_num_tokens=max_num_tokens,
        uniform_decode_query_len=uniform_decode_query_len,
        device=torch.device("cpu"),
        supports_mm_inputs=supports_mm_inputs,
        enable_prompt_embeds=enable_prompt_embeds,
        uses_mrope=uses_mrope,
        use_aux_hidden_state_outputs=use_aux,
        use_ascend_attention=use_ascend_attention,
        speculative_config=speculative_config,
        drafter=drafter,
        compilation_config=types.SimpleNamespace(
            cudagraph_specialize_lora=cudagraph_specialize_lora
        ),
        lora_config=lora_config,
        intermediate_tensors=intermediate_tensors,
        model=model,
        input_ids=_Buf(np.zeros(max_num_tokens, dtype=np.int32)),
        inputs_embeds=_Buf(
            np.zeros((max_num_tokens, 8), dtype=np.float32)
        ),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64),
        mrope_positions=types.SimpleNamespace(
            gpu=torch.zeros(3, max_num_tokens, dtype=torch.int64),
            cpu=torch.zeros(3, max_num_tokens, dtype=torch.int64),
        ),
        query_pos=_Buf(np.zeros(max_num_tokens, dtype=np.int64)),
        query_start_loc=_Buf(
            np.zeros(max_num_tokens + 1, dtype=np.int32)
        ),
        optimistic_seq_lens_cpu=torch.zeros(max_num_tokens, dtype=torch.int32),
        seq_lens=torch.zeros(max_num_tokens, dtype=torch.int32),
        input_batch=types.SimpleNamespace(
            block_table=types.SimpleNamespace(
                commit_block_table=lambda n: runner._commit_calls.append(n)
            )
        ),
    )
    if has_kv:
        runner.kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=[types.SimpleNamespace(layer_names=["l0"])]
        )

    def _determine(**kwargs):
        runner._det_calls.append(kwargs)
        return (
            mode,
            types.SimpleNamespace(
                num_tokens=batch_num_tokens,
                num_reqs=batch_num_reqs,
            ),
            should_ubatch,
            num_tokens_across_dp,
            None,
        )

    def _slot_mappings(**kwargs):
        runner._slot_calls.append(kwargs)
        if not has_kv:
            return None, None
        n = kwargs["num_tokens_padded"]
        tensor = (
            slot_tensor
            if slot_tensor is not None
            else torch.ones(n, dtype=torch.int64)
        )
        return {0: tensor}, {}

    def _build_attn(**kwargs):
        runner._attn_calls.append(kwargs)
        return {}, None

    def _rand(input_ids, inputs_embeds, randomize_inputs=False):
        runner._rand_calls.append((input_ids, inputs_embeds, randomize_inputs))
        return contextlib.nullcontext()

    def _pad(qsl, num_tokens_padded, num_reqs, batch_desc_num_reqs=None):
        runner._pad_calls.append(
            (num_tokens_padded, num_reqs, batch_desc_num_reqs)
        )
        base = (
            batch_desc_num_reqs
            if batch_desc_num_reqs is not None
            else num_reqs
        )
        return base + 1

    runner._rand_calls = []
    runner._determine_batch_execution_and_padding = _determine
    runner._get_slot_mappings = _slot_mappings
    runner._build_attention_metadata = _build_attn
    runner._init_model_kwargs = lambda: {}
    runner._prepare_mm_inputs = lambda n: (
        torch.zeros(n, dtype=torch.int64),
        torch.ones(n, 8),
    )
    runner._dummy_mm_kwargs = lambda n: {"mm_tag": n}
    runner._register_layerwise_nvtx_hooks = lambda: runner._nvtx.append(1)
    runner.eplb_step = lambda is_dummy=False, is_profile=False: (
        runner._eplb.append((is_dummy, is_profile))
    )
    runner.synchronize_input_prep = lambda: contextlib.nullcontext()
    runner.maybe_dummy_run_with_lora = lambda *a: contextlib.nullcontext()
    runner.maybe_randomize_inputs = _rand
    runner._get_cumsum_and_arange = (
        lambda nt, out, *a: np.cumsum(np.asarray(nt))
    )
    runner._pad_query_start_loc_for_fia = _pad
    runner.sync_and_gather_intermediate_tensors = lambda *a: (
        types.SimpleNamespace(marker="synced")
    )
    return runner


def _dummy_patches(runner, proposers=None):
    pp = types.SimpleNamespace(
        is_first_rank=runner._pp_first_rank, ranks=[0, 1]
    )

    def _fwd(*args, **kwargs):
        runner._fwd_calls.append((args, kwargs))
        return contextlib.nullcontext()

    return (
        (mr, "CUDAGraphMode", _MODES),
        (mr, "get_pp_group", lambda: pp),
        (mr, "get_dcp_dummy_context_len", lambda *a: runner._dcp_dummy_len),
        (
            mr,
            "maybe_create_ubatch_slices",
            lambda *a: (runner._ubatch_slices, runner._ubatch_slices_padded),
        ),
        (
            mr,
            "prepare_dcp_dummy_context_metadata",
            lambda **k: runner._dcp_calls.append(k),
        ),
        (mr, "set_forward_context", _fwd),
        *_proposer_patches(proposers or {}),
    )


def _ascend_dummy_module():
    full = "vllm_fl.dispatch.backends.vendor.ascend.impl.attention"
    fake = types.ModuleType(full)
    fake.get_ascend_dummy_attention_state = lambda *a, **k: "DUMMY_STATE"
    return full, fake


def _run_dummy(runner, num_tokens=4, *, ascend=None, proposers=None, **kw):
    stack = _dummy_patches(runner, proposers)
    if ascend is not None:
        stack = stack + ((sys.modules, ascend[0], ascend[1]),)
    with _stack(*stack):
        return mr.ModelRunnerFL._dummy_run(runner, num_tokens, **kw)


# ==========================================================================
# _dummy_run tests
# ==========================================================================
def test_g44_dummy_mm_encoder_only_early_return():
    runner = _dummy_runner(4, mm_encoder_only=True)
    a, b = _run_dummy(runner, 4)
    assert a.numel() == 0 and b.numel() == 0
    assert runner.model.calls == []


def test_g44_dummy_invalid_runtime_mode_assertion():
    runner = _dummy_runner(4)
    with pytest.raises(AssertionError):
        _run_dummy(runner, 4, cudagraph_runtime_mode=_Mode("BAD", valid=False))


def test_g44_dummy_normal_batching_split():
    runner = _dummy_runner(5, max_num_seqs=2)
    hidden, logits = _run_dummy(runner, 5)
    det = runner._det_calls[0]
    assert det["num_scheduled_tokens_np"].tolist() == [2, 3]
    assert det["max_num_scheduled_tokens"] == 5
    expected = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)[[1, 4]]
    assert torch.equal(logits, expected)
    assert runner._nvtx == [1]
    assert runner._eplb == [(True, False)]


def test_g44_dummy_normal_batching_remainder():
    runner = _dummy_runner(10, max_num_seqs=3)
    _run_dummy(runner, 10)
    assert runner._det_calls[0]["num_scheduled_tokens_np"].tolist() == [3, 3, 4]


def test_g44_dummy_uniform_decode_exact():
    runner = _dummy_runner(4, uniform_decode_query_len=2, max_num_seqs=8)
    _run_dummy(runner, 4, uniform_decode=True)
    det = runner._det_calls[0]
    assert det["num_scheduled_tokens_np"].tolist() == [2, 2]
    assert det["max_num_scheduled_tokens"] == 2


def test_g44_dummy_uniform_decode_remainder():
    runner = _dummy_runner(5, uniform_decode_query_len=2, max_num_seqs=8)
    _run_dummy(runner, 5, uniform_decode=True)
    assert runner._det_calls[0]["num_scheduled_tokens_np"].tolist() == [
        2,
        2,
        1,
    ]


def test_g44_dummy_create_mixed_batch_layout():
    runner = _dummy_runner(6, max_num_seqs=8)
    _run_dummy(runner, 6, create_mixed_batch=True)
    det = runner._det_calls[0]
    assert det["num_scheduled_tokens_np"].tolist() == [1, 1, 1, 3]
    assert det["max_num_scheduled_tokens"] == 3


def test_g44_dummy_create_mixed_batch_rejects_uniform_decode():
    runner = _dummy_runner(6, uniform_decode_query_len=2)
    with pytest.raises(AssertionError):
        _run_dummy(runner, 6, create_mixed_batch=True, uniform_decode=True)


def test_g44_dummy_mode_mismatch_assertion():
    runner = _dummy_runner(4, mode=_MODES.FULL)
    with pytest.raises(AssertionError):
        _run_dummy(runner, 4, cudagraph_runtime_mode=_MODES.NONE)


def test_g44_dummy_force_attention_default_seq_lens():
    runner = _dummy_runner(4, max_num_seqs=8)
    _run_dummy(runner, 4, force_attention=True)
    assert runner.optimistic_seq_lens_cpu[:4].tolist() == [4, 4, 4, 4]
    assert runner.seq_lens[:4].tolist() == [4, 4, 4, 4]
    assert runner.seq_lens[4:].tolist() == [0] * (128 - 4)
    assert runner.query_start_loc.np[:5].tolist() == [0, 1, 2, 3, 4]
    assert runner.query_start_loc.np[5] == 4
    assert len(runner._dcp_calls) == 2
    assert len(runner._attn_calls) == 1
    assert runner._attn_calls[0]["for_cudagraph_capture"] is False


def test_g44_dummy_force_attention_profile_seq_lens():
    runner = _dummy_runner(4)
    _run_dummy(runner, 4, force_attention=True, profile_seq_lens=17)
    assert runner.optimistic_seq_lens_cpu[:4].tolist() == [17, 17, 17, 17]


def test_g44_dummy_force_attention_dcp_context_seq_lens():
    runner = _dummy_runner(4, dcp_dummy_len=5)
    _run_dummy(runner, 4, force_attention=True)
    # max_query_len (== num_tokens, non-uniform) + dcp dummy context.
    assert runner.optimistic_seq_lens_cpu[:4].tolist() == [9, 9, 9, 9]


def test_g44_dummy_mixed_batch_dcp_seq_lens():
    runner = _dummy_runner(6, dcp_dummy_len=5, max_num_seqs=8)
    _run_dummy(runner, 6, create_mixed_batch=True, force_attention=True)
    # 3 decode (1+5) then 1 prefill (3+5).
    assert runner.optimistic_seq_lens_cpu[:4].tolist() == [6, 6, 6, 8]


def test_g44_dummy_mixed_batch_no_dcp_seq_lens():
    runner = _dummy_runner(6, dcp_dummy_len=0, max_num_seqs=8)
    _run_dummy(runner, 6, create_mixed_batch=True, force_attention=True)
    assert runner.optimistic_seq_lens_cpu[:4].tolist() == [1, 1, 1, 4]


def test_g44_dummy_full_mode_padding_and_capture_flag():
    runner = _dummy_runner(4, mode=_MODES.FULL, batch_num_tokens=8)
    _run_dummy(runner, 4)
    attn = runner._attn_calls[0]
    assert attn["num_tokens_padded"] == 8
    assert attn["for_cudagraph_capture"] is True
    assert attn["ubatch_slices"] is None
    assert attn["use_spec_decode"] is False


def test_g44_dummy_ascend_full_pads_query_start_loc():
    runner = _dummy_runner(
        4, mode=_MODES.FULL, use_ascend_attention=True, batch_num_tokens=8
    )
    _run_dummy(runner, 4, ascend=_ascend_dummy_module())
    assert runner._pad_calls
    assert runner._attn_calls[0]["ascend_attn_state"] == "DUMMY_STATE"
    assert runner._attn_calls[0]["for_cudagraph_capture"] is True


def test_g44_dummy_slot_mappings_zeroed():
    slot = torch.ones(8, dtype=torch.int64)
    runner = _dummy_runner(4, slot_tensor=slot)
    _run_dummy(runner, 4)
    assert torch.all(slot == -1)


def test_g44_dummy_mm_inputs_path():
    runner = _dummy_runner(4, supports_mm_inputs=True)
    _run_dummy(runner, 4)
    call = runner.model.calls[0]
    assert call["input_ids"] is not None
    assert call["inputs_embeds"] is not None
    assert call["mm_tag"] == 4  # num_reqs, not tokens


def test_g44_dummy_prompt_embeds_path():
    runner = _dummy_runner(4, enable_prompt_embeds=True)
    _run_dummy(runner, 4)
    call = runner.model.calls[0]
    assert call["input_ids"] is None
    assert call["inputs_embeds"].shape[0] == 4


def test_g44_dummy_uses_mrope_positions():
    runner = _dummy_runner(4, uses_mrope=True)
    runner.mrope_positions.gpu.fill_(7)
    _run_dummy(runner, 4)
    positions = runner.model.calls[0]["positions"]
    assert torch.all(positions == 7)
    assert positions.shape[-1] == 4


def test_g44_dummy_pp_not_first_rank_builds_intermediate():
    runner = _dummy_runner(4, pp_first_rank=False)
    _run_dummy(runner, 4)
    assert len(runner.model.empty_calls) == 1
    assert runner.model.calls[0]["intermediate_tensors"].marker == "synced"


def test_g44_dummy_pp_not_first_rank_reuses_intermediate():
    runner = _dummy_runner(
        4, pp_first_rank=False, intermediate_tensors=object()
    )
    _run_dummy(runner, 4)
    assert runner.model.empty_calls == []
    assert runner.model.calls[0]["intermediate_tensors"].marker == "synced"


def test_g44_dummy_ubatch_padded_adjusts_token_count():
    across_dp = torch.zeros(1, dtype=torch.int32)
    runner = _dummy_runner(
        4,
        should_ubatch=True,
        ubatch_slices=[types.SimpleNamespace(num_tokens=4, token_slice=slice(0, 4))],
        ubatch_slices_padded=[types.SimpleNamespace(num_tokens=3)],
        num_tokens_across_dp=across_dp,
    )
    _run_dummy(runner, 4)
    assert int(across_dp[0]) == 3
    assert runner._fwd_calls[0][1]["num_tokens"] == 3


def test_g44_dummy_aux_hidden_state_outputs():
    runner = _dummy_runner(4, use_aux=True)
    hidden, logits = _run_dummy(runner, 4)
    expected = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
    assert torch.equal(hidden, expected)
    assert torch.equal(logits, expected)


def test_g44_dummy_spec_decode_piecewise_capture_uses_cudagraphs():
    spec = _Spec(eagle=True)
    drafter = _Eagle()
    runner = _dummy_runner(
        4, mode=_MODES.PIECEWISE, speculative_config=spec, drafter=drafter
    )
    _run_dummy(
        runner,
        4,
        is_graph_capturing=True,
        proposers={"EagleProposer": _Eagle},
    )
    assert drafter.calls
    assert drafter.calls[0][1]["use_cudagraphs"] is True
    assert drafter.calls[0][1]["is_graph_capturing"] is True


def test_g44_dummy_spec_decode_none_mode_no_cudagraphs():
    spec = _Spec(eagle=True)
    drafter = _Eagle()
    runner = _dummy_runner(
        4, mode=_MODES.NONE, speculative_config=spec, drafter=drafter
    )
    _run_dummy(
        runner, 4, proposers={"EagleProposer": _Eagle}
    )
    assert drafter.calls[0][1]["use_cudagraphs"] is False


def test_g44_dummy_spec_decode_enforce_eager_disables():
    spec = _Spec(eagle=True, enforce_eager=True)
    drafter = _Eagle()
    runner = _dummy_runner(
        4, mode=_MODES.PIECEWISE, speculative_config=spec, drafter=drafter
    )
    _run_dummy(
        runner, 4, proposers={"EagleProposer": _Eagle}
    )
    assert drafter.calls[0][1]["use_cudagraphs"] is False


def test_g44_dummy_spec_decode_lora_specialize_disables():
    spec = _Spec(eagle=True)
    drafter = _Eagle()
    runner = _dummy_runner(
        4,
        mode=_MODES.PIECEWISE,
        speculative_config=spec,
        drafter=drafter,
        cudagraph_specialize_lora=True,
    )
    _run_dummy(
        runner,
        4,
        is_graph_capturing=True,
        num_active_loras=1,
        proposers={"EagleProposer": _Eagle},
    )
    assert drafter.calls[0][1]["use_cudagraphs"] is False


def test_g44_dummy_spec_decode_wrong_drafter_asserts():
    spec = _Spec(eagle=True)
    runner = _dummy_runner(
        4, mode=_MODES.NONE, speculative_config=spec, drafter=object()
    )
    with pytest.raises(AssertionError):
        _run_dummy(runner, 4, proposers={"EagleProposer": _Eagle})


def test_g44_dummy_skip_eplb_toggle():
    runner = _dummy_runner(4)
    _run_dummy(runner, 4, skip_eplb=True)
    assert runner._eplb == []

    runner2 = _dummy_runner(4)
    _run_dummy(runner2, 4, skip_eplb=False, is_profile=True)
    assert runner2._eplb == [(True, True)]


def test_g44_dummy_randomize_inputs_forwarded():
    runner = _dummy_runner(4)
    _run_dummy(runner, 4, randomize_inputs=True)
    assert runner._rand_calls
    assert runner._rand_calls[0][2] is True


# ==========================================================================
# _prepare_inputs remaining edges (g23/g37 own the rest)
# ==========================================================================
_MAX_REQS = 8
_MAX_TOKENS = 32
_TOKEN_STRIDE = 16


def _cumsum_and_arange(num_tokens, arange_out, cumsum_dtype=None):
    num_tokens = np.asarray(num_tokens)
    cu = np.cumsum(num_tokens, dtype=cumsum_dtype)
    total = int(cu[-1])
    if total:
        arange_out[:total] = np.concatenate(
            [np.arange(int(n)) for n in num_tokens]
        )
    return cu


def _fill_prev(holder, num_reqs):
    prev_map = holder.input_batch.prev_req_id_to_index
    out = holder.prev_positions.np[:num_reqs]
    if not prev_map:
        out.fill(-1)
        return
    for i, req_id in enumerate(holder.input_batch.req_ids[:num_reqs]):
        out[i] = prev_map.get(req_id, -1)


def _prep_runner(nst, *, total=None, **over):
    nst = np.asarray(nst, dtype=np.int32)
    num_reqs = int(nst.size)
    total = int(nst.sum()) if total is None else total
    req_ids = [f"r{i}" for i in range(num_reqs)]

    token_ids = np.zeros((_MAX_REQS, _TOKEN_STRIDE), dtype=np.int32)
    batch = types.SimpleNamespace(
        num_reqs=num_reqs,
        block_table=types.SimpleNamespace(
            commit_block_table=lambda n: None,
            compute_slot_mapping=lambda *a, **k: None,
        ),
        num_computed_tokens_cpu=np.zeros(_MAX_REQS, dtype=np.int32),
        num_computed_tokens_cpu_tensor=torch.zeros(_MAX_REQS, dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.zeros(_MAX_REQS, dtype=torch.int32),
        num_prompt_tokens=np.zeros(_MAX_REQS, dtype=np.int32),
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        token_ids_cpu=token_ids,
        token_ids_cpu_tensor=torch.from_numpy(token_ids),
        is_token_ids_tensor=torch.ones(
            _MAX_REQS * _TOKEN_STRIDE, dtype=torch.bool
        ),
        req_prompt_embeds={},
        prev_req_id_to_index={},
        num_accepted_tokens_cpu=np.ones(_MAX_REQS, dtype=np.int32),
        prev_sampled_token_ids=None,
    )
    runner = types.SimpleNamespace(
        input_batch=batch,
        arange_np=np.arange(_MAX_REQS, dtype=np.int64),
        use_ascend_attention=False,
        scheduler_config=types.SimpleNamespace(enable_chunked_prefill=False),
        speculative_config=None,
        query_pos=_Buf(np.zeros(_MAX_TOKENS, dtype=np.int64)),
        _get_cumsum_and_arange=_cumsum_and_arange,
        uses_mrope=False,
        input_ids=_Buf(np.zeros(_MAX_TOKENS, dtype=np.int32)),
        enable_prompt_embeds=False,
        is_token_ids=_Buf(np.ones(_MAX_TOKENS, dtype=bool)),
        inputs_embeds=_Buf(np.zeros((_MAX_TOKENS, 2), dtype=np.float32)),
        query_start_loc=_Buf(np.zeros(_MAX_REQS + 1, dtype=np.int32)),
        optimistic_seq_lens_cpu=torch.zeros(_MAX_REQS, dtype=torch.int32),
        discard_request_mask=_Buf(np.zeros(_MAX_REQS, dtype=bool)),
        num_accepted_tokens_event=None,
        use_async_scheduling=False,
        cache_config=types.SimpleNamespace(mamba_cache_mode="align"),
        num_accepted_tokens=_Buf(np.ones(_MAX_REQS, dtype=np.int32)),
        prev_positions=_Buf(np.full(_MAX_REQS, -1, dtype=np.int64)),
        prev_num_draft_tokens=_Buf(np.zeros(_MAX_REQS, dtype=np.int32)),
        use_async_spec_decode=False,
        valid_sampled_token_count_gpu=None,
        num_computed_tokens=torch.zeros(_MAX_REQS, dtype=torch.int32),
        device=torch.device("cpu"),
        req_indices=_Buf(np.zeros(_MAX_TOKENS, dtype=np.int64)),
        num_scheduled_tokens=_Buf(np.zeros(_MAX_REQS, dtype=np.int32)),
        positions=torch.zeros(_MAX_TOKENS, dtype=torch.int64),
        seq_lens=torch.zeros(_MAX_REQS, dtype=torch.int32),
        requests={
            rid: types.SimpleNamespace(num_tokens=int(nst[i]))
            for i, rid in enumerate(req_ids)
        },
        num_decode_draft_tokens=_Buf(np.full(_MAX_REQS, -1, dtype=np.int32)),
        lora_config=None,
        mrope_positions=types.SimpleNamespace(
            cpu=torch.zeros(3, _MAX_TOKENS + 1, dtype=torch.int64),
            gpu=torch.zeros(3, _MAX_TOKENS + 1, dtype=torch.int64),
        ),
        _prepare_input_ids=lambda *a: None,
    )
    runner._calc_spec_decode_metadata = (
        lambda ndt, cu: types.SimpleNamespace(
            logits_indices=torch.zeros(num_reqs, dtype=torch.int32)
        )
    )
    runner._calc_mrope_positions = lambda so: None
    runner._kl_guard_on = lambda: False
    runner._compute_prev_positions = lambda n, _r=runner: _fill_prev(_r, n)
    for key, value in over.items():
        setattr(runner, key, value)
    return runner, nst


def _sched(total, spec=None, extra=None):
    so = types.SimpleNamespace(
        total_num_scheduled_tokens=total,
        scheduled_spec_decode_tokens={} if spec is None else spec,
    )
    if extra:
        so.__dict__.update(extra)
    return so


def test_g44_prep_async_spec_decode_without_valid_count_copies_cpu():
    runner, nst = _prep_runner([2, 2])
    runner.use_async_spec_decode = True
    runner.valid_sampled_token_count_gpu = None  # fixup disabled
    runner.input_batch.prev_req_id_to_index = {"r0": 0, "r1": 1}
    runner.input_batch.num_computed_tokens_cpu[:2] = [5, 6]
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor(
        [5, 6], dtype=torch.int32
    )
    mr.ModelRunnerFL._prepare_inputs(runner, _sched(4), nst)
    assert runner.num_computed_tokens[:2].tolist() == [5, 6]


def test_g44_prep_async_spec_decode_empty_prev_map_copies_cpu():
    runner, nst = _prep_runner([2, 2])
    runner.use_async_spec_decode = True
    runner.valid_sampled_token_count_gpu = torch.zeros(2, dtype=torch.int32)
    runner.input_batch.prev_req_id_to_index = {}  # bool() -> False
    runner.input_batch.num_computed_tokens_cpu[:2] = [7, 8]
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor(
        [7, 8], dtype=torch.int32
    )
    mr.ModelRunnerFL._prepare_inputs(runner, _sched(4), nst)
    assert runner.num_computed_tokens[:2].tolist() == [7, 8]


# ==========================================================================
# _build_attention_metadata remaining edges
# ==========================================================================
class _CM:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def unpadded(self, num_tokens, num_reqs):
        return ("unpadded", num_tokens, num_reqs)


class _EncoderOnlyAttentionSpec:
    pass


class _UniformTypeKVCacheSpecs:
    def __init__(self, kv_cache_specs):
        self.kv_cache_specs = kv_cache_specs


class _BuilderBase:
    supports_update_block_table = False

    def __init__(self):
        self.built = 0
        self.updated = 0
        self.extras = []
        self.cms = []


class _Builder(_BuilderBase):
    def build_metadata(self, cm, ubid, cascade_len, **extra):
        self.built += 1
        self.cms.append(cm)
        self.extras.append(extra)
        return ("meta", cascade_len, ubid)

    def build_metadata_for_cudagraph_capture(self, cm, ubid):
        self.cms.append(cm)
        return ("cg", ubid)


class _EagleAttn:
    def __init__(self, kv_cache_gid=0):
        self.kv_cache_gid = kv_cache_gid


class _AttnGroup:
    def __init__(self, builder, marker, layers):
        self.layer_names = layers
        self.builder = builder
        self.marker = marker

    def get_metadata_builder(self, ubid):
        return self.builder

    def build_metadata(self, cm, ubid, cascade_len, **extra):
        self.builder.built += 1
        self.builder.extras.append(extra)
        self.builder.cms.append(cm)
        return (self.marker, cascade_len, ubid)

    def build_metadata_for_cudagraph_capture(self, cm, ubid):
        self.builder.cms.append(cm)
        return ("cg", self.marker, ubid)

    def update_block_table(self, cached, bt, sm):
        self.builder.updated += 1
        return ("updated", self.marker, bt)

    def map_to_kernel_block_table(self, bt, ubid):
        return ("mapped", bt)


class _BlockTable:
    def get_device_tensor(self, n):
        return torch.arange(n, dtype=torch.int32).reshape(n, 1)


def _attn_runner(*, groups=None, specs=None, **over):
    if specs is None:
        specs = [object()]
    if groups is None:
        groups = [[_AttnGroup(_Builder(), "g0", ["l0"])] for _ in specs]
    runner = types.SimpleNamespace(
        device=torch.device("cpu"),
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(kv_cache_spec=s) for s in specs
            ]
        ),
        max_model_len=100,
        optimistic_seq_lens_cpu=torch.tensor(
            [5, 6, 0, 0, 0, 0, 0, 0], dtype=torch.int32
        ),
        input_batch=types.SimpleNamespace(
            num_computed_tokens_cpu_tensor=torch.tensor(
                [2, 3, 0, 0, 0, 0, 0, 0], dtype=torch.int32
            ),
            num_prompt_tokens_cpu_tensor=torch.tensor(
                [2, 5, 0, 0, 0, 0, 0, 0], dtype=torch.int32
            ),
            block_table=[_BlockTable() for _ in specs],
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            replayssm_decode_base_cpu_tensor=torch.zeros(
                8, dtype=torch.int32
            ),
        ),
        routed_experts_initialized=False,
        use_async_spec_decode=False,
        is_mm_prefix_lm=False,
        model_config=types.SimpleNamespace(
            rswa_window=None, hf_text_config=types.SimpleNamespace()
        ),
        model=types.SimpleNamespace(mm_prefix_clamp_sliding_window=False),
        cache_config=types.SimpleNamespace(
            use_replayssm=False, kv_sharing_fast_prefill=False
        ),
        query_start_loc=types.SimpleNamespace(
            gpu=torch.zeros(9, dtype=torch.int32),
            cpu=torch.zeros(9, dtype=torch.int32),
        ),
        seq_lens=torch.zeros(8, dtype=torch.int32),
        positions=torch.zeros(24, dtype=torch.int64),
        use_ascend_attention=False,
        _ascend_attn_state=None,
        dcp_world_size=1,
        dcp_rank=0,
        parallel_config=types.SimpleNamespace(cp_kv_cache_interleave_size=1),
        attn_groups=groups,
        speculative_config=None,
        requests={},
    )
    runner._get_encoder_seq_lens = (
        lambda num_sched, spec, num_reqs, for_cudagraph_capture=False: (
            None,
            None,
        )
    )
    for key, value in over.items():
        setattr(runner, key, value)
    return runner


def _attn_base_patches(*extra):
    return (
        (mr, "CommonAttentionMetadata", _CM),
        (mr, "EncoderOnlyAttentionSpec", _EncoderOnlyAttentionSpec),
        (mr, "UniformTypeKVCacheSpecs", _UniformTypeKVCacheSpecs),
        (mr, "NULL_BLOCK_ID", -1),
    ) + extra


def _call_attn(runner, *, num_tokens=3, num_reqs=2, pad=None, **kwargs):
    pad = pad or {}
    slots = kwargs.pop("slot_mappings", None)
    if slots is None:
        slots = {
            i: torch.zeros(24, dtype=torch.int32)
            for i in range(len(runner.attn_groups))
        }
    return mr.ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        max_query_len=2,
        slot_mappings=slots,
        **pad,
        **kwargs,
    )


def test_g44_attn_gdn_and_bailing_spec_decode_extras():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(
        groups=[[g0]],
        speculative_config=object(),
        drafter=object(),
        num_accepted_tokens=_Buf(np.ones(8, dtype=np.int32)),
        num_decode_draft_tokens=_Buf(np.zeros(8, dtype=np.int32)),
    )
    with _stack(
        *_attn_base_patches(
            (mr, "Mamba2AttentionMetadataBuilder", _Builder),
            (mr, "GDNAttentionMetadataBuilder", _Builder),
            (mr, "BailingLinearAttentionMetadataBuilder", _Builder),
        ),
        *_proposer_patches({}),
    ):
        _call_attn(runner, use_spec_decode=True)
    extra = g0.builder.extras[0]
    assert "num_accepted_tokens" in extra
    assert "num_decode_draft_tokens_cpu" in extra


def test_g44_attn_drafter_gid_mismatch_returns_none():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(
        groups=[[g0]],
        speculative_config=object(),
        drafter=_EagleAttn(kv_cache_gid=5),
    )
    with _stack(
        *_attn_base_patches(),
        *_proposer_patches({"EagleProposer": _EagleAttn}),
    ):
        _, spec = _call_attn(
            runner,
            pad=dict(num_tokens_padded=5, num_reqs_padded=4),
        )
    assert spec is None
