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

"""G37: repair of the 12 failing ``test_prepare_inputs_*`` cases in the g28
(g17-based) suite, targeting ``ModelRunnerFL._prepare_inputs`` (814b
L2136-2484) and ``_build_attention_metadata`` (L2486-2828).

Root cause of the g28 failures (same one g23 fixed): the g28 fakes and helpers
did not match the production buffer contract.
  * ``query_pos`` / ``req_indices`` / ``prev_positions`` / ``positions`` /
    ``mrope_positions`` are **int64** in the runner (L942/L944/L996), not
    int32, and ``torch.index_select`` rejects an int32 index.
  * ``_get_cumsum_and_arange`` must mirror the real helper (write the batch
    arange into ``arange_out`` and return ``cu``), ``_compute_prev_positions``
    must derive ``prev_positions`` from ``prev_req_id_to_index`` (a no-op
    leaves stale values behind), and ``_prepare_inputs`` returns a **3-tuple**
    on this line while g28 asserted as if the values were always present.
  * ``num_accepted_tokens_cpu`` / prompt-embed counters are numpy arrays on
    the CPU side, so the fakes must expose numpy buffers.

This file reuses the proven g23 ``_attn_runner`` / ``_Patch`` / ``_Buf``
objects and the ``_call_prep`` 2-or-3-tuple normalisation.  All collaborators
are faked at the module boundary; CPU only, no NPU kernel or attention
backend is touched.
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


# --------------------------------------------------------------------------
# Shared fake CpuGpuBuffer (dtypes mirror model_runner.py L923-969)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# _prepare_inputs fakes
# --------------------------------------------------------------------------
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
    token_ids_t = torch.from_numpy(token_ids)
    computed_cpu = np.zeros(_MAX_REQS, dtype=np.int32)
    batch = types.SimpleNamespace(
        num_reqs=num_reqs,
        block_table=types.SimpleNamespace(
            commit_block_table=lambda n: None,
            compute_slot_mapping=lambda *a, **k: None,
        ),
        num_computed_tokens_cpu=computed_cpu,
        num_computed_tokens_cpu_tensor=torch.zeros(_MAX_REQS, dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.zeros(_MAX_REQS, dtype=torch.int32),
        num_prompt_tokens=np.zeros(_MAX_REQS, dtype=np.int32),
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        token_ids_cpu=token_ids,
        token_ids_cpu_tensor=token_ids_t,
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
        # arange_np is int64 in production (L995)
        arange_np=np.arange(_MAX_REQS, dtype=np.int64),
        use_ascend_attention=False,
        scheduler_config=types.SimpleNamespace(enable_chunked_prefill=False),
        speculative_config=None,
        # query_pos is int64 in production (L996)
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
        # prev_positions / req_indices are int64 (L942/L944)
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


def _call_prep(runner, sched, nst):
    """Call _prepare_inputs and normalise its 2- or 3-tuple return."""
    result = mr.ModelRunnerFL._prepare_inputs(runner, sched, nst)
    assert isinstance(result, tuple)
    if len(result) == 3:
        return result[0], result[1], result[2]
    assert len(result) == 2
    return result[0], result[1], None


def _ascend_module(record):
    full = "vllm_fl.dispatch.backends.vendor.ascend.impl.attention"
    fake = types.ModuleType(full)

    def classify(computed, scheduled, valid, chunked, method):
        record.append((computed.copy(), scheduled.copy(), valid.copy(),
                       chunked, method))
        return "ASCEND_PHASE"

    fake.classify_ascend_attention_state = classify
    return full, fake


# --------------------------------------------------------------------------
# _prepare_inputs tests (the 12 repaired cases + the g23 extras)
# --------------------------------------------------------------------------
def test_prepare_inputs_asserts_positive_total_tokens():
    runner, nst = _prep_runner([0])
    with pytest.raises(AssertionError):
        mr.ModelRunnerFL._prepare_inputs(runner, _sched(0), nst)


def test_prepare_inputs_basic_text_only():
    runner, nst = _prep_runner([2, 2])
    logits, spec, max_sampled = _call_prep(runner, _sched(4), nst)
    assert spec is None
    assert max_sampled == 1
    assert logits.tolist() == [1, 3]
    assert runner.query_start_loc.np[:3].tolist() == [0, 2, 4]
    assert runner.query_start_loc.np[3:].tolist() == [4] * (_MAX_REQS - 2)
    assert runner.seq_lens[:2].tolist() == [2, 2]
    assert runner.discard_request_mask.np[:2].tolist() == [False, False]
    assert runner.num_accepted_tokens.np.tolist() == [1] * _MAX_REQS


def test_prepare_inputs_ascend_classifies_state():
    runner, nst = _prep_runner([2, 2], use_ascend_attention=True)
    runner.scheduler_config.enable_chunked_prefill = True
    runner.speculative_config = types.SimpleNamespace(method="eagle")
    record = []
    full, fake = _ascend_module(record)
    with _stack((sys.modules, full, fake)):
        _call_prep(runner, _sched(4, spec={"r0": [7]}), nst)
    assert runner._ascend_attn_state == "ASCEND_PHASE"
    computed, scheduled, valid, chunked, method = record[0]
    assert chunked is True and method == "eagle"
    assert valid.tolist() == [1, 2]  # r0 lost one scheduled draft token


def test_prepare_inputs_ascend_state_without_speculative_config():
    runner, nst = _prep_runner([2, 2], use_ascend_attention=True)
    record = []
    full, fake = _ascend_module(record)
    with _stack((sys.modules, full, fake)):
        _call_prep(runner, _sched(4), nst)
    assert record[0][-1] is None


def test_prepare_inputs_prompt_embeds_partial_and_skips():
    runner, nst = _prep_runner([2, 2], enable_prompt_embeds=True)
    runner.input_batch.num_computed_tokens_cpu[:2] = [0, 4]
    embeds = torch.arange(15, dtype=torch.float32).reshape(5, 3)[:, :2]
    runner.input_batch.req_prompt_embeds = {1: embeds}
    _call_prep(runner, _sched(4), nst)
    assert torch.equal(runner.inputs_embeds.cpu[2], embeds[4])
    assert bool(runner.is_token_ids.cpu[:4].all())

    # zero-token request with embeds and a request reading past the embeds
    runner2, nst2 = _prep_runner([2, 0, 2], enable_prompt_embeds=True)
    runner2.input_batch.num_computed_tokens_cpu[2] = 9
    runner2.input_batch.req_prompt_embeds = {
        1: torch.ones(3, 2),   # num_sched == 0 -> skip
        2: torch.ones(2, 2),   # start_pos 9 >= 2 -> skip
    }
    _call_prep(runner2, _sched(4), nst2)
    assert torch.count_nonzero(runner2.inputs_embeds.cpu[:4]) == 0


def test_prepare_inputs_mrope_positions_copied():
    runner, nst = _prep_runner([2, 2], uses_mrope=True)
    seen = []
    runner._calc_mrope_positions = lambda so: seen.append(so)
    runner.mrope_positions.cpu.fill_(3)
    _call_prep(runner, _sched(4), nst)
    assert seen
    assert torch.all(runner.mrope_positions.gpu[:, :4] == 3)


def test_prepare_inputs_accepted_tokens_direct_copy():
    runner, nst = _prep_runner([2, 2])
    syncs = []
    runner.num_accepted_tokens_event = types.SimpleNamespace(
        synchronize=lambda: syncs.append(1)
    )
    runner.input_batch.num_accepted_tokens_cpu[:3] = [2, 3, 4]
    _call_prep(runner, _sched(4), nst)
    assert syncs == [1]
    assert runner.num_accepted_tokens.np[:4].tolist() == [2, 3, 1, 1]
    assert runner.num_computed_tokens[:2].tolist() == [0, 0]


def test_prepare_inputs_accepted_tokens_async_condense_mapping():
    runner, nst = _prep_runner([2, 2])
    runner.num_accepted_tokens_event = types.SimpleNamespace(
        synchronize=lambda: None
    )
    runner.use_async_scheduling = True
    runner.cache_config.mamba_cache_mode = "align"
    runner.input_batch.prev_req_id_to_index = {"r1": 0}
    runner.input_batch.num_accepted_tokens_cpu[:2] = [5, 6]
    _call_prep(runner, _sched(4), nst)
    # r0 is new (-1 -> 1), r1 maps to previous index 0 (5)
    assert runner.num_accepted_tokens.np[:2].tolist() == [1, 5]
    assert runner.input_batch.num_accepted_tokens_cpu[:2].tolist() == [1, 5]


def test_prepare_inputs_accepted_tokens_async_non_align_falls_back_to_ones():
    runner, nst = _prep_runner([2, 2])
    runner.num_accepted_tokens_event = types.SimpleNamespace(
        synchronize=lambda: pytest.fail("must not synchronize")
    )
    runner.use_async_scheduling = True
    runner.cache_config.mamba_cache_mode = "none"
    runner.input_batch.num_accepted_tokens_cpu[:2] = [9, 9]
    _call_prep(runner, _sched(4), nst)
    assert runner.num_accepted_tokens.np.tolist() == [1] * _MAX_REQS
    assert torch.all(runner.num_accepted_tokens.gpu == 1)


def test_prepare_inputs_async_spec_decode_fixup_kernel():
    runner, nst = _prep_runner([2, 2])
    runner.use_async_spec_decode = True
    runner.valid_sampled_token_count_gpu = torch.zeros(2, dtype=torch.int32)
    runner.input_batch.prev_req_id_to_index = {"r0": 0, "r1": 1}
    seen = []
    with _stack(
        (mr, "update_num_computed_tokens_for_batch_change",
         lambda *a: seen.append(a))
    ):
        _call_prep(runner, _sched(4), nst)
    assert len(seen) == 1
    args = seen[0]
    assert args[0] is runner.num_computed_tokens
    assert runner.prev_positions.gpu[:2].tolist() == [0, 1]
    assert runner.prev_num_draft_tokens.gpu.numel() == _MAX_REQS


def test_prepare_inputs_kl_guard_fills_positions_from_cpu():
    runner, nst = _prep_runner([2, 2])
    runner.input_batch.num_computed_tokens_cpu[:2] = [3, 4]
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor(
        [3, 4], dtype=torch.int32
    )
    runner._kl_guard_on = lambda: True
    _call_prep(runner, _sched(4), nst)
    assert runner.positions[:4].tolist() == [3, 4, 4, 5]
    assert runner.seq_lens[:2].tolist() == [5, 6]


def test_prepare_inputs_positions_gather_from_gpu():
    runner, nst = _prep_runner([2, 2])
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor(
        [7, 8], dtype=torch.int32
    )
    _call_prep(runner, _sched(4), nst)
    assert runner.num_computed_tokens[:2].tolist() == [7, 8]
    assert runner.positions[:4].tolist() == [7, 8, 8, 9]


def test_prepare_inputs_spec_decode_metadata_and_counts():
    runner, nst = _prep_runner([3, 1])
    seen = {}

    def _meta(ndt, cu):
        seen["ndt"] = ndt.copy()
        seen["cu"] = cu.copy()
        return types.SimpleNamespace(
            logits_indices=torch.tensor([2, 3], dtype=torch.int32)
        )

    runner._calc_spec_decode_metadata = _meta
    logits, spec, max_sampled = _call_prep(
        runner, _sched(4, spec={"r0": [9, 8]}), nst
    )
    assert logits.tolist() == [2, 3]
    assert spec is not None
    assert max_sampled == 3
    assert seen["ndt"].tolist() == [2, 0]
    assert runner.num_decode_draft_tokens.np[:3].tolist() == [2, -1, -1]


def test_prepare_inputs_spec_decode_chunked_keeps_negative_mask():
    runner, nst = _prep_runner([4, 1])
    logits, spec, max_sampled = _call_prep(
        runner, _sched(5, spec={"r0": [9, 8]}), nst
    )
    # draft_len + 1 == 3 != num_scheduled(4): chunked prefill keeps -1
    assert runner.num_decode_draft_tokens.np[:2].tolist() == [-1, -1]
    assert max_sampled == 3


def test_prepare_inputs_hot_swap_lora():
    runner, nst = _prep_runner([2, 2], lora_config=object())
    runner.vllm_config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=16)
    )
    calls = []
    runner.set_active_loras = lambda batch, ns, nsa: calls.append(
        (batch, ns.copy(), nsa.copy())
    )
    _call_prep(runner, _sched(4, spec={"r0": [5]}), nst)
    assert len(calls) == 1
    assert calls[0][2].tolist() == [2, 1]


def test_prepare_inputs_mrope_async_spec_decode_drift():
    runner, nst = _prep_runner([2, 2], uses_mrope=True)
    runner.use_async_spec_decode = True
    runner.valid_sampled_token_count_gpu = torch.zeros(2, dtype=torch.int32)
    runner.input_batch.prev_req_id_to_index = {"r0": 0, "r1": 1}
    runner.num_computed_tokens[:2] = torch.tensor([5, 6], dtype=torch.int32)
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor(
        [4, 5], dtype=torch.int32
    )
    with _stack(
        (mr, "update_num_computed_tokens_for_batch_change", lambda *a: None)
    ):
        _call_prep(runner, _sched(4), nst)
    # copied rows are 0, optimistic GPU computed tokens are +1 vs CPU
    assert torch.all(runner.mrope_positions.gpu[:, :4] == 1)
    assert torch.all(runner.mrope_positions.gpu[:, 4:] == 0)


# --------------------------------------------------------------------------
# _build_attention_metadata fakes
# --------------------------------------------------------------------------
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


class _NeverProposer:
    pass


class _AbsentProposer:
    """Stand-in for proposer classes that must NOT match `isinstance`."""

    pass


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


class _UpdateBuilder(_Builder):
    supports_update_block_table = True

    def update_block_table(self, cached, block_table, slot_mapping):
        self.updated += 1
        return ("updated", block_table)

    def map_to_kernel_block_table(self, block_table, ubid):
        return ("mapped", block_table)


class _Mamba(_Builder):
    pass


class _Eagle:
    def __init__(self, kv_cache_gid=0):
        self.kv_cache_gid = kv_cache_gid


class _Step3p5:
    def __init__(self):
        self.calls = []

    def set_per_group_attn_metadata(self, gid, block_table, slot_mapping):
        self.calls.append((gid, block_table, slot_mapping))


class _Gemma4:
    def __init__(self, kv_cache_gid=0):
        self.kv_cache_gid = kv_cache_gid
        self.calls = []

    def set_per_group_block_table(self, gid, block_table):
        self.calls.append((gid, block_table))


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


def _base_patches(*extra):
    return (
        (mr, "CommonAttentionMetadata", _CM),
        (mr, "EncoderOnlyAttentionSpec", _EncoderOnlyAttentionSpec),
        (mr, "UniformTypeKVCacheSpecs", _UniformTypeKVCacheSpecs),
        (mr, "NULL_BLOCK_ID", -1),
    ) + extra


def _proposer_patches(**mapping):
    pairs = []
    for name in (
        "EagleProposer",
        "DFlashProposer",
        "Gemma4Proposer",
        "ExtractHiddenStatesProposer",
        "Step3p5MTPProposer",
    ):
        pairs.append((mr, name, mapping.get(name, _AbsentProposer)))
    return tuple(pairs)


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


# ---- tests -------------------------------------------------------------
def test_attn_empty_without_kv_groups():
    runner = types.SimpleNamespace(
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[])
    )
    out = mr.ModelRunnerFL._build_attention_metadata(
        runner, num_tokens=3, num_reqs=2, max_query_len=2
    )
    assert out == ({}, None)


def test_attn_basic_multi_group_and_cascade_prefix_lens():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    g1 = _AttnGroup(_Builder(), "g1", ["l1"])
    runner = _attn_runner(groups=[[g0], [g1]], specs=[object(), object()])
    with _stack(*_base_patches()):
        attn, spec = _call_attn(
            runner, cascade_attn_prefix_lens=[[7], [8]]
        )
    assert spec is None
    assert attn["l0"] == ("g0", 7, 0)
    assert attn["l1"] == ("g1", 8, 0)
    assert g0.builder.cms[0].max_seq_len == 6
    assert g1.builder.cms[0].block_table_tensor.shape[0] == 2


def test_attn_encoder_only_block_table_padded():
    runner = _attn_runner(specs=[_EncoderOnlyAttentionSpec()])
    with _stack(*_base_patches()):
        attn, _ = _call_attn(
            runner,
            num_tokens=4,
            num_reqs=2,
            pad=dict(num_tokens_padded=4, num_reqs_padded=4),
        )
    cm = runner.attn_groups[0][0].builder.cms[0]
    assert cm.block_table_tensor.shape == (4, 1)
    assert cm.block_table_tensor[2:].tolist() == [[-1], [-1]]


def test_attn_uniform_specs_and_cached_block_table_update():
    spec = _UniformTypeKVCacheSpecs({"l0": object()})
    g0 = _AttnGroup(_UpdateBuilder(), "g0", ["l0"])
    g1 = _AttnGroup(_UpdateBuilder(), "g1", ["l0"])
    runner = _attn_runner(groups=[[g0], [g1]], specs=[spec, spec])
    with _stack(*_base_patches()):
        attn, _ = _call_attn(runner)
    assert attn["l0"][0] == "updated"
    assert g0.builder.built == 1
    assert g0.builder.updated == 0
    assert g1.builder.built == 0
    assert g1.builder.updated == 1


def test_attn_cudagraph_capture_uses_max_model_len():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    with _stack(*_base_patches()):
        attn, _ = _call_attn(runner, for_cudagraph_capture=True)
    assert attn["l0"] == ("cg", "g0", 0)
    assert g0.builder.cms[0].max_seq_len == 100


def test_attn_ubatch_slices_builds_list_metadata():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    with _stack(
        *_base_patches(
            (mr, "split_attn_metadata", lambda slices, cm: [cm, cm])
        )
    ):
        attn, _ = _call_attn(
            runner,
            ubatch_slices=[object(), object()],
            pad=dict(num_tokens_padded=4, num_reqs_padded=2),
        )
    assert isinstance(attn, list) and len(attn) == 2
    assert attn[0]["l0"] == ("g0", 0, 0)
    assert attn[1]["l0"] == ("g0", 0, 1)


def test_attn_routed_experts_snapshot():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.routed_experts_initialized = True
    runner.routed_experts_capturer = types.SimpleNamespace(attn_gid=0)
    runner.routed_experts_slot_mapping_device = torch.zeros(
        24, dtype=torch.int32
    )
    slots = {0: torch.arange(24, dtype=torch.int32)}
    with _stack(*_base_patches()):
        _call_attn(runner, slot_mappings=slots)
    assert runner.routed_experts_slot_mapping_device[:3].tolist() == [0, 1, 2]


def test_attn_mm_prefix_ranges_filtered():
    def _feature(modality, ranges):
        return types.SimpleNamespace(
            modality=modality,
            mm_position=types.SimpleNamespace(
                extract_embeds_range=lambda: list(ranges)
            ),
        )

    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.is_mm_prefix_lm = True
    runner.model_config.hf_text_config = types.SimpleNamespace(sliding_window=4)
    runner.model = types.SimpleNamespace(mm_prefix_clamp_sliding_window=False)
    runner.requests = {
        "r0": types.SimpleNamespace(
            mm_features=[
                _feature("audio", [(0, 9)]),
                _feature("image", [(0, 9), (0, 2)]),
            ]
        ),
        "r1": types.SimpleNamespace(mm_features=[]),
    }
    with _stack(*_base_patches()):
        _call_attn(runner)
    cm = g0.builder.cms[0]
    assert cm.mm_req_doc_ranges == {0: [(0, 2)], 1: []}

    g1 = _AttnGroup(_Builder(), "g1", ["l0"])
    runner2 = _attn_runner(groups=[[g1]])
    runner2.is_mm_prefix_lm = True
    runner2.model_config.hf_text_config = types.SimpleNamespace(sliding_window=4)
    runner2.model = types.SimpleNamespace(mm_prefix_clamp_sliding_window=True)
    runner2.requests = {
        "r0": types.SimpleNamespace(
            mm_features=[_feature("image", [(0, 9)])]
        ),
        "r1": types.SimpleNamespace(mm_features=[]),
    }
    with _stack(*_base_patches()):
        _call_attn(runner2)
    assert g1.builder.cms[0].mm_req_doc_ranges[0] == [(0, 9)]


def test_attn_rswa_and_replayssm():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.model_config.rswa_window = 8
    runner.cache_config.use_replayssm = True
    with _stack(*_base_patches()):
        _call_attn(runner)
    cm = g0.builder.cms[0]
    assert cm.rswa_prefix_lens is not None
    assert cm.rswa_prefix_lens.tolist()[:2] == [2, 5]
    assert cm.replayssm_decode_base_cpu is not None


def test_attn_async_spec_decode_and_prefilling_padding():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.use_async_spec_decode = True
    # r0 decode (2 computed >= 2 prompt), r1 still prefilling
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor(
        [2, 3], dtype=torch.int32
    )
    runner.input_batch.num_prompt_tokens_cpu_tensor[:2] = torch.tensor(
        [2, 5], dtype=torch.int32
    )
    with _stack(*_base_patches()):
        _call_attn(
            runner,
            pad=dict(num_tokens_padded=4, num_reqs_padded=4),
        )
    cm = g0.builder.cms[0]
    assert cm.query_start_loc_cpu is not None  # built before Noneing
    assert cm.seq_lens_cpu_upper_bound is not None
    assert cm.is_prefilling.tolist() == [False, True, False, False]


def test_attn_ascend_state_explicit_and_assertion():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]], use_ascend_attention=True)
    with _stack(*_base_patches()):
        _call_attn(runner, ascend_attn_state="PHASE")
    assert g0.builder.cms[0].attn_state == "PHASE"

    g1 = _AttnGroup(_Builder(), "g1", ["l0"])
    runner2 = _attn_runner(
        groups=[[g1]], use_ascend_attention=True, _ascend_attn_state="FALLBACK"
    )
    with _stack(*_base_patches()):
        _call_attn(runner2)
    assert g1.builder.cms[0].attn_state == "FALLBACK"

    g2 = _AttnGroup(_Builder(), "g2", ["l0"])
    runner3 = _attn_runner(groups=[[g2]], use_ascend_attention=True)
    with _stack(*_base_patches()):
        with pytest.raises(AssertionError):
            _call_attn(runner3)


def test_attn_dcp_and_kv_sharing_fast_prefill():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.dcp_world_size = 2
    runner.dcp_local_seq_lens = _Buf(np.zeros(8, dtype=np.int32))
    runner.cache_config.kv_sharing_fast_prefill = True
    runner._prepare_kv_sharing_fast_prefill = lambda li: "padded"
    with _stack(
        *_base_patches(
            (mr, "get_dcp_local_seq_lens",
             lambda seqs, ws, rank, interleave: torch.zeros(seqs.shape[0]))
        )
    ):
        _call_attn(
            runner,
            logits_indices=torch.zeros(3, dtype=torch.int32),
            max_num_sampled_tokens=2,
        )
    cm = g0.builder.cms[0]
    assert cm.dcp_local_seq_lens is not None
    assert cm.num_logits_indices == 3
    assert cm.max_logits_per_req == 2
    assert cm.logits_indices_padded == "padded"


def test_attn_mamba_builder_gets_spec_decode_args():
    g0 = _AttnGroup(_Mamba(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.speculative_config = object()
    runner.drafter = _NeverProposer()
    runner.num_accepted_tokens = _Buf(np.ones(8, dtype=np.int32))
    runner.num_decode_draft_tokens = _Buf(np.zeros(8, dtype=np.int32))
    with _stack(
        *_base_patches(
            (mr, "Mamba2AttentionMetadataBuilder", _Mamba),
            (mr, "GDNAttentionMetadataBuilder", _NeverProposer),
            (mr, "BailingLinearAttentionMetadataBuilder", _NeverProposer),
        ),
        *_proposer_patches(),
    ):
        _call_attn(runner, use_spec_decode=True)
    extra = g0.builder.extras[0]
    assert "num_accepted_tokens" in extra
    assert "num_decode_draft_tokens_cpu" in extra


def test_attn_drafter_eagle_unpads_padded_metadata():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.speculative_config = object()
    runner.drafter = _Eagle(kv_cache_gid=0)
    with _stack(*_base_patches(), *_proposer_patches(EagleProposer=_Eagle)):
        attn, spec = _call_attn(
            runner,
            pad=dict(num_tokens_padded=5, num_reqs_padded=4),
        )
    assert attn["l0"] == ("g0", 0, 0)
    assert spec == ("unpadded", 3, 2)


def test_attn_generic_drafter_keeps_metadata_unpadded_free():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.speculative_config = object()
    runner.drafter = _NeverProposer()
    with _stack(*_base_patches(), *_proposer_patches()):
        _, spec = _call_attn(
            runner,
            pad=dict(num_tokens_padded=5, num_reqs_padded=4),
        )
    # generic proposer returns unpadded metadata when padding was requested
    assert spec == ("unpadded", 3, 2)

    g1 = _AttnGroup(_Builder(), "g1", ["l0"])
    runner2 = _attn_runner(groups=[[g1]])
    runner2.speculative_config = object()
    runner2.drafter = _NeverProposer()
    with _stack(*_base_patches(), *_proposer_patches()):
        _, spec2 = _call_attn(runner2)
    assert hasattr(spec2, "query_start_loc")


def test_attn_step3p5_and_gemma4_per_group():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    g1 = _AttnGroup(_Builder(), "g1", ["l1"])
    runner = _attn_runner(groups=[[g0], [g1]], specs=[object(), object()])
    runner.speculative_config = object()
    runner.drafter = _Step3p5()
    with _stack(
        *_base_patches(),
        *_proposer_patches(Step3p5MTPProposer=_Step3p5),
    ):
        _call_attn(runner)
    assert [c[0] for c in runner.drafter.calls] == [0, 1]

    g2 = _AttnGroup(_Builder(), "g2", ["l0"])
    g3 = _AttnGroup(_Builder(), "g3", ["l1"])
    runner2 = _attn_runner(groups=[[g2], [g3]], specs=[object(), object()])
    runner2.speculative_config = object()
    runner2.drafter = _Gemma4(kv_cache_gid=0)
    with _stack(
        *_base_patches(),
        *_proposer_patches(Gemma4Proposer=_Gemma4),
    ):
        attn, _ = _call_attn(runner2)
    assert [c[0] for c in runner2.drafter.calls] == [0, 1]
    assert attn["l0"] == ("g2", 0, 0)
