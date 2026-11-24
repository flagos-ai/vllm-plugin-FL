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

"""G34 coverage: deep residual branches of ModelRunnerFL.

Targets ``vllm_fl/worker/model_runner.py`` (814b) after g7/g13/g14/g17/g23:
  * ``_prepare_inputs``            (L2136-2484)
  * ``_build_attention_metadata``  (L2486-2828)
  * ``execute_model``              (L4405-4784)

These cases deliberately avoid the scenarios already exercised by g14/g23
and instead drive the nested branch combinations left uncovered:
async-scheduling condense variants, the async-spec-decode fixup being
disabled by either a missing count or a missing prev-map, the kl-guard
optimisation interacting with the fixup, lora budget rejection, multi-group
encoder-only / drafter / routed-expert selection, cudagraph+ubatch and
ubatch cache updates, and execute_model's fast-prefill pass-through,
PP broadcast assertion, combined mamba ordering and within-group mixed
backends.

All fakes are module-boundary stand-ins matching production dtypes
(int64 for query_pos / req_indices / prev_positions, int32 for token
counts, bool for masks).  CPU only; no NPU kernel or attention backend.
"""

from __future__ import annotations

import contextlib
import types

import numpy as np
import pytest
import torch

import vllm_fl.worker.model_runner as mr


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


# ==========================================================================
# _prepare_inputs fakes (dtypes mirror production buffers)
# ==========================================================================
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
    batch = types.SimpleNamespace(
        num_reqs=num_reqs,
        block_table=types.SimpleNamespace(
            commit_block_table=lambda n: None,
            compute_slot_mapping=lambda *a, **k: None,
        ),
        num_computed_tokens_cpu=np.zeros(_MAX_REQS, dtype=np.int32),
        num_computed_tokens_cpu_tensor=torch.zeros(_MAX_REQS, dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.zeros(_MAX_REQS, dtype=torch.int32),
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


def _psched(total, spec=None, extra=None):
    so = types.SimpleNamespace(
        total_num_scheduled_tokens=total,
        scheduled_spec_decode_tokens={} if spec is None else spec,
    )
    if extra:
        so.__dict__.update(extra)
    return so


def _sync_event(record=None):
    def _sync():
        if record is not None:
            record.append(1)
    return types.SimpleNamespace(synchronize=_sync)


# ==========================================================================
# _build_attention_metadata fakes
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


class _NeverProposer:
    pass


class _AbsentProposer:
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


# ==========================================================================
# execute_model fakes
# ==========================================================================
class _Ctx:
    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


class _Intermediate:
    def __init__(self, tensors=None):
        self.tensors = tensors or {}

    def __getitem__(self, key):
        return self


class _TensorsHolder(_Intermediate):
    def __getitem__(self, key):
        return self


def _pp(is_last=True, send=None, broadcast=None, ranks=(0, 1)):
    return types.SimpleNamespace(
        is_last_rank=is_last,
        send_tensor_dict=send or (lambda *a, **k: None),
        broadcast_tensor_dict=broadcast or (lambda data, src=0: data),
        ranks=list(ranks),
    )


def _epatches(pp=None, ubatch=None, logger=None, **extra):
    pairs = [
        (mr, "record_function_or_nullcontext", lambda name: _Ctx()),
        (mr, "has_ec_transfer", lambda: False),
        (mr, "has_kv_transfer_group", lambda: False),
        (mr, "get_pp_group", lambda: pp or _pp()),
        (mr, "get_tp_group", lambda: "tp-group"),
        (mr, "set_forward_context", lambda *a, **k: _Ctx(k)),
        (mr, "maybe_create_ubatch_slices", ubatch or (lambda *a: (None, None))),
        (mr, "logger", logger or types.SimpleNamespace(debug=lambda *a: None)),
    ]
    pairs.extend(extra.items())
    return _stack(*pairs)


def _esched(total=4, **extra):
    base = dict(
        total_num_scheduled_tokens=total,
        num_scheduled_tokens={"a": 2, "b": 2},
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        kv_connector_metadata=None,
        num_common_prefix_blocks=0,
    )
    base.update(extra)
    return types.SimpleNamespace(**base)


def _erunner(**extra):
    base = dict(
        execute_model_state=None,
        speculative_config=None,
        vllm_config="vllm-cfg",
        encoder_cache={},
        synchronize_input_prep=lambda: _Ctx(),
        _update_states=lambda so: None,
        _prepare_inputs=lambda so, nst: (torch.arange(2), "spec-meta", 3),
        parallel_config=types.SimpleNamespace(
            distributed_executor_backend="mp",
            data_parallel_size=1,
            use_ubatching=False,
            num_ubatches=2,
        ),
        cache_config=types.SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode=None
        ),
        num_prompt_logprobs=0,
        input_batch=types.SimpleNamespace(
            num_reqs=2,
            req_ids=["a", "b"],
            num_computed_tokens_cpu=np.zeros(2, dtype=np.int64),
            num_accepted_tokens_cpu=np.zeros(2, dtype=np.int32),
        ),
        use_ascend_attention=False,
        _ascend_attn_state="prefill",
        cascade_attn_enabled=False,
        _compute_cascade_attn_prefix_lens=lambda nst, computed, blocks: "prefix",
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        _pad_query_start_loc_for_fia=lambda q, ntp, nr, npr: 99,
        _determine_batch_execution_and_padding=lambda **kw: (
            mr.CUDAGraphMode.NONE,
            types.SimpleNamespace(num_tokens=4, num_reqs=None),
            False,
            None,
            "stats",
        ),
        _allow_microbatching=lambda nr, nst: True,
        _get_slot_mappings=lambda **kw: ({"g": "slots"}, "flat-slots"),
        _build_attention_metadata=lambda **kw: ("attn-meta", "spec-common"),
        _preprocess=lambda so, ntp, it: (
            "ids",
            "embeds",
            "positions",
            it,
            {"extra": 1},
            "ec-conn",
        ),
        model_config=types.SimpleNamespace(is_encoder_decoder=False),
        eplb_state=None,
        maybe_get_kv_connector_output=lambda so, defer_finalize=False: _Ctx(
            "kv-out"
        ),
        _model_forward=lambda **kw: torch.zeros(4, 3),
        use_aux_hidden_state_outputs=False,
        broadcast_pp_output=False,
        is_pooling_model=False,
        _pool=lambda hs, nst, nstnp, kv: "pooled",
        model=types.SimpleNamespace(
            compute_logits=lambda shs: torch.zeros(shs.shape[0], 5)
        ),
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[]),
        attn_groups=[],
        mamba_state_idx=None,
        requests={},
        compilation_config=types.SimpleNamespace(static_forward_context={}),
        num_accepted_tokens=types.SimpleNamespace(
            np=np.zeros(2, dtype=np.int32), copy_to_gpu=lambda n: None
        ),
        _get_mamba_state_copy_funcs=lambda: "copy-funcs",
        _get_mamba_bufs=lambda: types.SimpleNamespace(
            preprocess="pre", postprocess_align=None
        ),
        _dummy_run=lambda n: None,
    )
    base.update(extra)
    return types.SimpleNamespace(**base)


# ==========================================================================
# _prepare_inputs tests
# ==========================================================================
def test_g34_prep_async_condense_all_rows_existing():
    runner, nst = _prep_runner([2, 2])
    runner.num_accepted_tokens_event = _sync_event()
    runner.use_async_scheduling = True
    runner.cache_config.mamba_cache_mode = "align"
    runner.input_batch.prev_req_id_to_index = {"r0": 0, "r1": 1}
    runner.input_batch.num_accepted_tokens_cpu[:2] = [5, 6]
    mr.ModelRunnerFL._prepare_inputs(runner, _psched(4), nst)
    assert runner.num_accepted_tokens.np[:2].tolist() == [5, 6]
    assert runner.input_batch.num_accepted_tokens_cpu[:2].tolist() == [5, 6]


def test_g34_prep_async_align_empty_prev_map_direct_copy():
    runner, nst = _prep_runner([2, 2])
    runner.num_accepted_tokens_event = _sync_event()
    runner.use_async_scheduling = True
    runner.cache_config.mamba_cache_mode = "align"
    runner.input_batch.prev_req_id_to_index = {}
    runner.input_batch.num_accepted_tokens_cpu[:2] = [5, 6]
    mr.ModelRunnerFL._prepare_inputs(runner, _psched(4), nst)
    assert runner.num_accepted_tokens.np[:2].tolist() == [5, 6]
    assert runner.input_batch.num_accepted_tokens_cpu[:2].tolist() == [5, 6]


def test_g34_prep_async_spec_without_valid_counts_copies_cpu():
    runner, nst = _prep_runner([2, 2])
    runner.use_async_spec_decode = True
    runner.valid_sampled_token_count_gpu = None
    runner.input_batch.prev_req_id_to_index = {"r0": 0, "r1": 1}
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor(
        [4, 5], dtype=torch.int32
    )
    with _stack(
        (mr, "update_num_computed_tokens_for_batch_change",
         lambda *a: pytest.fail("fixup must be disabled"))
    ):
        mr.ModelRunnerFL._prepare_inputs(runner, _psched(4), nst)
    assert runner.num_computed_tokens[:2].tolist() == [4, 5]
    assert runner.positions[:4].tolist() == [4, 5, 5, 6]
    assert runner.num_accepted_tokens.np.tolist() == [1] * _MAX_REQS


def test_g34_prep_async_spec_without_prev_map_copies_cpu():
    runner, nst = _prep_runner([2, 2])
    runner.use_async_spec_decode = True
    runner.valid_sampled_token_count_gpu = torch.zeros(2, dtype=torch.int32)
    runner.input_batch.prev_req_id_to_index = {}
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor(
        [7, 8], dtype=torch.int32
    )
    with _stack(
        (mr, "update_num_computed_tokens_for_batch_change",
         lambda *a: pytest.fail("fixup needs prev mapping"))
    ):
        mr.ModelRunnerFL._prepare_inputs(runner, _psched(4), nst)
    assert runner.num_computed_tokens[:2].tolist() == [7, 8]
    assert runner.positions[:4].tolist() == [7, 8, 8, 9]


def test_g34_prep_kl_guard_with_async_fixup_uses_gpu_positions():
    runner, nst = _prep_runner([2, 2])
    runner.use_async_spec_decode = True
    runner.valid_sampled_token_count_gpu = torch.zeros(2, dtype=torch.int32)
    runner.input_batch.prev_req_id_to_index = {"r0": 0, "r1": 1}
    runner._kl_guard_on = lambda: True
    runner.num_computed_tokens[:2] = torch.tensor([1, 2], dtype=torch.int32)
    seen = []
    with _stack(
        (mr, "update_num_computed_tokens_for_batch_change",
         lambda *a: seen.append(a))
    ):
        mr.ModelRunnerFL._prepare_inputs(runner, _psched(4), nst)
    assert len(seen) == 1
    # kl-guard CPU shortcut is skipped because the fixup branch is active.
    assert runner.positions[:4].tolist() == [1, 2, 2, 3]


def test_g34_prep_lora_sampled_tokens_exceed_budget_raises():
    runner, nst = _prep_runner([3, 2], lora_config=object())
    runner.vllm_config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=4)
    )
    runner.set_active_loras = lambda *a: None
    with pytest.raises(AssertionError):
        mr.ModelRunnerFL._prepare_inputs(
            runner, _psched(5, spec={"r0": [9, 8], "r1": [7]}), nst
        )


def test_g34_prep_prompt_embeds_two_requests_offsets():
    runner, nst = _prep_runner([2, 2], enable_prompt_embeds=True)
    runner.input_batch.num_computed_tokens_cpu[:2] = [0, 2]
    e0 = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    e1 = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 100
    runner.input_batch.req_prompt_embeds = {0: e0, 1: e1}
    mr.ModelRunnerFL._prepare_inputs(runner, _psched(4), nst)
    assert torch.equal(runner.inputs_embeds.cpu[0:2], e0[0:2])
    assert torch.equal(runner.inputs_embeds.cpu[2:4], e1[2:4])


def test_g34_prep_mrope_with_prompt_embeds_copies_rows():
    runner, nst = _prep_runner([2, 2], uses_mrope=True, enable_prompt_embeds=True)
    runner.input_batch.num_computed_tokens_cpu[:2] = [0, 0]
    e0 = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    runner.input_batch.req_prompt_embeds = {0: e0}
    runner._calc_mrope_positions = lambda so: None
    runner.mrope_positions.cpu.fill_(7)
    mr.ModelRunnerFL._prepare_inputs(runner, _psched(4), nst)
    assert torch.equal(runner.inputs_embeds.cpu[0:2], e0)
    assert torch.all(runner.mrope_positions.gpu[:, :4] == 7)


# ==========================================================================
# _build_attention_metadata tests
# ==========================================================================
def test_g34_attn_second_group_encoder_only_block_table_padded():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    g1 = _AttnGroup(_Builder(), "g1", ["l1"])
    runner = _attn_runner(
        groups=[[g0], [g1]],
        specs=[object(), _EncoderOnlyAttentionSpec()],
    )
    with _stack(*_base_patches()):
        attn, _ = _call_attn(
            runner,
            num_tokens=4,
            num_reqs=2,
            pad=dict(num_tokens_padded=4, num_reqs_padded=4),
        )
    assert "l0" in attn and "l1" in attn
    enc_cm = runner.attn_groups[1][0].builder.cms[0]
    assert enc_cm.block_table_tensor.shape == (4, 1)
    assert enc_cm.block_table_tensor[2:].tolist() == [[-1], [-1]]


def test_g34_attn_cudagraph_capture_with_ubatch_list():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    with _stack(
        *_base_patches(
            (mr, "split_attn_metadata", lambda slices, cm: [cm, cm])
        )
    ):
        attn, _ = _call_attn(
            runner,
            for_cudagraph_capture=True,
            ubatch_slices=[object(), object()],
            pad=dict(num_tokens_padded=4, num_reqs_padded=2),
        )
    assert isinstance(attn, list) and len(attn) == 2
    assert attn[0]["l0"] == ("cg", "g0", 0)
    assert attn[1]["l0"] == ("cg", "g0", 1)
    assert g0.builder.cms[0].max_seq_len == runner.max_model_len
    assert g0.builder.cms[0].num_reqs == 2


def test_g34_attn_eagle_drafter_selects_second_group():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    g1 = _AttnGroup(_Builder(), "g1", ["l1"])
    runner = _attn_runner(groups=[[g0], [g1]], specs=[object(), object()])
    runner.speculative_config = object()
    runner.drafter = _Eagle(kv_cache_gid=1)
    with _stack(*_base_patches(), *_proposer_patches(EagleProposer=_Eagle)):
        attn, spec = _call_attn(
            runner, pad=dict(num_tokens_padded=5, num_reqs_padded=4)
        )
    assert set(attn) == {"l0", "l1"}
    assert spec == ("unpadded", 3, 2)


def test_g34_attn_mamba_spec_decode_rejects_ubatching():
    g0 = _AttnGroup(_Mamba(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.speculative_config = object()
    runner.drafter = _NeverProposer()
    with _stack(
        *_base_patches(
            (mr, "Mamba2AttentionMetadataBuilder", _Mamba),
            (mr, "GDNAttentionMetadataBuilder", _NeverProposer),
            (mr, "BailingLinearAttentionMetadataBuilder", _NeverProposer),
            (mr, "split_attn_metadata", lambda slices, cm: [cm, cm]),
        ),
        *_proposer_patches(),
    ):
        with pytest.raises(AssertionError):
            _call_attn(
                runner,
                ubatch_slices=[object(), object()],
                use_spec_decode=True,
                pad=dict(num_tokens_padded=4, num_reqs_padded=2),
            )


def test_g34_attn_ubatch_cached_update_block_table():
    shared = object()
    g0 = _AttnGroup(_UpdateBuilder(), "g0", ["l0"])
    g1 = _AttnGroup(_UpdateBuilder(), "g1", ["l0"])
    runner = _attn_runner(
        groups=[[g0], [g1]], specs=[shared, shared]
    )
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
    assert "l0" in attn[0] and "l0" in attn[1]
    # gid0 builds once (ubid0) then updates (ubid1); gid1 always updates.
    assert g0.builder.built == 1
    assert g0.builder.updated == 1
    assert g1.builder.built == 0
    assert g1.builder.updated == 2


def test_g34_attn_eagle_unpads_when_only_tokens_padded():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.speculative_config = object()
    runner.drafter = _Eagle(kv_cache_gid=0)
    with _stack(*_base_patches(), *_proposer_patches(EagleProposer=_Eagle)):
        _, spec = _call_attn(
            runner, pad=dict(num_tokens_padded=5)
        )
    assert spec == ("unpadded", 3, 2)


def test_g34_attn_requires_slot_mappings():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    with _stack(*_base_patches()):
        with pytest.raises(AssertionError):
            mr.ModelRunnerFL._build_attention_metadata(
                runner,
                num_tokens=3,
                num_reqs=2,
                max_query_len=2,
                slot_mappings=None,
            )


def test_g34_attn_mm_prefix_clamp_without_sliding_window():
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
    runner.model_config.hf_text_config = types.SimpleNamespace()
    runner.model = types.SimpleNamespace(mm_prefix_clamp_sliding_window=True)
    runner.requests = {
        "r0": types.SimpleNamespace(
            mm_features=[_feature("image", [(0, 9)])]
        ),
        "r1": types.SimpleNamespace(mm_features=[]),
    }
    with _stack(*_base_patches()):
        _call_attn(runner)
    cm = g0.builder.cms[0]
    assert cm.mm_req_doc_ranges == {0: [(0, 9)], 1: []}


def test_g34_attn_routed_experts_second_attn_group():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    g1 = _AttnGroup(_Builder(), "g1", ["l1"])
    runner = _attn_runner(groups=[[g0], [g1]], specs=[object(), object()])
    runner.routed_experts_initialized = True
    runner.routed_experts_capturer = types.SimpleNamespace(attn_gid=1)
    runner.routed_experts_slot_mapping_device = torch.zeros(
        24, dtype=torch.int32
    )
    slots = {
        0: torch.zeros(24, dtype=torch.int32),
        1: torch.arange(24, dtype=torch.int32),
    }
    with _stack(*_base_patches()):
        _call_attn(runner, slot_mappings=slots)
    assert runner.routed_experts_slot_mapping_device[:3].tolist() == [0, 1, 2]


def test_g34_attn_dcp_without_logits_skips_kv_sharing():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.dcp_world_size = 2
    runner.dcp_local_seq_lens = _Buf(np.zeros(8, dtype=np.int32))
    runner.cache_config.kv_sharing_fast_prefill = True
    with _stack(
        *_base_patches(
            (mr, "get_dcp_local_seq_lens",
             lambda seqs, ws, rank, interleave: torch.zeros(seqs.shape[0]))
        )
    ):
        _call_attn(runner)
    cm = g0.builder.cms[0]
    assert cm.dcp_local_seq_lens is not None
    assert getattr(cm, "num_logits_indices", None) is None


def test_g34_attn_gid1_slot_mapping_used_per_group():
    shared = object()
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    g1 = _AttnGroup(_Builder(), "g1", ["l1"])
    runner = _attn_runner(groups=[[g0], [g1]], specs=[shared, shared])
    slot0 = torch.zeros(24, dtype=torch.int32)
    slot1 = torch.full((24,), 5, dtype=torch.int32)
    with _stack(*_base_patches()):
        _call_attn(runner, slot_mappings={0: slot0, 1: slot1})
    assert g0.builder.cms[0].slot_mapping is slot0
    assert g1.builder.cms[0].slot_mapping is slot1


# ==========================================================================
# execute_model tests
# ==========================================================================
def test_g34_exec_fast_prefill_zero_prompt_logprobs_proceeds():
    runner = _erunner(
        cache_config=types.SimpleNamespace(
            kv_sharing_fast_prefill=True, mamba_cache_mode=None
        ),
        num_prompt_logprobs=0,
    )
    with _epatches():
        out = mr.ModelRunnerFL.execute_model(runner, _esched())
    assert out is None
    assert runner.execute_model_state is not None


def test_g34_exec_broadcast_pp_none_raises():
    runner = _erunner(broadcast_pp_output=True)
    with _epatches(
        pp=_pp(is_last=True, broadcast=lambda data, src=0: None)
    ):
        with pytest.raises(AssertionError):
            mr.ModelRunnerFL.execute_model(runner, _esched())


def test_g34_exec_mamba_align_applies_deferred_before_preprocess():
    order = []
    runner = _erunner(
        cache_config=types.SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode="align"
        ),
        _update_states=lambda so: (lambda: order.append("deferred")),
        _get_mamba_bufs=lambda: types.SimpleNamespace(
            preprocess="pre", postprocess_align="fused"
        ),
    )
    with _epatches(), _Patch(
        mr.mamba_utils, "preprocess_mamba",
        lambda *a, **k: order.append("pre"),
    ), _Patch(
        mr.mamba_utils, "stage_postprocess_inputs_to_gpu",
        lambda *a: order.append("stage"),
    ):
        mr.ModelRunnerFL.execute_model(runner, _esched())
    assert order == ["deferred", "pre", "stage"]


def test_g34_exec_mixed_backends_within_group_pad_slots():
    captured = {}
    runner = _erunner(
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[types.SimpleNamespace(kv_cache_spec="spec")]
        ),
        attn_groups=[
            [
                types.SimpleNamespace(
                    backend=types.SimpleNamespace(
                        forward_includes_kv_cache_update=True
                    )
                ),
                types.SimpleNamespace(
                    backend=types.SimpleNamespace(
                        forward_includes_kv_cache_update=False
                    )
                ),
            ]
        ],
        _get_slot_mappings=lambda **kw: (
            captured.update(slots=kw) or ({"g": "slots"}, "flat-slots")
        ),
        _build_attention_metadata=lambda **kw: (
            captured.update(build=kw) or ("attn-meta", "spec-common")
        ),
        _determine_batch_execution_and_padding=lambda **kw: (
            mr.CUDAGraphMode.NONE,
            types.SimpleNamespace(num_tokens=8, num_reqs=4),
            False,
            None,
            "stats",
        ),
    )
    with _epatches():
        mr.ModelRunnerFL.execute_model(runner, _esched())
    assert captured["slots"]["num_tokens_padded"] == 8
    assert captured["slots"]["num_reqs_padded"] == 4
    assert captured["build"]["num_tokens_padded"] is None




# ==========================================================================
# Extra residual branches
# ==========================================================================
def test_g34_prep_missing_event_async_align_fills_ones():
    runner, nst = _prep_runner([2, 2])
    runner.num_accepted_tokens_event = None
    runner.use_async_scheduling = True
    runner.cache_config.mamba_cache_mode = "align"
    runner.input_batch.num_accepted_tokens_cpu[:2] = [9, 9]
    mr.ModelRunnerFL._prepare_inputs(runner, _psched(4), nst)
    assert runner.num_accepted_tokens.np.tolist() == [1] * _MAX_REQS
    assert torch.all(runner.num_accepted_tokens.gpu == 1)
    assert runner.input_batch.num_accepted_tokens_cpu[:2].tolist() == [9, 9]


def test_g34_attn_defaults_reqs_padded_independently():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    with _stack(*_base_patches()):
        _call_attn(
            runner,
            num_tokens=3,
            num_reqs=2,
            pad=dict(num_tokens_padded=6),
        )
    cm = g0.builder.cms[0]
    assert cm.num_actual_tokens == 6
    assert cm.num_reqs == 2


def test_g34_attn_uniform_specs_multi_layer_group():
    uniform = _UniformTypeKVCacheSpecs({"l0": object(), "l1": object()})
    g0 = _AttnGroup(_Builder(), "g0", ["l0", "l1"])
    runner = _attn_runner(groups=[[g0]], specs=[uniform])
    with _stack(*_base_patches()):
        attn, _ = _call_attn(runner)
    assert set(attn) == {"l0", "l1"}
    assert attn["l0"] is attn["l1"]
    assert g0.builder.built == 1
