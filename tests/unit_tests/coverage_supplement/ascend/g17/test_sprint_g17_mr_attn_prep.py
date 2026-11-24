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

"""G17 coverage: ModelRunnerFL._prepare_inputs (814b 2136-2484) and
_build_attention_metadata (2486-2828).

CPU-only. All collaborators are faked at the module boundary; no attention
backend, NPU kernel or model weights are touched. Fake buffers mimic
CpuGpuBuffer so the real code paths execute.
"""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
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
# Shared fake CpuGpuBuffer
# --------------------------------------------------------------------------
class _Buf:
    def __init__(self, np_arr, gpu=None, cpu=None):
        self.np = np_arr
        if gpu is None:
            gpu = torch.tensor(np_arr)
        if cpu is None:
            cpu = torch.tensor(np_arr)
        self.gpu = gpu
        self.cpu = cpu

    def copy_to_gpu(self, n=None):
        src = torch.tensor(self.np).reshape(-1)
        if self.gpu.numel() >= src.numel():
            self.gpu.reshape(-1)[: src.numel()].copy_(src)
        if self.cpu.numel() >= src.numel():
            self.cpu.reshape(-1)[: src.numel()].copy_(src)


# --------------------------------------------------------------------------
# _prepare_inputs
# --------------------------------------------------------------------------
_MAX_REQS = 6
_MAX_TOKENS = 24
_TOKEN_STRIDE = 8


def _cumsum_and_arange(nst, query_pos):
    cu = np.cumsum(np.asarray(nst, dtype=np.int32)).astype(np.int32)
    idx = 0
    for n in nst:
        query_pos.np[idx : idx + n] = np.arange(n, dtype=np.int32)
        idx += n
    return cu


def _prep_runner(nst, *, total=None, **over):
    nst = np.asarray(nst, dtype=np.int32)
    num_reqs = int(nst.size)
    total = int(nst.sum()) if total is None else total
    max_reqs, max_tokens = _MAX_REQS, _MAX_TOKENS
    req_ids = [f"r{i}" for i in range(num_reqs)]
    token_ids = np.zeros((max_reqs, _TOKEN_STRIDE), dtype=np.int32)
    token_ids_t = torch.from_numpy(token_ids)
    batch = types.SimpleNamespace(
        num_reqs=num_reqs,
        block_table=types.SimpleNamespace(
            commit_block_table=lambda n: None,
            compute_slot_mapping=lambda *a: None,
        ),
        num_computed_tokens_cpu=np.zeros(max_reqs, dtype=np.int32),
        num_computed_tokens_cpu_tensor=torch.zeros(max_reqs, dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.zeros(max_reqs, dtype=torch.int32),
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        token_ids_cpu=token_ids,
        token_ids_cpu_tensor=token_ids_t,
        is_token_ids_tensor=torch.ones(
            max_reqs * _TOKEN_STRIDE, dtype=torch.bool
        ),
        req_prompt_embeds={},
        prev_req_id_to_index={},
        num_accepted_tokens_cpu=np.ones(max_reqs, dtype=np.int32),
        prev_sampled_token_ids=None,
    )
    runner = types.SimpleNamespace(
        input_batch=batch,
        arange_np=np.arange(max_reqs),
        use_ascend_attention=False,
        scheduler_config=types.SimpleNamespace(enable_chunked_prefill=False),
        speculative_config=None,
        query_pos=_Buf(np.zeros(max_tokens, dtype=np.int32)),
        _get_cumsum_and_arange=_cumsum_and_arange,
        uses_mrope=False,
        input_ids=_Buf(
            np.zeros(max_tokens, dtype=np.int32),
            gpu=torch.zeros(max_tokens, dtype=torch.int32),
            cpu=torch.zeros(max_tokens, dtype=torch.int32),
        ),
        enable_prompt_embeds=False,
        is_token_ids=_Buf(
            np.ones(max_tokens, dtype=bool),
            gpu=torch.zeros(max_tokens, dtype=torch.bool),
            cpu=torch.zeros(max_tokens, dtype=torch.bool),
        ),
        inputs_embeds=_Buf(
            np.zeros((max_tokens, 2), dtype=np.float32),
            gpu=torch.zeros(max_tokens, 2),
            cpu=torch.zeros(max_tokens, 2),
        ),
        query_start_loc=_Buf(np.zeros(max_reqs + 1, dtype=np.int32)),
        optimistic_seq_lens_cpu=torch.zeros(max_reqs, dtype=torch.int32),
        discard_request_mask=_Buf(np.zeros(max_reqs, dtype=bool)),
        num_accepted_tokens_event=None,
        use_async_scheduling=False,
        cache_config=types.SimpleNamespace(mamba_cache_mode="align"),
        num_accepted_tokens=_Buf(np.ones(max_reqs, dtype=np.int32)),
        prev_positions=_Buf(np.full(max_reqs, -1, dtype=np.int32)),
        prev_num_draft_tokens=_Buf(np.zeros(max_reqs, dtype=np.int32)),
        use_async_spec_decode=False,
        valid_sampled_token_count_gpu=None,
        num_computed_tokens=torch.zeros(max_reqs, dtype=torch.int32),
        device=torch.device("cpu"),
        req_indices=_Buf(np.zeros(max_tokens, dtype=np.int32)),
        num_scheduled_tokens=_Buf(np.zeros(max_reqs, dtype=np.int32)),
        positions=torch.zeros(max_tokens, dtype=torch.int64),
        seq_lens=torch.zeros(max_reqs, dtype=torch.int32),
        requests={rid: types.SimpleNamespace(num_tokens=int(nst[i]))
                  for i, rid in enumerate(req_ids)},
        num_decode_draft_tokens=_Buf(np.full(max_reqs, -1, dtype=np.int32)),
        lora_config=None,
        mrope_positions=types.SimpleNamespace(
            cpu=torch.zeros(3, max_tokens),
            gpu=torch.zeros(3, max_tokens),
        ),
        _calc_spec_decode_metadata=lambda ndt, cu: types.SimpleNamespace(
            logits_indices=torch.zeros(num_reqs, dtype=torch.int32)
        ),
        _prepare_input_ids=lambda *a: None,
        _compute_prev_positions=lambda n: None,
    )
    runner._calc_mrope_positions = lambda so: None
    runner._kl_guard_on = lambda: False
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
        return {"updated": cached, "bt": block_table}

    def map_to_kernel_block_table(self, block_table, ubid):
        return ("mapped", block_table)


class _Mamba(_Builder):
    pass


class _Eagle:
    def __init__(self, kv_cache_gid=0):
        self.kv_cache_gid = kv_cache_gid


class _OtherProposer:
    pass


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
    def __init__(self):
        self.rows = []

    def get_device_tensor(self, n):
        t = torch.arange(n, dtype=torch.int32).reshape(n, 1)
        self.rows.append(n)
        return t


def _attn_runner(*, groups=None, specs=None, **over):
    if specs is None:
        specs = [object()]
    if groups is None:
        groups = [[_AttnGroup(_Builder(), "g0", ["l0"])] for _ in specs]
    block_tables = [_BlockTable() for _ in specs]
    runner = types.SimpleNamespace(
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(kv_cache_spec=s) for s in specs
            ]
        ),
        max_model_len=100,
        optimistic_seq_lens_cpu=torch.tensor([5, 6, 0, 0, 0, 0]),
        input_batch=types.SimpleNamespace(
            num_computed_tokens_cpu_tensor=torch.tensor(
                [2, 3, 0, 0, 0, 0], dtype=torch.int32
            ),
            num_prompt_tokens_cpu_tensor=torch.tensor(
                [2, 5, 0, 0, 0, 0], dtype=torch.int32
            ),
            block_table=block_tables,
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            replayssm_decode_base_cpu_tensor=torch.zeros(6, dtype=torch.int32),
        ),
        routed_experts_initialized=False,
        use_async_spec_decode=False,
        is_mm_prefix_lm=False,
        model_config=types.SimpleNamespace(
            rswa_window=None, hf_text_config=types.SimpleNamespace()
        ),
        cache_config=types.SimpleNamespace(
            use_replayssm=False,
            kv_sharing_fast_prefill=False,
            mamba_cache_mode="align",
        ),
        query_start_loc=types.SimpleNamespace(
            gpu=torch.zeros(9, dtype=torch.int32),
            cpu=torch.zeros(9, dtype=torch.int32),
        ),
        seq_lens=torch.zeros(8, dtype=torch.int32),
        positions=torch.zeros(24, dtype=torch.int64),
        use_ascend_attention=False,
        dcp_world_size=1,
        dcp_rank=0,
        parallel_config=types.SimpleNamespace(cp_kv_cache_interleave_size=1),
        attn_groups=groups,
        speculative_config=None,
        requests={},
        model=types.SimpleNamespace(),
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


def _patch_attn_symbols(*pairs):
    base = (
        (mr, "CommonAttentionMetadata", _CM),
        (mr, "EncoderOnlyAttentionSpec", _EncoderOnlyAttentionSpec),
        (mr, "UniformTypeKVCacheSpecs", _UniformTypeKVCacheSpecs),
        (mr, "NULL_BLOCK_ID", -1),
    )
    return base + pairs


def _call_attn(runner, *, num_tokens=3, num_reqs=2, padded=None, **kwargs):
    padded = padded or {}
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
        **padded,
        **kwargs,
    )


def test_attn_metadata_empty_without_kv_groups():
    runner = types.SimpleNamespace(
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[])
    )
    out = mr.ModelRunnerFL._build_attention_metadata(
        runner, num_tokens=3, num_reqs=2, max_query_len=2
    )
    assert out == ({}, None)


def test_attn_metadata_basic_and_cascade_prefix_lens():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    g1 = _AttnGroup(_Builder(), "g1", ["l1"])
    runner = _attn_runner(groups=[[g0], [g1]], specs=[object(), object()])
    with _stack(*_patch_attn_symbols()):
        attn, spec = _call_attn(
            runner, cascade_attn_prefix_lens=[[7], [8]]
        )
    assert spec is None
    assert attn["l0"] == ("g0", 7, 0)
    assert attn["l1"] == ("g1", 8, 0)
    assert g0.builder.cms[0].max_seq_len == 6
    assert g1.builder.cms[0].block_table_tensor.shape[0] == 2






def test_attn_metadata_cudagraph_capture_and_ubatch():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    with _stack(*_patch_attn_symbols()):
        attn, _ = _call_attn(runner, for_cudagraph_capture=True)
        assert attn["l0"] == ("cg", "g0", 0)
        assert g0.builder.cms[0].max_seq_len == 100

    g1 = _AttnGroup(_Builder(), "g1", ["l0"])
    runner2 = _attn_runner(groups=[[g1]])
    with _stack(
        *_patch_attn_symbols(
            (mr, "split_attn_metadata", lambda slices, cm: [cm])
        )
    ):
        attn, _ = _call_attn(
            runner2, ubatch_slices=[object()],
            padded=dict(num_tokens_padded=4, num_reqs_padded=2),
        )
    assert isinstance(attn, list)
    assert attn[0]["l0"] == ("g1", 0, 0)


def test_attn_metadata_routed_experts_snapshot():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.routed_experts_initialized = True
    runner.routed_experts_capturer = types.SimpleNamespace(attn_gid=0)
    runner.routed_experts_slot_mapping_device = torch.zeros(
        24, dtype=torch.int32
    )
    slots = {0: torch.arange(24, dtype=torch.int32)}
    with _stack(*_patch_attn_symbols()):
        _call_attn(runner, slot_mappings=slots)
    assert runner.routed_experts_slot_mapping_device[:3].tolist() == [0, 1, 2]


def test_attn_metadata_mm_prefix_ranges_filtered():
    def _feature(modality, ranges):
        return types.SimpleNamespace(
            modality=modality,
            mm_position=types.SimpleNamespace(
                extract_embeds_range=lambda: list(ranges)
            ),
        )

    runner = _attn_runner(groups=[[_AttnGroup(_Builder(), "g0", ["l0"])]])
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
    with _stack(*_patch_attn_symbols()):
        _call_attn(runner)
    cm = runner.attn_groups[0][0].builder.cms[0]
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
    with _stack(*_patch_attn_symbols()):
        _call_attn(runner2)
    assert g1.builder.cms[0].mm_req_doc_ranges[0] == [(0, 9)]


def test_attn_metadata_rswa_replayssm_async():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.model_config.rswa_window = 8
    runner.cache_config.use_replayssm = True
    runner.use_async_spec_decode = True
    with _stack(*_patch_attn_symbols()):
        _call_attn(runner)
    cm = g0.builder.cms[0]
    assert cm.rswa_prefix_lens is not None
    assert cm.replayssm_decode_base_cpu is not None
    assert cm.rswa_prefix_lens.tolist()[:2] == [2, 5]


def test_attn_metadata_ascend_state_explicit():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]], use_ascend_attention=True)
    with _stack(*_patch_attn_symbols()):
        _call_attn(runner, ascend_attn_state="PHASE")
    assert g0.builder.cms[0].attn_state == "PHASE"


def test_attn_metadata_dcp_and_kv_sharing_fast_prefill():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.dcp_world_size = 2
    runner.dcp_local_seq_lens = _Buf(np.zeros(6, dtype=np.int32))
    runner.cache_config.kv_sharing_fast_prefill = True
    with _stack(
        *_patch_attn_symbols(
            (mr, "get_dcp_local_seq_lens",
             lambda seqs, ws, rank, interleave: torch.zeros(seqs.shape[0]))
        )
    ):
        runner._prepare_kv_sharing_fast_prefill = lambda li: "padded"
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


def test_attn_metadata_eagle_drafter_unpadded_and_generic():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.speculative_config = object()
    runner.drafter = _Eagle(kv_cache_gid=0)
    with _stack(
        *_patch_attn_symbols(
            (mr, "EagleProposer", _Eagle),
            (mr, "DFlashProposer", _OtherProposer),
            (mr, "Gemma4Proposer", _OtherProposer),
            (mr, "ExtractHiddenStatesProposer", _OtherProposer),
        )
    ):
        attn, spec = _call_attn(
            runner,
            num_tokens=3,
            num_reqs=2,
            padded=dict(num_tokens_padded=5, num_reqs_padded=4),
        )
    assert attn["l0"] == ("g0", 0, 0)
    assert spec == ("unpadded", 3, 2)

    g1 = _AttnGroup(_Builder(), "g1", ["l0"])
    runner2 = _attn_runner(groups=[[g1]])
    runner2.speculative_config = object()
    runner2.drafter = _OtherProposer()
    with _stack(*_patch_attn_symbols()):
        _, spec2 = _call_attn(
            runner2,
            num_tokens=3,
            num_reqs=2,
            padded=dict(num_tokens_padded=5, num_reqs_padded=4),
        )
    assert spec2 == ("unpadded", 3, 2)


def test_attn_metadata_step3p5_and_gemma4_per_group():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    g1 = _AttnGroup(_Builder(), "g1", ["l1"])
    runner = _attn_runner(groups=[[g0], [g1]], specs=[object(), object()])
    runner.speculative_config = object()
    runner.drafter = _Step3p5()
    with _stack(
        *_patch_attn_symbols((mr, "Step3p5MTPProposer", _Step3p5))
    ):
        _call_attn(runner)
    assert [c[0] for c in runner.drafter.calls] == [0, 1]

    g2 = _AttnGroup(_Builder(), "g2", ["l0"])
    g3 = _AttnGroup(_Builder(), "g3", ["l1"])
    runner2 = _attn_runner(groups=[[g2], [g3]], specs=[object(), object()])
    runner2.speculative_config = object()
    runner2.drafter = _Gemma4(kv_cache_gid=0)
    with _stack(
        *_patch_attn_symbols(
            (mr, "Gemma4Proposer", _Gemma4),
            (mr, "EagleProposer", _OtherProposer),
            (mr, "DFlashProposer", _OtherProposer),
            (mr, "ExtractHiddenStatesProposer", _OtherProposer),
        )
    ):
        attn, _ = _call_attn(runner2)
    assert [c[0] for c in runner2.drafter.calls] == [0, 1]
    assert attn["l0"] == ("g2", 0, 0)


def test_attn_metadata_mamba_builder_gets_spec_decode_args():
    g0 = _AttnGroup(_Mamba(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.speculative_config = object()
    runner.drafter = _OtherProposer()
    runner.num_accepted_tokens = _Buf(np.ones(6, dtype=np.int32))
    runner.num_decode_draft_tokens = _Buf(np.zeros(6, dtype=np.int32))
    with _stack(
        *_patch_attn_symbols(
            (mr, "Mamba2AttentionMetadataBuilder", _Mamba),
            (mr, "GDNAttentionMetadataBuilder", _OtherProposer),
            (mr, "BailingLinearAttentionMetadataBuilder", _OtherProposer),
        )
    ):
        runner.attn_groups[0][0].build_metadata = (
            lambda cm, ubid, cascade_len, **extra: g0.builder.extras.append(
                extra
            )
        )
        _call_attn(runner, use_spec_decode=True)
    extra = g0.builder.extras[0]
    assert "num_accepted_tokens" in extra
    assert "num_decode_draft_tokens_cpu" in extra
