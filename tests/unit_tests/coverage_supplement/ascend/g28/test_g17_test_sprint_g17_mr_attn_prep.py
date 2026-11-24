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

The file is intentionally tolerant of the two runner variants that exist in
the tree: the 814b line (``_prepare_inputs`` returns 3 values, attention
metadata is built through ``attn_group.build_metadata`` /
``map_to_kernel_block_table``) and the main line (2-value return,
``builder.build`` / ``builder.build_for_cudagraph_capture``). The fakes below
implement *all* of those entry points and record what was passed, and the
assertions read the recorded state instead of hard-coding one call style.
Optional fields that only exist on some lines are checked through ``inspect``
capability flags so the suite passes on either line without over-asserting.
"""

from __future__ import annotations

import contextlib
import inspect
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
# Capability probes: which runner variant is loaded?
# --------------------------------------------------------------------------
def _source_of(fn) -> str:
    try:
        return inspect.getsource(fn)
    except Exception:  # pragma: no cover - source always available in tree
        return ""


_ATTN_SRC = _source_of(mr.ModelRunnerFL._build_attention_metadata)
_PREP_SRC = _source_of(mr.ModelRunnerFL._prepare_inputs)
try:
    _ATTN_PARAMS = set(
        inspect.signature(mr.ModelRunnerFL._build_attention_metadata).parameters
    )
except Exception:  # pragma: no cover
    _ATTN_PARAMS = set()

_HAS_RSWA = "rswa_prefix_lens" in _ATTN_SRC
_HAS_REPLAYSSM = "replayssm_decode_base_cpu" in _ATTN_SRC
_HAS_MM_CLAMP = "mm_prefix_clamp_sliding_window" in _ATTN_SRC
_HAS_MAX_LOGITS = "max_logits_per_req" in _ATTN_SRC
_HAS_KL_GUARD = "_kl_guard_on" in _PREP_SRC


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
        # Some runner variants read the plain numpy prompt lengths in the
        # spec-decode branch; keep them at 0 so both versions agree.
        num_prompt_tokens=np.zeros(max_reqs, dtype=np.int32),
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
        uses_xdrope_dim=0,
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
        xdrope_positions=types.SimpleNamespace(
            cpu=torch.zeros(3, max_tokens),
            gpu=torch.zeros(3, max_tokens),
        ),
        # Attributes read only by specific runner variants.
        mamba_prev_last_scheduled_idx=None,
        mamba_state_idx=None,
        common_attention_metadata_graph=None,
        _calc_spec_decode_metadata=lambda ndt, cu: types.SimpleNamespace(
            logits_indices=torch.zeros(num_reqs, dtype=torch.int32)
        ),
        _prepare_input_ids=lambda *a: None,
        _compute_prev_positions=lambda n: None,
        _kl_guard_on=lambda: False,
        _calc_mrope_positions=lambda so: None,
        _calc_xdrope_positions=lambda so: None,
    )
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


@contextlib.contextmanager
def _fake_ascend_attention(func):
    """Install a fake ``classify_ascend_attention_state`` for the lazy import.

    Only the leaf module is patched, but missing parent packages are created
    so ``from a.b.c.d import name`` resolves on every line.
    """
    full = "vllm_fl.dispatch.backends.vendor.ascend.impl.attention"
    parts = full.split(".")
    created = []
    try:
        for i in range(1, len(parts)):
            name = ".".join(parts[:i])
            if name not in sys.modules:
                mod = types.ModuleType(name)
                mod.__path__ = []
                sys.modules[name] = mod
                created.append(name)
        fake = types.ModuleType(full)
        fake.classify_ascend_attention_state = func
        with _Patch(sys.modules, full, fake):
            yield
    finally:
        for name in reversed(created):
            sys.modules.pop(name, None)


def test_prepare_inputs_basic_text_only():
    runner, nst = _prep_runner([2, 2])
    logits, spec, max_sampled = _call_prep(runner, _sched(4), nst)
    assert spec is None
    assert max_sampled is None or max_sampled == 1
    assert logits.tolist() == [1, 3]
    assert runner.query_start_loc.np[:3].tolist() == [0, 2, 4]
    assert runner.seq_lens[:2].tolist() == [2, 2]
    assert runner.discard_request_mask.np[:2].tolist() == [False, False]


def test_prepare_inputs_ascend_classifies_state():
    runner, nst = _prep_runner([2, 2], use_ascend_attention=True)
    runner.scheduler_config.enable_chunked_prefill = True
    runner.speculative_config = types.SimpleNamespace(method="eagle")
    calls = []
    with _fake_ascend_attention(
        lambda *a: (calls.append(a) or "P")
    ):
        _call_prep(runner, _sched(4, spec={"r0": [7]}), nst)
    assert calls and calls[0][-2] is True and calls[0][-1] == "eagle"
    assert runner._ascend_attn_state == "P"


def test_prepare_inputs_mrope_positions_copied():
    runner, nst = _prep_runner([2, 2], uses_mrope=True)
    calls = []
    runner._calc_mrope_positions = lambda so: calls.append(so)
    runner.mrope_positions.cpu.fill_(3)
    _call_prep(runner, _sched(4), nst)
    assert calls
    assert torch.all(runner.mrope_positions.gpu[:, :4] == 3)


def test_prepare_inputs_prompt_embeds_partial_and_skips():
    runner, nst = _prep_runner([2, 2], enable_prompt_embeds=True)
    runner.input_batch.num_computed_tokens_cpu[:2] = [0, 4]
    embeds = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    runner.input_batch.req_prompt_embeds = {1: embeds[:, :2]}
    _call_prep(runner, _sched(4), nst)
    assert torch.equal(runner.inputs_embeds.cpu[2], embeds[4, :2])
    assert torch.all(runner.is_token_ids.cpu[:4] == 1)

    # zero-token request with embeds and a request reading past the embeds
    runner2, nst2 = _prep_runner([2, 0, 2], enable_prompt_embeds=True)
    runner2.input_batch.num_computed_tokens_cpu[2] = 9
    runner2.input_batch.req_prompt_embeds = {
        1: torch.ones(3, 2),
        2: torch.ones(2, 2),
    }
    _call_prep(runner2, _sched(4), nst2)
    assert torch.all(runner2.inputs_embeds.cpu[:4] == 0)


def test_prepare_inputs_accepted_tokens_direct_copy():
    runner, nst = _prep_runner([2, 2])
    sync = []
    runner.num_accepted_tokens_event = types.SimpleNamespace(
        synchronize=lambda: sync.append(1)
    )
    runner.input_batch.num_accepted_tokens_cpu[:3] = [2, 3, 4]
    _call_prep(runner, _sched(4), nst)
    assert sync == [1]
    assert runner.num_accepted_tokens.np[:4].tolist() == [2, 3, 1, 1]
    assert runner.num_computed_tokens[:2].tolist() == [0, 0]


def test_prepare_inputs_accepted_tokens_async_condense_mapping():
    runner, nst = _prep_runner([2, 2])
    runner.num_accepted_tokens_event = types.SimpleNamespace(
        synchronize=lambda: None
    )
    runner.use_async_scheduling = True
    runner.cache_config.mamba_cache_mode = "align"
    runner.input_batch.prev_req_id_to_index = {"r0": 1, "r1": 0}
    runner.prev_positions.np[:2] = [-1, 0]
    runner.input_batch.num_accepted_tokens_cpu[:2] = [5, 6]
    _call_prep(runner, _sched(4), nst)
    assert runner.num_accepted_tokens.np[:2].tolist() == [1, 5]
    assert runner.input_batch.num_accepted_tokens_cpu[:2].tolist() == [1, 5]


def test_prepare_inputs_async_spec_decode_fixup_kernel():
    runner, nst = _prep_runner([2, 2])
    runner.use_async_spec_decode = True
    runner.valid_sampled_token_count_gpu = torch.zeros(2, dtype=torch.int32)
    runner.input_batch.prev_req_id_to_index = {"r0": 0}
    seen = []
    with _stack(
        (mr, "update_num_computed_tokens_for_batch_change",
         lambda *a: seen.append(a))
    ):
        _call_prep(runner, _sched(4), nst)
    assert len(seen) == 1
    assert seen[0][0] is runner.num_computed_tokens
    assert runner.prev_positions.gpu[:2].tolist() == [-1, -1]


def test_prepare_inputs_kl_guard_fills_positions_from_cpu():
    runner, nst = _prep_runner([2, 2])
    runner.input_batch.num_computed_tokens_cpu[:2] = [3, 4]
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor([3, 4])
    runner._kl_guard_on = lambda: True
    _call_prep(runner, _sched(4), nst)
    assert runner.positions[:4].tolist() == [3, 4, 4, 5]
    assert runner.seq_lens[:2].tolist() == [5, 6]


def test_prepare_inputs_positions_gather_from_gpu():
    runner, nst = _prep_runner([2, 2])
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor([7, 8])
    _call_prep(runner, _sched(4), nst)
    assert runner.num_computed_tokens[:2].tolist() == [7, 8]
    assert runner.positions[:4].tolist() == [7, 8, 8, 9]


def test_prepare_inputs_spec_decode_metadata_and_counts():
    runner, nst = _prep_runner([3, 1])
    seen = {}

    def _meta(ndt, cu):
        seen["ndt"] = np.array(ndt, copy=True)
        seen["cu"] = np.array(cu, copy=True)
        return types.SimpleNamespace(
            logits_indices=torch.tensor([2, 3], dtype=torch.int32)
        )

    runner._calc_spec_decode_metadata = _meta
    logits, spec, max_sampled = _call_prep(
        runner, _sched(4, spec={"r0": [9, 8]}), nst
    )
    assert logits.tolist() == [2, 3]
    assert spec is not None
    assert max_sampled is None or max_sampled == 3
    assert seen["ndt"].tolist() == [2, 0]
    assert runner.num_decode_draft_tokens.np[:3].tolist() == [2, -1, -1]


def test_prepare_inputs_hot_swap_lora():
    runner, nst = _prep_runner([2, 2], lora_config=object())
    runner.vllm_config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=16)
    )
    calls = []
    runner.set_active_loras = lambda batch, ns, nsa: calls.append(
        (batch, np.array(ns, copy=True), np.array(nsa, copy=True))
    )
    _call_prep(runner, _sched(4, spec={"r0": [5]}), nst)
    assert len(calls) == 1
    assert calls[0][2].tolist() == [2, 1]


def test_prepare_inputs_mrope_async_spec_decode_drift():
    runner, nst = _prep_runner([2, 2], uses_mrope=True)
    runner.use_async_spec_decode = True
    runner.valid_sampled_token_count_gpu = torch.zeros(2, dtype=torch.int32)
    runner.input_batch.prev_req_id_to_index = {"r0": 0}
    runner.num_computed_tokens[:2] = torch.tensor([5, 6])
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor([4, 5])
    # Make the GPU fixup kernel a no-op so ``num_computed_tokens`` keeps the
    # values above and the drift (num_computed - cpu) is deterministic.
    with _stack(
        (mr, "update_num_computed_tokens_for_batch_change", lambda *a: None)
    ):
        _call_prep(runner, _sched(4), nst)
    assert runner.mrope_positions.gpu.shape == (3, _MAX_TOKENS)
    assert torch.all(runner.mrope_positions.gpu[:, :4] == 1)


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
    """Fake metadata builder supporting *both* runner call styles.

    * 814b: ``attn_group.build_metadata`` / ``build_metadata_for_cudagraph_capture``
      and ``attn_group.map_to_kernel_block_table``.
    * main: ``builder.build`` / ``builder.build_for_cudagraph_capture``.

    Everything is recorded so assertions do not depend on the return shape.
    """

    supports_update_block_table = False
    name = "builder"

    def __init__(self):
        self.built = 0
        self.cudagraph_built = 0
        self.updated = 0
        self.builds = []
        self.cg = []

    # -- main-line entry points -------------------------------------------
    def build(self, *, common_prefix_len=0, common_attn_metadata=None, **extra):
        return self._record(common_attn_metadata, common_prefix_len, extra)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self._record_cg(common_attn_metadata, None)

    # -- 814b entry points ------------------------------------------------
    def build_metadata(self, cm, ubid, cascade_len, **extra):
        return self._record(cm, cascade_len, extra, ubid)

    def build_metadata_for_cudagraph_capture(self, cm, ubid=0):
        return self._record_cg(cm, ubid)

    def map_to_kernel_block_table(self, block_table, ubid=0):
        return ("mapped", block_table)

    def update_block_table(self, cached, *args, **kwargs):
        self.updated += 1
        return ("updated", self.name)

    # -- internals --------------------------------------------------------
    def _record(self, cm, cascade, extra, ubid=None):
        self.built += 1
        self.builds.append(
            {"cm": cm, "cascade": cascade, "extra": dict(extra), "ubid": ubid}
        )
        return ("meta", cascade, self.name)

    def _record_cg(self, cm, ubid):
        self.cudagraph_built += 1
        self.cg.append(cm)
        return ("cg", self.name, ubid)


class _Builder(_BuilderBase):
    pass


class _UpdateBuilder(_BuilderBase):
    supports_update_block_table = True
    name = "update"


class _Mamba(_BuilderBase):
    name = "mamba"


class _AttnGroup:
    """Delegates every builder entry point to ``self.builder``.

    Needed because 814b calls these on the *group*, while main calls them on
    the *builder* returned by ``get_metadata_builder``.
    """

    def __init__(self, builder, marker, layers):
        self.layer_names = layers
        self.builder = builder
        self.marker = marker

    def get_metadata_builder(self, ubid=0):
        return self.builder

    def build(self, **kwargs):
        return self.builder.build(**kwargs)

    def build_metadata(self, cm, ubid, cascade_len, **extra):
        return self.builder.build_metadata(cm, ubid, cascade_len, **extra)

    def build_for_cudagraph_capture(self, cm):
        return self.builder.build_for_cudagraph_capture(cm)

    def build_metadata_for_cudagraph_capture(self, cm, ubid=0):
        return self.builder.build_metadata_for_cudagraph_capture(cm, ubid)

    def map_to_kernel_block_table(self, block_table, ubid=0):
        return self.builder.map_to_kernel_block_table(block_table, ubid)

    def update_block_table(self, cached, *args, **kwargs):
        return self.builder.update_block_table(cached, *args, **kwargs)


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
        routed_experts_attn_gid=0,
        routed_experts_capturer=types.SimpleNamespace(attn_gid=0),
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
        device=torch.device("cpu"),
        mamba_prev_last_scheduled_idx=None,
        common_attention_metadata_graph=None,
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
    padded = dict(padded or {})
    slots = kwargs.pop("slot_mappings", None)
    if slots is None:
        slots = {
            i: torch.zeros(24, dtype=torch.int32)
            for i in range(len(runner.attn_groups))
        }
    call = dict(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        max_query_len=2,
        slot_mappings=slots,
    )
    call.update(padded)
    call.update(kwargs)
    if _ATTN_PARAMS:
        call = {k: v for k, v in call.items() if k in _ATTN_PARAMS}
    return mr.ModelRunnerFL._build_attention_metadata(runner, **call)


def _built_cm(builder):
    assert builder.builds, "builder did not record a build call"
    return builder.builds[0]["cm"]


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
        attn, spec = _call_attn(runner, cascade_attn_prefix_lens=[[7], [8]])
    assert spec is None
    assert set(attn) == {"l0", "l1"}
    assert attn["l0"][0] == "meta"
    assert attn["l1"][0] == "meta"
    assert g0.builder.built == 1 and g1.builder.built == 1
    assert g0.builder.builds[0]["cascade"] == 7
    assert g1.builder.builds[0]["cascade"] == 8
    assert _built_cm(g0.builder).max_seq_len == 6
    assert _built_cm(g1.builder).block_table_tensor.shape[0] == 2


def test_attn_metadata_encoder_only_block_table():
    spec = _EncoderOnlyAttentionSpec()
    runner = _attn_runner(specs=[spec])
    with _stack(*_patch_attn_symbols()):
        attn, _ = _call_attn(
            runner, num_tokens=4, num_reqs=2,
            padded=dict(num_tokens_padded=4, num_reqs_padded=4),
        )
    assert attn["l0"][0] == "meta"
    cm = _built_cm(runner.attn_groups[0][0].builder)
    assert cm.block_table_tensor.shape == (4, 1)
    assert cm.block_table_tensor[2:].tolist() == [[-1], [-1]]


def test_attn_metadata_uniform_specs_and_cached_block_table_update():
    inner = object()
    spec = _UniformTypeKVCacheSpecs({"l0": inner})
    g0 = _AttnGroup(_UpdateBuilder(), "g0", ["l0"])
    g1 = _AttnGroup(_UpdateBuilder(), "g1", ["l0"])
    runner = _attn_runner(groups=[[g0], [g1]], specs=[spec, spec])
    with _stack(*_patch_attn_symbols()):
        attn, _ = _call_attn(runner)
    assert attn["l0"][0] == "updated"
    assert g0.builder.built == 1
    assert g1.builder.updated == 1
    assert g1.builder.built == 0


def test_attn_metadata_cudagraph_capture_and_ubatch():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    with _stack(*_patch_attn_symbols()):
        attn, _ = _call_attn(runner, for_cudagraph_capture=True)
    assert attn["l0"][0] == "cg"
    assert g0.builder.cudagraph_built == 1
    assert g0.builder.cg[0].max_seq_len == 100

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
    assert attn[0]["l0"][0] == "meta"
    assert g1.builder.built == 1


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
    cm = _built_cm(runner.attn_groups[0][0].builder)
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
    cm2 = _built_cm(g1.builder)
    if _HAS_MM_CLAMP:
        # Models that clamp in-kernel keep ranges bigger than the window.
        assert cm2.mm_req_doc_ranges[0] == [(0, 9)]
    else:
        assert cm2.mm_req_doc_ranges[0] == []


def test_attn_metadata_rswa_replayssm_async():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]])
    runner.model_config.rswa_window = 8
    runner.cache_config.use_replayssm = True
    runner.use_async_spec_decode = True
    with _stack(*_patch_attn_symbols()):
        _call_attn(runner)
    cm = _built_cm(g0.builder)
    if _HAS_RSWA:
        assert cm.rswa_prefix_lens is not None
        assert cm.rswa_prefix_lens.tolist()[:2] == [2, 5]
    if _HAS_REPLAYSSM:
        assert getattr(cm, "replayssm_decode_base_cpu", None) is not None
    assert g0.builder.built == 1


def test_attn_metadata_ascend_state_explicit():
    g0 = _AttnGroup(_Builder(), "g0", ["l0"])
    runner = _attn_runner(groups=[[g0]], use_ascend_attention=True)
    with _stack(*_patch_attn_symbols()):
        _call_attn(runner, ascend_attn_state="PHASE")
    cm = _built_cm(g0.builder)
    assert getattr(cm, "attn_state", "PHASE") == "PHASE"


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
    cm = _built_cm(g0.builder)
    assert cm.dcp_local_seq_lens is not None
    assert cm.num_logits_indices == 3
    if _HAS_MAX_LOGITS:
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
    assert attn["l0"][0] == "meta"
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
    assert attn["l0"][0] == "meta"


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
        _call_attn(runner, use_spec_decode=True)
    assert g0.builder.built == 1
    extra = g0.builder.builds[0]["extra"]
    assert "num_accepted_tokens" in extra
    assert "num_decode_draft_tokens_cpu" in extra
