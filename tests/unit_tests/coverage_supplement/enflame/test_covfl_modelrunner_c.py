"""Coverage tests for input preparation / attention metadata in model_runner.py.

Scope (kept disjoint from the _a/_b/_d agent files):
  _prepare_inputs, _build_attention_metadata, _preprocess, _bookkeeping_sync,
  execute_model (main non-empty flow), _execute_mm_encoder, _gather_mm_embeddings,
  _prepare_mm_inputs and the deeper branches of _update_states.

Self contained: pytest + unittest.mock + stdlib + torch/numpy only.  The huge
``ModelRunnerFL.__init__`` is bypassed with ``object.__new__`` and only the
attributes each method touches are populated with SimpleNamespace / MagicMock.
No GCU kernel, weight load or process spawn happens.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:  # pragma: no cover - environment dependent
    from vllm_fl.worker import model_runner as mr
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"vllm_fl.worker.model_runner is not importable here: {exc!r}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_runner(**attrs):
    r = object.__new__(mr.ModelRunnerFL)
    r.__dict__.update(attrs)
    return r


class _Buf:
    """Minimal CpuGpuBuffer stand-in (``np`` CPU array + ``gpu`` torch buffer)."""

    def __init__(self, np_=None, gpu=None, cpu=None):
        self.np = np_ if np_ is not None else np.zeros(0, dtype=np.int64)
        self.gpu = gpu if gpu is not None else torch.zeros(0)
        self.cpu = cpu if cpu is not None else torch.zeros(0)
        self.calls = []

    def copy_to_gpu(self, n=None):
        self.calls.append(n)


# ---------------------------------------------------------------------------
# _prepare_mm_inputs
# ---------------------------------------------------------------------------
def test_prepare_mm_inputs_raw_and_embed_only():
    input_ids = _Buf(gpu=torch.arange(4))
    inputs_embeds = _Buf(gpu=torch.zeros(4, 2))
    r = make_runner(
        model=SimpleNamespace(requires_raw_input_tokens=True),
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
    )
    ids, emb = r._prepare_mm_inputs(3)
    assert torch.equal(ids, input_ids.gpu[:3])
    assert torch.equal(emb, inputs_embeds.gpu[:3])

    r.model.requires_raw_input_tokens = False
    ids, emb = r._prepare_mm_inputs(3)
    assert ids is None
    assert torch.equal(emb, inputs_embeds.gpu[:3])


# ---------------------------------------------------------------------------
# _prepare_inputs
# ---------------------------------------------------------------------------
def _mk_prepare_inputs(monkeypatch, *, spec=False, uses_mrope=False,
                       enable_prompt_embeds=False, req_prompt_embeds=None,
                       async_fixup=False, with_event=False):
    num_reqs = 2
    width = 8
    sched = np.array([3, 1] if spec else [2, 1], dtype=np.int32)
    total = int(sched.sum())
    cu = np.cumsum(sched, dtype=np.int64)
    qpos_np = np.concatenate([np.arange(int(n)) for n in sched]).astype(np.int64)
    req_indices = np.repeat(np.arange(num_reqs), sched).astype(np.int64)

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        block_table=SimpleNamespace(
            commit_block_table=Mock(), compute_slot_mapping=Mock()
        ),
        num_computed_tokens_cpu=np.zeros(num_reqs, dtype=np.int32),
        num_computed_tokens_cpu_tensor=torch.zeros(num_reqs, dtype=torch.int32),
        token_ids_cpu=np.zeros((num_reqs, width), dtype=np.int32),
        token_ids_cpu_tensor=torch.arange(
            num_reqs * width, dtype=torch.int32
        ).reshape(num_reqs, width),
        is_token_ids_tensor=torch.ones(num_reqs * width, dtype=torch.bool),
        num_prompt_tokens_cpu_tensor=torch.tensor([4, 1], dtype=torch.int32),
        req_prompt_embeds=req_prompt_embeds or {},
        prev_req_id_to_index=({"a": 0} if async_fixup else None),
        req_ids=["a", "b"],
        req_id_to_index={"a": 0, "b": 1},
        num_accepted_tokens_cpu=np.ones(num_reqs, dtype=np.int32),
    )

    runner = make_runner(
        input_batch=input_batch,
        requests={
            "a": SimpleNamespace(num_tokens=4),
            "b": SimpleNamespace(num_tokens=1),
        },
        arange_np=np.arange(64, dtype=np.int64),
        use_ascend_attention=False,
        uses_mrope=uses_mrope,
        enable_prompt_embeds=enable_prompt_embeds,
        query_pos=_Buf(
            np.zeros(32, dtype=np.int64),
            torch.tensor(qpos_np, dtype=torch.int64),
        ),
        query_start_loc=_Buf(
            np.zeros(8, dtype=np.int64),
            torch.tensor(np.concatenate([[0], cu]), dtype=torch.int64),
            torch.tensor(np.concatenate([[0], cu]), dtype=torch.int64),
        ),
        input_ids=_Buf(cpu=torch.zeros(total, dtype=torch.int32)),
        is_token_ids=_Buf(
            np.ones(total, dtype=bool), cpu=torch.ones(total, dtype=torch.bool)
        ),
        inputs_embeds=_Buf(cpu=torch.zeros(total, 8)),
        optimistic_seq_lens_cpu=torch.zeros(4, dtype=torch.int32),
        discard_request_mask=_Buf(np.zeros(4, dtype=bool)),
        num_accepted_tokens_event=(Mock() if with_event else None),
        use_async_scheduling=False,
        cache_config=SimpleNamespace(mamba_cache_mode=None),
        num_accepted_tokens=_Buf(
            np.zeros(4, dtype=np.int64), torch.zeros(4, dtype=torch.int64)
        ),
        prev_positions=_Buf(
            np.zeros(4, dtype=np.int64), torch.zeros(4, dtype=torch.int64)
        ),
        use_async_spec_decode=async_fixup,
        valid_sampled_token_count_gpu=(
            torch.zeros(num_reqs) if async_fixup else None
        ),
        device=torch.device("cpu"),
        num_computed_tokens=torch.zeros(4, dtype=torch.int64),
        prev_num_draft_tokens=_Buf(
            np.zeros(4, dtype=np.int32), torch.zeros(4, dtype=torch.int32)
        ),
        req_indices=_Buf(
            req_indices, torch.tensor(req_indices, dtype=torch.int64)
        ),
        num_scheduled_tokens=_Buf(
            np.zeros(4, dtype=np.int32),
            torch.tensor(sched, dtype=torch.int64),
        ),
        positions=torch.zeros(total + 2, dtype=torch.int64),
        seq_lens=torch.zeros(4, dtype=torch.int64),
        num_decode_draft_tokens=_Buf(
            np.zeros(4, dtype=np.int32), torch.zeros(4, dtype=torch.int32)
        ),
        lora_config=None,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=10)
        ),
        _calc_mrope_positions=Mock(),
        mrope_positions=_Buf(
            gpu=torch.zeros(3, total + 2), cpu=torch.zeros(3, total + 2)
        ),
        _prepare_input_ids=Mock(),
        _kl_guard_on=lambda: False,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=total,
        num_scheduled_tokens={"a": int(sched[0]), "b": int(sched[1])},
        scheduled_spec_decode_tokens=({"a": [10, 11]} if spec else {}),
    )
    return runner, scheduler_output, sched, cu


def test_prepare_inputs_decode_no_spec():
    r, so, sched, cu = _mk_prepare_inputs(None)
    logits_indices, spec_meta, max_sampled = r._prepare_inputs(so, sched)
    assert logits_indices.tolist() == [int(cu[0]) - 1, int(cu[1]) - 1]
    assert spec_meta is None
    assert max_sampled == 1
    r._prepare_input_ids.assert_called_once()
    r.input_batch.block_table.compute_slot_mapping.assert_called_once()
    r.input_batch.block_table.commit_block_table.assert_called_once_with(2)
    assert r.optimistic_seq_lens_cpu[:2].tolist() == [2, 1]
    assert r.discard_request_mask.np[:2].tolist() == [True, False]
    assert r.query_pos.calls == [3]


def test_prepare_inputs_with_spec_decode():
    r, so, sched, cu = _mk_prepare_inputs(None, spec=True)
    r._calc_spec_decode_metadata = Mock(
        return_value=SimpleNamespace(logits_indices=torch.tensor([2, 3]))
    )
    logits_indices, spec_meta, max_sampled = r._prepare_inputs(so, sched)
    assert logits_indices.tolist() == [2, 3]
    assert spec_meta is r._calc_spec_decode_metadata.return_value
    assert max_sampled == 3
    r._calc_spec_decode_metadata.assert_called_once()
    assert r.num_decode_draft_tokens.np[:2].tolist() == [2, -1]
    assert r.num_decode_draft_tokens.np[2:].tolist() == [-1, -1]
    assert r.num_decode_draft_tokens.calls == [None]


def test_prepare_inputs_prompt_embeds_and_mrope():
    req_embeds = {0: torch.zeros(4, 8)}
    r, so, sched, cu = _mk_prepare_inputs(
        None,
        uses_mrope=True,
        enable_prompt_embeds=True,
        req_prompt_embeds=req_embeds,
    )
    logits_indices, spec_meta, max_sampled = r._prepare_inputs(so, sched)
    assert spec_meta is None
    r._calc_mrope_positions.assert_called_once_with(so)
    assert r.mrope_positions.calls == [3]
    assert torch.equal(r.inputs_embeds.cpu[:2], req_embeds[0][:2])


def test_prepare_inputs_syncs_cpu_accepted_counts():
    r, so, sched, cu = _mk_prepare_inputs(None, with_event=True)
    r._prepare_inputs(so, sched)
    r.num_accepted_tokens_event.synchronize.assert_called_once()
    assert r.num_accepted_tokens.np[:2].tolist() == [1, 1]
    assert r.num_accepted_tokens.np[2:].tolist() == [1, 1]
    assert r.num_accepted_tokens.calls == [None]


def test_prepare_inputs_async_spec_decode_gpu_fixup(monkeypatch):
    called = {}

    def fake_update(computed, accepted, prev_pos, valid, prev_draft, cpu_vals):
        called["args"] = (computed, accepted, prev_pos, valid, prev_draft, cpu_vals)

    monkeypatch.setattr(
        mr, "update_num_computed_tokens_for_batch_change", fake_update
    )
    r, so, sched, cu = _mk_prepare_inputs(None, async_fixup=True)
    r._prepare_inputs(so, sched)
    assert "args" in called
    assert r.prev_positions.calls == [2]
    assert r.prev_num_draft_tokens.calls == [None]


# ---------------------------------------------------------------------------
# _build_attention_metadata
# ---------------------------------------------------------------------------
class _FakeCAM:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.unpadded_calls = []

    def unpadded(self, num_tokens, num_reqs):
        self.unpadded_calls.append((num_tokens, num_reqs))
        return self


class _EncOnly:
    pass


class _Uniform:
    def __init__(self, specs=None):
        self.kv_cache_specs = specs or {}


class _Cross:
    pass


def _patch_spec_types(monkeypatch):
    monkeypatch.setattr(mr, "CommonAttentionMetadata", _FakeCAM)
    monkeypatch.setattr(mr, "EncoderOnlyAttentionSpec", _EncOnly)
    monkeypatch.setattr(mr, "UniformTypeKVCacheSpecs", _Uniform)
    monkeypatch.setattr(mr, "CrossAttentionSpec", _Cross)


def _mk_attn_runner(*, groups, attn_groups, num_reqs_padded=2,
                    num_tokens_padded=4, speculative_config=None,
                    drafter=None, max_model_len=128):
    ib = SimpleNamespace(
        block_table={gid: Mock() for gid in range(len(groups))},
        num_computed_tokens_cpu_tensor=torch.tensor(
            [0, 0, 0, 0], dtype=torch.int32
        ),
        num_prompt_tokens_cpu_tensor=torch.tensor(
            [4, 1, 0, 0], dtype=torch.int32
        ),
        req_ids=["a", "b"],
        req_id_to_index={"a": 0, "b": 1},
    )
    for gid in range(len(groups)):
        tbl = ib.block_table[gid]
        tbl.get_device_tensor = Mock(
            return_value=torch.full((num_reqs_padded, 4), 7, dtype=torch.int32)
        )
    return make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        attn_groups=attn_groups,
        input_batch=ib,
        routed_experts_initialized=False,
        use_async_spec_decode=False,
        is_mm_prefix_lm=False,
        model_config=SimpleNamespace(rswa_window=None),
        cache_config=SimpleNamespace(
            use_replayssm=False, kv_sharing_fast_prefill=False
        ),
        use_ascend_attention=False,
        dcp_world_size=1,
        speculative_config=speculative_config,
        drafter=drafter,
        max_model_len=max_model_len,
        query_start_loc=_Buf(
            np.zeros(8, dtype=np.int64),
            torch.zeros(8, dtype=torch.int64),
            torch.zeros(8, dtype=torch.int64),
        ),
        seq_lens=torch.zeros(8, dtype=torch.int64),
        positions=torch.zeros(num_tokens_padded + 2, dtype=torch.int64),
        optimistic_seq_lens_cpu=torch.tensor([4, 1, 0, 0], dtype=torch.int32),
        device=torch.device("cpu"),
    )


def test_build_attention_metadata_no_kv_groups():
    r = make_runner(kv_cache_config=SimpleNamespace(kv_cache_groups=[]))
    assert r._build_attention_metadata(4, 2, 2) == ({}, None)


def test_build_attention_metadata_main_path(monkeypatch):
    _patch_spec_types(monkeypatch)
    spec = SimpleNamespace()
    builder = Mock()
    builder.supports_update_block_table = False
    ag = SimpleNamespace(
        get_metadata_builder=Mock(return_value=builder),
        build_metadata=Mock(return_value="meta0"),
        layer_names=["L0"],
    )
    group = SimpleNamespace(kv_cache_spec=spec)
    r = _mk_attn_runner(groups=[group], attn_groups=[[ag]])
    attn_metadata, spec_meta = r._build_attention_metadata(
        4, 2, 2, slot_mappings={0: torch.zeros(4)}
    )
    assert spec_meta is None
    assert attn_metadata == {"L0": "meta0"}
    ag.build_metadata.assert_called_once()


def test_build_attention_metadata_spec_decode_and_block_table_cache(monkeypatch):
    _patch_spec_types(monkeypatch)
    spec = SimpleNamespace()
    b0 = Mock()
    b0.supports_update_block_table = True
    b0.build_metadata = Mock(return_value="meta0")
    a0 = SimpleNamespace(
        get_metadata_builder=Mock(return_value=b0),
        build_metadata=b0.build_metadata,
        layer_names=["L0"],
    )
    b1 = Mock()
    b1.supports_update_block_table = True
    b1.update_block_table = Mock(return_value="meta1")
    a1 = SimpleNamespace(
        get_metadata_builder=Mock(return_value=b1),
        build_metadata=Mock(side_effect=AssertionError("should not build")),
        update_block_table=b1.update_block_table,
        map_to_kernel_block_table=Mock(return_value=torch.zeros(2, 4)),
        layer_names=["L1"],
    )
    groups = [
        SimpleNamespace(kv_cache_spec=spec),
        SimpleNamespace(kv_cache_spec=spec),
    ]
    r = _mk_attn_runner(
        groups=groups,
        attn_groups=[[a0], [a1]],
        num_reqs_padded=4,
        num_tokens_padded=8,
        speculative_config=SimpleNamespace(),
        drafter=object(),
    )
    attn_metadata, spec_meta = r._build_attention_metadata(
        4,
        2,
        2,
        num_tokens_padded=8,
        num_reqs_padded=4,
        slot_mappings={0: torch.zeros(8), 1: torch.zeros(8)},
    )
    assert attn_metadata["L0"] == "meta0"
    assert attn_metadata["L1"] == "meta1"
    b1.update_block_table.assert_called_once()
    a1.build_metadata.assert_not_called()
    assert spec_meta is not None
    assert spec_meta.unpadded_calls == [(4, 2)]


def test_build_attention_metadata_cudagraph_encoder_only(monkeypatch):
    _patch_spec_types(monkeypatch)
    spec = _EncOnly()
    builder = Mock()
    builder.supports_update_block_table = False
    ag = SimpleNamespace(
        get_metadata_builder=Mock(return_value=builder),
        build_metadata_for_cudagraph_capture=Mock(return_value="cgmeta"),
        layer_names=["enc"],
    )
    group = SimpleNamespace(kv_cache_spec=spec)
    r = _mk_attn_runner(
        groups=[group], attn_groups=[[ag]], num_reqs_padded=4,
        num_tokens_padded=8,
    )
    attn_metadata, spec_meta = r._build_attention_metadata(
        4,
        2,
        2,
        num_tokens_padded=8,
        num_reqs_padded=4,
        for_cudagraph_capture=True,
        slot_mappings={0: torch.zeros(8)},
    )
    assert attn_metadata == {"enc": "cgmeta"}
    ag.build_metadata_for_cudagraph_capture.assert_called_once()
    assert spec_meta is None


# ---------------------------------------------------------------------------
# _execute_mm_encoder
# ---------------------------------------------------------------------------
def test_execute_mm_encoder_no_kwargs():
    r = make_runner(
        _batch_mm_inputs_from_scheduler=Mock(return_value=([], [], []))
    )
    assert r._execute_mm_encoder(SimpleNamespace()) == []


def test_execute_mm_encoder_prompt_embeds_passthrough(monkeypatch):
    monkeypatch.setattr(mr, "async_tensor_h2d", lambda t, device=None: t)
    pe = SimpleNamespace(data=torch.ones(2, 3))
    r = make_runner(
        _batch_mm_inputs_from_scheduler=Mock(
            return_value=(
                ["h"],
                [("prompt_embeds", {"embedding": pe})],
                [("r", SimpleNamespace())],
            )
        ),
        _cache_encoder_output=Mock(),
        device=torch.device("cpu"),
    )
    so = SimpleNamespace(
        ec_manager_metadata=None,
        free_encoder_mm_hashes=[],
        scheduled_encoder_inputs={},
    )
    assert r._execute_mm_encoder(so) == []
    r._cache_encoder_output.assert_called_once()
    assert r._cache_encoder_output.call_args.args[0] == "h"


def _mm_encoder_runner(**over):
    base = dict(
        _batch_mm_inputs_from_scheduler=Mock(
            return_value=(
                ["h1", "h2"],
                [("image", SimpleNamespace()), ("image", SimpleNamespace())],
                [("r", SimpleNamespace()), ("r", SimpleNamespace())],
            )
        ),
        observability_config=None,
        lora_config=None,
        is_multimodal_pruning_enabled=False,
        requires_sequential_video_encoding=False,
        encoder_cudagraph_manager=None,
        device=torch.device("cpu"),
        model=SimpleNamespace(
            embed_multimodal=Mock(
                return_value=[torch.zeros(1, 2), torch.zeros(1, 2)]
            )
        ),
        _cache_encoder_output=Mock(),
        timed_encoder_operation=lambda *a, **k: nullcontext(),
    )
    base.update(over)
    return make_runner(**base)


def test_execute_mm_encoder_regular_batch(monkeypatch):
    monkeypatch.setattr(
        mr,
        "group_and_batch_mm_kwargs",
        lambda kws, device=None, pin_memory=None: iter(
            [("image", len(kws), {"k": 1})]
        ),
    )
    sanity = Mock()
    monkeypatch.setattr(mr, "sanity_check_mm_encoder_outputs", sanity)
    r = _mm_encoder_runner()
    so = SimpleNamespace(
        ec_manager_metadata=None,
        free_encoder_mm_hashes=[],
        scheduled_encoder_inputs={"r": [0]},
    )
    out = r._execute_mm_encoder(so)
    assert len(out) == 2
    r.model.embed_multimodal.assert_called_once()
    sanity.assert_called_once()
    assert r._cache_encoder_output.call_count == 2


def test_execute_mm_encoder_sequential_video(monkeypatch):
    def fake_group(kws, device=None, pin_memory=None):
        if len(kws) == 1:
            return iter([("video", 1, {"m": kws[0][1]})])
        return iter([("video", len(kws), {"m": None})])

    monkeypatch.setattr(mr, "group_and_batch_mm_kwargs", fake_group)
    monkeypatch.setattr(mr, "sanity_check_mm_encoder_outputs", Mock())
    r = _mm_encoder_runner(is_multimodal_pruning_enabled=True)
    r.model.embed_multimodal = Mock(return_value=[torch.zeros(1, 2)])
    so = SimpleNamespace(
        ec_manager_metadata=None,
        free_encoder_mm_hashes=[],
        scheduled_encoder_inputs={"r": [0]},
    )
    out = r._execute_mm_encoder(so)
    assert len(out) == 2
    assert r.model.embed_multimodal.call_count == 2


def test_execute_mm_encoder_lora_tower_and_connector(monkeypatch):
    monkeypatch.setattr(mr, "LoRAMapping", lambda *a, **k: ("LORA", a, k))
    monkeypatch.setattr(
        mr,
        "LoRAMappingType",
        SimpleNamespace(TOWER="tower", CONNECTOR="connector"),
    )
    monkeypatch.setattr(
        mr,
        "group_and_batch_mm_kwargs",
        lambda kws, device=None, pin_memory=None: iter(
            [("image", 1, {"k": 1})]
        ),
    )
    monkeypatch.setattr(mr, "sanity_check_mm_encoder_outputs", Mock())
    pos_info = SimpleNamespace(get_num_embeds=Mock(return_value=1))
    model = SimpleNamespace(
        embed_multimodal=Mock(return_value=[torch.zeros(1, 2)]),
        get_mm_lora_token_counts=Mock(return_value=(2, 3)),
        get_mm_mapping=Mock(return_value=SimpleNamespace(connector=True)),
    )
    ib = SimpleNamespace(
        req_id_to_index={"r": 0},
        request_lora_mapping=np.array([1]),
        lora_id_to_lora_request={1: "REQ"},
    )
    lora_manager = SimpleNamespace(
        supports_tower_connector_lora=Mock(return_value=True),
        set_active_adapters=Mock(),
    )
    r = _mm_encoder_runner(
        lora_config=SimpleNamespace(),
        lora_manager=lora_manager,
        model=model,
        input_batch=ib,
        _batch_mm_inputs_from_scheduler=Mock(
            return_value=(
                ["h"],
                [("image", SimpleNamespace())],
                [("r", pos_info)],
            )
        ),
    )
    so = SimpleNamespace(
        ec_manager_metadata=None,
        free_encoder_mm_hashes=[],
        scheduled_encoder_inputs={"r": [0]},
    )
    out = r._execute_mm_encoder(so)
    assert len(out) == 1
    assert lora_manager.set_active_adapters.call_count == 2


# ---------------------------------------------------------------------------
# _gather_mm_embeddings
# ---------------------------------------------------------------------------
def _gather_runner(feat, *, encoder_cache, pruning=False, mrope=False, **over):
    req_state = SimpleNamespace(
        num_computed_tokens=0,
        mm_features=[feat],
        prompt_token_ids=[1, 2, 3],
        mrope_positions=torch.zeros(3, 1),
        mrope_position_delta=0.0,
    )
    req_state.__dict__.update(over.pop("req_state", {}))
    base = dict(
        input_batch=SimpleNamespace(req_ids=["r"]),
        requests={"r": req_state},
        encoder_cache=encoder_cache,
        is_multimodal_pruning_enabled=pruning,
        uses_mrope=mrope,
        model=SimpleNamespace(
            recompute_mrope_positions=Mock(
                return_value=([torch.zeros(1, 4)], torch.zeros(3, 1), 0.5)
            )
        ),
        mrope_positions=_Buf(gpu=torch.zeros(3, 8), cpu=torch.zeros(3, 8)),
        _calc_mrope_positions=Mock(),
    )
    base.update(over)
    return make_runner(**base), req_state


def _patch_gather(monkeypatch):
    monkeypatch.setattr(
        mr,
        "get_mm_features_in_window",
        lambda feats, start, end: (0, len(feats)),
    )
    monkeypatch.setattr(
        mr, "set_mm_embedding_modality", lambda tensor, modality: None
    )
    monkeypatch.setattr(
        mr, "copy_mm_embedding_modality", lambda src, dst: dst
    )


def _feat(*, offset=0, length=3, is_embed=None, ident="h"):
    return SimpleNamespace(
        modality="image",
        identifier=ident,
        mm_position=SimpleNamespace(
            offset=offset,
            length=length,
            is_embed=is_embed,
            get_embeds_indices_in_range=lambda a, b: (a, b),
        ),
    )


def test_gather_mm_embeddings_plain_mask(monkeypatch):
    _patch_gather(monkeypatch)
    feat = _feat()
    r, _ = _gather_runner(
        feat, encoder_cache={"h": torch.arange(12).reshape(3, 4)}
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=3, num_scheduled_tokens={"r": 3}
    )
    embeds, mask = r._gather_mm_embeddings(so)
    assert len(embeds) == 1
    assert mask.tolist() == [True, True, True]


def test_gather_mm_embeddings_is_embed_mask(monkeypatch):
    _patch_gather(monkeypatch)
    is_embed = torch.tensor([True, False, True])
    feat = _feat(is_embed=is_embed)
    r, _ = _gather_runner(
        feat, encoder_cache={"h": torch.arange(12).reshape(3, 4)}
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=3, num_scheduled_tokens={"r": 3}
    )
    _, mask = r._gather_mm_embeddings(so)
    assert mask.tolist() == [True, False, True]


def test_gather_mm_embeddings_cache_miss_raises(monkeypatch):
    _patch_gather(monkeypatch)
    feat = _feat()
    r, _ = _gather_runner(feat, encoder_cache={})
    so = SimpleNamespace(
        total_num_scheduled_tokens=3, num_scheduled_tokens={"r": 3}
    )
    with pytest.raises(RuntimeError):
        r._gather_mm_embeddings(so)


def test_gather_mm_embeddings_lookahead_skips(monkeypatch):
    _patch_gather(monkeypatch)
    feat = _feat(offset=5, length=3)
    r, _ = _gather_runner(feat, encoder_cache={})
    so = SimpleNamespace(
        total_num_scheduled_tokens=2, num_scheduled_tokens={"r": 2}
    )
    embeds, mask = r._gather_mm_embeddings(so, shift_computed_tokens=5)
    assert embeds == []
    assert mask.tolist() == [False, False]


def test_gather_mm_embeddings_pruning_mrope(monkeypatch):
    _patch_gather(monkeypatch)
    feat = _feat()
    r, req_state = _gather_runner(
        feat,
        encoder_cache={"h": torch.arange(12).reshape(3, 4)},
        pruning=True,
        mrope=True,
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=3, num_scheduled_tokens={"r": 3}
    )
    embeds, _ = r._gather_mm_embeddings(so)
    assert len(embeds) == 1
    r.model.recompute_mrope_positions.assert_called_once()
    assert req_state.mrope_position_delta == 0.5
    r._calc_mrope_positions.assert_called_once_with(so)
    assert r.mrope_positions.calls == [3]


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------
def _pp(monkeypatch, *, first=True, last=True):
    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=first, is_last_rank=last),
    )


def _ec_ctx(scheduler_output, encoder_cache=None):
    @contextmanager
    def _cm():
        yield "EC"

    return _cm()


def test_preprocess_multimodal_first_rank(monkeypatch):
    _pp(monkeypatch)

    def fake_extract(so):
        return {"b": 2}

    r = make_runner(
        model_config=SimpleNamespace(is_encoder_decoder=False),
        speculative_config=None,
        supports_mm_inputs=True,
        enable_prompt_embeds=False,
        input_batch=SimpleNamespace(req_prompt_embeds={}),
        uses_mrope=False,
        positions=torch.zeros(6),
        input_ids=_Buf(gpu=torch.zeros(3, dtype=torch.long)),
        inputs_embeds=_Buf(gpu=torch.zeros(4, 5)),
        maybe_get_ec_connector_output=_ec_ctx,
        _execute_mm_encoder=Mock(return_value=[]),
        _gather_mm_embeddings=Mock(
            return_value=([], torch.zeros(3, dtype=torch.bool))
        ),
        _prepare_mm_inputs=Mock(return_value=(None, "EMB")),
        _init_model_kwargs=Mock(return_value={"a": 1}),
        _extract_mm_kwargs=fake_extract,
        model=SimpleNamespace(
            embed_input_ids=Mock(return_value=torch.zeros(3, 5))
        ),
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=3, scheduled_encoder_inputs={}
    )
    out = r._preprocess(so, 4)
    assert out[0] is None
    assert out[1] == "EMB"
    assert out[2].shape == (4,)
    assert out[4] == {"a": 1, "b": 2}
    assert out[5] == "EC"
    r._execute_mm_encoder.assert_called_once_with(so)
    r._gather_mm_embeddings.assert_called_once_with(so)


def test_preprocess_prompt_embeds_text_rank(monkeypatch):
    _pp(monkeypatch)
    monkeypatch.setattr(
        mr,
        "async_tensor_h2d",
        lambda t, device=None: torch.as_tensor(t, dtype=torch.long),
    )
    r = make_runner(
        model_config=SimpleNamespace(is_encoder_decoder=False),
        speculative_config=None,
        supports_mm_inputs=False,
        enable_prompt_embeds=True,
        input_batch=SimpleNamespace(req_prompt_embeds={}),
        uses_mrope=False,
        positions=torch.zeros(6),
        input_ids=_Buf(gpu=torch.tensor([3, 4, 5])),
        is_token_ids=_Buf(np.array([True, False, True])),
        inputs_embeds=_Buf(gpu=torch.zeros(6, 5)),
        model=SimpleNamespace(
            embed_input_ids=Mock(return_value=torch.ones(2, 5))
        ),
        _init_model_kwargs=Mock(return_value={"a": 1}),
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=3, scheduled_encoder_inputs={}
    )
    out = r._preprocess(so, 4)
    assert out[0] is None
    assert out[4] == {"a": 1}
    r.model.embed_input_ids.assert_called_once()
    assert torch.equal(r.inputs_embeds.gpu[[0, 2]], torch.ones(2, 5))


def test_preprocess_text_only_clamps_spec_placeholders(monkeypatch):
    _pp(monkeypatch)
    r = make_runner(
        model_config=SimpleNamespace(is_encoder_decoder=False),
        speculative_config=SimpleNamespace(),
        supports_mm_inputs=False,
        enable_prompt_embeds=False,
        uses_mrope=False,
        positions=torch.zeros(6),
        input_ids=_Buf(gpu=torch.tensor([-1, 2, -1, -1])),
        _init_model_kwargs=Mock(return_value={}),
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=3, scheduled_encoder_inputs={}
    )
    out = r._preprocess(so, 4)
    assert out[0] is not None
    assert out[1] is None
    assert int(out[0].min()) == 0


def test_preprocess_non_first_rank_gathers_intermediate(monkeypatch):
    _pp(monkeypatch, first=False)
    sentinel = SimpleNamespace()
    r = make_runner(
        model_config=SimpleNamespace(is_encoder_decoder=False),
        speculative_config=None,
        supports_mm_inputs=False,
        enable_prompt_embeds=False,
        uses_mrope=False,
        positions=torch.zeros(6),
        input_ids=_Buf(gpu=torch.arange(4)),
        _init_model_kwargs=Mock(return_value={}),
        sync_and_gather_intermediate_tensors=Mock(return_value=sentinel),
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=3, scheduled_encoder_inputs={}
    )
    out = r._preprocess(so, 4, intermediate_tensors=SimpleNamespace())
    assert out[3] is sentinel
    r.sync_and_gather_intermediate_tensors.assert_called_once()


def test_preprocess_encoder_decoder_runs_encoder(monkeypatch):
    _pp(monkeypatch)
    r = make_runner(
        model_config=SimpleNamespace(is_encoder_decoder=True),
        speculative_config=None,
        supports_mm_inputs=False,
        enable_prompt_embeds=False,
        uses_mrope=False,
        positions=torch.zeros(6),
        input_ids=_Buf(gpu=torch.arange(4)),
        _init_model_kwargs=Mock(return_value={"x": 1}),
        _execute_mm_encoder=Mock(return_value=["enc"]),
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=3, scheduled_encoder_inputs={"r": 1}
    )
    out = r._preprocess(so, 4)
    assert out[4] == {"x": 1, "encoder_outputs": ["enc"]}
    r._execute_mm_encoder.assert_called_once_with(so)


# ---------------------------------------------------------------------------
# _bookkeeping_sync
# ---------------------------------------------------------------------------
def _bookkeeping_runner(**over):
    gen = Mock()
    gen.device = SimpleNamespace(type="cpu")
    gen.set_state = Mock()
    gen.get_offset = Mock(return_value=8)
    ib = SimpleNamespace(
        num_reqs=2,
        generators={0: gen},
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        vocab_size=10,
        prev_sampled_token_ids=None,
        prev_req_id_to_index=None,
        token_ids_cpu=np.zeros((2, 10), dtype=np.int32),
        is_token_ids=np.zeros((2, 10), dtype=bool),
        num_tokens_no_spec=np.zeros(2, dtype=np.int32),
    )
    base = dict(
        use_async_scheduling=False,
        input_batch=ib,
        requests={
            "r0": SimpleNamespace(output_token_ids=[]),
            "r1": SimpleNamespace(output_token_ids=[]),
        },
        max_model_len=20,
        discard_request_mask=SimpleNamespace(np=np.array([True, False])),
        _cpu_generator_states={0: "S"},
        routed_experts_initialized=False,
        _to_list=Mock(return_value=[[5], [6]]),
        _get_prompt_logprobs_dict=Mock(return_value={}),
        _get_nans_in_logits=Mock(return_value={"r1": 2}),
    )
    base.update(over)
    return make_runner(**base), gen


def _bookkeeping_patches(monkeypatch, *, nans=False):
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: nullcontext())
    monkeypatch.setattr(
        mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", nans, raising=False
    )


def test_bookkeeping_sync_single_token(monkeypatch):
    _bookkeeping_patches(monkeypatch)
    r, gen = _bookkeeping_runner()
    sampler_output = SimpleNamespace(
        sampled_token_ids=torch.tensor([[5], [6]]), logprobs_tensors=None
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 2, "r1": 2}
    )
    out = r._bookkeeping_sync(
        so, sampler_output, None, torch.zeros(4, 3), 4
    )
    num_nans_in_logits, num_nans, logprobs_lists, valid, prompt_logprobs, \
        req_ids_copy, idx_copy, invalid = out
    assert num_nans_in_logits == {}
    assert num_nans is None
    assert logprobs_lists is None
    assert valid[0] == []
    assert valid[1] == [6]
    assert r.requests["r1"].output_token_ids == [6]
    assert r.input_batch.num_tokens_no_spec[1] == 1
    assert r.input_batch.token_ids_cpu[1, 0] == 6
    assert gen.set_state.call_args.args[0] == "S"
    assert req_ids_copy == ["r0", "r1"]
    assert idx_copy == {"r0": 0, "r1": 1}
    assert invalid == []
    r._get_prompt_logprobs_dict.assert_called_once()


def test_bookkeeping_sync_spec_decode_and_routed_experts(monkeypatch):
    _bookkeeping_patches(monkeypatch)
    monkeypatch.setattr(
        mr,
        "RejectionSampler",
        SimpleNamespace(parse_output=Mock(return_value=([[5], [7]], None))),
    )
    capturer = SimpleNamespace(
        get_device_buffer=Mock(return_value=torch.zeros(4, 2))
    )
    r, gen = _bookkeeping_runner(
        routed_experts_initialized=True,
        routed_experts_capturer=capturer,
        routed_experts_cpu=torch.zeros(4, 2),
        routed_experts_slot_mapping_device=torch.zeros(4),
        routed_experts_slot_mapping_cpu=torch.zeros(4),
    )
    sampler_output = SimpleNamespace(
        sampled_token_ids=torch.tensor([[5, 6], [7, 8]]), logprobs_tensors=None
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 2, "r1": 2}
    )
    out = r._bookkeeping_sync(
        so, sampler_output, None, torch.zeros(4, 3), 4
    )
    assert out[3] == [[5], [7]]
    mr.RejectionSampler.parse_output.assert_called_once()
    capturer.get_device_buffer.assert_called_once()
    assert r.routed_experts_cpu.sum() == 0


def test_bookkeeping_sync_async_scheduling(monkeypatch):
    _bookkeeping_patches(monkeypatch)
    r, gen = _bookkeeping_runner(use_async_scheduling=True)
    sampled = torch.tensor([[5], [6]])
    sampler_output = SimpleNamespace(
        sampled_token_ids=sampled, logprobs_tensors=None
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 2, "r1": 2}
    )
    out = r._bookkeeping_sync(
        so, sampler_output, None, torch.zeros(4, 3), 4
    )
    assert out[3] == []
    assert out[7] == [0]
    assert r.input_batch.prev_sampled_token_ids is sampled
    assert r.input_batch.prev_req_id_to_index == {"r1": 1}
    assert r.requests["r1"].output_token_ids == [-1]


def test_bookkeeping_sync_nans_sync(monkeypatch):
    _bookkeeping_patches(monkeypatch, nans=True)
    r, gen = _bookkeeping_runner()
    logits = torch.zeros(2, 5)
    sampler_output = SimpleNamespace(
        sampled_token_ids=torch.tensor([[5], [6]]), logprobs_tensors=None
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 2, "r1": 2}
    )
    out = r._bookkeeping_sync(so, sampler_output, logits, torch.zeros(4, 3), 4)
    assert out[0] == {"r1": 2}
    assert out[1] is None
    r._get_nans_in_logits.assert_called_once_with(logits)


# ---------------------------------------------------------------------------
# execute_model (main non-empty flow)
# ---------------------------------------------------------------------------
def _exec_patches(monkeypatch, *, is_last_rank=True):
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(
        mr, "maybe_create_ubatch_slices", lambda *a, **k: (None, None)
    )
    monkeypatch.setattr(
        mr, "set_forward_context", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(
        mr, "record_function_or_nullcontext", lambda name: nullcontext()
    )
    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_last_rank=is_last_rank,
            ranks=[0],
            send_tensor_dict=Mock(),
            broadcast_tensor_dict=Mock(),
        ),
    )


def _base_execute_runner(monkeypatch, *, is_last_rank=True):
    _exec_patches(monkeypatch, is_last_rank=is_last_rank)

    @contextmanager
    def kv_ctx(scheduler_output, defer_finalize=False):
        yield "KV"

    batch_desc = SimpleNamespace(num_tokens=4, num_reqs=1)
    r = make_runner(
        execute_model_state=None,
        speculative_config=None,
        synchronize_input_prep=lambda: nullcontext(),
        _update_states=Mock(return_value=None),
        cache_config=SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode=None
        ),
        parallel_config=SimpleNamespace(
            use_ubatching=False,
            distributed_executor_backend="mp",
            data_parallel_size=1,
            num_ubatches=2,
        ),
        input_batch=SimpleNamespace(num_reqs=1, req_ids=["r"]),
        _prepare_inputs=Mock(return_value=(torch.tensor([2]), None, 1)),
        use_ascend_attention=False,
        cascade_attn_enabled=False,
        _allow_microbatching=Mock(return_value=True),
        _determine_batch_execution_and_padding=Mock(
            return_value=(None, batch_desc, False, None, None)
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _get_slot_mappings=Mock(
            return_value=({0: torch.zeros(4)}, torch.zeros(4))
        ),
        _build_attention_metadata=Mock(return_value=({}, None)),
        _preprocess=Mock(
            return_value=(None, None, torch.zeros(4), None, {}, None)
        ),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        eplb_state=None,
        maybe_get_kv_connector_output=kv_ctx,
        _model_forward=Mock(return_value=torch.zeros(4, 8)),
        use_aux_hidden_state_outputs=False,
        broadcast_pp_output=False,
        is_pooling_model=False,
        model=SimpleNamespace(
            compute_logits=Mock(return_value=torch.zeros(1, 10))
        ),
        vllm_config=SimpleNamespace(),
    )
    so = SimpleNamespace(
        total_num_scheduled_tokens=3,
        num_scheduled_tokens={"r": 3},
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        kv_connector_metadata=None,
    )
    return r, so


def test_execute_model_main_flow(monkeypatch):
    r, so = _base_execute_runner(monkeypatch)
    assert r.execute_model(so) is None
    st = r.execute_model_state
    assert st is not None
    assert st.logits is r.model.compute_logits.return_value
    assert st.spec_decode_metadata is None
    assert st.ec_connector_output is None
    r._update_states.assert_called_once_with(so)
    r._prepare_inputs.assert_called_once()
    r._build_attention_metadata.assert_called_once()
    r._preprocess.assert_called_once_with(so, 4, None)
    r._model_forward.assert_called_once()
    assert r.kv_connector_output == "KV"


def test_execute_model_pp_non_last_rank_returns_hidden(monkeypatch):
    class _DummyTensor:
        pass

    monkeypatch.setattr(mr, "IntermediateTensors", _DummyTensor)
    r, so = _base_execute_runner(monkeypatch, is_last_rank=False)
    hidden = _DummyTensor()
    r._model_forward = Mock(return_value=hidden)
    assert r.execute_model(so) is hidden
    assert r.kv_connector_output == "KV"


def test_execute_model_pooling_branch(monkeypatch):
    r, so = _base_execute_runner(monkeypatch)
    r.is_pooling_model = True
    r._pool = Mock(return_value="POOL")
    assert r.execute_model(so) == "POOL"
    r._pool.assert_called_once()


# ---------------------------------------------------------------------------
# _update_states deeper branches
# ---------------------------------------------------------------------------
def test_update_states_cached_request_not_last_rank(monkeypatch):
    monkeypatch.setattr(
        mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    ib = SimpleNamespace(
        req_id_to_index={"r1": 0},
        req_ids=["r1"],
        remove_request=Mock(),
        add_request=Mock(),
        update_req_spec_token_ids=Mock(),
        condense=Mock(),
        refresh_metadata=Mock(),
        prev_req_id_to_index=None,
        block_table=SimpleNamespace(append_row=Mock()),
        num_computed_tokens_cpu=np.zeros(2, dtype=np.int32),
        token_ids_cpu=np.zeros((1, 16), dtype=np.int32),
        is_token_ids=np.zeros((1, 16), dtype=bool),
        num_tokens_no_spec=np.array([5], dtype=np.int32),
        vocab_size=100,
    )
    req_state = SimpleNamespace(
        num_computed_tokens=0,
        output_token_ids=[],
        block_ids=[[1]],
        prev_num_draft_len=0,
        num_tokens=5,
    )
    late = SimpleNamespace(
        on_requests_finished=Mock(), register_request=Mock()
    )
    r = make_runner(
        requests={"r1": req_state},
        num_prompt_logprobs={},
        input_batch=ib,
        late_interaction_runner=late,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        is_pooling_model=False,
        uses_mrope=False,
        _may_reorder_batch=Mock(),
        encoder_cache={},
        _on_request_state_removed=Mock(),
    )
    so = SimpleNamespace(
        finished_req_ids=[],
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={"r1": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r1"],
            num_computed_tokens=[5],
            new_block_ids=[[[2]]],
            resumed_req_ids=set(),
            num_output_tokens=[0],
            new_token_ids=[[9]],
        ),
        scheduled_spec_decode_tokens={},
    )
    assert r._update_states(so) is None
    assert req_state.output_token_ids == [9]
    assert req_state.block_ids == [[1, 2]]
    assert ib.num_computed_tokens_cpu[0] == 5
    ib.block_table.append_row.assert_called_once()
    assert ib.num_tokens_no_spec[0] == 6
    assert ib.token_ids_cpu[0, 5] == 9
    ib.condense.assert_called_once()
    ib.refresh_metadata.assert_called_once()


def test_update_states_resumed_preemption_and_truncation(monkeypatch):
    monkeypatch.setattr(
        mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    ib = SimpleNamespace(
        req_id_to_index={},
        req_ids=[],
        remove_request=Mock(),
        add_request=Mock(),
        update_req_spec_token_ids=Mock(),
        condense=Mock(),
        refresh_metadata=Mock(),
        prev_req_id_to_index=None,
        block_table=SimpleNamespace(append_row=Mock()),
        num_computed_tokens_cpu=np.zeros(2, dtype=np.int32),
        token_ids_cpu=np.zeros((1, 16), dtype=np.int32),
        is_token_ids=np.zeros((1, 16), dtype=bool),
        num_tokens_no_spec=np.zeros(1, dtype=np.int32),
        vocab_size=100,
    )
    req_state = SimpleNamespace(
        num_computed_tokens=0,
        output_token_ids=[1, 2, 3, 4],
        block_ids=[[1]],
        prev_num_draft_len=0,
        num_tokens=5,
    )
    late = SimpleNamespace(
        on_requests_finished=Mock(), register_request=Mock()
    )
    r = make_runner(
        requests={"r1": req_state},
        num_prompt_logprobs={},
        input_batch=ib,
        late_interaction_runner=late,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=True,
        is_pooling_model=False,
        uses_mrope=False,
        _may_reorder_batch=Mock(),
        encoder_cache={},
        _on_request_state_removed=Mock(),
    )
    so = SimpleNamespace(
        finished_req_ids=[],
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={"r1": 2},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r1"],
            num_computed_tokens=[5],
            new_block_ids=[[[7]]],
            resumed_req_ids={"r1"},
            num_output_tokens=[2],
            new_token_ids=[[9]],
            all_token_ids={"r1": [1, 2, 3, 4]},
        ),
        scheduled_spec_decode_tokens={},
    )
    assert r._update_states(so) is None
    assert req_state.block_ids == [[7]]
    assert req_state.output_token_ids == [3, 4]
    ib.add_request.assert_called_once()
    ib.update_req_spec_token_ids.assert_called()


# end of file




