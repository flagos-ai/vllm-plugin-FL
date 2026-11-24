"""Coverage for _build_attention_metadata / _build_attn_group_metadata (w10)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.worker import model_runner as mr


SENTINEL = object()


def make_runner(**attrs):
    runner = object.__new__(mr.ModelRunnerFL)
    runner.__dict__.update(attrs)
    return runner


def buf(size, np_dtype=np.int32, torch_dtype=torch.int32, value=0):
    cpu = torch.full((size,), value, dtype=torch_dtype)
    npv = cpu.numpy()
    gpu = torch.full((size,), value, dtype=torch_dtype)

    def copy_to_gpu(n=None):
        n = gpu.numel() if n is None else n
        gpu[:n].copy_(cpu[:n])

    return SimpleNamespace(np=npv, cpu=cpu, gpu=gpu, copy_to_gpu=copy_to_gpu)


class Builder:
    supports_update_block_table = False

    def __init__(self):
        self.build_calls = []
        self.update_calls = []

    def update_block_table(self, cached, block_table_tensor, slot_mapping):
        self.update_calls.append(
            SimpleNamespace(
                cached=cached, block_table_tensor=block_table_tensor,
                slot_mapping=slot_mapping,
            )
        )
        return ("updated", len(self.update_calls))


class Group:
    def __init__(self, layer_names, builder):
        self.layer_names = layer_names
        self.builder = builder

    def get_metadata_builder(self, ubid=0):
        return self.builder

    def build_metadata(self, cm, ubid, cascade, **extra):
        self.builder.build_calls.append(
            SimpleNamespace(cm=cm, ubid=ubid, cascade=cascade, extra=extra)
        )
        return ("built", len(self.builder.build_calls))

    def build_metadata_for_cudagraph_capture(self, cm, ubid=0):
        self.builder.build_calls.append(
            SimpleNamespace(cm=cm, ubid=ubid, cascade=0, extra={})
        )
        return ("capture", len(self.builder.build_calls))

    def map_to_kernel_block_table(self, block_table_tensor, ubid=0):
        return ("kernel_bt", block_table_tensor)


@pytest.fixture(autouse=True)
def _cpu_pinned(monkeypatch):
    monkeypatch.setattr(mr, "PIN_MEMORY", False, raising=False)


@pytest.fixture
def cm_factory(monkeypatch):
    made = []

    def factory(**kw):
        ns = SimpleNamespace(**kw)
        ns.unpadded = lambda t, r: ("unpadded", t, r)
        made.append(ns)
        return ns

    monkeypatch.setattr(mr, "CommonAttentionMetadata", factory)
    return made


def setup(num_reqs=2, num_tokens=2, padded=4, specs=None, groups=None, **over):
    if specs is None:
        specs = [object()]
    if groups is None:
        groups = [[Group(["L0"], Builder())] for _ in specs]
    block_tables = [
        SimpleNamespace(
            get_device_tensor=lambda n: torch.zeros(
                (n, 2), dtype=torch.int32, device=torch.device("cpu")
            )
        )
        for _ in specs
    ]
    input_batch = SimpleNamespace(
        block_table=block_tables,
        num_computed_tokens_cpu_tensor=torch.tensor([0, 1, 0, 0], dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.tensor([3, 3, 0, 0], dtype=torch.int32),
        req_ids=[f"r{i}" for i in range(num_reqs)],
        req_id_to_index={f"r{i}": i for i in range(num_reqs)},
        replayssm_decode_base_cpu_tensor=torch.arange(8, dtype=torch.int32),
    )
    attrs = dict(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in specs]
        ),
        attn_groups=groups,
        input_batch=input_batch,
        optimistic_seq_lens_cpu=torch.tensor([5, 6, 0, 0], dtype=torch.int32),
        device=torch.device("cpu"),
        query_start_loc=buf(16),
        seq_lens=torch.zeros(16, dtype=torch.int32),
        positions=torch.zeros(16, dtype=torch.int64),
        use_async_spec_decode=False,
        is_mm_prefix_lm=False,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(sliding_window=None), rswa_window=None
        ),
        model=SimpleNamespace(mm_prefix_clamp_sliding_window=False),
        requests={},
        cache_config=SimpleNamespace(
            use_replayssm=False, kv_sharing_fast_prefill=False
        ),
        use_ascend_attention=False,
        _ascend_attn_state=None,
        dcp_world_size=1,
        dcp_rank=0,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        speculative_config=None,
        drafter=None,
        num_accepted_tokens=buf(16),
        num_decode_draft_tokens=buf(16),
        routed_experts_initialized=False,
        max_model_len=128,
        _get_encoder_seq_lens=lambda *a, **k: (None, None),
    )
    attrs.update(over)
    return make_runner(**attrs)


def slot_map(n=8, groups=1):
    return {g: torch.zeros(n, dtype=torch.int32) for g in range(groups)}


def test_build_basic_single_group(cm_factory):
    r = setup()
    slot_mappings = slot_map()
    out, spec = r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        num_tokens_padded=4,
        num_reqs_padded=4,
        slot_mappings=slot_mappings,
    )
    assert spec is None
    assert out["L0"][0] == "built"
    cm = cm_factory[0]
    assert cm.max_seq_len == 6
    assert cm.num_reqs == 4
    assert cm.num_actual_tokens == 4
    assert cm.max_query_len == 2
    assert cm.is_prefilling.tolist() == [True, True, False, False]
    # padded block-table rows cleared to NULL_BLOCK_ID
    bt = cm.block_table_tensor
    assert (bt[2:4] == mr.NULL_BLOCK_ID).all()
    assert cm.causal is True
    assert cm.mm_req_doc_ranges is None
    assert cm.rswa_prefix_lens is None


def test_build_attention_free_early_return():
    r = make_runner(kv_cache_config=SimpleNamespace(kv_cache_groups=[]))
    assert r._build_attention_metadata(1, 1, 1) == ({}, None)


def test_build_cudagraph_capture_max_model_len(cm_factory):
    r = setup(max_model_len=777)
    r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        for_cudagraph_capture=True,
        slot_mappings=slot_map(),
    )
    assert cm_factory[0].max_seq_len == 777


def test_build_encoder_only_spec_uses_zero_block_table(cm_factory, monkeypatch):
    class DummyEncOnly:
        pass

    monkeypatch.setattr(mr, "EncoderOnlyAttentionSpec", DummyEncOnly)
    spec = DummyEncOnly()
    r = setup(specs=[spec])
    r.input_batch.block_table[0].get_device_tensor = lambda n: (_ for _ in ()).throw(
        AssertionError("should not be used for encoder-only")
    )
    r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        num_reqs_padded=4,
        slot_mappings=slot_map(),
    )
    bt = cm_factory[0].block_table_tensor
    assert bt.shape == (4, 1)


def test_build_uniform_type_specs_collapses_to_inner(cm_factory, monkeypatch):
    class DummyUniform:
        def __init__(self, inner):
            self.kv_cache_specs = {"L0": inner}

    inner = object()
    monkeypatch.setattr(mr, "UniformTypeKVCacheSpecs", DummyUniform)
    r = setup(specs=[DummyUniform(inner)])
    r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        slot_mappings=slot_map(),
    )
    # build still succeeded (cache key resolves inner spec)
    assert cm_factory[0].num_reqs == 2


def test_build_cache_reuse_update_block_table(cm_factory):
    spec = object()
    b0 = Builder()
    b0.supports_update_block_table = True
    b1 = Builder()
    b1.supports_update_block_table = True
    r = setup(specs=[spec, spec], groups=[[Group(["L0"], b0)], [Group(["L1"], b1)]])
    out, _ = r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        slot_mappings=slot_map(groups=2),
    )
    assert out["L0"][0] == "built"
    assert len(b1.build_calls) == 0
    assert len(b1.update_calls) == 1
    # second group reuses cached metadata from first (same spec & builder type)
    assert out["L1"][0] == "updated"
    assert b1.update_calls[0].cached == ("built", 1)


def test_build_spec_decode_common_metadata_unpadded(cm_factory):
    r = setup(
        speculative_config=SimpleNamespace(),
        drafter=SimpleNamespace(),
    )
    out, spec = r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        num_tokens_padded=4,
        num_reqs_padded=4,
        slot_mappings=slot_map(),
    )
    assert spec == ("unpadded", 2, 2)


def test_build_ascend_sets_attn_state(cm_factory):
    r = setup(use_ascend_attention=True, _ascend_attn_state=None)
    r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        slot_mappings=slot_map(),
        ascend_attn_state="phase",
    )
    assert cm_factory[0].attn_state == "phase"


def test_build_ascend_requires_state(cm_factory):
    r = setup(use_ascend_attention=True, _ascend_attn_state=None)
    with pytest.raises(AssertionError):
        r._build_attention_metadata(
            num_tokens=2,
            num_reqs=2,
            max_query_len=2,
            slot_mappings=slot_map(),
        )


def test_build_dcp_local_seq_lens(cm_factory, monkeypatch):
    dcp = buf(16)
    calls = {}

    def fake(seq_lens, world, rank, interleave):
        calls["args"] = (world, rank, interleave)
        # Real get_dcp_local_seq_lens returns a torch tensor (assigned into a
        # CPU torch buffer at model_runner.py:2665), not a numpy array.
        return torch.tensor([1, 2], dtype=torch.int32)

    monkeypatch.setattr(mr, "get_dcp_local_seq_lens", fake)
    r = setup(
        dcp_world_size=2,
        dcp_rank=1,
        dcp_local_seq_lens=dcp,
    )
    r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        num_reqs_padded=4,
        slot_mappings=slot_map(),
    )
    cm = cm_factory[0]
    assert calls["args"] == (2, 1, 1)
    assert cm.dcp_local_seq_lens_cpu[:2].tolist() == [1, 2]
    assert cm.dcp_local_seq_lens_cpu[2:].tolist() == [0, 0]


def test_build_mm_prefix_doc_ranges(cm_factory):
    audio = SimpleNamespace(modality="audio", mm_position=None)
    img = SimpleNamespace(
        modality="image",
        mm_position=SimpleNamespace(extract_embeds_range=lambda: [(0, 10), (1, 2)]),
    )
    r = setup(
        is_mm_prefix_lm=True,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(sliding_window=4), rswa_window=None
        ),
        model=SimpleNamespace(mm_prefix_clamp_sliding_window=False),
        requests={
            "r0": SimpleNamespace(mm_features=[audio, img]),
            "r1": SimpleNamespace(mm_features=[]),
        },
    )
    r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        slot_mappings=slot_map(),
    )
    # (0,10) spans 11 > sliding_window 4 -> skipped; (1,2) kept; audio skipped
    assert cm_factory[0].mm_req_doc_ranges[0] == [(1, 2)]


def test_build_rswa_replayssm_routed_kvsharing(cm_factory):
    r = setup(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(sliding_window=None), rswa_window=16
        ),
        cache_config=SimpleNamespace(
            use_replayssm=True, kv_sharing_fast_prefill=True
        ),
        routed_experts_initialized=True,
    )
    r.routed_experts_capturer = SimpleNamespace(attn_gid=0)
    r.routed_experts_slot_mapping_device = torch.zeros(8, dtype=torch.int32)
    r._prepare_kv_sharing_fast_prefill = lambda li: torch.zeros(8, dtype=torch.int64)
    logits_indices = torch.tensor([1, 3])
    r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        logits_indices=logits_indices,
        max_num_sampled_tokens=3,
        slot_mappings=slot_map(),
    )
    cm = cm_factory[0]
    assert cm.rswa_prefix_lens.tolist()[:2] == [3, 3]
    assert cm.replayssm_decode_base_cpu is not None
    assert cm.num_logits_indices == 2
    assert cm.max_logits_per_req == 3
    assert cm.logits_indices_padded is not None
    assert r.routed_experts_slot_mapping_device[:2].numel() == 2


def test_build_ubatch_slices_split(cm_factory, monkeypatch):
    sub = SimpleNamespace(num_reqs=1)
    monkeypatch.setattr(
        mr, "split_attn_metadata", lambda slices, cm: [sub, sub]
    )
    r = setup()
    ubatch_slices = [object(), object()]
    out, _ = r._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=2,
        ubatch_slices=ubatch_slices,
        slot_mappings=slot_map(),
    )
    assert isinstance(out, list) and len(out) == 2
    assert out[0]["L0"][0] == "built"
    assert out[1]["L0"][0] == "built"
