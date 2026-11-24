"""Coverage tests for ModelRunnerFL KV-cache / attention / metadata helpers.

Source under test: vllm_fl/worker/model_runner.py @ d12e9d9e (lines ~5895-8224).

All runners are built with ``object.__new__(ModelRunnerFL)`` plus hand-set
attributes.  Attention backends, KV cache allocation and graph machinery are
replaced with local fakes so no real device work happens.
"""

import importlib
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_fl.worker import model_runner as mr

MR = mr.ModelRunnerFL


def new_runner(**attrs):
    runner = object.__new__(MR)
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


def _group(threshold):
    builder = SimpleNamespace(reorder_batch_threshold=threshold)
    return SimpleNamespace(get_metadata_builder=lambda: builder)


# ---------------------------------------------------------------------------
# _to_list / encoder timing
# ---------------------------------------------------------------------------
def test_to_list_copies_into_pinned_buffer_and_returns_list():
    buf = mr.torch.zeros((4, 2), dtype=mr.torch.int64)
    event = MagicMock()
    runner = new_runner(sampled_token_ids_pinned_cpu=buf, transfer_event=event)
    src = mr.torch.tensor([[1, 2], [3, 4]], dtype=mr.torch.int64)

    out = MR._to_list(runner, src)

    assert out == [[1, 2], [3, 4]]
    event.record.assert_called_once()
    event.synchronize.assert_called_once()


def test_get_encoder_timing_stats_returns_and_clears():
    registry = {"r1": SimpleNamespace(to_dict=lambda: {"encoder_forward_secs": 1.5})}
    runner = new_runner(
        _encoder_timing_lock=threading.Lock(), encoder_timing_registry=registry
    )

    out = MR.get_encoder_timing_stats(runner)

    assert out == {"r1": {"encoder_forward_secs": 1.5}}
    assert runner.encoder_timing_registry == {}


def test_timed_encoder_operation_disabled_is_noop():
    runner = new_runner(
        _encoder_timing_lock=threading.Lock(), encoder_timing_registry={}
    )
    with MR.timed_encoder_operation(runner, False, [("r1", None)], 0, 1):
        pass
    assert runner.encoder_timing_registry == {}


def test_timed_encoder_operation_accumulates_per_request(monkeypatch):
    monkeypatch.setattr(
        mr.torch,
        "accelerator",
        SimpleNamespace(synchronize=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(mr.time, "perf_counter", MagicMock(side_effect=[10.0, 10.4]))
    runner = new_runner(
        _encoder_timing_lock=threading.Lock(), encoder_timing_registry={}
    )
    refs = [("r1", None), ("r2", None)]

    with MR.timed_encoder_operation(runner, True, refs, 0, 2):
        pass

    assert set(runner.encoder_timing_registry) == {"r1", "r2"}
    for stat in runner.encoder_timing_registry.values():
        assert stat.num_encoder_calls == 1
        assert stat.encoder_forward_secs == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# reorder threshold + attn group iterators
# ---------------------------------------------------------------------------
def test_calculate_reorder_threshold_without_attention_groups():
    runner = new_runner(attn_groups=[])
    MR.calculate_reorder_batch_threshold(runner)
    assert runner.reorder_batch_threshold is None


def test_calculate_reorder_threshold_takes_min_ignoring_none():
    runner = new_runner(attn_groups=[[_group(8)], [_group(None), _group(4)]])
    MR.calculate_reorder_batch_threshold(runner)
    assert runner.reorder_batch_threshold == 4


def test_calculate_reorder_threshold_all_none():
    runner = new_runner(attn_groups=[[_group(None)], [_group(None)]])
    MR.calculate_reorder_batch_threshold(runner)
    assert runner.reorder_batch_threshold is None


def test_attn_group_iterator_flattens_groups():
    g1, g2, g3 = object(), object(), object()
    runner = new_runner(attn_groups=[[g1, g2], [g3]])
    assert list(MR._attn_group_iterator(runner)) == [g1, g2, g3]


def test_kv_cache_spec_attn_group_iterator_empty_and_populated():
    g1, g2 = object(), object()
    empty = new_runner(
        attn_groups=[[g1]], kv_cache_config=SimpleNamespace(kv_cache_groups=[])
    )
    assert list(MR._kv_cache_spec_attn_group_iterator(empty)) == []

    full = new_runner(
        attn_groups=[[g1], [g2]],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
    )
    assert list(MR._kv_cache_spec_attn_group_iterator(full)) == [g1, g2]


# ---------------------------------------------------------------------------
# KV cache tensor allocation
# ---------------------------------------------------------------------------
def test_initialize_kv_cache_tensors_delegates_to_v032():
    runner = new_runner()
    runner._initialize_kv_cache_tensors_v032 = MagicMock(return_value={"k": 1})
    out = MR.initialize_kv_cache_tensors(runner, "cfg", [4], "ctx")
    assert out == {"k": 1}
    runner._initialize_kv_cache_tensors_v032.assert_called_once_with("cfg", [4], "ctx")


def _patch_v032_allocation(monkeypatch, layout="NHD"):
    gmr = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    vu = importlib.import_module("vllm.v1.worker.utils")
    calls = {}

    def alloc(cfg, device, layout_arg, kernel_blocks):
        calls["alloc"] = (cfg, device, layout_arg, kernel_blocks)
        return {"layer_a": "tensor-a", "layer_x": "tensor-x"}

    def bind(kv_caches, forward_ctx, kv_caches_list, num_attn_module, **kwargs):
        calls["bind"] = (kv_caches, forward_ctx, kv_caches_list, num_attn_module, kwargs)

    monkeypatch.setattr(vu, "allocate_kv_cache", alloc, raising=False)
    monkeypatch.setattr(gmr, "bind_kv_cache", bind, raising=False)
    monkeypatch.setattr(
        gmr,
        "map_kv_caches_to_kernel_blocks",
        lambda kv, ag: {"mapped": tuple(ag)},
        raising=False,
    )
    return calls


def _v032_runner(model_type="llama", speculative_config=None, **extra):
    attrs = dict(
        device="dev",
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: "NHD"),
        shared_kv_cache_layers={"layer_b": "layer_a"},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type)
        ),
        attn_groups=[["g1"]],
        speculative_config=speculative_config,
        compilation_config=SimpleNamespace(static_forward_context={"ctx": 1}),
        kv_caches=[],
    )
    attrs.update(extra)
    return new_runner(**attrs)


def test_initialize_kv_cache_tensors_v032_main(monkeypatch):
    calls = _patch_v032_allocation(monkeypatch)
    runner = _v032_runner()
    cfg = SimpleNamespace(kv_cache_groups=["grp"])

    out = MR._initialize_kv_cache_tensors_v032(runner, cfg, [16])

    assert out["layer_b"] == "tensor-a"
    assert out["layer_a"] == "tensor-a"
    assert calls["alloc"][1] == "dev"
    assert calls["alloc"][2] == "NHD"
    assert calls["alloc"][3] == [16]
    assert calls["bind"][3] == 1
    assert calls["bind"][4]["kv_cache_groups"] == ["grp"]
    assert calls["bind"][4]["layer_kv_caches"] == {"mapped": ("g1",)}


def test_initialize_kv_cache_tensors_v032_longcat_and_context(monkeypatch):
    calls = _patch_v032_allocation(monkeypatch)
    entered = []
    ctx = nullcontext()
    runner = _v032_runner(model_type="longcat_flash")
    cfg = SimpleNamespace(kv_cache_groups=["grp"])

    MR._initialize_kv_cache_tensors_v032(runner, cfg, [8], kv_cache_allocation_context=ctx)

    assert calls["bind"][3] == 2


# ---------------------------------------------------------------------------
# get_kv_cache_spec
# ---------------------------------------------------------------------------
def test_get_kv_cache_spec_ec_producer_returns_empty(monkeypatch):
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: True)
    monkeypatch.setattr(
        mr, "get_ec_transfer", lambda: SimpleNamespace(is_consumer=False)
    )
    runner = new_runner()
    assert MR.get_kv_cache_spec(runner) == {}


def test_get_kv_cache_spec_skips_sharing_and_empty_layers(monkeypatch):
    class FakeAttention:
        pass

    class FakeSpec:
        def __init__(self, tag):
            self.tag = tag

    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(mr, "Attention", FakeAttention)
    monkeypatch.setattr(mr, "AttentionSpec", FakeSpec)
    monkeypatch.setattr(mr, "set_current_vllm_config", lambda cfg: nullcontext())
    monkeypatch.setattr(mr, "UniformTypeKVCacheSpecs", type("U", (), {}))
    monkeypatch.setattr(mr, "replace", lambda spec, **kw: spec)

    spec2 = FakeSpec("s2")
    backend = SimpleNamespace(
        indexes_kv_by_block_stride=lambda: "block_stride",
        customize_spec=lambda s: SimpleNamespace(customized=True, base=s),
    )

    class A1(FakeAttention):
        kv_sharing_target_layer_name = "target"

    class A2(FakeAttention):
        kv_sharing_target_layer_name = None

        def get_kv_cache_spec(self, cfg):
            return spec2

        def get_attn_backend(self):
            return backend

    class A3(FakeAttention):
        kv_sharing_target_layer_name = None

        def get_kv_cache_spec(self, cfg):
            return None

    layers = {"a1": A1(), "a2": A2(), "a3": A3()}
    monkeypatch.setattr(
        mr, "get_layers_from_vllm_config", lambda cfg, layer_type: layers
    )

    runner = new_runner(
        vllm_config=SimpleNamespace(),
        shared_kv_cache_layers={},
        use_ascend_hybrid_cache=False,
    )
    out = MR.get_kv_cache_spec(runner)

    assert runner.shared_kv_cache_layers == {"a1": "target"}
    assert set(out) == {"a2"}
    assert out["a2"].customized is True


# ---------------------------------------------------------------------------
# may_reinitialize_input_batch
# ---------------------------------------------------------------------------
def _spec(block_size, kind, max_blocks):
    return SimpleNamespace(
        block_size=block_size,
        kind=kind,
        max_num_blocks_per_req=lambda cfg, max_len: max_blocks,
    )


def test_may_reinitialize_input_batch_no_change_skips_encoder_only(monkeypatch):
    enc_kind = object()
    normal_kind = object()
    monkeypatch.setattr(mr, "get_kv_cache_spec_kind", lambda spec: spec.kind)
    monkeypatch.setattr(
        mr,
        "KVCacheSpecKind",
        SimpleNamespace(ENCODER_ONLY_ATTENTION=enc_kind, MAMBA=object()),
    )
    monkeypatch.setattr(
        mr,
        "SlotMappingMode",
        SimpleNamespace(NONE="NONE", TOKEN_TO_KV_SLOT="TOKEN_TO_KV_SLOT"),
    )
    constructed = []
    monkeypatch.setattr(mr, "InputBatch", lambda **kw: constructed.append(kw) or "ib")

    groups = [
        SimpleNamespace(kv_cache_spec=_spec(16, enc_kind, 0)),
        SimpleNamespace(kv_cache_spec=_spec(16, normal_kind, 5)),
    ]
    runner = new_runner(
        max_model_len=1000,
        max_encoder_len=10,
        vllm_config=SimpleNamespace(),
        _init_block_sizes=[16],
        _init_kernel_block_sizes=[8],
        _init_max_num_blocks=[5],
        _init_slot_mapping_modes=["TOKEN_TO_KV_SLOT"],
    )
    MR.may_reinitialize_input_batch(
        runner, SimpleNamespace(kv_cache_groups=groups), [8]
    )
    assert constructed == []
    assert runner._init_block_sizes == [16]


def test_may_reinitialize_input_batch_rebuilds_on_change(monkeypatch):
    normal_kind = object()
    mamba_kind = object()
    monkeypatch.setattr(mr, "get_kv_cache_spec_kind", lambda spec: spec.kind)
    monkeypatch.setattr(
        mr,
        "KVCacheSpecKind",
        SimpleNamespace(ENCODER_ONLY_ATTENTION=object(), MAMBA=mamba_kind),
    )
    monkeypatch.setattr(
        mr,
        "SlotMappingMode",
        SimpleNamespace(NONE="NONE", TOKEN_TO_KV_SLOT="TOKEN_TO_KV_SLOT"),
    )

    captured = {}

    class FakeInputBatch:
        def __init__(self, **kw):
            captured.update(kw)
            self.logitsprocs = kw.get("logitsprocs")
            self.logitsprocs_need_output_token_ids = kw.get(
                "logitsprocs_need_output_token_ids"
            )

    monkeypatch.setattr(mr, "InputBatch", FakeInputBatch)

    groups = [
        SimpleNamespace(kv_cache_spec=_spec(32, normal_kind, 9)),
        SimpleNamespace(kv_cache_spec=_spec(64, mamba_kind, 3)),
    ]
    runner = new_runner(
        max_model_len=500,
        max_encoder_len=0,
        max_num_reqs=4,
        max_num_tokens=128,
        device="dev",
        num_spec_tokens=0,
        is_pooling_model=False,
        model_config=SimpleNamespace(get_vocab_size=lambda: 1000),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        vllm_config=SimpleNamespace(reasoning_config=None),
        cache_config=SimpleNamespace(use_replayssm=False),
        jit_warmup_registry=SimpleNamespace(activate=lambda: nullcontext()),
        input_batch=SimpleNamespace(logitsprocs="lp", logitsprocs_need_output_token_ids=False),
        _init_block_sizes=[],
        _init_kernel_block_sizes=[],
        _init_max_num_blocks=[],
        _init_slot_mapping_modes=[],
    )
    MR.may_reinitialize_input_batch(
        runner, SimpleNamespace(kv_cache_groups=groups), [8, 8]
    )

    assert runner._init_block_sizes == [32, 64]
    assert runner._init_max_num_blocks == [9, 3]
    assert runner._init_slot_mapping_modes == ["TOKEN_TO_KV_SLOT", "NONE"]
    assert captured["block_sizes"] == [32, 64]
    assert captured["max_num_blocks_per_req"] == [9, 3]
    assert captured["logitsprocs"] == "lp"


# ---------------------------------------------------------------------------
# metadata builders
# ---------------------------------------------------------------------------
def test_initialize_metadata_builders_basic(monkeypatch):
    group = SimpleNamespace(create_metadata_builders=MagicMock())
    runner = new_runner(
        attn_groups=[[group]],
        parallel_config=SimpleNamespace(use_ubatching=False),
        speculative_config=None,
        vllm_config=SimpleNamespace(),
        device="dev",
    )
    runner.calculate_reorder_batch_threshold = MagicMock()

    MR.initialize_metadata_builders(
        runner, SimpleNamespace(kv_cache_groups=[object()]), [16]
    )

    group.create_metadata_builders.assert_called_once_with(
        runner.vllm_config, "dev", 16, num_metadata_builders=1
    )
    runner.calculate_reorder_batch_threshold.assert_called_once()


def test_initialize_metadata_builders_ubatching_and_drafter(monkeypatch):
    class Proposer:
        pass

    for name in ("EagleProposer", "DFlashProposer", "DraftModelProposer", "Gemma4Proposer"):
        monkeypatch.setattr(mr, name, Proposer)

    drafter = Proposer()
    drafter.initialize_attn_backend = MagicMock()
    group = SimpleNamespace(create_metadata_builders=MagicMock())
    spec = SimpleNamespace(use_eagle=lambda: True, uses_draft_model=lambda: False)
    runner = new_runner(
        attn_groups=[[group]],
        parallel_config=SimpleNamespace(use_ubatching=True, num_ubatches=2),
        speculative_config=spec,
        drafter=drafter,
        vllm_config=SimpleNamespace(),
        device="dev",
    )
    runner.calculate_reorder_batch_threshold = MagicMock()
    cfg = SimpleNamespace(kv_cache_groups=[object()])

    MR.initialize_metadata_builders(runner, cfg, [])

    group.create_metadata_builders.assert_called_once_with(
        runner.vllm_config, "dev", None, num_metadata_builders=2
    )
    drafter.initialize_attn_backend.assert_called_once_with(cfg, [])


# ---------------------------------------------------------------------------
# _check_and_update_cudagraph_mode
# ---------------------------------------------------------------------------
def test_check_and_update_cudagraph_mode_picks_min_and_dispatches(monkeypatch):
    weak = SimpleNamespace(value=1)
    always = SimpleNamespace(value=3)

    class Strong:
        __name__ = "Strong"

        @staticmethod
        def get_builder_cls():
            return SimpleNamespace(
                get_cudagraph_support=lambda cfg, spec: always
            )

    class Weak:
        __name__ = "Weak"

        @staticmethod
        def get_builder_cls():
            return SimpleNamespace(
                get_cudagraph_support=lambda cfg, spec: weak
            )

    runner = new_runner(
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=MagicMock(return_value="RESOLVED")
        ),
        cudagraph_dispatcher=SimpleNamespace(initialize_cudagraph_keys=MagicMock()),
        uniform_decode_query_len=1,
        kv_cache_config="kvcfg",
        max_num_reqs=4,
        vllm_config=SimpleNamespace(),
        speculative_config=None,
    )
    MR._check_and_update_cudagraph_mode(
        runner, [{Strong, Weak}], [SimpleNamespace(kv_cache_spec="spec")]
    )

    args, kwargs = runner.compilation_config.resolve_cudagraph_mode_and_sizes.call_args
    assert args[0] is weak
    assert args[1] == "Weak"
    assert kwargs["use_v2_model_runner"] is False
    assert kwargs["is_profiling"] is False
    runner.cudagraph_dispatcher.initialize_cudagraph_keys.assert_called_once_with(
        "RESOLVED", 1
    )


def test_check_and_update_cudagraph_mode_initializes_drafter(monkeypatch):
    class Proposer:
        pass

    for name in (
        "EagleProposer",
        "DFlashProposer",
        "DraftModelProposer",
        "ExtractHiddenStatesProposer",
        "Gemma4Proposer",
    ):
        monkeypatch.setattr(mr, name, Proposer)

    drafter = Proposer()
    drafter.initialize_cudagraph_keys = MagicMock()
    spec = SimpleNamespace(
        use_eagle=lambda: False,
        uses_draft_model=lambda: False,
        uses_extract_hidden_states=lambda: True,
    )
    always = SimpleNamespace(value=2)

    class OnlyBackend:
        @staticmethod
        def get_builder_cls():
            return SimpleNamespace(get_cudagraph_support=lambda cfg, s: always)

    runner = new_runner(
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=MagicMock(return_value="MODE")
        ),
        cudagraph_dispatcher=SimpleNamespace(initialize_cudagraph_keys=MagicMock()),
        uniform_decode_query_len=2,
        kv_cache_config="kvcfg",
        max_num_reqs=4,
        vllm_config=SimpleNamespace(),
        speculative_config=spec,
        drafter=drafter,
    )
    MR._check_and_update_cudagraph_mode(
        runner, [{OnlyBackend}], [SimpleNamespace(kv_cache_spec="spec")], is_profiling=True
    )
    drafter.initialize_cudagraph_keys.assert_called_once_with("MODE")


# ---------------------------------------------------------------------------
# initialize_attn_backend
# ---------------------------------------------------------------------------
def test_initialize_attn_backend_creates_groups(monkeypatch):
    class FakeBackend:
        def __init__(self, name):
            self.name = name

        def full_cls_name(self):
            return self.name

    backend = FakeBackend("plain")

    class FakeAttentionLayer:
        num_heads = 8
        kv_sharing_target_layer_name = None

        def get_attn_backend(self):
            return backend

    layer = FakeAttentionLayer()

    class FakeGroup:
        def __init__(self, attn_backend, layer_names, kv_cache_spec, group_id):
            self.attn_backend = attn_backend
            self.layer_names = layer_names
            self.kv_cache_spec = kv_cache_spec
            self.group_id = group_id

    monkeypatch.setattr(
        mr,
        "get_layers_from_vllm_config",
        lambda cfg, layer_type, names: {n: layer for n in names},
    )
    monkeypatch.setattr(mr, "check_attention_cp_compatibility", lambda cfg: None)
    monkeypatch.setattr(mr, "UniformTypeKVCacheSpecs", type("U", (), {}))
    monkeypatch.setattr(mr, "AttentionGroup", FakeGroup)

    runner = new_runner(
        attn_groups=[],
        kv_sharing_fast_prefill_eligible_layers=set(),
        vllm_config=SimpleNamespace(),
    )
    runner._check_and_update_cudagraph_mode = MagicMock()
    cfg = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["l1"], kv_cache_spec=object())]
    )

    MR.initialize_attn_backend(runner, cfg, is_profiling=True)

    assert len(runner.attn_groups) == 1
    assert len(runner.attn_groups[0]) == 1
    assert runner.attn_groups[0][0].layer_names == ["l1"]
    assert runner.attn_groups[0][0].group_id == 0
    runner._check_and_update_cudagraph_mode.assert_called_once()


def test_initialize_attn_backend_fast_prefill_backend(monkeypatch):
    class FakeBackend:
        def full_cls_name(self):
            return "plain"

    class FastBackend:
        def full_cls_name(self):
            return "fast"

    layer = SimpleNamespace(
        num_heads=4,
        kv_sharing_target_layer_name=None,
        get_attn_backend=lambda: FakeBackend(),
    )

    class FakeGroup:
        def __init__(self, attn_backend, layer_names, kv_cache_spec, group_id):
            self.attn_backend = attn_backend
            self.layer_names = layer_names

    monkeypatch.setattr(
        mr,
        "get_layers_from_vllm_config",
        lambda cfg, layer_type, names: {n: layer for n in names},
    )
    monkeypatch.setattr(mr, "check_attention_cp_compatibility", lambda cfg: None)
    monkeypatch.setattr(mr, "UniformTypeKVCacheSpecs", type("U", (), {}))
    monkeypatch.setattr(mr, "AttentionGroup", FakeGroup)
    monkeypatch.setattr(
        mr,
        "create_fast_prefill_custom_backend",
        lambda name, base: FastBackend(),
    )

    runner = new_runner(
        attn_groups=[],
        kv_sharing_fast_prefill_eligible_layers={"l1"},
        vllm_config=SimpleNamespace(),
    )
    runner._check_and_update_cudagraph_mode = MagicMock()
    cfg = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["l1"], kv_cache_spec=object())]
    )

    MR.initialize_attn_backend(runner, cfg)

    assert runner.attn_groups[0][0].attn_backend.full_cls_name() == "fast"


# ---------------------------------------------------------------------------
# kv sharing / encoder-only helpers
# ---------------------------------------------------------------------------
def test_maybe_add_kv_sharing_no_layers_returns(monkeypatch):
    called = []
    monkeypatch.setattr(
        mr,
        "add_kv_sharing_layers_to_kv_cache_groups",
        lambda *a: called.append(a),
    )
    runner = new_runner(shared_kv_cache_layers={})
    MR.maybe_add_kv_sharing_layers_to_kv_cache_groups(
        runner, SimpleNamespace(kv_cache_groups=[])
    )
    assert called == []


def test_maybe_add_kv_sharing_fast_prefill_suffix(monkeypatch):
    monkeypatch.setattr(
        mr, "add_kv_sharing_layers_to_kv_cache_groups", lambda *a: None
    )
    monkeypatch.setattr(
        mr,
        "get_layers_from_vllm_config",
        lambda cfg, layer_type: {"l1": object(), "l2": object()},
    )
    runner = new_runner(
        shared_kv_cache_layers={"l2": "l0"},
        runner_only_attn_layers=set(),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=True),
        kv_sharing_fast_prefill_eligible_layers=set(),
        vllm_config=SimpleNamespace(),
    )
    MR.maybe_add_kv_sharing_layers_to_kv_cache_groups(
        runner, SimpleNamespace(kv_cache_groups=[])
    )
    assert runner.kv_sharing_fast_prefill_eligible_layers == {"l2"}


def test_may_add_encoder_only_layers(monkeypatch):
    monkeypatch.setattr(mr, "Attention", type("FakeAttn", (), {}))
    monkeypatch.setattr(
        mr, "EncoderOnlyAttentionSpec", lambda **kw: ("spec", kw["block_size"])
    )
    monkeypatch.setattr(
        mr, "AttentionType", SimpleNamespace(ENCODER_ONLY="enc", DECODER="dec")
    )

    class FakeGroup:
        def __init__(self, layer_names, kv_cache_spec):
            self.layer_names = layer_names
            self.kv_cache_spec = kv_cache_spec

    monkeypatch.setattr(mr, "KVCacheGroupSpec", FakeGroup)
    layers = {
        "l1": SimpleNamespace(attn_type="enc", num_kv_heads=2, head_size=8),
        "l2": SimpleNamespace(attn_type="dec"),
    }
    monkeypatch.setattr(
        mr, "get_layers_from_vllm_config", lambda cfg, layer_type: layers
    )
    runner = new_runner(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(block_size=16)),
        kv_cache_dtype="float16",
        runner_only_attn_layers=set(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )

    MR.may_add_encoder_only_layers_to_kv_cache_config(runner)

    assert runner.runner_only_attn_layers == {"l1"}
    assert len(runner.kv_cache_config.kv_cache_groups) == 1
    assert runner.kv_cache_config.kv_cache_groups[0].layer_names == ["l1"]


# ---------------------------------------------------------------------------
# initialize_kv_cache orchestration
# ---------------------------------------------------------------------------
def _orchestration_runner():
    runner = new_runner(
        vllm_config=SimpleNamespace(mamba_config=object()),
        cache_config=SimpleNamespace(num_gpu_blocks=None),
        attn_groups=[],
        graph_runtime=None,
        speculative_config=None,
        cross_layers_kv_cache=None,
        cross_layers_attn_backend=None,
        kv_caches=[],
        jit_warmup_registry=SimpleNamespace(activate=lambda: nullcontext()),
    )
    runner.may_add_encoder_only_layers_to_kv_cache_config = MagicMock()
    runner.maybe_add_kv_sharing_layers_to_kv_cache_groups = MagicMock()
    runner.initialize_attn_backend = MagicMock()
    runner.initialize_metadata_builders = MagicMock()
    runner.may_reinitialize_input_batch = MagicMock()
    runner.initialize_kv_cache_tensors = MagicMock(return_value={"k": 1})
    return runner


def test_initialize_kv_cache_orchestrates(monkeypatch):
    monkeypatch.setattr(mr, "deepcopy", lambda cfg: cfg)
    monkeypatch.setattr(mr, "initialize_mamba_ssu_backend", lambda *a: None)
    monkeypatch.setattr(mr, "prepare_kernel_block_sizes", lambda cfg, ag: [16])
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)
    runner = _orchestration_runner()
    cfg = SimpleNamespace(kv_cache_groups=[object()])

    MR.initialize_kv_cache(runner, cfg, is_profiling=True)

    assert runner.kv_cache_config is cfg
    assert runner._kernel_block_sizes == [16]
    runner.initialize_attn_backend.assert_called_once_with(cfg, is_profiling=True)
    runner.initialize_kv_cache_tensors.assert_called_once_with(cfg, [16])


def test_initialize_kv_cache_registers_kv_transfer(monkeypatch):
    monkeypatch.setattr(mr, "deepcopy", lambda cfg: cfg)
    monkeypatch.setattr(mr, "initialize_mamba_ssu_backend", lambda *a: None)
    monkeypatch.setattr(mr, "prepare_kernel_block_sizes", lambda cfg, ag: [8])
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(mr, "copy_kv_blocks", lambda: None)
    transfer = SimpleNamespace(
        register_cross_layers_kv_cache=MagicMock(),
        register_kv_caches=MagicMock(),
        set_host_xfer_buffer_ops=MagicMock(),
    )
    monkeypatch.setattr(mr, "get_kv_transfer_group", lambda: transfer)
    runner = _orchestration_runner()

    MR.initialize_kv_cache(
        runner, SimpleNamespace(kv_cache_groups=[object()]), is_profiling=False
    )

    transfer.register_kv_caches.assert_called_once_with({"k": 1})
    transfer.set_host_xfer_buffer_ops.assert_called_once()


def test_initialize_kv_cache_cross_layer_transfer(monkeypatch):
    monkeypatch.setattr(mr, "deepcopy", lambda cfg: cfg)
    monkeypatch.setattr(mr, "initialize_mamba_ssu_backend", lambda *a: None)
    monkeypatch.setattr(mr, "prepare_kernel_block_sizes", lambda cfg, ag: [8])
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(mr, "copy_kv_blocks", lambda: None)
    transfer = SimpleNamespace(
        register_cross_layers_kv_cache=MagicMock(),
        register_kv_caches=MagicMock(),
        set_host_xfer_buffer_ops=MagicMock(),
    )
    monkeypatch.setattr(mr, "get_kv_transfer_group", lambda: transfer)
    runner = _orchestration_runner()
    runner.cross_layers_kv_cache = "cross"
    runner.cross_layers_attn_backend = "backend"

    MR.initialize_kv_cache(
        runner, SimpleNamespace(kv_cache_groups=[object()]), is_profiling=False
    )

    transfer.register_cross_layers_kv_cache.assert_called_once_with("cross", "backend")
    transfer.register_kv_caches.assert_not_called()
