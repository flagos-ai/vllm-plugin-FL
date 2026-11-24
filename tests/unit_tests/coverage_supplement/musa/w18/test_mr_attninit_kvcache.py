"""Extra coverage for KV-cache tensors / init / encoder timing / capturer.

Complements w3d/test_mr_kv_attn_cache.py by covering the remaining branches:

* ``_initialize_kv_cache_tensors_v032``: speculative draft attn groups are
  appended only on the last PP rank.
* ``initialize_kv_cache``: ``graph_runtime.bind_attention_groups`` and the
  ``uses_extract_hidden_states`` drafter validation branch.
* ``get_kv_cache_spec``: backend without the ``indexes_kv_by_block_stride``
  hook (no ``replace``) and the ``use_ascend_hybrid_cache`` padding path.
* ``may_add_encoder_only_layers_to_kv_cache_config``: no encoder-only layers.
* ``may_reinitialize_input_batch``: max() of model/encoder len with a MAMBA
  spec selecting ``SlotMappingMode.NONE``.
* ``timed_encoder_operation``: duplicate / empty request groups and the
  exception path that must still record timings.
* ``init_routed_experts_capturer``: full method with fake capturer + buffers.

All runners are ``object.__new__(ModelRunnerFL)``; allocation, distrib and
torch buffers are doubles so nothing runs on a device.
"""

import importlib
import sys
import threading
import types
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


# ---------------------------------------------------------------------------
# _initialize_kv_cache_tensors_v032: speculative draft groups on last rank
# ---------------------------------------------------------------------------
def _patch_v032(monkeypatch):
    gmr = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    vu = importlib.import_module("vllm.v1.worker.utils")
    seen = {}

    def alloc(cfg, device, layout, kernel_blocks):
        seen["alloc"] = (device, layout, kernel_blocks)
        return {"layer_a": "tensor-a"}

    def mapper(kv_caches, attn_groups):
        seen["attn_groups"] = list(attn_groups)
        return {"mapped": list(attn_groups)}

    monkeypatch.setattr(vu, "allocate_kv_cache", alloc, raising=False)
    monkeypatch.setattr(gmr, "bind_kv_cache", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(
        gmr, "map_kv_caches_to_kernel_blocks", mapper, raising=False
    )
    return seen


def _v032_runner(speculative_config):
    return new_runner(
        device="dev",
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: "NHD"),
        shared_kv_cache_layers={},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="llama")),
        attn_groups=[["g1"]],
        speculative_config=speculative_config,
        drafter=SimpleNamespace(draft_attn_groups=["dg"]),
        compilation_config=SimpleNamespace(static_forward_context={"ctx": 1}),
        kv_caches=[],
    )


def test_v032_appends_draft_attn_groups_on_last_rank(monkeypatch):
    seen = _patch_v032(monkeypatch)
    monkeypatch.setattr(
        mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    cfg = SimpleNamespace(kv_cache_groups=["grp"])

    MR._initialize_kv_cache_tensors_v032(_v032_runner(object()), cfg, [16])

    assert seen["attn_groups"] == ["g1", "dg"]


def test_v032_skips_draft_attn_groups_off_last_rank(monkeypatch):
    seen = _patch_v032(monkeypatch)
    monkeypatch.setattr(
        mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    cfg = SimpleNamespace(kv_cache_groups=["grp"])

    MR._initialize_kv_cache_tensors_v032(_v032_runner(object()), cfg, [16])

    assert seen["attn_groups"] == ["g1"]


# ---------------------------------------------------------------------------
# initialize_kv_cache: graph runtime + extract-hidden-states drafter
# ---------------------------------------------------------------------------
class _FakeExtract:
    def __init__(self):
        self.validate_same_kv_cache_group = MagicMock()


def _orchestration_runner(**attrs):
    base = dict(
        vllm_config=SimpleNamespace(mamba_config=object()),
        cache_config=SimpleNamespace(num_gpu_blocks=None),
        attn_groups=[["g"]],
        graph_runtime=None,
        speculative_config=None,
        cross_layers_kv_cache=None,
        cross_layers_attn_backend=None,
        kv_caches=[],
        jit_warmup_registry=SimpleNamespace(activate=lambda: nullcontext()),
    )
    base.update(attrs)
    runner = new_runner(**base)
    runner.may_add_encoder_only_layers_to_kv_cache_config = MagicMock()
    runner.maybe_add_kv_sharing_layers_to_kv_cache_groups = MagicMock()
    runner.initialize_attn_backend = MagicMock()
    runner.initialize_metadata_builders = MagicMock()
    runner.may_reinitialize_input_batch = MagicMock()
    runner.initialize_kv_cache_tensors = MagicMock(return_value={"k": 1})
    return runner


def _patch_orchestration(monkeypatch):
    monkeypatch.setattr(mr, "deepcopy", lambda cfg: cfg)
    monkeypatch.setattr(mr, "initialize_mamba_ssu_backend", lambda *a: None)
    monkeypatch.setattr(mr, "prepare_kernel_block_sizes", lambda cfg, ag: [8])
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)


def test_initialize_kv_cache_binds_graph_runtime_and_validation(monkeypatch):
    _patch_orchestration(monkeypatch)
    monkeypatch.setattr(mr, "ExtractHiddenStatesProposer", _FakeExtract)
    drafter = _FakeExtract()
    graph_runtime = SimpleNamespace(bind_attention_groups=MagicMock())
    runner = _orchestration_runner(
        graph_runtime=graph_runtime,
        speculative_config=SimpleNamespace(
            uses_extract_hidden_states=lambda: True
        ),
        drafter=drafter,
    )
    cfg = SimpleNamespace(kv_cache_groups=[object()])

    MR.initialize_kv_cache(runner, cfg, is_profiling=False)

    graph_runtime.bind_attention_groups.assert_called_once_with(runner.attn_groups)
    drafter.validate_same_kv_cache_group.assert_called_once_with(cfg)


# ---------------------------------------------------------------------------
# get_kv_cache_spec: no stride hook + ascend hybrid padding
# ---------------------------------------------------------------------------
def test_get_kv_cache_spec_backend_without_stride_hook(monkeypatch):
    class FakeAttention:
        pass

    class FakeSpec:
        pass

    custom = FakeSpec()

    class Layer(FakeAttention):
        kv_sharing_target_layer_name = None

        def get_kv_cache_spec(self, cfg):
            return custom

        def get_attn_backend(self):
            return SimpleNamespace(customize_spec=lambda s: ("customized", s))

    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(mr, "Attention", FakeAttention)
    monkeypatch.setattr(mr, "AttentionSpec", FakeSpec)
    monkeypatch.setattr(mr, "set_current_vllm_config", lambda cfg: nullcontext())
    monkeypatch.setattr(mr, "replace", MagicMock())
    monkeypatch.setattr(
        mr, "get_layers_from_vllm_config", lambda cfg, lt: {"l": Layer()}
    )
    runner = new_runner(
        vllm_config=SimpleNamespace(),
        shared_kv_cache_layers={},
        use_ascend_hybrid_cache=False,
    )

    out = MR.get_kv_cache_spec(runner)

    # No ``indexes_kv_by_block_stride`` -> ``replace`` is skipped; customize still runs.
    mr.replace.assert_not_called()
    assert out == {"l": ("customized", custom)}


def test_get_kv_cache_spec_ascend_hybrid_pads(monkeypatch):
    class FakeAttention:
        pass

    class Layer(FakeAttention):
        kv_sharing_target_layer_name = None

        def get_kv_cache_spec(self, cfg):
            return "plain-spec"

    hyb = types.ModuleType("vllm_fl.kv_cache.ascend.hybrid")
    hyb.pad_attention_specs = lambda spec, pad: {"spec": spec, "pad": pad}
    asc = types.ModuleType("vllm_fl.kv_cache.ascend")
    asc.hybrid = hyb
    pkg = types.ModuleType("vllm_fl.kv_cache")
    pkg.ascend = asc
    for name, mod in (
        ("vllm_fl.kv_cache", pkg),
        ("vllm_fl.kv_cache.ascend", asc),
        ("vllm_fl.kv_cache.ascend.hybrid", hyb),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(mr, "Attention", FakeAttention)
    monkeypatch.setattr(
        mr, "get_layers_from_vllm_config", lambda cfg, lt: {"l": Layer()}
    )
    runner = new_runner(
        vllm_config=SimpleNamespace(),
        shared_kv_cache_layers={},
        use_ascend_hybrid_cache=True,
        cache_config=SimpleNamespace(mamba_page_size_padded=128),
    )

    out = MR.get_kv_cache_spec(runner)

    assert out == {"spec": {"l": "plain-spec"}, "pad": 128}


# ---------------------------------------------------------------------------
# may_add_encoder_only_layers_to_kv_cache_config: no encoder-only layers
# ---------------------------------------------------------------------------
def test_may_add_encoder_only_layers_noop(monkeypatch):
    monkeypatch.setattr(mr, "Attention", type("FakeAttn", (), {}))
    monkeypatch.setattr(
        mr, "AttentionType", SimpleNamespace(ENCODER_ONLY="enc", DECODER="dec")
    )
    monkeypatch.setattr(
        mr,
        "get_layers_from_vllm_config",
        lambda cfg, lt: {"l1": SimpleNamespace(attn_type="dec")},
    )
    monkeypatch.setattr(
        mr,
        "EncoderOnlyAttentionSpec",
        MagicMock(side_effect=AssertionError("should not build a spec")),
    )
    groups = []
    runner = new_runner(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(block_size=16)),
        kv_cache_dtype="float16",
        runner_only_attn_layers=set(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
    )

    MR.may_add_encoder_only_layers_to_kv_cache_config(runner)

    assert runner.kv_cache_config.kv_cache_groups == []
    assert runner.runner_only_attn_layers == set()


# ---------------------------------------------------------------------------
# may_reinitialize_input_batch: max model/encoder len with MAMBA spec
# ---------------------------------------------------------------------------
def test_may_reinitialize_input_batch_mamba_uses_max_len(monkeypatch):
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
    seen = {}

    class FakeInputBatch:
        def __init__(self, **kw):
            seen.update(kw)
            self.logitsprocs = kw.get("logitsprocs")
            self.logitsprocs_need_output_token_ids = kw.get(
                "logitsprocs_need_output_token_ids"
            )

    monkeypatch.setattr(mr, "InputBatch", FakeInputBatch)

    def max_blocks(cfg, max_len):
        seen["max_len"] = max_len
        return max_len + 1

    spec = SimpleNamespace(
        block_size=32, kind=mamba_kind, max_num_blocks_per_req=max_blocks
    )
    runner = new_runner(
        max_model_len=100,
        max_encoder_len=250,
        max_num_reqs=2,
        max_num_tokens=64,
        device="dev",
        num_spec_tokens=0,
        is_pooling_model=False,
        model_config=SimpleNamespace(get_vocab_size=lambda: 10),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        vllm_config=SimpleNamespace(reasoning_config=None),
        cache_config=SimpleNamespace(use_replayssm=False),
        jit_warmup_registry=SimpleNamespace(activate=lambda: nullcontext()),
        input_batch=SimpleNamespace(
            logitsprocs="lp", logitsprocs_need_output_token_ids=False
        ),
        _init_block_sizes=[],
        _init_kernel_block_sizes=[],
        _init_max_num_blocks=[],
        _init_slot_mapping_modes=[],
    )

    MR.may_reinitialize_input_batch(
        runner, SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]), [8]
    )

    assert seen["max_len"] == 250
    assert runner._init_block_sizes == [32]
    assert runner._init_max_num_blocks == [251]
    assert runner._init_slot_mapping_modes == ["NONE"]
    assert seen["slot_mapping_modes"] == ["NONE"]


# ---------------------------------------------------------------------------
# timed_encoder_operation: duplicate / empty / exception groups
# ---------------------------------------------------------------------------
def _patch_perf(monkeypatch, times):
    monkeypatch.setattr(mr.torch, "accelerator", SimpleNamespace(synchronize=MagicMock()), raising=False)
    monkeypatch.setattr(mr.time, "perf_counter", MagicMock(side_effect=list(times)))


def test_timed_encoder_operation_dedupes_duplicate_request_ids(monkeypatch):
    _patch_perf(monkeypatch, [1.0, 1.6])
    runner = new_runner(
        _encoder_timing_lock=threading.Lock(), encoder_timing_registry={}
    )
    refs = [("r", None), ("r", "pos")]

    with MR.timed_encoder_operation(runner, True, refs, 0, 2):
        pass

    assert set(runner.encoder_timing_registry) == {"r"}
    stat = runner.encoder_timing_registry["r"]
    assert stat.num_encoder_calls == 1
    assert stat.encoder_forward_secs == pytest.approx(0.6)


def test_timed_encoder_operation_empty_group_records_nothing(monkeypatch):
    _patch_perf(monkeypatch, [2.0, 2.5])
    runner = new_runner(
        _encoder_timing_lock=threading.Lock(), encoder_timing_registry={}
    )

    with MR.timed_encoder_operation(runner, True, [("a", None)], 0, 0):
        pass

    assert runner.encoder_timing_registry == {}


def test_timed_encoder_operation_records_on_exception(monkeypatch):
    _patch_perf(monkeypatch, [0.0, 1.0])
    runner = new_runner(
        _encoder_timing_lock=threading.Lock(), encoder_timing_registry={}
    )

    with pytest.raises(RuntimeError):
        with MR.timed_encoder_operation(runner, True, [("e", None)], 0, 1):
            raise RuntimeError("boom")

    stat = runner.encoder_timing_registry["e"]
    assert stat.num_encoder_calls == 1
    assert stat.encoder_forward_secs == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# init_routed_experts_capturer
# ---------------------------------------------------------------------------
def test_init_routed_experts_capturer_builds_buffers(monkeypatch):
    monkeypatch.setattr(mr, "PIN_MEMORY", True, raising=False)

    class FakeCapturer:
        def __init__(self, **kw):
            self.init_kw = kw
            self.device_buffer = SimpleNamespace(shape=(8, 4), dtype="float16")

    created = {}

    def fake_capturer(**kw):
        cap = FakeCapturer(**kw)
        created["cap"] = cap
        return cap

    bound = MagicMock()
    monkeypatch.setattr(mr, "RoutedExpertsCapturer", fake_capturer)
    monkeypatch.setattr(mr, "bind_routed_experts_capturer", bound)

    empty_calls = []

    def fake_empty(*args, **kwargs):
        empty_calls.append((args, kwargs))
        shape = args[0]
        return SimpleNamespace(
            shape=shape,
            dtype=kwargs.get("dtype"),
            device=kwargs.get("device"),
            pin_memory=kwargs.get("pin_memory"),
        )

    monkeypatch.setattr(mr.torch, "empty", fake_empty, raising=False)

    model = object()
    vllm_config = object()
    kv_cache_config = object()
    runner = new_runner(
        model_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        model=model,
        device="musa:0",
    )

    MR.init_routed_experts_capturer(runner)

    cap = created["cap"]
    assert cap.init_kw == {
        "max_num_batched_tokens": 32,
        "vllm_config": vllm_config,
        "kv_cache_config": kv_cache_config,
    }
    assert runner.routed_experts_capturer is cap
    bound.assert_called_once_with(model, cap)
    assert runner.routed_experts_initialized is True

    # Pinned CPU mirror of the device buffer.
    assert runner.routed_experts_cpu.shape == (8, 4)
    assert runner.routed_experts_cpu.dtype == "float16"
    assert runner.routed_experts_cpu.device == "cpu"
    assert runner.routed_experts_cpu.pin_memory is True

    # int64 slot mapping on CPU and device.
    assert runner.routed_experts_slot_mapping_cpu.shape == (32,)
    assert runner.routed_experts_slot_mapping_cpu.dtype is mr.torch.int64
    assert runner.routed_experts_slot_mapping_cpu.device == "cpu"
    assert runner.routed_experts_slot_mapping_device.shape == (32,)
    assert runner.routed_experts_slot_mapping_device.device == "musa:0"
    assert len(empty_calls) == 3
