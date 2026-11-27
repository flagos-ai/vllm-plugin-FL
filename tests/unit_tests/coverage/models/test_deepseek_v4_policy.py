# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU policy tests for the DeepSeek-V4 Ascend adapter (live module import).

``vllm_fl/models/deepseek_v4.py`` is an Ascend 910C execution adapter: its
import needs ``vllm.forward_context``, the upstream ``vllm.models.deepseek_v4``
classes, ``vllm.platforms.current_platform`` and (on npu) the
``flaggems_vllm`` HC-head kernel — none installed on CPU-only machines.
This module loads the *real* production source under stubbed external
modules so the actual Python branches execute:

* ``_validate_short_context_attention`` — every rejection branch plus the
  valid-config passes, with expectations derived by hand;
* the registry surface: backend names, builder classes, compute
  capability, supported kv dtypes, tile scheduler, padded heads;
* ``_install_hc_head_fallback`` — patch installation, idempotence and the
  ``forward_native`` numeric contract against a recording
  ``flaggems_vllm`` stub;
* ``DeepseekV4FLAttention.__init__`` — the ``torch.cuda.Event`` swap around
  ``super().__init__`` (unwound by the ``finally`` even when upstream
  raises) and kwarg/positional vllm_config resolution;
* ``DeepseekV4ForCausalLM.__init__`` — validation before the upstream
  constructor and the resulting inheritance wiring.

The module-level ``current_platform.device_type`` is stubbed to ``"npu"``
so both npu-gated patches (HCHead fallback and the XPU model override) run;
``torch.npu.Event`` is provided via a temporary attribute on ``torch``
(removed afterwards) because CPU torch has no ``torch.npu``.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[4] / "vllm_fl/models/deepseek_v4.py"
_MODULE_NAME = "supp_deepseek_v4_under_test"



class _UpstreamBase:
    """Recording stand-in for upstream DeepseekV4Attention.

    Counts constructions so tests can prove validation rejected a config
    *before* ``super().__init__`` ran, and raises on ``boom=True`` so the
    FL constructor's ``finally`` branch is exercised with a real failure.
    """

    _constructions = 0

    def __init__(self, *args, **kwargs):
        _UpstreamBase._constructions += 1
        if kwargs.get("boom"):
            raise RuntimeError("upstream boom")
        self.init_args = args
        self.init_kwargs = kwargs
        # The Event class visible while super().__init__ runs.
        self.event_seen = torch.cuda.Event


class _HCHeadOp:
    """Stand-in for vllm.model_executor.layers.mhc.HCHeadOp."""


class _DeepseekV4SparseMLAMetadataBuilder:
    """Stand-in for the upstream sparse-MLA metadata builder."""


class _XpuModelBase:
    """Stand-in for the upstream XPU causal-LM base class."""

    _constructions = 0

    def __init__(self, *, vllm_config, prefix=""):
        _XpuModelBase._constructions += 1
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.constructed = True


def _install_stubs(monkeypatch):
    """Install the fake vllm/flaggems_vllm tree; returns shared fixtures."""
    hc_calls = []

    def hc_head_fused_kernel(
        flat, hc_fn, hc_scale, hc_base, out, hidden_size, eps, hc_eps, hc_mult
    ):
        hc_calls.append(
            (
                tuple(flat.shape),
                hc_fn,
                hc_scale,
                hc_base,
                tuple(out.shape),
                hidden_size,
                hc_mult,
            )
        )
        out.fill_(2.0)

    modules = {}

    def leaf(name, **attrs):
        parts = name.split(".")
        for length in range(1, len(parts) + 1):
            partial = ".".join(parts[:length])
            if partial not in modules:
                module = ModuleType(partial)
                module.__path__ = []
                modules[partial] = module
        for length in range(1, len(parts)):
            setattr(
                modules[".".join(parts[:length])],
                parts[length],
                modules[".".join(parts[: length + 1])],
            )
        for key, value in attrs.items():
            setattr(modules[name], key, value)
        return modules[name]

    leaf("vllm.forward_context", get_forward_context=lambda: SimpleNamespace())
    leaf(
        "vllm.models.deepseek_v4.attention",
        DeepseekV4Attention=_UpstreamBase,
    )
    leaf(
        "vllm.models.deepseek_v4.sparse_mla",
        DeepseekV4FlashMLAMetadata=type("DeepseekV4FlashMLAMetadata", (), {}),
        DeepseekV4SparseMLABackend=type("DeepseekV4SparseMLABackend", (), {}),
        DeepseekV4SparseMLAMetadataBuilder=_DeepseekV4SparseMLAMetadataBuilder,
    )
    leaf("vllm.platforms", current_platform=SimpleNamespace(device_type="npu"))
    leaf(
        "vllm.v1.attention.backends.mla.sparse_swa",
        DeepseekSparseSWABackend=type("DeepseekSparseSWABackend", (), {}),
        DeepseekSparseSWAMetadata=type("DeepseekSparseSWAMetadata", (), {}),
        DeepseekSparseSWAMetadataBuilder=type(
            "DeepseekSparseSWAMetadataBuilder", (), {}
        ),
    )
    leaf("vllm.model_executor.layers.mhc", HCHeadOp=_HCHeadOp)
    leaf("flaggems_vllm", hc_head_fused_kernel=hc_head_fused_kernel)

    xpu_model = SimpleNamespace(
        DeepseekV4ForCausalLM=_XpuModelBase,
        DeepseekV4XPUAttention=object,
    )
    leaf("vllm.models.deepseek_v4.xpu", model=xpu_model)

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(hc_calls=hc_calls, xpu_model=xpu_model)


@pytest.fixture
def deepseek_mod(monkeypatch):
    """Load a fresh copy of the real deepseek_v4.py under stubs.

    Also installs a temporary ``torch.npu`` namespace (CPU torch has none)
    and removes it afterwards. Any HCHeadOp patch a previous module load
    left on the shared stub class is stripped first so every test observes
    a fresh install (monkeypatch restores the stripped state at teardown).
    """
    stubs = _install_stubs(monkeypatch)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=object), raising=False)
    monkeypatch.delattr(_HCHeadOp, "forward_native", raising=False)
    monkeypatch.delattr(_HCHeadOp, "_vllm_fl_ascend", raising=False)

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    spec.loader.exec_module(module)
    module._stubs = stubs
    return module


def _vllm_config(ratios, layers, sliding_window, max_model_len, index_topk):
    """Build the config shape _validate_short_context_attention reads."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                compress_ratios=list(ratios),
                num_hidden_layers=layers,
                sliding_window=sliding_window,
                index_topk=index_topk,
            ),
            max_model_len=max_model_len,
        )
    )


def _valid_config():
    """A config that passes every validation branch.

    Derived by hand: all ratios in {0, 1, 4, 128}; max_model_len 128 equals
    min(sliding_window 4096, 128); with ratio 4 present,
    (128 + 3) // 4 = 32 <= index_topk 64.
    """
    return _vllm_config([0, 1, 4, 128], 4, 4096, 128, 64)


# ---------------------------------------------------------------------------
# _validate_short_context_attention
# ---------------------------------------------------------------------------


def test_validate_accepts_all_supported_ratios_and_boundary_len(deepseek_mod):
    """Ratios {0,1,4,128} with max_model_len == min(window, 128) pass."""
    deepseek_mod._validate_short_context_attention(_valid_config())
    # The boundary itself: max_model_len == supported_context is allowed,
    # including a sliding window smaller than the hard 128 cap.
    deepseek_mod._validate_short_context_attention(_vllm_config([0], 1, 4096, 128, 64))
    deepseek_mod._validate_short_context_attention(_vllm_config([0], 1, 64, 64, 64))
    # Ratios past num_hidden_layers are ignored (the slice guard).
    deepseek_mod._validate_short_context_attention(
        _vllm_config([0, 99], 1, 4096, 128, 64)
    )


@pytest.mark.parametrize(
    ("bad_ratio", "layer_id"),
    [(2, 0), (8, 1), (3, 2)],
)
def test_validate_rejects_unsupported_compress_ratio(deepseek_mod, bad_ratio, layer_id):
    """A compress ratio outside {0, 1, 4, 128} raises NotImplementedError.

    The error must name the offending layer id — derived from enumerate()
    over the first num_hidden_layers ratios.
    """
    ratios = [0] * 4
    ratios[layer_id] = bad_ratio
    with pytest.raises(NotImplementedError, match=f"layer {layer_id}"):
        deepseek_mod._validate_short_context_attention(
            _vllm_config(ratios, 4, 4096, 128, 64)
        )


@pytest.mark.parametrize(
    ("sliding_window", "max_model_len"),
    [(4096, 129), (64, 65), (128, 200)],
)
def test_validate_rejects_context_longer_than_supported(
    deepseek_mod, sliding_window, max_model_len
):
    """max_model_len above min(sliding_window, 128) raises ValueError.

    supported_context = min(sliding_window, 128); the message must echo it
    as the suggested --max-model-len.
    """
    with pytest.raises(ValueError) as excinfo:
        deepseek_mod._validate_short_context_attention(
            _vllm_config([0], 1, sliding_window, max_model_len, 64)
        )
    supported = min(sliding_window, 128)
    assert f"--max-model-len {supported}" in str(excinfo.value)


@pytest.mark.parametrize(("max_model_len", "index_topk"), [(128, 31), (125, 31)])
def test_validate_rejects_c4a_candidates_exceeding_index_topk(
    deepseek_mod, max_model_len, index_topk
):
    """With ratio 4, (max_model_len + 3) // 4 must fit within index_topk.

    Independent oracle: (125 + 3) // 4 = (128 + 3) // 4 = 32 compressed
    candidates; index_topk 31 cannot hold them, so the C4A gate raises.
    """
    assert (max_model_len + 3) // 4 == 32
    with pytest.raises(NotImplementedError, match="index_topk"):
        deepseek_mod._validate_short_context_attention(
            _vllm_config([4], 1, 4096, max_model_len, index_topk)
        )


def test_validate_c4a_boundary_passes_exactly(deepseek_mod):
    """(max_model_len + 3) // 4 == index_topk is the passing boundary."""
    assert (125 + 3) // 4 == 32
    deepseek_mod._validate_short_context_attention(_vllm_config([4], 1, 4096, 125, 32))
    deepseek_mod._validate_short_context_attention(_vllm_config([4], 1, 4096, 128, 32))


def test_validate_ignores_ratio_4_when_beyond_hidden_layers(deepseek_mod):
    """The index_topk gate only fires when ratio 4 is within the slice."""
    deepseek_mod._validate_short_context_attention(
        _vllm_config([0, 0, 4], 2, 4096, 128, 1)
    )


# ---------------------------------------------------------------------------
# Cache helpers (live execution)
# ---------------------------------------------------------------------------


def test_cache_helpers_reject_non_page_shaped_cache(deepseek_mod):
    """Both helpers require a 3D [blocks, block_size, ...] page layout.

    These cases execute the helpers inside the loaded module so the
    live lines run. Independent oracle: ndim != 3 or shape[1] != block
    size must raise before any indexing happens.
    """
    cache_2d = torch.zeros(8, 16)
    cache_wrong_bs = torch.zeros(4, 2, 16)
    slots = torch.tensor([0, 3])
    for bad_cache in (cache_2d, cache_wrong_bs):
        with pytest.raises(ValueError, match="KV cache shape"):
            deepseek_mod._cache_rows(bad_cache, slots, 8)
        with pytest.raises(ValueError, match="KV cache shape"):
            deepseek_mod._store_cache_rows(bad_cache, slots, torch.zeros(2, 16), 8)


def test_cache_helpers_round_trip_through_non_contiguous_pages(deepseek_mod):
    """Rows map slots as (slots // bs, slots % bs) across scattered pages.

    Independent oracle for block size 2 over a [3, 2, 4] cache: slots
    [0, 3, 5] address rows (0,0), (1,1), (2,1) — the pages need not be
    contiguous — and the stored values must read back identically.
    """
    cache = torch.zeros(3, 2, 4)
    slots = torch.tensor([0, 3, 5])
    rows = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    deepseek_mod._store_cache_rows(cache, slots, rows, 2)
    read_back = deepseek_mod._cache_rows(cache, slots, 2)

    assert torch.equal(read_back, rows)
    # The exact page/row addressing the slot arithmetic must produce.
    expected = torch.zeros(3, 2, 4)
    expected[0, 0] = rows[0]
    expected[1, 1] = rows[1]
    expected[2, 1] = rows[2]
    assert torch.equal(cache, expected)


def test_store_then_read_round_trips_through_scrambled_slot_order(deepseek_mod):
    """Store and read logical rows through non-contiguous page assignment.

    Ported from the retired supplementary ``test_deepseek_v4_contracts.py``
    (its only variant not otherwise covered): the slot list deliberately
    maps to (page, offset) pairs in non-ascending order, so the page/offset
    split cannot shortcut via sorted or contiguous assumptions.
    """
    num_blocks = 3
    block_size = 2
    cache = torch.zeros((num_blocks, block_size, 5), dtype=torch.float32)
    width = cache.shape[-1]
    # Scrambled logical order: slot i maps to (page, offset) pairs that are
    # deliberately not in ascending order.
    targets = [(2, block_size - 1), (0, 0), (1, min(block_size - 1, 1))]
    slots = torch.tensor([page * block_size + offset for page, offset in targets])
    rows = torch.arange(3 * width, dtype=torch.float32).reshape(3, width) + 1.0

    deepseek_mod._store_cache_rows(cache, slots, rows, block_size)

    read_back = deepseek_mod._cache_rows(cache, slots, block_size)
    torch.testing.assert_close(read_back, rows, rtol=0, atol=0)
    # Nothing leaked into rows we did not write.
    expected_cache = torch.zeros_like(cache)
    for (page, offset), row in zip(targets, rows):
        expected_cache[page, offset] = row
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)


def test_store_cache_rows_casts_to_cache_dtype(deepseek_mod):
    """Float32 rows written into a BF16 cache come back as BF16 values.

    Independent oracle: 0.5 and 1.0 are exactly representable in bfloat16,
    so the round trip must be bit-exact.
    """
    cache = torch.zeros(2, 2, 2, dtype=torch.bfloat16)
    slots = torch.tensor([1, 2])
    rows = torch.tensor([[0.5, 1.0], [1.0, 0.5]], dtype=torch.float32)

    deepseek_mod._store_cache_rows(cache, slots, rows, 2)

    assert cache.dtype is torch.bfloat16
    assert torch.equal(
        deepseek_mod._cache_rows(cache, slots, 2),
        rows.to(torch.bfloat16),
    )


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------


def test_swa_backend_registry_surface(deepseek_mod):
    """The SWA cache backend name and builder are wired to the FL classes."""
    assert (
        deepseek_mod.DeepseekV4FLSWABackend.get_name() == "FL_ASCEND_V4_BF16_SWA_CACHE"
    )
    assert (
        deepseek_mod.DeepseekV4FLSWABackend.get_builder_cls()
        is deepseek_mod.DeepseekV4FLMetadataBuilder
    )


def test_mla_backend_registry_surface(deepseek_mod):
    """The sparse MLA backend name, builder, capability and dtypes."""
    assert deepseek_mod.DeepseekV4FLBackend.get_name() == "FL_ASCEND_V4_BF16_SWA"
    assert (
        deepseek_mod.DeepseekV4FLBackend.get_builder_cls()
        is _DeepseekV4SparseMLAMetadataBuilder
    )
    # The capability gate accepts anything (the 910C target is fixed).
    assert (
        deepseek_mod.DeepseekV4FLBackend.supports_compute_capability(object()) is True
    )
    assert deepseek_mod.DeepseekV4FLBackend.supports_compute_capability((9, 10)) is True
    assert deepseek_mod.DeepseekV4FLBackend.supported_kv_cache_dtypes == ["auto"]


def test_tile_scheduler_disables_all_three_engines(deepseek_mod):
    """build_tile_scheduler returns all-None for any decode token count."""
    builder = object.__new__(deepseek_mod.DeepseekV4FLMetadataBuilder)
    for num_decode_tokens in (0, 1, 128, 4096):
        assert builder.build_tile_scheduler(num_decode_tokens) == {
            "swaonly": None,
            "c4a": None,
            "c128a": None,
        }


def test_get_padded_num_q_heads_is_identity(deepseek_mod):
    """The BF16 path does not pad query heads."""
    for num_heads in (1, 7, 128):
        assert (
            deepseek_mod.DeepseekV4FLAttention.get_padded_num_q_heads(num_heads)
            == num_heads
        )


def test_attention_class_attributes(deepseek_mod):
    """The FL attention pins backend classes and the BF16 layout flag."""
    assert (
        deepseek_mod.DeepseekV4FLAttention.backend_cls
        is deepseek_mod.DeepseekV4FLBackend
    )
    assert (
        deepseek_mod.DeepseekV4FLAttention.swa_backend_cls
        is deepseek_mod.DeepseekV4FLSWABackend
    )
    assert deepseek_mod.DeepseekV4FLAttention.use_fp8_ds_mla_layout is False


def test_module_all_exports(deepseek_mod):
    """__all__ exports the causal-LM wrapper and the attention class."""
    assert deepseek_mod.__all__ == [
        "DeepseekV4ForCausalLM",
        "DeepseekV4FLAttention",
    ]


# ---------------------------------------------------------------------------
# HC-head fallback
# ---------------------------------------------------------------------------


def test_hc_head_fallback_installed_at_import(deepseek_mod):
    """Import under device_type "npu" patches HCHeadOp and the XPU symbol."""
    assert _HCHeadOp._vllm_fl_ascend is True
    assert callable(_HCHeadOp.forward_native)
    assert _HCHeadOp.forward_native.__name__ == "forward_native"
    # The XPU attention symbol was swapped for the FL class.
    assert deepseek_mod._stubs.xpu_model.DeepseekV4XPUAttention is (
        deepseek_mod.DeepseekV4FLAttention
    )


def test_hc_head_fallback_is_idempotent(deepseek_mod):
    """A second _install_hc_head_fallback call leaves the patch untouched."""
    forward_native = _HCHeadOp.forward_native
    deepseek_mod._install_hc_head_fallback()
    assert _HCHeadOp.forward_native is forward_native


def test_hc_head_forward_native_contract(deepseek_mod):
    """forward_native reshapes [..., hc_mult, hidden] and fills the output.

    Independent oracle: a [2, 3, 5] input (hc_mult=3, hidden_size=5) must
    produce a [2, 5] output — the [..., hc_mult, hidden] tail collapses —
    and the kernel receives the same flattened view plus every scale arg.
    """
    head = object.__new__(_HCHeadOp)
    hidden_states = torch.randn(2, 3, 5)
    hc_calls = deepseek_mod._stubs.hc_calls
    hc_calls.clear()

    out = head.forward_native(hidden_states, "fn", 1.5, 0.5, 1e-6, 1e-6)

    assert out.shape == (2, 5)
    assert out.dtype == hidden_states.dtype
    assert bool(torch.all(out == 2.0))  # the stub kernel fills 2.0
    assert len(hc_calls) == 1
    (
        flat_shape,
        hc_fn,
        hc_scale,
        hc_base,
        out_shape,
        hidden_size,
        hc_mult,
    ) = hc_calls[0]
    assert flat_shape == (2, 3, 5)
    assert out_shape == (2, 5)
    assert hidden_size == 5
    assert hc_mult == 3
    assert (hc_fn, hc_scale, hc_base) == ("fn", 1.5, 0.5)


def test_hc_head_forward_native_higher_rank_input(deepseek_mod):
    """A 4D [..., hc_mult, hidden] input keeps its leading dimensions.

    Independent oracle: [4, 2, 3, 5] flattens to [8, 3, 5] batch rows,
    the kernel writes a [8, 5] buffer, and the returned view restores the
    [4, 2, 5] leading shape.
    """
    head = object.__new__(_HCHeadOp)
    hidden_states = torch.randn(4, 2, 3, 5)
    hc_calls = deepseek_mod._stubs.hc_calls
    hc_calls.clear()

    out = head.forward_native(hidden_states, None, 1.0, 0.0, 1e-6, 1e-6)

    assert out.shape == (4, 2, 5)
    assert len(hc_calls) == 1
    assert hc_calls[0][0] == (8, 3, 5)
    assert hc_calls[0][4] == (8, 5)


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention.__init__
# ---------------------------------------------------------------------------


def test_attention_init_swaps_cuda_event_around_super(deepseek_mod):
    """super().__init__ sees torch.npu.Event; the original is restored.

    The upstream constructor records the ``torch.cuda.Event`` class it
    observed; with the swap active that must be the fake npu Event, and
    afterwards the real cuda Event must be back in place.
    """
    real_cuda_event = torch.cuda.Event
    config = _valid_config()
    attention = deepseek_mod.DeepseekV4FLAttention(
        vllm_config=config, prefix="model.layers.0"
    )

    assert attention.event_seen is torch.npu.Event
    assert torch.cuda.Event is real_cuda_event
    assert torch.cuda.Event is not torch.npu.Event
    # The kwargs reached the upstream constructor unchanged.
    assert attention.init_kwargs == {
        "vllm_config": config,
        "prefix": "model.layers.0",
    }


def test_attention_init_resolves_positional_vllm_config(deepseek_mod):
    """A positional vllm_config (args[0]) is accepted as well."""
    config = _valid_config()
    attention = deepseek_mod.DeepseekV4FLAttention(config)
    assert attention.init_args == (config,)
    assert attention.init_kwargs == {}
    assert attention.event_seen is torch.npu.Event


def test_attention_init_requires_vllm_config(deepseek_mod):
    """Neither kwargs nor args: the constructor asserts vllm_config."""
    with pytest.raises(AssertionError):
        deepseek_mod.DeepseekV4FLAttention()


def test_attention_init_runs_validation_before_super(deepseek_mod):
    """An invalid config fails in validation, before the upstream super.

    The upstream base counts constructions; a validation failure must
    leave the counter untouched.
    """
    before = _UpstreamBase._constructions
    with pytest.raises(NotImplementedError, match="layer 0"):
        deepseek_mod.DeepseekV4FLAttention(
            vllm_config=_vllm_config([7], 1, 4096, 128, 64)
        )
    assert _UpstreamBase._constructions == before


def test_attention_init_restores_cuda_event_when_super_raises(deepseek_mod):
    """The finally block unwinds the Event swap when upstream raises.

    Validation passes first, so the swap is active when the upstream
    constructor raises; the original cuda Event must still be restored.
    """
    real_cuda_event = torch.cuda.Event
    with pytest.raises(RuntimeError, match="upstream boom"):
        deepseek_mod.DeepseekV4FLAttention(vllm_config=_valid_config(), boom=True)
    assert torch.cuda.Event is real_cuda_event


# ---------------------------------------------------------------------------
# DeepseekV4ForCausalLM
# ---------------------------------------------------------------------------


def test_causal_lm_subclasses_xpu_base(deepseek_mod):
    """The wrapper subclasses the upstream XPU causal-LM base class."""
    assert issubclass(
        deepseek_mod.DeepseekV4ForCausalLM,
        deepseek_mod._stubs.xpu_model.DeepseekV4ForCausalLM,
    )


def test_causal_lm_init_delegates_to_super(deepseek_mod):
    """A valid config reaches the upstream constructor unchanged."""
    before = _XpuModelBase._constructions
    config = _valid_config()
    model = deepseek_mod.DeepseekV4ForCausalLM(vllm_config=config, prefix="model")
    assert _XpuModelBase._constructions == before + 1
    assert model.prefix == "model"
    assert model.vllm_config is config
    assert model.constructed is True


def test_causal_lm_validates_before_super(deepseek_mod):
    """An invalid config fails in validation, before the upstream super."""
    before = _XpuModelBase._constructions
    with pytest.raises(ValueError, match="--max-model-len 128"):
        deepseek_mod.DeepseekV4ForCausalLM(
            vllm_config=_vllm_config([0], 1, 4096, 512, 64)
        )
    assert _XpuModelBase._constructions == before
