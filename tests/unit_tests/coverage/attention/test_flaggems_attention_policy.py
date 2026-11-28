# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU policy/branch tests for the FlagGems attention backend module.

``vllm_fl/dispatch/backends/flaggems/impl/attention.py`` implements the
FlagOS "CUSTOM" attention backend. Its import surface is the external
boundary: ``flaggems_vllm`` plus a dozen ``vllm.*`` modules, none of which
are installed on CPU-only test machines. Following the pattern of
``test_mm_encoder_attention_adapters.py``, these tests install stub modules
for that boundary (restored afterwards by ``monkeypatch``), execute the real
production source from disk, and drive the CPU-reachable logic:

* backend capability predicates (name, attn types, kv cache dtypes, head
  sizes, combinations, kernel block sizes);
* ``use_cascade_attention`` — every early rejection, both non-flash-decoding
  returns, and both outcomes of the flash-decoding CTA performance model
  (expectations derived by hand arithmetic, never by re-running the code);
* ``_get_sliding_window_configs`` and ``_split_kv_cache``;
* ``AttentionFLImpl.__init__`` attribute surface and rejection branches;
* ``AttentionFLMetadataBuilder.__init__``/``build`` metadata branches
  (plain, cascade, DCP, AOT sliding-window population);
* the ``forward`` dispatch branches and the encoder/cascade/DCP call
  contracts against a recording ``flaggems_vllm`` stub (the kernels
  themselves need a GPU and are covered by the functional suite).

Known source defects pinned here (also documented in the functional suite):

* ``AttentionFLImpl.__init__`` never assigns ``dcp_world_size``;
* ``forward``'s cascade branch passes an ``s_aux`` keyword that
  ``cascade_attention`` does not accept.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "vllm_fl/dispatch/backends/flaggems/impl/attention.py"
)
_MODULE_NAME = "supp_flaggems_attention_under_test"


_HEAD_SIZE = 64
_NUM_HEADS = 8
_NUM_KV_HEADS = 4
_SCALE = _HEAD_SIZE**-0.5


# ---------------------------------------------------------------------------
# External-boundary stubs
# ---------------------------------------------------------------------------


class _StubState:
    """Mutable knobs shared between the stub modules and the tests."""

    def __init__(self):
        self.current_config = None
        self.layers = {}
        self.group = SimpleNamespace(
            world_size=1,
            rank_in_group=0,
            all_gather=lambda tensor, dim=0: tensor,
        )
        self.kernel_calls = []
        self.merge_calls = []
        self.writer_calls = []


def _make_stub_classes():
    """Build the generic-subscriptable stand-ins for the vllm base classes.

    The production module annotates subclasses like
    ``AttentionMetadataBuilder[AttentionFLMetadata]`` and returns
    ``type[AttentionFLImpl]`` from methods; real vllm classes support ``[]``
    via ``__class_getitem__``, so the stubs must too.
    """
    import enum

    class AttentionType(enum.Enum):
        DECODER = "decoder"
        ENCODER = "encoder"
        ENCODER_ONLY = "encoder_only"
        ENCODER_DECODER = "encoder_decoder"

    class MultipleOf:
        def __init__(self, multiple):
            self.multiple = multiple

    class AttentionBackend:
        pass

    class AttentionImpl:
        pass

    class AttentionMetadataBuilder:
        def __init__(self, *args, **kwargs):
            pass

    for cls in (AttentionBackend, AttentionImpl, AttentionMetadataBuilder):
        cls.__class_getitem__ = classmethod(lambda cls, item: cls)

    class CacheDType(enum.Enum):
        AUTO = "auto"

    return SimpleNamespace(
        AttentionType=AttentionType,
        MultipleOf=MultipleOf,
        AttentionBackend=AttentionBackend,
        AttentionImpl=AttentionImpl,
        AttentionMetadataBuilder=AttentionMetadataBuilder,
        CacheDType=CacheDType,
    )


def _install_stubs(monkeypatch):
    """Install the fake vllm/flaggems_vllm module tree; returns shared state."""
    classes = _make_stub_classes()
    state = _StubState()

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

    def get_current_vllm_config(*args, **kwargs):
        return state.current_config

    def get_layers_from_vllm_config(vllm_config, cls):
        return dict(state.layers)

    def get_dcp_group():
        return state.group

    def get_dcp_local_seq_lens(seq_lens, world_size, rank, interleave):
        # Independent of the code under test: ceil(len / world_size).
        return [-(-int(length) // world_size) for length in seq_lens.tolist()]

    def flash_attn_varlen_func(**kwargs):
        """Recording stand-in for the flaggems varlen kernel.

        Returns a zeroed ``out`` (or a fresh zeros tensor) so the caller's
        buffer stays valid; LSE requests return ``[tokens, heads]`` zeros.
        """
        state.kernel_calls.append(kwargs)
        query = kwargs["q"]
        out = kwargs.get("out")
        if out is None:
            out = torch.zeros_like(query)
        else:
            out.zero_()
        if kwargs.get("return_softmax_lse"):
            return out, torch.zeros(query.shape[0], query.shape[1])
        return out

    def merge_attn_states(output, prefix_out, prefix_lse, suffix_out, suffix_lse):
        state.merge_calls.append(
            (
                tuple(output.shape),
                tuple(prefix_out.shape),
                tuple(prefix_lse.shape),
                tuple(suffix_out.shape),
                tuple(suffix_lse.shape),
            )
        )
        output.copy_(prefix_out.reshape(output.shape))

    def cp_lse_ag_out_rs(out, lse, group, return_lse=True):
        return out, lse

    leaf("vllm.envs", VLLM_BATCH_INVARIANT=False)
    leaf("vllm.config", VllmConfig=object)
    cfg_mod = modules["vllm.config"]
    cfg_mod.get_current_vllm_config = get_current_vllm_config
    cfg_mod.get_layers_from_vllm_config = get_layers_from_vllm_config
    leaf("vllm.config.cache", CacheDType=classes.CacheDType)
    ps_mod = leaf("vllm.distributed.parallel_state")
    ps_mod.get_dcp_group = get_dcp_group
    leaf("vllm.logger", init_logger=lambda name: SimpleNamespace())
    leaf(
        "vllm.model_executor.layers.attention.attention",
        Attention=object,
    )
    leaf("vllm.platforms.interface", DeviceCapability=SimpleNamespace)
    leaf("vllm.utils.math_utils", cdiv=lambda a, b: -(-a // b))
    leaf(
        "vllm.utils.torch_utils",
        is_quantized_kv_cache=lambda dtype: (
            isinstance(dtype, str) and dtype not in ("auto", "bf16")
        ),
    )
    leaf(
        "vllm.v1.attention.backend",
        AttentionBackend=classes.AttentionBackend,
        AttentionImpl=classes.AttentionImpl,
        AttentionMetadataBuilder=classes.AttentionMetadataBuilder,
        AttentionType=classes.AttentionType,
        MultipleOf=classes.MultipleOf,
        AttentionCGSupport=SimpleNamespace(UNIFORM_BATCH="uniform_batch"),
    )
    utils_mod = leaf("vllm.v1.attention.backends.utils")
    utils_mod.CommonAttentionMetadata = object
    utils_mod.get_dcp_local_seq_lens = get_dcp_local_seq_lens
    leaf("vllm.v1.attention.ops.dcp", cp_lse_ag_out_rs=cp_lse_ag_out_rs)
    merge_mod = leaf("vllm.v1.attention.ops.merge_attn_states")
    merge_mod.merge_attn_states = merge_attn_states
    leaf("vllm.v1.kv_cache_interface", AttentionSpec=object)

    # The triton writer is imported lazily by do_kv_cache_update.
    triton_mod = leaf(
        "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
    )

    def triton_reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, dtype, k_scale, v_scale
    ):
        state.writer_calls.append(
            (
                tuple(key.shape),
                tuple(key_cache.shape),
                tuple(value_cache.shape),
                slot_mapping.tolist(),
            )
        )

    triton_mod.triton_reshape_and_cache_flash = triton_reshape_and_cache_flash
    leaf("flaggems_vllm", flash_attn_varlen_func=flash_attn_varlen_func)

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(state=state, classes=classes)


@pytest.fixture
def attention_mod(monkeypatch):
    """Load a fresh copy of the real attention.py under the stubbed boundary."""
    stubs = _install_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    spec.loader.exec_module(module)
    module._stubs = stubs
    return module


@pytest.fixture
def impl(attention_mod):
    """A DECODER impl with the attributes forward() reads.

    ``dcp_world_size`` is set explicitly because ``__init__`` omits it (a
    known source defect also pinned by the functional GPU suite); a single
    device never takes the DCP branch, so 1 is the honest value.
    """
    made = _make_impl(attention_mod)
    return made


def _make_impl(module, attn_type=None, sliding_window=None, kv_cache_dtype="auto"):
    attention_type = module._stubs.classes.AttentionType
    impl = module.AttentionFLImpl(
        num_heads=_NUM_HEADS,
        head_size=_HEAD_SIZE,
        scale=_SCALE,
        num_kv_heads=_NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype=kv_cache_dtype,
        logits_soft_cap=None,
        attn_type=attn_type or attention_type.DECODER,
        kv_sharing_target_layer_name=None,
    )
    impl.dcp_world_size = 1
    return impl


def _make_builder(module, cudagraph_full=False, dcp_world_size=1, device_set=True):
    """Build a metadata builder over a minimal fake VllmConfig."""
    AttentionType = module._stubs.classes.AttentionType
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=False,
            get_num_attention_heads=lambda pc: _NUM_HEADS,
            get_num_kv_heads=lambda pc: _NUM_KV_HEADS,
            get_head_size=lambda: _HEAD_SIZE,
        ),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=2),
        cache_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: cudagraph_full),
            max_cudagraph_capture_size=4,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    kv_cache_spec = SimpleNamespace(dtype=torch.bfloat16, block_size=16)
    module._stubs.state.group.world_size = dcp_world_size
    module._stubs.state.group.rank_in_group = dcp_world_size - 1
    builder = module.AttentionFLMetadataBuilder(
        kv_cache_spec, ["model.layers.0"], vllm_config, torch.device("cpu")
    )
    if device_set:
        # The real base class stores the device; the stub base does not, and
        # build() needs it for cascade tensor allocation.
        builder.device = torch.device("cpu")
    # Same for vllm_config, which _get_sliding_window_configs reads when the
    # AOT scheduler is on.
    builder.vllm_config = vllm_config
    del AttentionType
    return builder


def _common_attn_metadata(
    num_reqs=2, num_tokens=6, max_query_len=3, max_seq_len=5, seq_lens=(5, 4)
):
    """CommonAttentionMetadata stand-in for build()."""
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32)[: num_reqs + 1],
        seq_lens=torch.tensor(list(seq_lens), dtype=torch.int32),
        block_table_tensor=torch.zeros(num_reqs, 2, dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
        causal=True,
    )


def _fake_layer():
    return SimpleNamespace(
        _q_scale=torch.ones(1),
        _k_scale=torch.ones(1),
        _v_scale=torch.ones(1),
    )


# ---------------------------------------------------------------------------
# Backend capability surface
# ---------------------------------------------------------------------------


def test_backend_name_and_class_accessors(attention_mod):
    """The backend reports "CUSTOM" and resolves its impl/builder classes."""
    backend = attention_mod.AttentionFLBackend
    assert backend.get_name() == "CUSTOM"
    assert backend.get_impl_cls() is attention_mod.AttentionFLImpl
    assert backend.get_builder_cls() is attention_mod.AttentionFLMetadataBuilder
    assert backend.accept_output_buffer is True
    assert backend.forward_includes_kv_cache_update is False
    assert backend.supported_dtypes == [torch.float16, torch.bfloat16]


def test_backend_supports_all_decoder_encoder_attn_types(attention_mod):
    """Every standard AttentionType is accepted; unknown strings are not."""
    AttentionType = attention_mod._stubs.classes.AttentionType
    supported = {
        AttentionType.DECODER,
        AttentionType.ENCODER,
        AttentionType.ENCODER_ONLY,
        AttentionType.ENCODER_DECODER,
    }
    for attn_type in supported:
        assert attention_mod.AttentionFLBackend.supports_attn_type(attn_type) is True
    assert attention_mod.AttentionFLBackend.supports_attn_type("bogus") is False


def test_backend_sink_and_kv_cache_dtype_and_head_size(attention_mod):
    """Sink is rejected; only "auto"/None kv dtypes and 8-aligned heads pass."""
    backend = attention_mod.AttentionFLBackend
    assert backend.supports_sink() is False
    # kv cache dtype: None and "auto" accepted, anything quantized rejected.
    assert backend.supports_kv_cache_dtype(None) is True
    assert backend.supports_kv_cache_dtype("auto") is True
    assert backend.supports_kv_cache_dtype("fp8") is False
    assert backend.supports_kv_cache_dtype("int8") is False
    # head size: multiple of 8 and at most 256 (0 % 8 == 0 passes, so only
    # non-multiples and >256 values are rejected).
    for head_size in (8, 64, 256):
        assert backend.supports_head_size(head_size) is True
    for head_size in (60, 264):
        assert backend.supports_head_size(head_size) is False


def test_backend_supports_combination_rejects_sink_and_mm_prefix(attention_mod):
    """The combination gate rejects sink and multimodal prefix, else None."""
    result = attention_mod.AttentionFLBackend.supports_combination(
        _HEAD_SIZE, torch.float16, "auto", 16, False, True, False, False, None
    )
    assert result == "not support sink"
    result = attention_mod.AttentionFLBackend.supports_combination(
        _HEAD_SIZE,
        torch.float16,
        "auto",
        16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=True,
        device_capability=None,
    )
    assert result == ("FlagGems FA2 does not support multimodal prefix attention")
    assert (
        attention_mod.AttentionFLBackend.supports_combination(
            _HEAD_SIZE,
            torch.float16,
            "auto",
            16,
            False,
            False,
            False,
            False,
            None,
        )
        is None
    )


def test_backend_kernel_block_sizes_follow_hybrid_mamba_dtypes(attention_mod):
    """Hybrid models with float32 mamba caches get fixed block sizes."""
    stubs = attention_mod._stubs
    backend = attention_mod.AttentionFLBackend

    def set_config(is_hybrid, ssm_dtype="auto", mamba_dtype="auto"):
        stubs.state.current_config = SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=is_hybrid),
            cache_config=SimpleNamespace(
                mamba_ssm_cache_dtype=ssm_dtype, mamba_cache_dtype=mamba_dtype
            ),
        )

    def as_block_sizes(result):
        """Normalize a result entry to its integer value for comparison."""
        return [getattr(entry, "multiple", entry) for entry in result]

    MultipleOf = stubs.classes.MultipleOf

    set_config(False)
    assert as_block_sizes(backend.get_supported_kernel_block_sizes()) == [16]
    assert isinstance(backend.get_supported_kernel_block_sizes()[0], MultipleOf)
    set_config(None)
    sizes = backend.get_supported_kernel_block_sizes()
    assert as_block_sizes(sizes) == [16] and isinstance(sizes[0], MultipleOf)

    set_config(True, ssm_dtype="auto", mamba_dtype="float32")
    assert backend.get_supported_kernel_block_sizes() == [16, 32, 64]
    set_config(True, ssm_dtype="float32", mamba_dtype="auto")
    assert backend.get_supported_kernel_block_sizes() == [16, 32, 64]
    set_config(True, ssm_dtype="auto", mamba_dtype="auto")
    sizes = backend.get_supported_kernel_block_sizes()
    assert as_block_sizes(sizes) == [16] and isinstance(sizes[0], MultipleOf)


# ---------------------------------------------------------------------------
# use_cascade_attention
# ---------------------------------------------------------------------------


def _cascade_decider(module):
    return module.use_cascade_attention


@pytest.mark.parametrize("common_prefix_len", [0, 128, 255])
def test_cascade_attention_rejects_short_common_prefix(
    attention_mod, common_prefix_len
):
    """A common prefix below the 256-token threshold is never worth it."""
    decider = _cascade_decider(attention_mod)
    assert (
        decider(
            common_prefix_len,
            np.ones(16, dtype=np.int64),
            32,
            8,
            False,
            False,
            False,
            148,
            1,
        )
        is False
    )


@pytest.mark.parametrize("flag", ["alibi", "sliding_window", "local_attention"])
def test_cascade_attention_rejects_variants(attention_mod, flag):
    """Alibi, sliding window and local attention disable cascade entirely."""
    assert (
        attention_mod.use_cascade_attention(
            1024,
            np.ones(16, dtype=np.int64),
            32,
            8,
            flag == "alibi",
            flag == "sliding_window",
            flag == "local_attention",
            148,
            1,
        )
        is False
    )


def test_cascade_attention_rejects_small_batch_and_dcp(attention_mod):
    """Fewer than 8 requests, or DCP rank layouts, disable cascade."""
    decider = attention_mod.use_cascade_attention
    assert (
        decider(1024, np.ones(7, dtype=np.int64), 32, 8, False, False, False, 148, 1)
        is False
    )
    assert (
        decider(1024, np.ones(16, dtype=np.int64), 32, 8, False, False, False, 148, 2)
        is False
    )


def test_cascade_attention_true_without_flash_decoding(attention_mod):
    """Without FlashDecoding (MHA or mixed prefill) cascade is chosen."""
    decider = attention_mod.use_cascade_attention
    # MHA: num_queries_per_kv == 1, so use_flash_decoding is False.
    assert (
        decider(1024, np.ones(16, dtype=np.int64), 32, 32, False, False, False, 148, 1)
        is True
    )
    # Mixed prefill: not all query lens are 1.
    mixed = np.array([3, 5, 7, 9, 11, 13, 15, 17], dtype=np.int64)
    assert decider(1024, mixed, 32, 8, False, False, False, 148, 1) is True


def test_cascade_attention_flash_decoding_performance_model(attention_mod):
    """Both performance-model outcomes with hand-derived CTA counts.

    Case "cascade wins": 16 requests, GQA 4:1 (32 q heads / 8 kv heads),
    1024-token prefix, 148 SMs, all query lens 1.
      prefix tiles        = ceil(1024 / 128) = 8
      cascade CTAs       = 32 * ceil(16 / 128) = 32, waves = ceil(32/148) = 1
      cascade time       = 1 * 8 = 8
      FlashDecoding CTAs = 16 * 8 * ceil(4/128) = 128, * 8 tiles = 1024
      FlashDecoding time = ceil(1024 / 148) = 7
      8 < 7 is False -> the heuristic keeps FlashDecoding.

    Case "flash decoding wins": 8 requests, GQA 32:1, 256-token prefix,
    1 SM, all query lens 1.
      prefix tiles        = ceil(256 / 128) = 2
      cascade CTAs       = 32 * ceil(8 / 128) = 32, waves = 32
      cascade time       = 32 * 2 = 64
      FlashDecoding CTAs = 8 * 1 * ceil(32/128) = 8, * 2 = 16
      FlashDecoding time = 16
      64 < 16 is False -> FlashDecoding.

    Case "cascade wins under flash decoding": 8 requests, GQA 2:1, 256-token
    prefix, 1 SM.
      cascade time       = 32 * ceil(8/128)=32 waves * 2 tiles = 64
      FlashDecoding      = 8 * 16 * 1 * 2 = 256 CTAs, time = 256
      64 < 256 is True -> cascade.
    """
    decider = attention_mod.use_cascade_attention
    ones16 = np.ones(16, dtype=np.int64)
    ones8 = np.ones(8, dtype=np.int64)

    # Derived above: cascade_time 8 vs FlashDecoding 7.
    assert decider(1024, ones16, 32, 8, False, False, False, 148, 1) is False
    # Derived above: cascade_time 64 vs FlashDecoding 16.
    assert decider(256, ones8, 32, 1, False, False, False, 1, 1) is False
    # Derived above: cascade_time 64 vs FlashDecoding 256.
    assert decider(256, ones8, 32, 16, False, False, False, 1, 1) is True


def test_cascade_attention_accepts_python_list_query_lens(attention_mod):
    """len(query_lens) works for plain lists too (duck-typed input)."""
    assert (
        attention_mod.use_cascade_attention(
            1024, [1] * 16, 32, 32, False, False, False, 148, 1
        )
        is True
    )


# ---------------------------------------------------------------------------
# _get_sliding_window_configs / _split_kv_cache
# ---------------------------------------------------------------------------


def test_get_sliding_window_configs_collects_impl_windows(attention_mod):
    """The layer walker collects one entry per distinct sliding window."""
    module = attention_mod

    def layer_with(window):
        impl = _make_impl(module, sliding_window=window)
        return SimpleNamespace(impl=impl)

    # sliding_window=None -> (-1, -1); sliding_window=4 -> (3, 0) for DECODER.
    module._stubs.state.layers = {
        "a": layer_with(None),
        "b": layer_with(4),
    }
    assert module._get_sliding_window_configs(object()) == {(-1, -1), (3, 0)}

    module._stubs.state.layers = {"a": layer_with(6)}
    assert module._get_sliding_window_configs(object()) == {(5, 0)}

    module._stubs.state.layers = {"a": layer_with(6), "b": layer_with(8)}
    assert module._get_sliding_window_configs(object()) == {(5, 0), (7, 0)}

    module._stubs.state.layers = {}
    assert module._get_sliding_window_configs(object()) == set()


def test_split_kv_cache_returns_strided_k_and_v_views(attention_mod):
    """The 4D [B, H, N, K+V] cache splits into two [B, N, H, D] views."""
    num_blocks, heads, block_size = 2, 3, 4
    head_size = 8  # 2 * head_size == 16 (the packed last dim below)
    cache = torch.arange(
        num_blocks * heads * block_size * 2 * head_size, dtype=torch.float32
    ).reshape(num_blocks, heads, block_size, 2 * head_size)
    key_cache, value_cache = attention_mod._split_kv_cache(cache, head_size)

    assert key_cache.shape == (num_blocks, block_size, heads, head_size)
    assert value_cache.shape == key_cache.shape
    # Views alias the cache storage (no copy): writing through them is
    # visible in the packed cache.
    probe = torch.full_like(cache, -1.0)
    key_cache.copy_(probe.transpose(1, 2)[..., :head_size])
    assert torch.equal(cache[..., :head_size].transpose(1, 2), key_cache)
    cache.copy_(
        torch.arange(
            num_blocks * heads * block_size * 2 * head_size, dtype=torch.float32
        ).reshape(cache.shape)
    )
    # K is the first half, V the second half of the packed last dim.
    expected = cache.transpose(1, 2)
    torch.testing.assert_close(key_cache, expected[..., :head_size])
    torch.testing.assert_close(value_cache, expected[..., head_size:])

    with pytest.raises(ValueError, match="equal K/V head sizes"):
        attention_mod._split_kv_cache(cache, head_size + 1)
    with pytest.raises(ValueError, match="Unsupported FlagGems KV cache shape"):
        attention_mod._split_kv_cache(torch.zeros(2, 3, 4), head_size)


# ---------------------------------------------------------------------------
# AttentionFLImpl.__init__ / do_kv_cache_update
# ---------------------------------------------------------------------------


def test_impl_init_attribute_surface_and_sliding_window_encoding(attention_mod):
    """__init__ encodes sliding windows, softcap defaults and GQA ratio."""
    AttentionType = attention_mod._stubs.classes.AttentionType

    impl = attention_mod.AttentionFLImpl(
        num_heads=_NUM_HEADS,
        head_size=_HEAD_SIZE,
        scale=_SCALE,
        num_kv_heads=_NUM_KV_HEADS,
        alibi_slopes=[0.25, 0.5],
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    assert impl.num_heads == _NUM_HEADS
    assert impl.head_size == _HEAD_SIZE
    assert impl.scale == _SCALE
    assert impl.num_kv_heads == _NUM_KV_HEADS
    assert impl.sliding_window == (-1, -1)
    assert impl.alibi_slopes.tolist() == [0.25, 0.5]
    assert impl.alibi_slopes.dtype == torch.float32
    assert impl.logits_soft_cap == 0
    assert impl.num_queries_per_kv == _NUM_HEADS // _NUM_KV_HEADS
    assert impl.vllm_flash_attn_version == 2
    assert impl.supports_quant_query_input is False
    assert impl.can_return_lse_for_decode is True

    encoder_only = _make_impl(
        attention_mod,
        attn_type=AttentionType.ENCODER_ONLY,
        sliding_window=1024,
    )
    assert encoder_only.sliding_window == (1023, 1023)
    decoder = _make_impl(attention_mod, sliding_window=512)
    assert decoder.sliding_window == (511, 0)

    softcap = attention_mod.AttentionFLImpl(
        num_heads=_NUM_HEADS,
        head_size=_HEAD_SIZE,
        scale=_SCALE,
        num_kv_heads=_NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=30.0,
    )
    assert softcap.logits_soft_cap == 30.0


def test_impl_init_missing_dcp_world_size_is_pinned(attention_mod):
    """Pin the source defect: __init__ never assigns dcp_world_size."""
    impl = attention_mod.AttentionFLImpl(
        num_heads=_NUM_HEADS,
        head_size=_HEAD_SIZE,
        scale=_SCALE,
        num_kv_heads=_NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    assert not hasattr(impl, "dcp_world_size")


def test_impl_init_rejects_quantized_kv_cache(attention_mod):
    """A quantized kv-cache dtype is rejected at construction time."""
    with pytest.raises(NotImplementedError, match="quantization kv-cache"):
        _make_impl(attention_mod, kv_cache_dtype="fp8")


def test_impl_do_kv_cache_update_skips_encoder_types(attention_mod):
    """Encoder / ENCODER_ONLY attention never writes the paged cache."""
    AttentionType = attention_mod._stubs.classes.AttentionType
    for attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
        encoder_impl = _make_impl(attention_mod, attn_type=attn_type)
        layer = _fake_layer()
        cache = torch.zeros(2, _NUM_KV_HEADS, 16, 2 * _HEAD_SIZE)
        before = attention_mod._stubs.state.writer_calls.copy()

        encoder_impl.do_kv_cache_update(
            layer,
            torch.randn(4, _NUM_KV_HEADS, _HEAD_SIZE),
            torch.randn(4, _NUM_KV_HEADS, _HEAD_SIZE),
            cache,
            torch.arange(4),
        )
        assert attention_mod._stubs.state.writer_calls == before


def test_impl_do_kv_cache_update_writes_split_pages(attention_mod):
    """The decoder path hands the split K/V views to the Triton writer."""
    decoder_impl = _make_impl(attention_mod)
    layer = _fake_layer()
    num_blocks, block_size = 2, 16
    cache = torch.zeros(num_blocks, _NUM_KV_HEADS, block_size, 2 * _HEAD_SIZE)
    key = torch.randn(4, _NUM_KV_HEADS, _HEAD_SIZE)
    value = torch.randn(4, _NUM_KV_HEADS, _HEAD_SIZE)
    slot_mapping = torch.arange(4)

    decoder_impl.do_kv_cache_update(layer, key, value, cache, slot_mapping)

    state = attention_mod._stubs.state
    assert len(state.writer_calls) == 1
    key_shape, key_cache_shape, value_cache_shape, slots = state.writer_calls[0]
    assert key_shape == (4, _NUM_KV_HEADS, _HEAD_SIZE)
    # The writer receives the split [B, N, H, D] views, not the packed cache.
    assert key_cache_shape == (num_blocks, block_size, _NUM_KV_HEADS, _HEAD_SIZE)
    assert value_cache_shape == key_cache_shape
    assert slots == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# MetadataBuilder __init__ / build
# ---------------------------------------------------------------------------


def test_builder_init_attribute_surface(attention_mod):
    """The builder snapshots model/parallel/cache/compilation config."""
    builder = _make_builder(attention_mod)
    assert builder.num_heads_q == _NUM_HEADS
    assert builder.num_heads_kv == _NUM_KV_HEADS
    assert builder.headdim == _HEAD_SIZE
    assert builder.block_size == 16
    assert builder.kv_cache_dtype == torch.bfloat16
    assert builder.max_num_splits == 0
    assert builder.aot_schedule is False
    assert builder.dcp_world_size == 1
    assert builder.dcp_rank == 0
    assert builder.cp_kv_cache_interleave_size == 2
    assert builder.use_full_cuda_graph is False
    assert builder.max_cudagraph_size == 4
    assert builder.aot_sliding_window is None
    assert builder._cudagraph_support == "uniform_batch"


def test_builder_init_dcp_group_from_parallel_state(attention_mod):
    """DCP world size and rank come from the (stubbed) dcp group."""
    builder = _make_builder(attention_mod, dcp_world_size=2)
    assert builder.dcp_world_size == 2
    assert builder.dcp_rank == 1
    # Restore the single-GPU group for the other tests in this module.
    attention_mod._stubs.state.group.world_size = 1
    attention_mod._stubs.state.group.rank_in_group = 0


def test_builder_build_plain_metadata(attention_mod):
    """A zero common prefix yields a non-cascade metadata with no extras."""
    builder = _make_builder(attention_mod)
    cam = _common_attn_metadata()

    metadata = builder.build(0, cam)

    assert metadata.num_actual_tokens == cam.num_actual_tokens
    assert metadata.max_query_len == cam.max_query_len
    assert metadata.max_seq_len == cam.max_seq_len
    assert metadata.query_start_loc is cam.query_start_loc
    assert metadata.seq_lens is cam.seq_lens
    assert metadata.block_table is cam.block_table_tensor
    assert metadata.slot_mapping is cam.slot_mapping
    assert metadata.causal is True
    assert metadata.use_cascade is False
    assert metadata.common_prefix_len == 0
    assert metadata.max_dcp_context_kv_len == 0
    assert metadata.dcp_context_kv_lens is None
    assert metadata.cu_prefix_query_lens is None
    assert metadata.prefix_kv_lens is None
    assert metadata.suffix_kv_lens is None
    assert metadata.scheduler_metadata is None
    assert metadata.prefix_scheduler_metadata is None
    assert metadata.max_num_splits == 0


def test_builder_build_cascade_metadata(attention_mod):
    """A positive common prefix yields the cascade prefix/suffix tensors."""
    builder = _make_builder(attention_mod)
    cam = _common_attn_metadata()
    common_prefix_len = 64

    metadata = builder.build(common_prefix_len, cam)

    assert metadata.use_cascade is True
    assert metadata.common_prefix_len == common_prefix_len
    # cu_prefix_query_lens covers all tokens as one packed query group.
    assert metadata.cu_prefix_query_lens.tolist() == [0, cam.num_actual_tokens]
    assert metadata.prefix_kv_lens.tolist() == [common_prefix_len]
    assert metadata.prefix_kv_lens.dtype == torch.int32
    # suffix = seq_lens[:num_reqs] - common_prefix_len, computed independently.
    expected_suffix = [
        int(seq_len) - common_prefix_len for seq_len in cam.seq_lens[: cam.num_reqs]
    ]
    assert metadata.suffix_kv_lens.tolist() == expected_suffix
    assert metadata.max_num_splits == 0


def test_builder_build_dcp_metadata(attention_mod):
    """A DCP world > 1 distributes context lens and bounds the workspace.

    All numbers are derived independently of the code under test. With
    query_start_loc=[0, 3, 6] the per-request query lens are [3, 3]; the
    context lens are seq_lens - query_lens = [5-3, 4-3] = [2, 1]. The
    stubbed ``get_dcp_local_seq_lens`` oracle distributes them as
    ceil(len / world_size) = [1, 1]. The workspace bound:
    num_partitions = world_size * interleave = 2 * 2 = 4, and
    ceil(max_seq_len / num_partitions) * interleave = ceil(5/4) * 2 = 4.
    """
    builder = _make_builder(attention_mod, dcp_world_size=2)
    cam = _common_attn_metadata(max_seq_len=5, seq_lens=(5, 4))

    metadata = builder.build(0, cam)

    assert metadata.use_cascade is False
    assert metadata.max_dcp_context_kv_len == 4
    query_kv_lens = [
        int(end) - int(start)
        for start, end in zip(cam.query_start_loc[:-1], cam.query_start_loc[1:])
    ]
    context_kv_lens = [
        int(seq_len) - query_kv_len
        for seq_len, query_kv_len in zip(cam.seq_lens, query_kv_lens)
    ]
    assert query_kv_lens == [3, 3]
    assert context_kv_lens == [2, 1]
    assert list(metadata.dcp_context_kv_lens) == [
        -(-context_kv_len // 2) for context_kv_len in context_kv_lens
    ]
    assert list(metadata.dcp_context_kv_lens) == [1, 1]
    assert metadata.cu_prefix_query_lens is None
    # Restore the single-GPU group for the other tests in this module.
    attention_mod._stubs.state.group.world_size = 1
    attention_mod._stubs.state.group.rank_in_group = 0


def test_builder_build_populates_aot_sliding_window(attention_mod):
    """A single uniform sliding-window layer pins the AOT window tuple.

    The builder starts with ``aot_schedule`` False (FA2 only), so the
    ``build`` call takes the ``self.aot_sliding_window is None`` branch and
    seeds it with the (-1, -1) sentinel without consulting the layers;
    a second build keeps the value (populate-once contract).
    """
    builder = _make_builder(attention_mod)
    assert builder.aot_sliding_window is None

    first = builder.build(0, _common_attn_metadata())
    assert builder.aot_sliding_window == (-1, -1)
    assert first.max_num_splits == 0

    builder.build(64, _common_attn_metadata())
    assert builder.aot_sliding_window == (-1, -1)


def test_builder_use_cascade_attention_delegates_to_module_function(attention_mod):
    """The builder method forwards to the module-level heuristic."""
    builder = _make_builder(attention_mod)
    assert (
        builder.use_cascade_attention(
            1024, np.ones(16, dtype=np.int64), 32, 32, False, False, False, 148, 1
        )
        is True
    )


def test_builder_init_falls_back_when_dcp_group_not_initialized(
    attention_mod, monkeypatch
):
    """A missing DCP group degrades to single-device world size and rank 0.

    ``__init__`` imports ``get_dcp_group`` from the (stubbed) parallel_state
    module inside its try block; making that stub raise AssertionError —
    what real vllm raises before DCP init — must land in the documented
    single-GPU fallback instead of failing construction.
    """
    import vllm.distributed.parallel_state  # the stub installed by _install_stubs

    def uninitialized_group():
        raise AssertionError("DCP not initialized")

    monkeypatch.setattr(
        vllm.distributed.parallel_state, "get_dcp_group", uninitialized_group
    )
    builder = _make_builder(attention_mod)
    assert builder.dcp_world_size == 1
    assert builder.dcp_rank == 0


def test_builder_build_cudagraph_num_splits(attention_mod):
    """Full cudagraph builds lift max_num_splits from the AOT workspace.

    With ``use_full_cuda_graph`` forced on and a token count at or below
    the capture size (4), ``build`` must copy ``self.max_num_splits`` (the
    AOT pre-allocated split bound) into the returned metadata; above the
    capture size — or without cudagraphs — it stays 0 (FA2 heuristics).
    """
    builder = _make_builder(attention_mod, cudagraph_full=True)
    assert builder.use_full_cuda_graph is True
    assert builder.max_num_splits == 0

    small_cam = _common_attn_metadata(num_tokens=2, max_query_len=1)
    assert small_cam.num_actual_tokens <= builder.max_cudagraph_size
    small = builder.build(0, small_cam)
    assert small.max_num_splits == builder.max_num_splits

    big_cam = _common_attn_metadata(num_tokens=16)
    assert big_cam.num_actual_tokens > builder.max_cudagraph_size
    big = builder.build(0, big_cam)
    assert big.max_num_splits == 0

    plain = _make_builder(attention_mod)
    assert plain.build(0, _common_attn_metadata()).max_num_splits == 0


def test_builder_build_populates_aot_window_from_uniform_layers(attention_mod):
    """AOT schedules pin the sliding window from a single layer config.

    ``aot_schedule`` defaults to False (FA2 only); flipping it on with a
    single uniform sliding-window layer must make the first ``build`` adopt
    that layer's window instead of the (-1, -1) sentinel, and a second
    build keep the pinned value (populate-once).
    """
    builder = _make_builder(attention_mod)
    builder.aot_schedule = True

    made = _make_impl(attention_mod, sliding_window=1024)
    attention_mod._stubs.state.layers["model.layers.0"] = SimpleNamespace(impl=made)

    assert builder.aot_sliding_window is None
    first = builder.build(0, _common_attn_metadata())
    assert builder.aot_sliding_window == (1023, 0)
    assert first.max_num_splits == 0

    builder.build(64, _common_attn_metadata())
    assert builder.aot_sliding_window == (1023, 0)


def test_builder_build_disables_aot_when_layers_have_mixed_windows(attention_mod):
    """Conflicting sliding windows force the AOT scheduler off."""
    builder = _make_builder(attention_mod)
    builder.aot_schedule = True
    layers = attention_mod._stubs.state.layers
    layers["model.layers.0"] = SimpleNamespace(
        impl=_make_impl(attention_mod, sliding_window=1024)
    )
    layers["model.layers.1"] = SimpleNamespace(
        impl=_make_impl(attention_mod, sliding_window=512)
    )

    builder.build(0, _common_attn_metadata())

    assert builder.aot_schedule is False
    assert builder.aot_sliding_window == (-1, -1)


# ---------------------------------------------------------------------------
# forward() dispatch
# ---------------------------------------------------------------------------


def _forward_inputs(module, num_tokens=6):
    query = torch.randn(num_tokens, _NUM_HEADS, _HEAD_SIZE)
    key = torch.randn(num_tokens, _NUM_KV_HEADS, _HEAD_SIZE)
    value = torch.randn(num_tokens, _NUM_KV_HEADS, _HEAD_SIZE)
    output = torch.zeros(num_tokens, _NUM_HEADS * _HEAD_SIZE)
    cache = torch.zeros(2, _NUM_KV_HEADS, 16, 2 * _HEAD_SIZE)
    return query, key, value, output, cache


def test_forward_plain_decoder_calls_varlen_kernel(attention_mod, impl):
    """The non-cascade decoder path calls the kernel with paged inputs."""
    builder = _make_builder(attention_mod)
    metadata = builder.build(0, _common_attn_metadata())
    query, key, value, output, cache = _forward_inputs(attention_mod)
    layer = _fake_layer()
    state = attention_mod._stubs.state
    state.kernel_calls.clear()

    returned = impl.forward(layer, query, key, value, cache, metadata, output=output)

    assert returned is output
    assert len(state.kernel_calls) == 1
    kwargs = state.kernel_calls[0]
    # q/out are the [:num_actual_tokens] slices of the caller tensors.
    assert kwargs["q"].shape == query.shape
    assert torch.equal(kwargs["q"], query)
    assert kwargs["k"] is not key  # the paged cache views, not the raw keys
    assert kwargs["out"].shape == output.shape
    assert torch.equal(kwargs["out"], output)
    assert kwargs["cu_seqlens_q"] is metadata.query_start_loc
    assert kwargs["seqused_k"] is metadata.seq_lens
    assert kwargs["max_seqlen_q"] == metadata.max_query_len
    assert kwargs["max_seqlen_k"] == metadata.max_seq_len
    assert kwargs["block_table"] is metadata.block_table
    assert kwargs["softmax_scale"] == impl.scale
    assert kwargs["causal"] is True
    assert kwargs["window_size"] == (-1, -1)
    assert kwargs["softcap"] == 0
    assert kwargs["fa_version"] == 2
    assert kwargs["num_splits"] == 0
    assert kwargs["s_aux"] is None
    # Descale tensors are expanded to (num_seqs, num_kv_heads); num_seqs is
    # cu_seqlens_q.shape[0] - 1 = 2 here.
    assert kwargs["q_descale"].shape == (2, _NUM_KV_HEADS)
    assert kwargs["k_descale"].shape == (2, _NUM_KV_HEADS)
    assert torch.all(kwargs["q_descale"] == 1.0)


def test_forward_profiling_run_fills_zero(attention_mod, impl):
    """attn_metadata=None is the profiling run: zero-fill, no kernel."""
    query, key, value, output, cache = _forward_inputs(attention_mod)
    state = attention_mod._stubs.state
    state.kernel_calls.clear()

    returned = impl.forward(
        _fake_layer(), query, key, value, cache, None, output=output
    )

    assert returned is output
    assert torch.all(output == 0)
    assert state.kernel_calls == []


@pytest.mark.parametrize("scale_arg", ["output_scale", "output_block_scale"])
def test_forward_rejects_fused_output_quantization(attention_mod, impl, scale_arg):
    """Fused output quantization scales are not supported."""
    builder = _make_builder(attention_mod)
    metadata = builder.build(0, _common_attn_metadata())
    query, key, value, output, cache = _forward_inputs(attention_mod)

    with pytest.raises(NotImplementedError, match="fused output quantization"):
        impl.forward(
            _fake_layer(),
            query,
            key,
            value,
            cache,
            metadata,
            output=output,
            **{scale_arg: torch.ones(1)},
        )


def test_forward_requires_output_tensor(attention_mod, impl):
    """forward asserts the caller-provided output buffer."""
    builder = _make_builder(attention_mod)
    metadata = builder.build(0, _common_attn_metadata())
    query, key, value, _output, cache = _forward_inputs(attention_mod)

    with pytest.raises(AssertionError, match="Output tensor must be provided"):
        impl.forward(_fake_layer(), query, key, value, cache, metadata)


def test_forward_encoder_attention_skips_kv_cache(attention_mod):
    """Encoder attention feeds raw Q/K/V slices to the varlen kernel."""
    AttentionType = attention_mod._stubs.classes.AttentionType
    encoder_impl = _make_impl(attention_mod, attn_type=AttentionType.ENCODER_ONLY)
    builder = _make_builder(attention_mod)
    metadata = builder.build(0, _common_attn_metadata())
    query, key, value, output, cache = _forward_inputs(attention_mod)
    state = attention_mod._stubs.state
    state.kernel_calls.clear()

    returned = encoder_impl.forward(
        _fake_layer(), query, key, value, cache, metadata, output=output
    )

    assert torch.equal(returned, output) and returned.shape == output.shape
    assert len(state.kernel_calls) == 1
    kwargs = state.kernel_calls[0]
    assert kwargs["q"].shape == query.shape
    assert torch.equal(kwargs["q"], query)
    assert torch.equal(kwargs["k"], key)
    assert torch.equal(kwargs["v"], value)
    assert kwargs["cu_seqlens_q"] is metadata.query_start_loc
    assert kwargs["cu_seqlens_k"] is metadata.query_start_loc
    assert kwargs["causal"] is False  # bidirectional encoder attention
    assert kwargs["max_seqlen_q"] == metadata.max_query_len
    assert kwargs["max_seqlen_k"] == metadata.max_query_len


def test_forward_encoder_attention_rejects_fp8_kv_cache(attention_mod):
    """The encoder path rejects fp8 kv-cache dtypes before the kernel."""
    AttentionType = attention_mod._stubs.classes.AttentionType
    # Bypass __init__'s generic quant rejection by patching the attribute.
    encoder_impl = _make_impl(attention_mod, attn_type=AttentionType.ENCODER_ONLY)
    encoder_impl.kv_cache_dtype = "fp8"
    builder = _make_builder(attention_mod)
    metadata = builder.build(0, _common_attn_metadata())
    query, key, value, output, cache = _forward_inputs(attention_mod)
    state = attention_mod._stubs.state
    state.kernel_calls.clear()

    with pytest.raises(NotImplementedError, match="encoder attention"):
        encoder_impl.forward(
            _fake_layer(), query, key, value, cache, metadata, output=output
        )
    assert state.kernel_calls == []


def test_forward_cascade_branch_s_aux_defect_is_pinned(attention_mod):
    """Pin the source defect: forward's cascade branch passes ``s_aux``.

    ``cascade_attention`` has no ``s_aux`` parameter, so the cascade branch
    of ``forward`` raises TypeError whenever vLLM actually selects cascade
    attention. The direct ``cascade_attention`` contract is covered by
    ``test_cascade_attention_prefix_suffix_contract`` below; this test keeps
    the wiring defect visible (mirroring the GPU functional suite).
    """
    impl = _make_impl(attention_mod)
    builder = _make_builder(attention_mod)
    metadata = builder.build(64, _common_attn_metadata())
    query, key, value, output, cache = _forward_inputs(attention_mod)

    with pytest.raises(TypeError, match="s_aux"):
        impl.forward(_fake_layer(), query, key, value, cache, metadata, output=output)


def test_cascade_attention_prefix_suffix_contract(attention_mod):
    """cascade_attention issues prefix + suffix kernels and merges them.

    With a 64-token common prefix over a 16-token block cache:
    num_common_kv_blocks = 64 // 16 = 4, so the prefix kernel reads the
    first block-table row only and the suffix kernel reads the remaining
    columns (num_blocks - 4). The prefix is bidirectional over the shared
    prefix; the suffix is causal over each request's tail.
    """
    impl = _make_impl(attention_mod)
    builder = _make_builder(attention_mod)
    metadata = builder.build(64, _common_attn_metadata())
    query, key, value, output, cache = _forward_inputs(attention_mod)
    key_cache, value_cache = attention_mod._split_kv_cache(cache, _HEAD_SIZE)
    state = attention_mod._stubs.state
    state.kernel_calls.clear()
    state.merge_calls.clear()

    attention_mod.cascade_attention(
        output,
        query,
        key_cache,
        value_cache,
        cu_query_lens=metadata.query_start_loc,
        max_query_len=metadata.max_query_len,
        cu_prefix_query_lens=metadata.cu_prefix_query_lens,
        prefix_kv_lens=metadata.prefix_kv_lens,
        suffix_kv_lens=metadata.suffix_kv_lens,
        max_kv_len=metadata.max_seq_len,
        softmax_scale=impl.scale,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=impl.logits_soft_cap,
        block_table=metadata.block_table,
        common_prefix_len=64,
        fa_version=2,
    )

    assert len(state.kernel_calls) == 2
    prefix, suffix = state.kernel_calls
    assert prefix["causal"] is False
    assert suffix["causal"] is True
    assert prefix["seqused_k"] is metadata.prefix_kv_lens
    assert suffix["seqused_k"] is metadata.suffix_kv_lens
    assert prefix["block_table"].shape[0] == 1
    # 64 prefix tokens over 16-token blocks drop the first 4 columns.
    assert suffix["block_table"].shape == (
        metadata.block_table.shape[0],
        max(metadata.block_table.shape[1] - 4, 0),
    )
    assert prefix["return_softmax_lse"] is True
    assert suffix["return_softmax_lse"] is True
    assert len(state.merge_calls) == 1
    out_shape, p_shape, pl_shape, s_shape, sl_shape = state.merge_calls[0]
    assert out_shape == tuple(output.shape)
    assert p_shape == s_shape
    assert pl_shape == sl_shape


@pytest.mark.parametrize(
    ("kwargs_name", "kwargs_value"),
    [
        ("alibi_slopes", torch.ones(1)),
        ("sliding_window", (5, 0)),
    ],
)
def test_cascade_attention_rejects_alibi_and_sliding_window(
    attention_mod, kwargs_name, kwargs_value
):
    """cascade_attention asserts against ALiBi and sliding windows."""
    impl = _make_impl(attention_mod)
    builder = _make_builder(attention_mod)
    metadata = builder.build(64, _common_attn_metadata())
    query, key, value, output, cache = _forward_inputs(attention_mod)
    key_cache, value_cache = attention_mod._split_kv_cache(cache, _HEAD_SIZE)

    base = dict(
        cu_query_lens=metadata.query_start_loc,
        max_query_len=metadata.max_query_len,
        cu_prefix_query_lens=metadata.cu_prefix_query_lens,
        prefix_kv_lens=metadata.prefix_kv_lens,
        suffix_kv_lens=metadata.suffix_kv_lens,
        max_kv_len=metadata.max_seq_len,
        softmax_scale=impl.scale,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=impl.logits_soft_cap,
        block_table=metadata.block_table,
        common_prefix_len=64,
        fa_version=2,
    )
    base[kwargs_name] = kwargs_value

    with pytest.raises(AssertionError):
        attention_mod.cascade_attention(output, query, key_cache, value_cache, **base)


def test_forward_with_dcp_contract(attention_mod):
    """The DCP path gathers context attention then merges with local tails.

    With world_size=2 the context kernel attends bidirectionally over the
    distributed context lens (returning softmax LSE) and the local kernel is
    causal over the query range with ``cu_seqlens_k == cu_seqlens_q``; the
    two results are merged exactly once.
    """
    impl = _make_impl(attention_mod)
    impl.dcp_world_size = 2
    builder = _make_builder(attention_mod, dcp_world_size=2)
    metadata = builder.build(0, _common_attn_metadata())
    query, key, value, output, cache = _forward_inputs(attention_mod)
    state = attention_mod._stubs.state
    state.kernel_calls.clear()
    state.merge_calls.clear()

    returned = impl.forward(
        _fake_layer(), query, key, value, cache, metadata, output=output
    )

    assert returned is output
    assert len(state.kernel_calls) == 2
    context, local = state.kernel_calls
    assert context["causal"] is False
    assert context["return_softmax_lse"] is True
    assert context["seqused_k"] is metadata.dcp_context_kv_lens
    assert context["max_seqlen_k"] == metadata.max_dcp_context_kv_len
    assert local["causal"] is True
    assert local["return_softmax_lse"] is True
    assert local["cu_seqlens_k"] is metadata.query_start_loc
    assert local["cu_seqlens_q"] is metadata.query_start_loc
    assert len(state.merge_calls) == 1
    # Restore the single-GPU group for the other tests in this module.
    state.group.world_size = 1
    state.group.rank_in_group = 0
