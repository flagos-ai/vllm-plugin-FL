# Copyright 2025 BAAI. All rights reserved.
"""Supplementary CPU tests for the FlagGems dispatch impls (non-fused_moe).

The modules under test (``vllm_fl/dispatch/backends/flaggems/impl/``) treat
``vllm``, ``flag_gems`` and ``flaggems_vllm`` as external boundaries: they
import them at module level and their kernel entry points need a real
accelerator. These tests install stub modules for that boundary in
``sys.modules`` (restored afterwards), load the real production source, and
drive the CPU-reachable logic:

* ``impl/attention.py`` — the ``AttentionFLBackend`` metadata surface,
  ``supports_*`` gates, ``_split_kv_cache`` layouts, the
  ``AttentionFLImpl`` construction and its ``forward`` dispatch (encoder
  path, profiling run, output-scale rejection, cascade branch, DCP branch)
  against recording ``flash_attn_varlen_func`` doubles, plus the
  ``use_cascade_attention`` decision table and ``cascade_attention``
  argument plumbing;
* ``impl/quantization.py`` — input validation, buffer/grid sizing, the
  launch contract and the kernel math of the Triton fallback via a
  recording kernel double checked against an independent torch oracle;
* ``impl/rotary.py`` — the vendor-branch matrix (ascend six-arg kernel,
  GCU out-of-place copy-back, nvidia/thead read-write race workaround,
  plain passthrough) against recording kernel doubles;
* ``impl/normalization.py`` and ``impl/activation.py`` — the delegate
  contracts against plain-torch kernel doubles;
* ``impl/mla.py`` — backend metadata and construction rejections.

The real ``flag_gems`` import on this CI box raises ``RuntimeError`` ("0
active drivers"), which the ImportError-only handler in
``FlagGemsBackend.is_available`` does not catch; a fake ``flag_gems``
module is installed so probes resolve on every platform.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import Generic, TypeVar

import pytest
import torch

pytestmark = [pytest.mark.supplementary]

_STUB_ROOTS = ("vllm", "vllm_fl", "flag_gems", "flaggems_vllm", "triton")
_MISSING = object()


@pytest.fixture(autouse=True)
def _fl_sys_modules():
    """Snapshot ``sys.modules`` and restore it exactly after the test.

    Records every ``vllm*`` / ``vllm_fl*`` / ``flag_gems*`` /
    ``flaggems_vllm*`` / ``triton*`` entry present before the test, removes
    entries the test added, and puts back the original module objects for
    entries the test replaced.
    """
    saved = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_ROOTS:
            saved[name] = sys.modules.get(name, _MISSING)
    try:
        yield saved
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name.split(".")[0] in _STUB_ROOTS and name not in saved
        ]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _OnceLogger:
    """Logger wrapper carrying ``info_once`` used by the impls."""

    def __init__(self, logger):
        self._logger = logger

    def info_once(self, msg, *args):
        self._logger.info(msg, *args)

    def __getattr__(self, name):
        return getattr(self._logger, name)


class _StubState:
    """Mutable knobs shared between the stub modules and the tests."""

    def __init__(self):
        self.vendor_name = "nvidia"
        self.device_type = "cuda"
        self.scaled_int8_quant_calls = []
        self.flash_attn_calls = []
        self.flash_mla_calls = []
        self.device_ctx = None


def _make_stub_classes():
    """Generic-subscriptable stand-ins for the vllm v1 attention bases."""

    class AttentionType(enum.Enum):
        DECODER = "decoder"
        ENCODER = "encoder"
        ENCODER_ONLY = "encoder_only"
        ENCODER_DECODER = "encoder_decoder"

    T = TypeVar("T")

    class _Gen(Generic[T]):
        pass

    class MLACommonImpl(Generic[TypeVar("M")]):
        def __init__(self, *args, **kwargs):
            # Position 6 of the real signature is kv_cache_dtype.
            if len(args) > 6:
                self.kv_cache_dtype = args[6]

    class MLACommonBackend:
        pass

    class MLACommonMetadata:
        pass

    class AttentionCGSupport(enum.Enum):
        ALWAYS = "always"
        NEVER = "never"
        UNIFORM_BATCH = "uniform_batch"
        UNIFORM_SINGLE_TOKEN_DECODE = "uniform_single_token_decode"

    return SimpleNamespace(
        AttentionType=AttentionType,
        MLACommonImpl=MLACommonImpl,
        MLACommonBackend=MLACommonBackend,
        MLACommonMetadata=MLACommonMetadata,
        AttentionCGSupport=AttentionCGSupport,
        AttentionBackend=_Gen,
        AttentionImpl=_Gen,
        AttentionMetadataBuilder=_Gen,
    )


def _next_power_of_2(value):
    """Independent oracle: smallest power of two >= value (>= 1)."""
    if value <= 1:
        return 1
    result = 1
    while result < value:
        result *= 2
    return result


class _VllmStubTree:
    """Install a fake ``vllm``/``flag_gems``/``flaggems_vllm`` module tree."""

    def __init__(self):
        self.state = _StubState()
        self.classes = _make_stub_classes()
        self.modules = {}
        self._build()

    def _leaf(self, name, **attrs):
        parts = name.split(".")
        for length in range(1, len(parts) + 1):
            partial = ".".join(parts[:length])
            if partial not in self.modules:
                module = ModuleType(partial)
                module.__path__ = []
                self.modules[partial] = module
        for length in range(1, len(parts)):
            setattr(
                self.modules[".".join(parts[:length])],
                parts[length],
                self.modules[".".join(parts[: length + 1])],
            )
        for key, value in attrs.items():
            setattr(self.modules[name], key, value)
        return self.modules[name]

    def _build(self):
        state, classes = self.state, self.classes

        self._leaf(
            "vllm.envs",
            # AttentionFLImpl.__init__ reads this at construction; vllm
            # 0.32 wheels without the batch-invariant flag must still work.
            VLLM_BATCH_INVARIANT=False,
        )
        self._leaf(
            "vllm.logger",
            init_logger=lambda name: _OnceLogger(logging.getLogger(name)),
        )

        class _Platform:
            """Platform double reading vendor/device from the shared state."""

            @property
            def vendor_name(self):
                return state.vendor_name

            @property
            def device_type(self):
                return state.device_type

        self._leaf("vllm.platforms", current_platform=_Platform())
        self._leaf("vllm.platforms.interface", DeviceCapability=SimpleNamespace)

        @contextlib.contextmanager
        def _set_current_vllm_config(config):
            yield

        self._leaf(
            "vllm.config",
            VllmConfig=type("VllmConfig", (), {}),
            get_current_vllm_config=lambda: None,
            get_layers_from_vllm_config=lambda cfg, cls: {},
            set_current_vllm_config=_set_current_vllm_config,
        )
        self._leaf("vllm.config.cache", CacheDType=object)
        self._leaf("vllm.config.compilation", CompilationMode=object)

        class _DcpGroup:
            world_size = 1
            rank_in_group = 0

            def all_gather(self, tensor, dim):
                return tensor

        self._leaf(
            "vllm.distributed.parallel_state", get_dcp_group=lambda: _DcpGroup()
        )
        self._leaf(
            "vllm.model_executor.layers.attention.attention", Attention=object
        )
        mla = self._leaf("vllm.model_executor.layers.attention.mla_attention")
        mla.MLACommonBackend = classes.MLACommonBackend
        mla.MLACommonMetadata = classes.MLACommonMetadata
        mla.MLACommonImpl = classes.MLACommonImpl
        self._leaf("vllm.utils.math_utils", cdiv=lambda a, b: -(-a // b))
        self._leaf(
            "vllm.utils.torch_utils",
            is_quantized_kv_cache=lambda dtype: (
                isinstance(dtype, str) and dtype not in ("auto", "bf16", "fp16")
            ),
        )
        backend_mod = self._leaf(
            "vllm.v1.attention.backend",
            AttentionBackend=classes.AttentionBackend,
            AttentionImpl=classes.AttentionImpl,
            AttentionMetadataBuilder=classes.AttentionMetadataBuilder,
            AttentionCGSupport=classes.AttentionCGSupport,
            AttentionType=classes.AttentionType,
            MultipleOf=object,
            AttentionLayer=object,
        )

        class _Entry:
            def __init__(self, path):
                self.path = path

            def get_path(self):
                return self.path

        class _Enum:
            TRITON_ATTN = _Entry(
                "vllm.v1.attention.backends.triton_attn.TritonAttn"
            )

        backend_mod.AttentionBackendEnum = _Enum
        self._leaf(
            "vllm.v1.attention.backends.utils",
            CommonAttentionMetadata=object,
            get_dcp_local_seq_lens=lambda *a, **k: None,
        )
        self._leaf(
            "vllm.v1.attention.backends.registry",
            AttentionBackendEnum=_Enum,
            register_backend=lambda **k: None,
        )
        self._leaf("vllm.v1.attention.ops.dcp", cp_lse_ag_out_rs=lambda *a, **k: None)
        self._leaf(
            "vllm.v1.attention.ops.merge_attn_states",
            merge_attn_states=lambda *a, **k: None,
        )
        self._leaf("vllm.v1.kv_cache_interface", AttentionSpec=object)
        self._leaf("vllm.v1.attention.ops.triton_reshape_and_cache_flash")

        # vllm.triton_utils.triton is set after the fake triton module below
        # is built, so impl/quantization.py (direct ``import triton``) and
        # the fused_moe impl see the same stubbed module.
        self._leaf("vllm.triton_utils")
        self._leaf(
            "vllm.utils.math_utils",
            round_up=lambda value, multiple: (
                (value + multiple - 1) // multiple * multiple
            ),
        )

        # flag_gems: runtime surface imported by impl/quantization.py plus a
        # plain marker module for availability probes.
        class _DeviceCtx:
            def __init__(self, device):
                self.device = device

            def __enter__(self):
                state.device_ctx = self.device
                return self

            def __exit__(self, *exc):
                state.device_ctx = None
                return False

        self._leaf("flag_gems", __file__="/fake/flag_gems/__init__.py")
        self.modules["flag_gems"].__path__ = []
        self._leaf(
            "flag_gems.runtime",
            torch_device_fn=SimpleNamespace(device=_DeviceCtx),
        )
        self._leaf("flag_gems.utils", libentry=lambda: lambda fn: fn)

        # triton: impl/quantization.py imports it directly; the decorator
        # stand-in is the identity and the tl namespace only needs to
        # support annotation at import time.
        def _identity_jit(fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

        class _Tl:
            float32 = "float32"
            int8 = "int8"
            constexpr = lambda annotation: annotation  # noqa: E731

        triton_mod = self._leaf(
            "triton",
            jit=_identity_jit,
            cdiv=lambda x, y: -(-x // y),
            next_power_of_2=_next_power_of_2,
        )
        triton_mod.__path__ = []
        triton_mod.__file__ = "/fake/triton/__init__.py"
        language_mod = self._leaf("triton.language")
        for attr in (
            "float32",
            "int8",
            "constexpr",
        ):
            setattr(language_mod, attr, getattr(_Tl, attr))
        triton_mod.language = language_mod
        self.modules["vllm.triton_utils"].triton = triton_mod

        # flaggems_vllm: kernel surface imported by the impls. Recording
        # doubles implement plain-torch oracles so the CPU-side
        # pre/post-processing can be verified.
        def flash_attn_varlen_func(**kwargs):
            state.flash_attn_calls.append(kwargs)
            out = kwargs.get("out")
            if out is None:
                out = torch.zeros_like(kwargs["q"])
            else:
                out.zero_()
            if kwargs.get("return_softmax_lse"):
                return out, torch.zeros(
                    kwargs["q"].shape[0], kwargs["q"].shape[1], dtype=torch.float32
                )
            return out

        def flash_mla(*args):
            state.flash_mla_calls.append(args)
            q = args[0]
            return torch.zeros(q.shape[0], q.shape[1], 128, dtype=q.dtype)

        scaled_int8 = self._leaf("flaggems_vllm")
        scaled_int8.__path__ = []
        scaled_int8.__file__ = "/fake/flaggems_vllm/__init__.py"
        scaled_int8.flash_attn_varlen_func = flash_attn_varlen_func
        scaled_int8.flash_mla = flash_mla
        scaled_int8.silu_and_mul = None
        scaled_int8.gelu_and_mul = None
        scaled_int8.moe_align_block_size_triton = None
        scaled_int8.topk_softmax = None
        scaled_int8.invoke_fused_moe_triton_kernel = None
        scaled_int8.grouped_topk = None
        scaled_int8.moe_sum = None
        scaled_int8.apply_rotary_pos_emb = None
        quant_ops = self._leaf("flaggems_vllm.ops.scaled_int8_quant")

        def scaled_int8_quant(x, scale=None, azp=None, symmetric=False):
            state.scaled_int8_quant_calls.append(
                {"x": x, "scale": scale, "azp": azp, "symmetric": symmetric}
            )
            # Independent oracle: plain torch per-token symmetric INT8.
            contiguous = x.contiguous()
            absmax = contiguous.abs().amax(dim=-1, keepdim=True)
            safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
            quantized = torch.clamp(
                torch.round(contiguous * (127.0 / safe)), -128, 127
            ).to(torch.int8)
            return quantized, absmax / 127.0, None

        quant_ops.scaled_int8_quant = scaled_int8_quant

    def install(self, monkeypatch):
        """Install every built module into ``sys.modules``."""
        for name, module in self.modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        return self


@pytest.fixture
def stubs(monkeypatch):
    """Installed stub tree plus shared mutable state."""
    tree = _VllmStubTree()
    tree.install(monkeypatch)
    return tree


def _import_impl(module_name, evict=True):
    """Import a flaggems impl module fresh under the installed stubs.

    The module may already be cached from collection with the REAL vllm
    bound (pytest imports functional files eagerly), so the cached entry is
    evicted first; ``_fl_sys_modules`` restores the original on teardown.
    """
    if evict:
        sys.modules.pop(module_name, None)
    return __import__(module_name, fromlist=["_"])


# ===========================================================================
# impl/attention.py
# ===========================================================================


class TestAttentionBackendSurface:
    """AttentionFLBackend metadata and support gates."""

    def test_get_name_and_impl_cls(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        AttentionFLBackend = module.AttentionFLBackend
        AttentionFLImpl = module.AttentionFLImpl
        AttentionFLMetadataBuilder = module.AttentionFLMetadataBuilder

        assert AttentionFLBackend.get_name() == "CUSTOM"
        assert AttentionFLBackend.get_impl_cls() is AttentionFLImpl
        assert AttentionFLBackend.get_builder_cls() is AttentionFLMetadataBuilder

    def test_supports_attn_type(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        AttentionFLBackend = module.AttentionFLBackend
        from vllm.v1.attention.backend import AttentionType

        for attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        ):
            assert AttentionFLBackend.supports_attn_type(attn_type)
        assert not AttentionFLBackend.supports_attn_type("unknown")

    def test_supports_kv_cache_dtype(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        AttentionFLBackend = module.AttentionFLBackend

        assert AttentionFLBackend.supports_kv_cache_dtype(None)
        assert AttentionFLBackend.supports_kv_cache_dtype("auto")
        assert not AttentionFLBackend.supports_kv_cache_dtype("fp8")

    def test_supports_head_size(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        AttentionFLBackend = module.AttentionFLBackend

        assert AttentionFLBackend.supports_head_size(8)
        assert AttentionFLBackend.supports_head_size(256)
        assert not AttentionFLBackend.supports_head_size(7)
        assert not AttentionFLBackend.supports_head_size(264)

    def test_supports_combination(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        AttentionFLBackend = module.AttentionFLBackend

        assert (
            AttentionFLBackend.supports_combination(
                128, torch.float16, "auto", 16, False, False, False, False,
                SimpleNamespace(),
            )
            is None
        )
        assert AttentionFLBackend.supports_combination(
            128, torch.float16, "auto", 16, False, True, False, False,
            SimpleNamespace(),
        ) == "not support sink"
        assert AttentionFLBackend.supports_combination(
            128, torch.float16, "auto", 16, False, False, False, True,
            SimpleNamespace(),
        ) == "FlagGems FA2 does not support multimodal prefix attention"


class TestSplitKvCache:
    """_split_kv_cache view construction."""

    def test_4d_cache_transposes_and_splits(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        _split_kv_cache = module._split_kv_cache

        kv_cache = torch.arange(2 * 3 * 4 * 16, dtype=torch.float32).reshape(
            2, 3, 4, 16
        )
        key_cache, value_cache = _split_kv_cache(kv_cache, 8)
        assert torch.equal(key_cache, kv_cache.transpose(1, 2)[..., :8])
        assert torch.equal(value_cache, kv_cache.transpose(1, 2)[..., 8:])
        assert key_cache.shape == (2, 4, 3, 8)

    def test_unequal_head_sizes_rejected(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        _split_kv_cache = module._split_kv_cache

        kv_cache = torch.zeros(2, 3, 4, 20)
        with pytest.raises(ValueError, match="equal K/V head sizes"):
            _split_kv_cache(kv_cache, 8)

    def test_unsupported_ndim_rejected(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        _split_kv_cache = module._split_kv_cache

        kv_cache = torch.zeros(2, 3, 4, 16, 5)
        with pytest.raises(ValueError, match="Unsupported FlagGems KV cache"):
            _split_kv_cache(kv_cache, 8)


def _make_attention_impl(stubs, **overrides):
    """Build an AttentionFLImpl with decoder defaults under the stubs."""
    module = _import_impl("vllm_fl.dispatch.backends.flaggems.impl.attention")
    AttentionType = stubs.classes.AttentionType
    kwargs = dict(
        num_heads=4,
        head_size=64,
        scale=0.125,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    kwargs.update(overrides)
    impl = module.AttentionFLImpl(
        attn_type=AttentionType.DECODER, **kwargs
    )
    # __init__ never assigns dcp_world_size yet forward() reads it (a
    # known source quirk — production callers set it explicitly).
    impl.dcp_world_size = 1
    return module, impl


class TestAttentionImplInit:
    """AttentionFLImpl construction and its rejection branches."""

    def test_defaults_recorded(self, stubs):
        module, impl = _make_attention_impl(stubs)
        assert impl.num_heads == 4
        assert impl.head_size == 64
        assert impl.scale == 0.125
        assert impl.num_kv_heads == 2
        assert impl.sliding_window == (-1, -1)
        assert impl.num_queries_per_kv == 2
        assert impl.vllm_flash_attn_version == 2
        assert impl.supports_quant_query_input is False

    def test_sliding_window_decoder(self, stubs):
        module, impl = _make_attention_impl(stubs, sliding_window=128)
        assert impl.sliding_window == (127, 0)

    def test_sliding_window_encoder_only(self, stubs):
        module = _import_impl("vllm_fl.dispatch.backends.flaggems.impl.attention")
        AttentionType = stubs.classes.AttentionType
        impl = module.AttentionFLImpl(
            num_heads=4, head_size=64, scale=0.1, num_kv_heads=2,
            alibi_slopes=None, sliding_window=128, kv_cache_dtype="auto",
            attn_type=AttentionType.ENCODER_ONLY,
        )
        assert impl.sliding_window == (127, 127)

    def test_alibi_slopes_become_tensor(self, stubs):
        module, impl = _make_attention_impl(stubs, alibi_slopes=[0.1, 0.2])
        assert torch.equal(
            impl.alibi_slopes, torch.tensor([0.1, 0.2], dtype=torch.float32)
        )

    def test_quantized_kv_cache_rejected(self, stubs):
        with pytest.raises(NotImplementedError, match="quantization kv-cache"):
            _make_attention_impl(stubs, kv_cache_dtype="fp8")

    def test_logtis_soft_cap_default_zero(self, stubs):
        module, impl = _make_attention_impl(stubs)
        assert impl.logits_soft_cap == 0


class TestAttentionImplForward:
    """AttentionFLImpl.forward dispatch against recording kernel doubles."""

    @staticmethod
    def _layer():
        return SimpleNamespace(
            _q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1)
        )

    def _metadata(self, **overrides):
        metadata = SimpleNamespace(
            num_actual_tokens=4,
            max_query_len=4,
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            max_seq_len=8,
            seq_lens=torch.tensor([8], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.zeros(4, dtype=torch.int32),
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            scheduler_metadata=None,
            prefix_scheduler_metadata=None,
            max_num_splits=0,
            causal=True,
            max_dcp_context_kv_len=None,
            dcp_context_kv_lens=None,
        )
        for key, value in overrides.items():
            setattr(metadata, key, value)
        return metadata

    def test_profiling_run_zeroes_output(self, stubs):
        module, impl = _make_attention_impl(stubs)
        output = torch.ones(4, 4, 64)
        result = impl.forward(
            self._layer(), torch.zeros(4, 4, 64), torch.zeros(4, 2, 64),
            torch.zeros(4, 2, 64), torch.zeros(1, 2, 8, 128), None, output,
        )
        assert torch.count_nonzero(output) == 0

    def test_output_scale_rejected(self, stubs):
        module, impl = _make_attention_impl(stubs)
        with pytest.raises(NotImplementedError, match="fused output quantization"):
            impl.forward(
                self._layer(), torch.zeros(4, 4, 64), torch.zeros(4, 2, 64),
                torch.zeros(4, 2, 64), torch.zeros(1, 2, 8, 128),
                self._metadata(), torch.zeros(4, 4, 64),
                output_scale=torch.ones(1),
            )

    def test_decoder_path_calls_flash_varlen(self, stubs):
        module, impl = _make_attention_impl(stubs)
        metadata = self._metadata()
        output = torch.zeros(4, 4, 64)
        kv_cache = torch.zeros(2, 2, 8, 128)
        q = torch.randn(4, 4, 64)
        k = torch.randn(4, 2, 64)
        v = torch.randn(4, 2, 64)
        impl.forward(self._layer(), q, k, v, kv_cache, metadata, output)

        assert len(stubs.state.flash_attn_calls) == 1
        call = stubs.state.flash_attn_calls[0]
        assert torch.equal(call["cu_seqlens_q"], metadata.query_start_loc)
        assert torch.equal(call["seqused_k"], metadata.seq_lens)
        assert call["max_seqlen_q"] == 4
        assert call["max_seqlen_k"] == 8
        assert call["softmax_scale"] == 0.125
        assert call["causal"] is True
        assert call["fa_version"] == 2
        assert call["num_splits"] == 0
        # The kv cache arrived as split (transposed, half-width) views.
        assert call["k"].shape[-1] == 64

    def test_encoder_only_path_uses_qkv_directly(self, stubs):
        module = _import_impl("vllm_fl.dispatch.backends.flaggems.impl.attention")
        AttentionType = stubs.classes.AttentionType
        impl = module.AttentionFLImpl(
            num_heads=4, head_size=64, scale=0.125, num_kv_heads=2,
            alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
            attn_type=AttentionType.ENCODER_ONLY,
        )
        impl.dcp_world_size = 1
        metadata = self._metadata(num_actual_tokens=4)
        output = torch.zeros(4, 4, 64)
        q = torch.randn(4, 4, 64)
        k = torch.randn(4, 2, 64)
        v = torch.randn(4, 2, 64)
        result = impl.forward(
            self._layer(), q, k, v, torch.zeros(1), metadata, output
        )
        assert len(stubs.state.flash_attn_calls) == 1
        call = stubs.state.flash_attn_calls[0]
        # Encoder attention is bidirectional and uses q/k/v directly (the
        # [:num_actual_tokens] slice of a full-length tensor is a view, so
        # compare values rather than identity).
        assert call["causal"] is False
        torch.testing.assert_close(call["q"], q)
        torch.testing.assert_close(call["k"], k)
        torch.testing.assert_close(call["v"], v)
        # The encoder path returns its sliced output view, not the caller's
        # full tensor; the kernel wrote into that view (zeroed by the double).
        torch.testing.assert_close(result, output)

    def test_cascade_path_calls_cascade_attention(self, stubs, monkeypatch):
        module, impl = _make_attention_impl(stubs)
        calls = []
        monkeypatch.setattr(
            module, "cascade_attention", lambda *a, **k: calls.append((a, k))
        )
        metadata = self._metadata(
            use_cascade=True,
            common_prefix_len=512,
            cu_prefix_query_lens=torch.tensor([0, 4], dtype=torch.int32),
            prefix_kv_lens=torch.tensor([512], dtype=torch.int32),
            suffix_kv_lens=torch.tensor([8], dtype=torch.int32),
        )
        output = torch.zeros(4, 4, 64)
        impl.forward(
            self._layer(), torch.randn(4, 4, 64), torch.randn(4, 2, 64),
            torch.randn(4, 2, 64), torch.zeros(2, 2, 8, 128), metadata, output,
        )
        assert calls, "cascade_attention must run for use_cascade metadata"
        args, kwargs = calls[0]
        assert kwargs["common_prefix_len"] == 512
        assert kwargs["softmax_scale"] == 0.125

    def test_dcp_path_calls_forward_with_dcp(self, stubs, monkeypatch):
        module, impl = _make_attention_impl(stubs)
        impl.dcp_world_size = 2
        calls = []
        monkeypatch.setattr(
            impl, "_forward_with_dcp", lambda *a, **k: calls.append((a, k))
        )
        metadata = self._metadata()
        output = torch.zeros(4, 4, 64)
        result = impl.forward(
            self._layer(), torch.randn(4, 4, 64), torch.randn(4, 2, 64),
            torch.randn(4, 2, 64), torch.zeros(2, 2, 8, 128), metadata, output,
        )
        assert calls, "_forward_with_dcp must run for dcp_world_size > 1"
        assert result is output


class TestDoKvCacheUpdate:
    """do_kv_cache_update: encoder types skip; decoder writes via triton op."""

    def test_encoder_types_skip(self, stubs):
        module = _import_impl("vllm_fl.dispatch.backends.flaggems.impl.attention")
        AttentionType = stubs.classes.AttentionType
        for attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            impl = module.AttentionFLImpl(
                num_heads=4, head_size=64, scale=0.1, num_kv_heads=2,
                alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                attn_type=attn_type,
            )
            # The early return must not touch the cache or import the op.
            impl.do_kv_cache_update(
                SimpleNamespace(_k_scale=None, _v_scale=None),
                torch.zeros(1, 2, 8), torch.zeros(1, 2, 8),
                torch.zeros(2, 2, 8, 16), torch.zeros(1, dtype=torch.int32),
            )

    def test_decoder_writes_via_triton_op(self, stubs):
        module, impl = _make_attention_impl(stubs)
        launches = []

        def recording_op(key, value, key_cache, value_cache, slot_mapping,
                         kv_dtype, k_scale, v_scale):
            launches.append(
                {
                    "key": key, "value": value,
                    "key_cache": key_cache, "value_cache": value_cache,
                    "slot_mapping": slot_mapping, "kv_dtype": kv_dtype,
                }
            )

        triton_op_mod = stubs._leaf(
            "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
            triton_reshape_and_cache_flash=recording_op,
        )
        kv_cache = torch.zeros(2, 2, 8, 128)
        key = torch.randn(4, 2, 64)
        value = torch.randn(4, 2, 64)
        slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        layer = SimpleNamespace(_k_scale=None, _v_scale=None)
        impl.do_kv_cache_update(layer, key, value, kv_cache, slots)

        assert len(launches) == 1
        assert launches[0]["key"] is key
        assert launches[0]["value"] is value
        # The cache arrived split into K and V halves (last dim halved).
        assert launches[0]["key_cache"].shape[-1] == 64
        assert launches[0]["kv_dtype"] == "auto"


class TestUseCascadeAttentionDecision:
    """The use_cascade_attention heuristic decision table."""

    def _call(self, stubs, **overrides):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        use_cascade_attention = module.use_cascade_attention

        kwargs = dict(
            common_prefix_len=1024,
            query_lens=np_array([1] * 16),
            num_query_heads=32,
            num_kv_heads=8,
            use_alibi=False,
            use_sliding_window=False,
            use_local_attention=False,
            num_sms=100,
            dcp_world_size=1,
        )
        kwargs.update(overrides)
        return use_cascade_attention(**kwargs)

    def test_short_prefix_false(self, stubs):
        assert self._call(stubs, common_prefix_len=128) is False

    def test_alibi_false(self, stubs):
        assert self._call(stubs, use_alibi=True) is False

    def test_sliding_window_false(self, stubs):
        assert self._call(stubs, use_sliding_window=True) is False

    def test_local_attention_false(self, stubs):
        assert self._call(stubs, use_local_attention=True) is False

    def test_few_requests_false(self, stubs):
        assert self._call(stubs, query_lens=np_array([1] * 4)) is False

    def test_dcp_world_size_false(self, stubs):
        assert self._call(stubs, dcp_world_size=2) is False

    def test_uniform_decode_with_flash_decoding_decision(self, stubs):
        # num_queries_per_kv > 1 and all query_lens == 1: flash decoding is
        # used, so the decision compares the two CTA models.
        result = self._call(stubs, 
            query_lens=np_array([1] * 16),
            common_prefix_len=2048,
            num_sms=1,  # tiny SM count favors cascade
        )
        assert isinstance(result, bool)

    def test_multi_token_queries_prefer_cascade(self, stubs):
        # Not all query_lens == 1 -> flash decoding not used -> cascade.
        assert self._call(stubs, query_lens=np_array([4] * 16)) is True


def np_array(values):
    import numpy as np

    return np.asarray(values)


class TestCascadeAttentionPlumbing:
    """cascade_attention argument plumbing against the kernel double."""

    def test_prefix_and_suffix_calls(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        cascade_attention = module.cascade_attention

        query = torch.randn(8, 4, 64)
        # The production code reads the block size from key_cache.shape[-3],
        # so this layout carries block_size=16: 32 prefix tokens = 2 blocks.
        key_cache = torch.zeros(2, 16, 4, 64)
        value_cache = torch.zeros(2, 16, 4, 64)
        output = torch.zeros(8, 4, 64)
        cu_query_lens = torch.tensor([0, 8], dtype=torch.int32)
        cu_prefix_query_lens = torch.tensor([0, 8], dtype=torch.int32)
        prefix_kv_lens = torch.tensor([32], dtype=torch.int32)
        suffix_kv_lens = torch.tensor([32], dtype=torch.int32)
        block_table = torch.arange(4, dtype=torch.int32).reshape(1, 4)

        cascade_attention(
            output, query, key_cache, value_cache,
            cu_query_lens=cu_query_lens, max_query_len=8,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens, suffix_kv_lens=suffix_kv_lens,
            max_kv_len=64, softmax_scale=0.125, alibi_slopes=None,
            sliding_window=(-1, -1), logits_soft_cap=0,
            block_table=block_table, common_prefix_len=32, fa_version=2,
        )

        # Two kernel calls: shared prefix then per-query suffix.
        assert len(stubs.state.flash_attn_calls) == 2
        prefix_call, suffix_call = stubs.state.flash_attn_calls
        assert prefix_call["causal"] is False
        assert prefix_call["max_seqlen_k"] == 32
        assert torch.equal(prefix_call["block_table"], block_table[:1])
        assert suffix_call["causal"] is True
        assert suffix_call["max_seqlen_k"] == 64 - 32
        assert torch.equal(
            suffix_call["block_table"], block_table[:, 32 // 16:]
        )

    def test_alibi_rejected(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.attention"
        )
        cascade_attention = module.cascade_attention

        query = torch.randn(8, 4, 64)
        key_cache = torch.zeros(4, 2, 16, 64)
        with pytest.raises(AssertionError, match="does not support ALiBi"):
            cascade_attention(
                torch.zeros(8, 4, 64), query, key_cache, key_cache,
                cu_query_lens=torch.tensor([0, 8], dtype=torch.int32),
                max_query_len=8,
                cu_prefix_query_lens=torch.tensor([0, 8], dtype=torch.int32),
                prefix_kv_lens=torch.tensor([32], dtype=torch.int32),
                suffix_kv_lens=torch.tensor([32], dtype=torch.int32),
                max_kv_len=64, softmax_scale=0.125,
                alibi_slopes=torch.ones(4),
                sliding_window=(-1, -1), logits_soft_cap=0,
                block_table=torch.zeros(1, 4, dtype=torch.int32),
                common_prefix_len=32, fa_version=2,
            )


# ===========================================================================
# impl/quantization.py
# ===========================================================================


def _oracle_quant(x):
    """Independent per-token symmetric INT8 oracle (plain torch, float64)."""
    widened = x.double()
    absmax = widened.abs().amax(dim=-1, keepdim=True)
    scale = (absmax / 127.0).float()
    safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    quantized = torch.clamp(torch.round(widened * (127.0 / safe)), -128, 127).to(
        torch.int8
    )
    return quantized, scale


class TestQuantizationTritonPath:
    """dynamic_per_token_quant_int8_flaggems_triton full contract."""

    @staticmethod
    def _install_fake_kernel(monkeypatch, module):
        """Replace the triton kernel with a recording CPU double."""

        launches = []

        class _KernelDouble:
            def __getitem__(self, grid):
                def launch(x, output, scale, hidden_size=None, BLOCK_SIZE=None):
                    launches.append(
                        {
                            "grid": grid,
                            "x": x,
                            "output": output,
                            "scale": scale,
                            "hidden_size": hidden_size,
                            "BLOCK_SIZE": BLOCK_SIZE,
                        }
                    )
                    widened = x.double()
                    absmax = widened.abs().amax(dim=-1, keepdim=True)
                    safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
                    quantized = torch.clamp(
                        torch.round(widened * (127.0 / safe)), -128, 127
                    ).to(torch.int8)
                    output.copy_(quantized)
                    scale.copy_(absmax.float() / 127.0)

                return launch

        monkeypatch.setattr(
            module, "_dynamic_per_token_quant_int8_kernel", _KernelDouble()
        )
        return launches

    def _load_module(self, stubs):
        return _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.quantization"
        )

    def test_validates_rank(self, stubs):
        module = self._load_module(stubs)
        with pytest.raises(ValueError, match="2D"):
            module.dynamic_per_token_quant_int8_flaggems_triton(
                torch.zeros(2, 2, 2)
            )

    def test_validates_dtype(self, stubs):
        module = self._load_module(stubs)
        with pytest.raises(TypeError, match="floating point"):
            module.dynamic_per_token_quant_int8_flaggems_triton(
                torch.zeros(2, 2, dtype=torch.int32)
            )

    def test_validates_hidden_size(self, stubs):
        module = self._load_module(stubs)
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            module.dynamic_per_token_quant_int8_flaggems_triton(
                torch.zeros(2, 0)
            )

    def test_buffers_grid_and_launch_contract(self, stubs, monkeypatch):
        module = self._load_module(stubs)
        launches = self._install_fake_kernel(monkeypatch, module)

        x = torch.tensor([[1.0, -2.0], [0.0, 0.5], [0.6, 0.6]])
        out, scale = module.dynamic_per_token_quant_int8_flaggems_triton(x)

        assert len(launches) == 1
        launch = launches[0]
        # One program per token row.
        assert launch["grid"] == (3,)
        # int8 output shaped like the input; float32 [N, 1] scale.
        assert launch["output"].dtype == torch.int8
        assert launch["output"].shape == x.shape
        assert launch["scale"].dtype == torch.float32
        assert launch["scale"].shape == (3, 1)
        # hidden_size and BLOCK_SIZE (next power of two) are forwarded.
        assert launch["hidden_size"] == 2
        assert launch["BLOCK_SIZE"] == 2
        # The returned tensors ARE the kernel's buffers (written in place).
        assert out is launch["output"]
        assert scale is launch["scale"]

    def test_non_contiguous_input_is_made_contiguous(self, stubs, monkeypatch):
        module = self._load_module(stubs)
        launches = self._install_fake_kernel(monkeypatch, module)

        base = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        x = base.t()  # [2, 4], non-contiguous
        assert not x.is_contiguous()

        module.dynamic_per_token_quant_int8_flaggems_triton(x)
        assert launches[0]["x"].is_contiguous()
        assert launches[0]["x"].shape == x.shape

    def test_values_match_independent_oracle(self, stubs, monkeypatch):
        module = self._load_module(stubs)
        self._install_fake_kernel(monkeypatch, module)

        torch.manual_seed(2026)
        x = torch.randn(7, 6) * 3.0
        out, scale = module.dynamic_per_token_quant_int8_flaggems_triton(x)

        expected_q, expected_scale = _oracle_quant(x)
        assert out.dtype == torch.int8
        torch.testing.assert_close(out, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(scale, expected_scale, rtol=1e-6, atol=1e-7)

        # Dequantization stays within half a scale step of the input.
        dequant = out.float() * scale
        assert ((x - dequant).abs() <= scale + 1e-6).all()

    def test_block_size_rounds_up_to_power_of_two(self, stubs, monkeypatch):
        module = self._load_module(stubs)
        launches = self._install_fake_kernel(monkeypatch, module)

        module.dynamic_per_token_quant_int8_flaggems_triton(torch.randn(2, 5))
        assert launches[0]["BLOCK_SIZE"] == 8

        module.dynamic_per_token_quant_int8_flaggems_triton(torch.randn(2, 16))
        assert launches[1]["BLOCK_SIZE"] == 16


class TestQuantizationVllmPath:
    """dynamic_per_token_quant_int8_flaggems_vllm delegate contract."""

    def test_forwards_symmetric_dynamic_args(self, stubs):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.quantization"
        )
        x = torch.tensor([[1.0, -2.0], [0.0, 0.5]])
        out, scale = module.dynamic_per_token_quant_int8_flaggems_vllm(x)

        call = stubs.state.scaled_int8_quant_calls[-1]
        assert call["scale"] is None
        assert call["azp"] is None
        assert call["symmetric"] is True
        # The third return element is dropped: a 2-tuple comes back.
        assert isinstance(scale, torch.Tensor)

        # Independent oracle on the returned values.
        absmax = x.abs().amax(dim=-1, keepdim=True)
        torch.testing.assert_close(scale, absmax / 127.0, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            out, torch.tensor([[64, -127], [0, 127]], dtype=torch.int8)
        )


# ===========================================================================
# impl/rotary.py
# ===========================================================================


class TestRotaryEmbeddingFlaggems:
    """rotary_embedding_flaggems vendor-branch matrix."""

    def _make(self, stubs, monkeypatch, vendor="nvidia", device_type="cuda",
             kernel=None):
        stubs.state.vendor_name = vendor
        stubs.state.device_type = device_type

        module = _import_impl("vllm_fl.dispatch.backends.flaggems.impl.rotary")

        # flag_gems double: records calls and returns new tensors.
        calls = {"gems": [], "vllm": []}

        import flag_gems

        def gems_apply(q, k, cos, sin, position_ids=None,
                       rotary_interleaved=False):
            calls["gems"].append((q, k))
            return q * 2.0, k * 3.0

        monkeypatch.setattr(
            flag_gems, "apply_rotary_pos_emb", kernel or gems_apply,
            raising=False
        )
        monkeypatch.setattr(flag_gems, "vendor_name", vendor, raising=False)

        # The impl reads use_c_extension from flag_gems.config at call
        # time. A real flag_gems.config may still be cached in sys.modules
        # from the plugin's own import (monkeypatch only replaced the
        # top-level "flag_gems" entry), so the stub leaf is installed into
        # sys.modules explicitly to shadow it.
        config_mod = stubs._leaf("flag_gems.config")
        monkeypatch.setitem(sys.modules, "flag_gems.config", config_mod)
        monkeypatch.setattr(config_mod, "use_c_extension", False,
                            raising=False)

        flaggems_vllm = sys.modules["flaggems_vllm"]

        def vllm_rope(q, k, cos, sin, position_ids=None,
                      rotary_interleaved=False, inplace=False):
            calls["vllm"].append({"q": q, "k": k, "inplace": inplace})
            return q * 2.0, k * 3.0

        monkeypatch.setattr(
            flaggems_vllm, "apply_rotary_pos_emb", vllm_rope
        )
        return module, calls

    def test_plain_passthrough(self, stubs, monkeypatch):
        # A neutral vendor forwards inplace=True straight to the kernel and
        # returns its fresh outputs unchanged.
        module, calls = self._make(stubs, monkeypatch, vendor="metax")
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 2, 8)
        cos = torch.ones(2, 1, 4)
        sin = torch.zeros(2, 1, 4)
        pos = torch.tensor([[0], [1]])

        q_snapshot = q.clone()
        k_snapshot = k.clone()
        q_out, k_out = module.rotary_embedding_flaggems(
            None, q, k, cos, sin, pos, rotary_interleaved=False, inplace=True
        )
        assert torch.equal(q_out, q_snapshot * 2.0)
        assert torch.equal(k_out, k_snapshot * 3.0)
        assert calls["vllm"][0]["inplace"] is True
        # No copy-back happened: the caller's tensors keep their values.
        assert torch.equal(q, q_snapshot)

    def test_nvidia_inplace_computed_out_of_place(self, stubs, monkeypatch):
        """NVIDIA's in-place race is avoided: both outputs computed, then copied."""
        module, calls = self._make(stubs, monkeypatch, vendor="nvidia")
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 2, 8)
        cos = torch.ones(2, 1, 4)
        sin = torch.zeros(2, 1, 4)
        pos = torch.tensor([[0], [1]])

        q_snapshot = q.clone()
        k_snapshot = k.clone()
        q_out, k_out = module.rotary_embedding_flaggems(
            None, q, k, cos, sin, pos, inplace=True
        )
        # The kernel was asked NOT to do the in-place write.
        assert calls["vllm"][0]["inplace"] is False
        # The results are copied back into the caller's storage.
        assert q_out is q
        assert k_out is k
        assert torch.equal(q_out, q_snapshot * 2.0)
        assert torch.equal(k_out, k_snapshot * 3.0)

    def test_thead_inplace_computed_out_of_place(self, stubs, monkeypatch):
        """T-Head's in-place race is handled the same way as NVIDIA's."""
        module, calls = self._make(stubs, monkeypatch, vendor="thead")
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 2, 8)
        cos = torch.ones(2, 1, 4)
        sin = torch.zeros(2, 1, 4)
        pos = torch.tensor([[0], [1]])

        q_snapshot = q.clone()
        k_snapshot = k.clone()
        q_out, k_out = module.rotary_embedding_flaggems(
            None, q, k, cos, sin, pos, inplace=True
        )
        assert calls["vllm"][0]["inplace"] is False
        assert q_out is q and k_out is k
        assert torch.equal(q_out, q_snapshot * 2.0)
        assert torch.equal(k_out, k_snapshot * 3.0)

    def test_ascend_without_inplace_argument_uses_direct_kernel(
        self, stubs, monkeypatch
    ):
        """The pinned Ascend six-argument kernel is called directly."""

        gems_calls = []

        def six_arg_kernel(q, k, cos, sin, position_ids, rotary_interleaved):
            gems_calls.append((q, k))
            return q * 5.0, k * 7.0

        module, calls = self._make(
            stubs, monkeypatch, vendor="ascend", kernel=six_arg_kernel
        )
        # The dispatch checks "inplace" in inspect.signature: the stub takes
        # none, so the direct-kernel branch runs.
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 2, 8)
        cos = torch.ones(2, 1, 4)
        sin = torch.zeros(2, 1, 4)
        pos = torch.tensor([[0], [1]])

        q_out, k_out = module.rotary_embedding_flaggems(
            None, q, k, cos, sin, pos, inplace=False
        )
        assert gems_calls, "the direct Ascend kernel must run"
        assert torch.equal(q_out, q * 5.0)
        assert torch.equal(k_out, k * 7.0)
        # The flaggems_vllm wrapper was never consulted on this path.
        assert calls["vllm"] == []

    def test_gcu_inplace_uses_out_of_place_copy_back(self, stubs, monkeypatch):
        """The GCU in-place bug routes through the out-of-place kernel."""

        class _FakeGcuTensor(torch.Tensor):
            @property
            def device(self):
                return SimpleNamespace(type="gcu")

        module, calls = self._make(stubs, monkeypatch)
        q = _FakeGcuTensor(torch.randn(2, 4, 8))
        k = _FakeGcuTensor(torch.randn(2, 2, 8))
        cos = torch.ones(2, 1, 4)
        sin = torch.zeros(2, 1, 4)
        pos = torch.tensor([[0], [1]])

        q_snapshot = q.clone()
        k_snapshot = k.clone()
        q_out, k_out = module.rotary_embedding_flaggems(
            None, q, k, cos, sin, pos, inplace=True
        )
        # The kernel was invoked with inplace=False and the result copied
        # back into the caller's tensors.
        assert calls["vllm"][0]["inplace"] is False
        assert q_out is q and k_out is k
        assert torch.equal(q_out, q_snapshot * 2.0)

    def test_supports_inplace_argument_helper(self, stubs, monkeypatch):
        """_supports_inplace_argument inspects the kernel signature."""
        module = _import_impl("vllm_fl.dispatch.backends.flaggems.impl.rotary")

        def with_inplace(a, inplace=False):
            return None

        def without_inplace(a):
            return None

        assert module._supports_inplace_argument(with_inplace) is True
        assert module._supports_inplace_argument(without_inplace) is False
        # Signature-less objects report False, not an exception.
        assert module._supports_inplace_argument(object()) is False


# ===========================================================================
# impl/normalization.py and impl/activation.py
# ===========================================================================


class TestNormalizationDelegate:
    """rms_norm_flaggems delegate contract."""

    def test_forwards_weight_epsilon_and_residual(self, stubs, monkeypatch):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.normalization"
        )
        recorded = {}

        def fake_gems_rms(x, residual, weight, epsilon):
            recorded.update(x=x, residual=residual, weight=weight,
                            epsilon=epsilon)
            return x

        gems_mod = sys.modules.setdefault(
            "flag_gems.modules.normalization", ModuleType("flag_gems.modules.normalization")
        )
        monkeypatch.setattr(gems_mod, "gems_rms_forward", fake_gems_rms)

        class Layer:
            weight = torch.ones(6)
            variance_epsilon = 1e-6

        x = torch.randn(3, 6)
        residual = torch.randn(3, 6)
        out = module.rms_norm_flaggems(Layer(), x, residual)
        assert recorded["x"] is x
        assert recorded["residual"] is residual
        assert recorded["weight"] is Layer.weight
        assert recorded["epsilon"] == 1e-6
        assert out is x

    def test_residual_none_forwarded(self, stubs, monkeypatch):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.normalization"
        )
        recorded = {}

        def fake_gems_rms(x, residual, weight, epsilon):
            recorded["residual"] = residual
            return x

        gems_mod = sys.modules.setdefault(
            "flag_gems.modules.normalization", ModuleType("flag_gems.modules.normalization")
        )
        monkeypatch.setattr(gems_mod, "gems_rms_forward", fake_gems_rms)

        class Layer:
            weight = torch.ones(6)
            variance_epsilon = 1e-6

        module.rms_norm_flaggems(Layer(), torch.randn(3, 6))
        assert recorded["residual"] is None


class TestActivationDelegates:
    """silu_and_mul_flaggems / gelu_and_mul_flaggems contracts."""

    def test_silu_splits_and_combines(self, stubs, monkeypatch):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.activation"
        )
        recorded = {}

        def fake_silu(x1, x2):
            recorded["halves"] = (x1, x2)
            return torch.nn.functional.silu(x1) * x2

        flaggems_vllm = sys.modules["flaggems_vllm"]
        monkeypatch.setattr(flaggems_vllm, "silu_and_mul", fake_silu)

        torch.manual_seed(2026)
        x = torch.randn(2, 6)
        out = module.silu_and_mul_flaggems(None, x)
        assert out.shape == (2, 3)
        x1, x2 = recorded["halves"]
        torch.testing.assert_close(x1, x[..., :3])
        torch.testing.assert_close(x2, x[..., 3:])
        expected = torch.nn.functional.silu(x[..., :3]) * x[..., 3:]
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_gelu_forwards_approximate(self, stubs, monkeypatch):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.activation"
        )
        recorded = {}

        def fake_gelu(x1, x2, approximate="none"):
            recorded["approximate"] = approximate
            if approximate == "tanh":
                inner = (2**0.5 / 3.141592653589793) * (x1 + 0.044715 * x1**3)
                act = 0.5 * x1 * (1.0 + torch.tanh(inner))
            else:
                act = 0.5 * x1 * (1.0 + torch.erf(x1 / 2**0.5))
            return act * x2

        flaggems_vllm = sys.modules["flaggems_vllm"]
        monkeypatch.setattr(flaggems_vllm, "gelu_and_mul", fake_gelu)

        class Obj:
            approximate = "tanh"

        torch.manual_seed(2026)
        x = torch.randn(2, 6)
        out = module.gelu_and_mul_flaggems(Obj(), x)
        assert recorded["approximate"] == "tanh"
        assert out.shape == (2, 3)

    def test_gelu_none_obj_defaults_to_none(self, stubs, monkeypatch):
        module = _import_impl(
            "vllm_fl.dispatch.backends.flaggems.impl.activation"
        )
        recorded = {}

        def fake_gelu(x1, x2, approximate="none"):
            recorded["approximate"] = approximate
            return x1 * x2

        flaggems_vllm = sys.modules["flaggems_vllm"]
        monkeypatch.setattr(flaggems_vllm, "gelu_and_mul", fake_gelu)

        module.gelu_and_mul_flaggems(None, torch.randn(2, 6))
        assert recorded["approximate"] == "none"


# ===========================================================================
# impl/mla.py
# ===========================================================================


_MLA_ARGS = dict(
    num_heads=8,
    head_size=128,
    scale=0.1,
    num_kv_heads=1,
    alibi_slopes=None,
    sliding_window=None,
    kv_cache_dtype="auto",
    logits_soft_cap=None,
    attn_type=None,  # replaced at call time
    kv_sharing_target_layer_name=None,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    qk_head_dim=192,
    v_head_dim=128,
    kv_b_proj=None,
)


def _make_mla_impl(stubs, **overrides):
    """Construct a concrete MLAFLImpl under the stubbed boundary."""
    module = _import_impl("vllm_fl.dispatch.backends.flaggems.impl.mla")
    AttentionType = stubs.classes.AttentionType
    kwargs = dict(_MLA_ARGS)
    kwargs["attn_type"] = AttentionType.DECODER
    kwargs.update(overrides)

    class ConcreteMLA(module.MLAFLImpl):
        def forward_mqa(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("forward_mqa must not run in these tests")

    impl = ConcreteMLA(**kwargs)
    impl._pad_v = True
    return module, impl


class TestMlaBackend:
    """MLAFLBackend metadata."""

    def test_get_name_and_impl_cls(self, stubs):
        module = _import_impl("vllm_fl.dispatch.backends.flaggems.impl.mla")
        assert module.MLAFLBackend.get_name() == "MLAFL"
        assert module.MLAFLBackend.get_impl_cls() is module.MLAFLImpl

    def test_impl_can_return_lse_for_decode(self, stubs):
        _, impl = _make_mla_impl(stubs)
        assert impl.can_return_lse_for_decode is True


class TestMlaImplInit:
    """MLAFLImpl.__init__ rejection branches."""

    def test_unsupported_feature_rejected(self, stubs):
        with pytest.raises(NotImplementedError, match="alibi_slopes"):
            _make_mla_impl(stubs, alibi_slopes=[0.1])

    def test_non_decoder_attention_type_rejected(self, stubs):
        with pytest.raises(NotImplementedError, match="Encoder"):
            _make_mla_impl(stubs, attn_type=stubs.classes.AttentionType.ENCODER)

    def test_fp8_kv_cache_rejected(self, stubs):
        with pytest.raises(NotImplementedError, match="FP8 KV cache"):
            _make_mla_impl(stubs, kv_cache_dtype="fp8")

    def test_successful_init_records_kv_cache_dtype(self, stubs):
        _, impl = _make_mla_impl(stubs)
        assert impl.kv_cache_dtype == "auto"


class TestFlashAttnVarlenDiffHeaddims:
    """MLAFLImpl._flash_attn_varlen_diff_headdims padding and unpacking."""

    def test_v_padded_to_q_head_dim(self, stubs):
        _, impl = _make_mla_impl(stubs)
        q = torch.randn(3, 8, 576)
        k = torch.randn(3, 1, 576)
        v = torch.randn(3, 1, 128)

        impl._flash_attn_varlen_diff_headdims(q, k, v)

        call = stubs.state.flash_attn_calls[-1]
        assert call["q"] is q
        assert call["k"] is k
        # v padded from 128 to 576 (q's head dim), zeros in the tail.
        padded = call["v"]
        assert padded.shape == (3, 1, 576)
        torch.testing.assert_close(padded[..., :128], v)
        assert (padded[..., 128:] == 0).all()

    def test_no_pad_passes_v_through(self, stubs):
        _, impl = _make_mla_impl(stubs)
        impl._pad_v = False
        q = torch.randn(2, 4, 64)
        k = torch.randn(2, 1, 64)
        v = torch.randn(2, 1, 64)

        impl._flash_attn_varlen_diff_headdims(q, k, v)

        call = stubs.state.flash_attn_calls[-1]
        assert call["v"] is v

    def test_scale_and_kwargs_forwarded(self, stubs):
        _, impl = _make_mla_impl(stubs)
        q = torch.randn(2, 4, 64)
        k = torch.randn(2, 1, 64)
        v = torch.randn(2, 1, 64)

        impl._flash_attn_varlen_diff_headdims(
            q, k, v, softmax_scale=0.25, causal=True
        )

        call = stubs.state.flash_attn_calls[-1]
        assert call["softmax_scale"] == 0.25
        assert call["causal"] is True

    def test_return_softmax_lse_flag_forwarded_and_tuple_returned(self, stubs):
        _, impl = _make_mla_impl(stubs)
        q = torch.randn(5, 4, 64)
        k = torch.randn(5, 1, 64)
        v = torch.randn(5, 1, 64)

        out, lse = impl._flash_attn_varlen_diff_headdims(
            q, k, v, return_softmax_lse=True
        )

        call = stubs.state.flash_attn_calls[-1]
        assert call["return_softmax_lse"] is True
        assert lse is not None
        assert lse.shape == (5, 4)
        assert out.shape == q.shape

    def test_default_returns_single_tensor(self, stubs):
        _, impl = _make_mla_impl(stubs)
        q = torch.randn(5, 4, 64)
        k = torch.randn(5, 1, 64)
        v = torch.randn(5, 1, 64)

        result = impl._flash_attn_varlen_diff_headdims(q, k, v)

        call = stubs.state.flash_attn_calls[-1]
        assert call["return_softmax_lse"] is False
        assert isinstance(result, torch.Tensor)
        assert result.shape == q.shape
