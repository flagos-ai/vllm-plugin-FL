# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the FlagGems dispatch backend, MLA impl and INT8 quant.

The modules under test (``vllm_fl/dispatch/backends/flaggems``) treat
``vllm``, ``flag_gems`` and ``flaggems_vllm`` as external boundaries:
``mla.py`` and the ``impl/*`` helpers import them at module level, and the
kernel entry points they wrap need a real accelerator.  These tests install
stub modules for that boundary in ``sys.modules`` (restored afterwards),
load the real production source, and drive the CPU-reachable logic:

* ``FlagGemsBackend`` — name, availability caching, delegate pass-through;
* ``attention_backend`` — CUDA guard, MLA/sparse rejection, the
  ``VLLM_FL_USE_FLAGGEMS_ATTN`` switch, the TRITON_ATTN default;
* ``impl/quantization.py`` — input validation, buffer/grid sizing, the
  launch contract, and the numeric math of the Triton fallback kernel via a
  recording kernel double checked against an independent torch oracle;
* ``impl/mla.py`` — backend metadata, construction rejections, the
  varlen-with-different-head-dims padding/unpacking contract and the
  decode-path ``flash_mla`` call contract;
* ``register_ops.py`` — full wiring into a fresh ``OpRegistry`` plus the
  ``use_flaggems_op`` filtering.

The real ``flag_gems`` import on the Linux CI box raises ``RuntimeError``
("No device were detected"), which is NOT caught by the ImportError-only
handler in ``FlagGemsBackend.is_available``; a fake ``flag_gems`` module is
installed so the probe resolves on both platforms.

Known source quirks pinned here (documented, not mocked away):

* ``MLAFLImpl.__init__`` never assigns ``_pad_v`` or ``dcp_world_size``,
  yet ``_flash_attn_varlen_diff_headdims``/``_forward_decode`` read them —
  the tests set them externally the way the production callers do;
* ``_forward_decode`` passes the last ROW of the (concatenated) query as
  the ``head_dim`` argument to ``flash_mla`` (a tensor, not an int) — the
  recorded contract pins what is actually passed today.
"""

import contextlib
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import Generic, TypeVar

import pytest
import torch

pytestmark = pytest.mark.supplementary

_STUB_ROOTS = ("vllm", "vllm_fl", "flag_gems", "flaggems_vllm", "triton")
_MISSING = object()


@pytest.fixture(autouse=True)
def _fl_sys_modules():
    """Snapshot ``sys.modules`` around stub installs and restore exactly.

    Records every ``vllm*`` / ``vllm_fl*`` / ``flag_gems*`` /
    ``flaggems_vllm*`` entry present before the test, removes entries the
    test added, and puts back the original module objects for entries the
    test replaced — same contract as the ops/worker supplementary conftest
    fixtures, inlined here because this directory's conftest cannot be
    modified.
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


@pytest.fixture(autouse=True)
def _clean_flaggems_env(monkeypatch):
    """Clear every FlagGems policy env var so filtering defaults are stable."""
    for key in (
        "USE_FLAGGEMS",
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
        "VLLM_FL_PREFER",
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_FL_USE_FLAGGEMS_ATTN",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# External-boundary stubs
# ---------------------------------------------------------------------------


class _OnceLogger:
    """Logger wrapper carrying the ``info_once`` used by register_ops."""

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
        self.cuda_available = False
        self.scaled_int8_quant_calls = []
        self.flash_attn_calls = []
        self.flash_mla_calls = []


def _make_stub_classes():
    """Generic-subscriptable stand-ins for the vllm v1 attention bases.

    ``mla.py`` subclasses ``MLACommonImpl[MLACommonMetadata]`` and the
    ``vllm.v1.attention.backend`` generics, so every stub base must support
    ``cls[...]`` the way the real classes do.
    """
    import enum

    class AttentionType(enum.Enum):
        DECODER = "decoder"
        ENCODER = "encoder"
        ENCODER_ONLY = "encoder_only"
        ENCODER_DECODER = "encoder_decoder"

    T = TypeVar("T")

    class _Gen(Generic[T]):
        pass

    class MLACommonImpl(Generic[TypeVar("M")]):  # noqa: UP040
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


class _VllmStubTree:
    """Install a fake ``vllm``/``flag_gems``/``flaggems_vllm`` module tree.

    Only the names the modules under test import are provided; knobs on
    ``state`` let tests steer the probe results.  On Linux, where a real
    ``vllm`` is installed, the real modules stay visible unless a test
    explicitly overrides them with the stub entries — the same names are
    used, so behaviour is identical on both platforms.
    """

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

        self._leaf("vllm.envs")
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
        self._leaf("vllm.distributed.parallel_state", get_dcp_group=lambda: None)
        self._leaf("vllm.model_executor.layers.attention.attention", Attention=object)
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

        # FlagGemsBackend.attention_backend imports this lazily; install a
        # controllable enum surface.  Lazily-built so a test can override.
        class _Entry:
            def __init__(self, path):
                self.path = path

            def get_path(self):
                return self.path

        class _Enum:
            TRITON_ATTN = _Entry("vllm.v1.attention.backends.triton_attn.TritonAttn")

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

        # vllm.triton_utils.triton is set after the fake triton module below
        # is built, so both impl/quantization.py (direct ``import triton``)
        # and impl/fused_moe.py (``from vllm.triton_utils import triton``)
        # see the same stubbed module.
        self._leaf("vllm.triton_utils")
        self._leaf(
            "vllm.utils.math_utils" if False else "vllm.utils.math_utils",
            round_up=lambda value, multiple: (
                (value + multiple - 1) // multiple * multiple
            ),
        )

        # flag_gems: runtime surface imported by impl/quantization.py plus a
        # plain marker module for FlagGemsBackend.is_available().
        class _DeviceCtx:
            def __init__(self, device):
                self.device = device

            def __enter__(self):
                state.device_ctx = self.device
                return self

            def __exit__(self, *exc):
                state.device_ctx = None
                return False

        self._leaf(
            "flag_gems",
            __file__="/fake/flag_gems/__init__.py",
        )
        self.modules["flag_gems"].__path__ = []
        self._leaf(
            "flag_gems.runtime",
            torch_device_fn=SimpleNamespace(device=_DeviceCtx),
        )
        self._leaf("flag_gems.utils", libentry=lambda: lambda fn: fn)

        # triton: impl/quantization.py imports it directly and decorates its
        # kernels with @triton.jit. The real package is GPU-only and absent
        # on CPU-only hosts; the decorator stand-in is the identity (every
        # test that reaches a kernel replaces it with a CPU double anyway),
        # and the tl namespace only needs to support annotation at import.
        def _identity_jit(fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

        class _Tl:
            float32 = "float32"
            int8 = "int8"
            constexpr = lambda annotation: annotation  # noqa: E731

            @staticmethod
            def program_id(axis):
                return ("program_id", axis)

            @staticmethod
            def arange(start, end):
                return ("arange", start, end)

            @staticmethod
            def load(*args, **kwargs):
                return ("load", args, kwargs)

            @staticmethod
            def store(*args, **kwargs):
                return ("store", args, kwargs)

            @staticmethod
            def max(value, axis=None):
                return ("max", value, axis)

            @staticmethod
            def abs(value):
                return ("abs", value)

            @staticmethod
            def where(cond, a, b):
                return ("where", cond, a, b)

            @staticmethod
            def floor(value):
                return ("floor", value)

            @staticmethod
            def minimum(a, b):
                return ("minimum", a, b)

            @staticmethod
            def maximum(a, b):
                return ("maximum", a, b)

        def _next_power_of_2(value):
            # Independent oracle: smallest power of two >= value (>= 1).
            if value <= 1:
                return 1
            result = 1
            while result < value:
                result *= 2
            return result

        triton_mod = self._leaf(
            "triton",
            jit=_identity_jit,
            cdiv=lambda x, y: -(-x // y),
            next_power_of_2=_next_power_of_2,
        )
        triton_mod.__path__ = []
        triton_mod.__file__ = "/fake/triton/__init__.py"
        # impl/quantization.py does ``import triton.language as tl`` and then
        # calls ``tl.floor`` / ``tl.constexpr`` / ... on the module itself, so
        # the language module carries the namespace directly.
        language_mod = self._leaf("triton.language")
        for attr in (
            "float32",
            "int8",
            "constexpr",
            "program_id",
            "arange",
            "load",
            "store",
            "max",
            "abs",
            "where",
            "floor",
            "minimum",
            "maximum",
        ):
            setattr(language_mod, attr, getattr(_Tl, attr))
        triton_mod.language = language_mod
        self.modules["vllm.triton_utils"].triton = triton_mod

        # flaggems_vllm: kernel surface imported by impl/mla.py.  Recording
        # doubles so the call contracts can be asserted on both platforms.
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

        # CUDA availability knob for attention_backend's guard.
        self.cuda_available = False

        def _cuda_is_available():
            return state.cuda_available

        torch_stub = self.modules.setdefault("vllm", ModuleType("vllm"))
        torch_stub._fl_cuda_is_available = _cuda_is_available

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


@pytest.fixture
def flaggems_backend(stubs, monkeypatch):
    """A FlagGemsBackend whose availability cache starts empty.

    ``torch.cuda.is_available`` is pinned False (the honest value on the
    CPU CI machines) so the attention-backend CUDA guard is exercised.
    """
    from vllm_fl.dispatch.backends.flaggems import FlagGemsBackend

    monkeypatch.setattr(FlagGemsBackend, "_available", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    return FlagGemsBackend()


# ---------------------------------------------------------------------------
# FlagGemsBackend metadata and availability
# ---------------------------------------------------------------------------


class TestFlagGemsBackendAvailability:
    """FlagGemsBackend identity, probe and availability caching."""

    def test_name_is_flagos(self, flaggems_backend):
        """The backend is named 'flagos' and is not a vendor backend."""
        assert flaggems_backend.name == "flagos"
        assert flaggems_backend.vendor is None

    def test_is_available_true_with_importable_flag_gems(self, flaggems_backend, stubs):
        """A stubbed flag_gems import resolves the probe to True."""
        # The stub tree installs a fake flag_gems, so the probe succeeds on
        # both the Windows (no flag_gems) and Linux (broken flag_gems)
        # machines.
        assert flaggems_backend.is_available() is True

    def test_availability_is_cached_on_class(self, flaggems_backend):
        """The probe memoizes its result on the class."""
        from vllm_fl.dispatch.backends.flaggems import FlagGemsBackend

        flaggems_backend.is_available()
        assert FlagGemsBackend._available is True
        # A second call returns the cached value without re-probing.
        assert flaggems_backend.is_available() is True


# ---------------------------------------------------------------------------
# FlagGemsBackend.attention_backend
# ---------------------------------------------------------------------------


class TestFlagGemsAttentionBackend:
    """FlagGemsBackend.attention_backend branch matrix."""

    def test_no_cuda_raises_runtime_error(self, flaggems_backend):
        """Without CUDA the Triton backend is refused with RuntimeError."""
        with pytest.raises(RuntimeError, match="requires CUDA"):
            flaggems_backend.attention_backend()

    def test_default_returns_triton_attn_path(self, flaggems_backend, stubs):
        """With CUDA and no policy overrides the TRITON_ATTN path is used."""
        stubs.state.cuda_available = True
        monkeypatch_cuda = torch.cuda.is_available
        try:
            torch.cuda.is_available = lambda: True
            path = flaggems_backend.attention_backend()
        finally:
            torch.cuda.is_available = monkeypatch_cuda
        assert path == "vllm.v1.attention.backends.triton_attn.TritonAttn"

    def test_use_mla_rejected(self, flaggems_backend, stubs, monkeypatch):
        """MLA is not supported by the FlagGems attention backend."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with pytest.raises(NotImplementedError, match="NOT support mla now!"):
            flaggems_backend.attention_backend(use_mla=True)

    def test_sparse_without_mla_rejected(self, flaggems_backend, monkeypatch):
        """use_sparse without use_mla is a ValueError."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with pytest.raises(ValueError, match="use_sparse=True requires use_mla=True"):
            flaggems_backend.attention_backend(use_sparse=True)

    def test_env_switch_selects_flaggems_backend(self, flaggems_backend, monkeypatch):
        """VLLM_FL_USE_FLAGGEMS_ATTN=1 returns the in-tree backend path."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "1")
        path = flaggems_backend.attention_backend()
        assert (
            path
            == "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
        )


# ---------------------------------------------------------------------------
# impl/quantization.py — vllm path
# ---------------------------------------------------------------------------


class TestQuantizationVllmPath:
    """dynamic_per_token_quant_int8_flaggems_vllm delegate contract."""

    def test_forwards_symmetric_dynamic_args(self, flaggems_backend, stubs):
        """The kernel is invoked with scale=None, azp=None, symmetric=True."""
        x = torch.tensor([[1.0, -2.0], [0.0, 0.5]])
        out, scale = flaggems_backend.dynamic_per_token_quant_int8(x)

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


# ---------------------------------------------------------------------------
# impl/quantization.py — triton path
# ---------------------------------------------------------------------------


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
        """Replace the triton kernel with a recording CPU double.

        The double implements the documented kernel math in plain torch
        (round-half-even via torch.round, clamp, scale=absmax/127) so the
        CPU-side pre/post-processing and the launch contract can both be
        asserted without an accelerator.
        """
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
        """Import impl.quantization fresh under the installed stubs."""
        import importlib

        mod = importlib.import_module(
            "vllm_fl.dispatch.backends.flaggems.impl.quantization"
        )
        return importlib.reload(mod)

    def test_validates_rank(self, flaggems_backend):
        """A 3-D tensor is rejected with the documented ValueError."""
        with pytest.raises(ValueError, match="2D"):
            flaggems_backend.dynamic_per_token_quant_int8_triton(torch.zeros(2, 2, 2))

    def test_validates_dtype(self, flaggems_backend):
        """An integer input is rejected with the documented TypeError."""
        with pytest.raises(TypeError, match="floating point"):
            flaggems_backend.dynamic_per_token_quant_int8_triton(
                torch.zeros(2, 2, dtype=torch.int32)
            )

    def test_validates_hidden_size(self, flaggems_backend):
        """A zero hidden dimension is rejected."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            flaggems_backend.dynamic_per_token_quant_int8_triton(torch.zeros(2, 0))

    def test_buffers_grid_and_launch_contract(
        self, flaggems_backend, stubs, monkeypatch
    ):
        """Grid, buffer shapes/dtypes and keyword args match the source."""
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

    def test_non_contiguous_input_is_made_contiguous(
        self, flaggems_backend, stubs, monkeypatch
    ):
        """A transposed (non-contiguous) input is copied before launch."""
        module = self._load_module(stubs)
        launches = self._install_fake_kernel(monkeypatch, module)

        base = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        x = base.t()  # [2, 4], non-contiguous
        assert not x.is_contiguous()

        module.dynamic_per_token_quant_int8_flaggems_triton(x)
        assert launches[0]["x"].is_contiguous()
        assert launches[0]["x"].shape == x.shape

    def test_values_match_independent_oracle(
        self, flaggems_backend, stubs, monkeypatch
    ):
        """Kernel-double outputs match a plain-torch INT8 quant oracle."""
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

    def test_block_size_rounds_up_to_power_of_two(
        self, flaggems_backend, stubs, monkeypatch
    ):
        """BLOCK_SIZE is triton.next_power_of_2 of the hidden size."""
        module = self._load_module(stubs)
        launches = self._install_fake_kernel(monkeypatch, module)

        module.dynamic_per_token_quant_int8_flaggems_triton(torch.randn(2, 5))
        assert launches[0]["BLOCK_SIZE"] == 8

        module.dynamic_per_token_quant_int8_flaggems_triton(torch.randn(2, 16))
        assert launches[1]["BLOCK_SIZE"] == 16


# ---------------------------------------------------------------------------
# impl/mla.py — MLAFLBackend / MLAFLImpl
# ---------------------------------------------------------------------------


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


def _make_impl(stubs, **overrides):
    """Construct a concrete MLAFLImpl under the stubbed boundary.

    ``forward_mqa`` is abstract in vllm 0.32.0, so the test subclass fills
    it in.  ``_pad_v`` is set explicitly because ``__init__`` never assigns
    it (see module docstring).
    """
    from vllm_fl.dispatch.backends.flaggems.impl.mla import MLAFLImpl

    AttentionType = stubs.classes.AttentionType
    kwargs = dict(_MLA_ARGS)
    kwargs["attn_type"] = AttentionType.DECODER
    kwargs.update(overrides)

    class ConcreteMLA(MLAFLImpl):
        def forward_mqa(self, *args, **kwargs):  # pragma: no cover - abstract
            raise AssertionError("forward_mqa must not run in these tests")

    impl = ConcreteMLA(**kwargs)
    impl._pad_v = True
    return impl


class TestMLAFLBackend:
    """MLAFLBackend metadata."""

    def test_get_name_and_impl_cls(self, stubs):
        """The backend is named MLAFL and returns the impl class."""
        from vllm_fl.dispatch.backends.flaggems.impl.mla import (
            MLAFLBackend,
            MLAFLImpl,
        )

        assert MLAFLBackend.get_name() == "MLAFL"
        assert MLAFLBackend.get_impl_cls() is MLAFLImpl

    def test_impl_can_return_lse_for_decode(self, stubs):
        """The impl advertises decode LSE support."""
        impl = _make_impl(stubs)
        assert impl.can_return_lse_for_decode is True


class TestMLAFLImplInit:
    """MLAFLImpl.__init__ rejection branches."""

    def test_unsupported_feature_rejected(self, stubs):
        """alibi_slopes/sliding_window/logits_soft_cap are rejected."""
        with pytest.raises(NotImplementedError, match="alibi_slopes"):
            _make_impl(stubs, alibi_slopes=[0.1])

    def test_sliding_window_rejected(self, stubs):
        """A sliding window is rejected with the same error."""
        with pytest.raises(NotImplementedError, match="alibi_slopes"):
            _make_impl(stubs, sliding_window=512)

    def test_non_decoder_attention_type_rejected(self, stubs):
        """Encoder attention is rejected."""
        with pytest.raises(NotImplementedError, match="Encoder"):
            _make_impl(stubs, attn_type=stubs.classes.AttentionType.ENCODER)

    def test_fp8_kv_cache_rejected(self, stubs):
        """A quantized KV cache dtype is rejected."""
        with pytest.raises(NotImplementedError, match="FP8 KV cache"):
            _make_impl(stubs, kv_cache_dtype="fp8")

    def test_successful_init_records_kv_cache_dtype(self, stubs):
        """The impl stores the kv_cache_dtype passed to the base."""
        impl = _make_impl(stubs)
        assert impl.kv_cache_dtype == "auto"


class TestFlashAttnVarlenDiffHeaddims:
    """_flash_attn_varlen_diff_headdims padding and unpacking."""

    def test_v_padded_to_q_head_dim(self, stubs):
        """With _pad_v the value tensor is zero-padded on the last dim."""
        impl = _make_impl(stubs)
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
        """With _pad_v=False the value tensor is passed unchanged."""
        impl = _make_impl(stubs)
        impl._pad_v = False
        q = torch.randn(2, 4, 64)
        k = torch.randn(2, 1, 64)
        v = torch.randn(2, 1, 64)

        impl._flash_attn_varlen_diff_headdims(q, k, v)

        call = stubs.state.flash_attn_calls[-1]
        assert call["v"] is v

    def test_scale_and_kwargs_forwarded(self, stubs):
        """softmax_scale and extra kwargs reach the kernel unchanged."""
        impl = _make_impl(stubs)
        q = torch.randn(2, 4, 64)
        k = torch.randn(2, 1, 64)
        v = torch.randn(2, 1, 64)

        impl._flash_attn_varlen_diff_headdims(q, k, v, softmax_scale=0.25, causal=True)

        call = stubs.state.flash_attn_calls[-1]
        assert call["softmax_scale"] == 0.25
        assert call["causal"] is True

    def test_return_softmax_lse_flag_forwarded_and_tuple_returned(self, stubs):
        """return_softmax_lse is forced into kwargs and unpacks the tuple."""
        impl = _make_impl(stubs)
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
        """Without return_softmax_lse only the output tensor comes back."""
        impl = _make_impl(stubs)
        q = torch.randn(5, 4, 64)
        k = torch.randn(5, 1, 64)
        v = torch.randn(5, 1, 64)

        result = impl._flash_attn_varlen_diff_headdims(q, k, v)

        call = stubs.state.flash_attn_calls[-1]
        assert call["return_softmax_lse"] is False
        assert isinstance(result, torch.Tensor)
        assert result.shape == q.shape


class TestMLAForwardDecode:
    """_forward_decode flash_mla call contract."""

    @staticmethod
    def _make_metadata():
        """Minimal metadata double exposing decode.block_table/seq_lens."""
        return SimpleNamespace(
            decode=SimpleNamespace(
                block_table=torch.arange(4, dtype=torch.int32).reshape(1, 4),
                seq_lens=torch.tensor([32], dtype=torch.int32),
            )
        )

    def test_single_tensor_q_contract(self, stubs):
        """A plain q tensor goes straight to flash_mla; lse is zeros."""
        impl = _make_impl(stubs)
        metadata = self._make_metadata()
        cache = torch.randn(4, 8, 1024)  # [pages, page_size, d]
        q = torch.randn(2, 8, 192)

        o, lse = impl._forward_decode(q, cache, metadata, layer=None)

        # lse is [B, heads] zeros in the query dtype.
        torch.testing.assert_close(lse, torch.zeros(2, 8))
        assert o.shape[0] == 2 and o.shape[1] == 8

        args = stubs.state.flash_mla_calls[-1]
        q_arg, block_table, cache_arg = args[0], args[1], args[2]
        assert q_arg is q
        assert torch.equal(block_table, metadata.decode.block_table)
        # The cache gained a head dim of size 1: [pages, page, 1, d].
        assert cache_arg.shape == (4, 8, 1, 1024)
        # Positions: None cache, PAGE_SIZE, B, 1, seq_lens, heads, None, ...
        assert args[3] is None
        assert args[4] == 8  # PAGE_SIZE = cache.size(1)
        assert args[5] == 2  # B
        assert args[6] == 1
        assert torch.equal(args[7], metadata.decode.seq_lens)
        assert args[8] == 8  # q_num_heads
        assert args[9] is None
        assert args[12] is True

    def test_tuple_q_is_concatenated_on_last_dim(self, stubs):
        """q_nope/q_pe tuple is concatenated; head_dim_v comes from q[0]."""
        impl = _make_impl(stubs)
        metadata = self._make_metadata()
        cache = torch.randn(4, 8, 1024)
        q_nope = torch.randn(2, 8, 128)
        q_pe = torch.randn(2, 8, 64)

        impl._forward_decode((q_nope, q_pe), cache, metadata, layer=None)

        args = stubs.state.flash_mla_calls[-1]
        concatenated = args[0]
        assert concatenated.shape == (2, 8, 192)
        torch.testing.assert_close(concatenated[..., :128], q_nope)
        torch.testing.assert_close(concatenated[..., 128:], q_pe)
        # head_dim_v (args[11]) is q[0][-1]: the last row of the nope part.
        assert torch.equal(args[11], q_nope[-1])

    def test_fp8_kv_cache_rejected(self, stubs):
        """A quantized KV cache raises before any kernel call.

        ``__init__`` itself rejects fp8, so the decode-path guard is
        reached by mutating the dtype attribute after construction (the
        production base class allows this state drift).
        """
        impl = _make_impl(stubs)
        impl.kv_cache_dtype = "fp8"
        calls_before = len(stubs.state.flash_mla_calls)
        with pytest.raises(NotImplementedError, match="FP8 MLA FL"):
            impl._forward_decode(
                torch.randn(1, 2, 128),
                torch.randn(2, 4, 64),
                self._make_metadata(),
                layer=None,
            )
        assert len(stubs.state.flash_mla_calls) == calls_before

    def test_empty_cache_asserts(self, stubs):
        """An empty cache fails the numel>0 assertion."""
        impl = _make_impl(stubs)
        with pytest.raises(AssertionError):
            impl._forward_decode(
                torch.randn(1, 2, 128),
                torch.zeros(0),
                self._make_metadata(),
                layer=None,
            )

    def test_none_decode_metadata_asserts(self, stubs):
        """A metadata without a decode schedule fails the assertion."""
        impl = _make_impl(stubs)
        with pytest.raises(AssertionError):
            impl._forward_decode(
                torch.randn(1, 2, 128),
                torch.randn(2, 4, 64),
                SimpleNamespace(decode=None),
                layer=None,
            )


# ---------------------------------------------------------------------------
# FlagGemsBackend delegate pass-throughs (impl/activation, normalization,
# rotary, fused_moe) — each delegate forwards to its impl helper, which
# forwards to the flaggems_vllm kernel boundary.  The kernel doubles below
# implement plain-torch oracles so the pre/post-processing on the CPU side
# (splitting the last dim, argument reordering) is verified.
# ---------------------------------------------------------------------------


class TestFlagGemsBackendDelegates:
    """FlagGemsBackend delegate methods over working kernel doubles."""

    @pytest.fixture
    def kernel_stubs(self, stubs, monkeypatch):
        """Install plain-torch kernel doubles on the flaggems_vllm surface."""
        import flaggems_vllm

        recorded = {}

        def silu_and_mul(x1, x2):
            recorded["silu"] = (x1, x2)
            return x1 * torch.sigmoid(x1) * x2

        def gelu_and_mul(x1, x2, approximate):
            recorded["gelu"] = (x1, x2, approximate)
            if approximate == "tanh":
                inner = (2**0.5 / 3.141592653589793) * (x1 + 0.044715 * x1**3)
                act = 0.5 * x1 * (1.0 + torch.tanh(inner))
            else:
                act = 0.5 * x1 * (1.0 + torch.erf(x1 / 2**0.5))
            return act * x2

        def topk_softmax(weights, indices, src_rows, gating, renormalize):
            recorded["topk"] = {
                "gating": gating,
                "renormalize": renormalize,
                "weights": weights,
                "indices": indices,
            }
            probs = torch.softmax(gating, dim=-1)
            top_p, top_i = torch.topk(probs, weights.shape[-1], dim=-1)
            weights.copy_(top_p)
            indices.copy_(top_i)
            return weights, indices

        def moe_sum(inp, out):
            recorded["moe_sum"] = (inp, out)
            out.copy_(inp.sum(dim=1))

        def grouped_topk(
            scores, n_group, topk_group, topk, renormalize, factor, bias, sf
        ):
            recorded["grouped_topk"] = (scores, n_group, topk_group, topk, sf)
            return "flaggems-grouped-sentinel"

        def moe_align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, ntp):
            recorded["moe_align"] = (topk_ids, num_experts, block_size)
            sorted_ids.fill_(topk_ids.numel())
            expert_ids.fill_(0)
            ntp.fill_(0)

        def invoke_kernel(*args, **kwargs):
            recorded["invoke"] = (args, kwargs)

        monkeypatch.setattr(flaggems_vllm, "silu_and_mul", silu_and_mul)
        monkeypatch.setattr(flaggems_vllm, "gelu_and_mul", gelu_and_mul)
        monkeypatch.setattr(flaggems_vllm, "topk_softmax", topk_softmax)
        monkeypatch.setattr(flaggems_vllm, "moe_sum", moe_sum)
        monkeypatch.setattr(flaggems_vllm, "grouped_topk", grouped_topk)
        monkeypatch.setattr(flaggems_vllm, "moe_align_block_size_triton", moe_align)
        monkeypatch.setattr(
            flaggems_vllm, "invoke_fused_moe_triton_kernel", invoke_kernel
        )
        return recorded

    def test_silu_and_mul_splits_and_combines(self, flaggems_backend, kernel_stubs):
        """The delegate splits [..., 2d] into halves for the kernel."""
        torch.manual_seed(2026)
        x = torch.randn(2, 6)
        out = flaggems_backend.silu_and_mul(None, x)
        assert out.shape == (2, 3)
        x1, x2 = kernel_stubs["silu"]
        torch.testing.assert_close(x1, x[..., :3])
        torch.testing.assert_close(x2, x[..., 3:])
        expected = x[..., :3] * torch.sigmoid(x[..., :3]) * x[..., 3:]
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_gelu_and_mul_forwards_approximate(self, flaggems_backend, kernel_stubs):
        """The delegate forwards the obj's approximate mode to the kernel."""
        torch.manual_seed(2026)
        x = torch.randn(2, 6)

        class Obj:
            approximate = "tanh"

        out = flaggems_backend.gelu_and_mul(Obj(), x)
        assert kernel_stubs["gelu"][2] == "tanh"
        assert out.shape == (2, 3)

    def test_topk_softmax_delegates(self, flaggems_backend, kernel_stubs):
        """topk_softmax forwards args and returns the kernel's pair."""
        torch.manual_seed(2026)
        gating = torch.randn(2, 4)
        weights = torch.zeros(2, 2)
        indices = torch.zeros(2, 2, dtype=torch.int64)
        src_rows = torch.zeros(2, 2, dtype=torch.int64)

        out_w, out_i = flaggems_backend.topk_softmax(
            weights, indices, src_rows, gating, renormalize=True
        )
        assert kernel_stubs["topk"]["renormalize"] is True
        assert out_w is weights and out_i is indices
        # Independent oracle on the kernel double's softmax+topk result.
        probs = torch.softmax(gating, dim=-1)
        for row in range(2):
            for slot in range(2):
                torch.testing.assert_close(
                    out_w[row, slot].item(),
                    probs[row, out_i[row, slot].item()].item(),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_moe_sum_delegates(self, flaggems_backend, kernel_stubs):
        """moe_sum passes (inp, out) through to the kernel."""
        inp = torch.ones(2, 3, 4)
        out = torch.zeros(2, 4)
        flaggems_backend.moe_sum(inp, out)
        assert kernel_stubs["moe_sum"][0] is inp
        assert kernel_stubs["moe_sum"][1] is out
        torch.testing.assert_close(out, torch.full((2, 4), 3.0))

    def test_grouped_topk_delegates(self, flaggems_backend, kernel_stubs):
        """grouped_topk forwards all args and returns the kernel result."""
        scores = torch.randn(1, 8)
        out = flaggems_backend.grouped_topk(scores, 2, 1, 4, True, 2.5, None, 1)
        assert out == "flaggems-grouped-sentinel"
        args = kernel_stubs["grouped_topk"]
        assert args[0] is scores
        assert args[1:3] == (2, 1)
        assert args[4] == 1

    def test_moe_align_block_size_delegates(self, flaggems_backend, kernel_stubs):
        """moe_align_block_size allocates buffers and calls the kernel."""
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
        sorted_ids, expert_ids, num_post = flaggems_backend.moe_align_block_size(
            topk_ids, block_size=2, num_experts=2
        )
        # numel=2, experts=2, block=2: 2 + 2*1 = 4.
        assert sorted_ids.numel() == 4
        assert expert_ids.numel() == 2
        assert kernel_stubs["moe_align"][1:] == (2, 2)

    def test_invoke_fused_moe_triton_kernel_delegates(
        self, flaggems_backend, kernel_stubs
    ):
        """invoke_fused_moe_triton_kernel forwards everything."""
        flaggems_backend.invoke_fused_moe_triton_kernel(
            "A",
            "B",
            "C",
            "As",
            "Bs",
            "tw",
            "sti",
            "ei",
            "ntp",
            True,
            2,
            "cfg",
            "ct",
            False,
            False,
            False,
            False,
            False,
            block_shape=[64, 64],
            B_bias=None,
        )
        args, kwargs = kernel_stubs["invoke"]
        assert args[0] == "A" and args[8] == "ntp"
        assert args[10] == 2
        assert kwargs == {"block_shape": [64, 64], "B_bias": None}

    def test_is_available_false_when_import_fails(self, flaggems_backend, monkeypatch):
        """The ImportError branch caches False when flag_gems is missing."""
        import builtins

        from vllm_fl.dispatch.backends.flaggems import FlagGemsBackend

        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "flag_gems":
                raise ImportError("no flag_gems")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(FlagGemsBackend, "_available", None)
        monkeypatch.setattr(builtins, "__import__", failing_import)
        try:
            assert flaggems_backend.is_available() is False
            assert FlagGemsBackend._available is False
        finally:
            monkeypatch.setattr(builtins, "__import__", real_import)

    def test_rms_norm_delegates_with_residual(self, flaggems_backend, monkeypatch):
        """The rms_norm delegate forwards (obj, x, residual) to the impl."""
        import vllm_fl.dispatch.backends.flaggems.impl.normalization as norm_impl

        captured = {}

        def fake_flaggems_norm(obj, x, residual):
            captured["residual"] = residual
            return x, residual

        monkeypatch.setattr(norm_impl, "rms_norm_flaggems", fake_flaggems_norm)
        x = torch.randn(2, 4)
        residual = torch.randn(2, 4)
        out_x, out_res = flaggems_backend.rms_norm(None, x, residual)
        assert captured["residual"] is residual
        assert out_x is x and out_res is residual

    def test_rotary_embedding_delegates(self, flaggems_backend, monkeypatch):
        """The rotary delegate forwards every argument by position/kw."""
        import vllm_fl.dispatch.backends.flaggems.impl.rotary as rotary_impl

        captured = {}

        def fake_rotary(
            obj, query, key, cos, sin, position_ids, rotary_interleaved, inplace
        ):
            captured.update(
                query=query,
                key=key,
                cos=cos,
                sin=sin,
                position_ids=position_ids,
                rotary_interleaved=rotary_interleaved,
                inplace=inplace,
            )
            return query, key

        monkeypatch.setattr(rotary_impl, "rotary_embedding_flaggems", fake_rotary)
        q = torch.randn(1, 2, 4)
        k = torch.randn(1, 1, 4)
        cos, sin = torch.randn(4, 2), torch.randn(4, 2)
        pos = torch.tensor([0])

        out_q, out_k = flaggems_backend.rotary_embedding(
            None, q, k, cos, sin, pos, rotary_interleaved=True, inplace=False
        )
        assert captured["query"] is q
        assert captured["key"] is k
        assert captured["cos"] is cos
        assert captured["sin"] is sin
        assert captured["position_ids"] is pos
        assert captured["rotary_interleaved"] is True
        assert captured["inplace"] is False
        assert out_q is q and out_k is k

    def test_moe_sum_falls_back_to_flag_gems_on_metax(
        self, flaggems_backend, kernel_stubs, monkeypatch
    ):
        """moe_sum falls back to the flag_gems kernel on MetaX failure."""
        import flag_gems
        import flaggems_vllm

        def broken_moe_sum(inp, out):
            raise RuntimeError("no working kernel on MACA")

        def flag_gems_moe_sum(inp, out):
            out.copy_(inp.sum(dim=1) * 2.0)

        monkeypatch.setattr(flaggems_vllm, "moe_sum", broken_moe_sum)
        monkeypatch.setattr(flag_gems, "vendor_name", "MetaX", raising=False)
        monkeypatch.setattr(flag_gems, "moe_sum", flag_gems_moe_sum, raising=False)

        inp = torch.ones(2, 3, 4)
        out = torch.zeros(2, 4)
        flaggems_backend.moe_sum(inp, out)
        # The fallback kernel reduced with its own (doubling) arithmetic.
        torch.testing.assert_close(out, torch.full((2, 4), 6.0))

    def test_moe_sum_reraises_on_other_vendors(
        self, flaggems_backend, kernel_stubs, monkeypatch
    ):
        """A kernel failure on a non-MetaX vendor is re-raised, not swallowed."""
        import flag_gems
        import flaggems_vllm

        def broken_moe_sum(inp, out):
            raise RuntimeError("kernel exploded")

        monkeypatch.setattr(flaggems_vllm, "moe_sum", broken_moe_sum)
        monkeypatch.setattr(flag_gems, "vendor_name", "nvidia", raising=False)
        inp = torch.ones(2, 3, 4)
        with pytest.raises(RuntimeError, match="kernel exploded"):
            flaggems_backend.moe_sum(inp, torch.zeros(2, 4))

    def test_topk_softmax_npu_normalizes_after_kernel(
        self, flaggems_backend, kernel_stubs, monkeypatch
    ):
        """On npu devices renormalization is applied with plain torch."""
        import torch.nn.functional as F

        recorded = {}

        def no_renorm_topk(weights, indices, src_rows, gating, renormalize):
            recorded["kernel_renormalize"] = renormalize
            probs = F.softmax(gating, dim=-1)
            top_p, top_i = torch.topk(probs, weights.shape[-1], dim=-1)
            weights.copy_(top_p)
            indices.copy_(top_i)

        monkeypatch.setattr(
            torch.Tensor, "device", property(lambda self: SimpleNamespace(type="npu"))
        )
        # Note: monkeypatching the device property is invasive; restore in
        # teardown via monkeypatch's own undo.
        import flaggems_vllm

        monkeypatch.setattr(flaggems_vllm, "topk_softmax", no_renorm_topk)

        gating = torch.tensor([[3.0, 1.0, 0.0]])
        weights = torch.zeros(1, 2)
        indices = torch.zeros(1, 2, dtype=torch.int64)
        src_rows = torch.zeros(1, 2, dtype=torch.int64)
        out_w, _ = flaggems_backend.topk_softmax(
            weights, indices, src_rows, gating, renormalize=True
        )
        assert recorded["kernel_renormalize"] is False
        # After-kernel normalization makes the selected weights sum to 1.
        torch.testing.assert_close(out_w.sum().item(), 1.0, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# register_ops.py
# ---------------------------------------------------------------------------


class TestFlagGemsRegisterOps:
    """register_builtins wiring and use_flaggems_op filtering."""

    @pytest.fixture
    def registered(self, stubs):
        """A fresh registry populated by register_builtins."""
        from vllm_fl.dispatch.backends.flaggems.register_ops import register_builtins
        from vllm_fl.dispatch.registry import OpRegistry

        registry = OpRegistry()
        register_builtins(registry)
        return registry

    def test_all_ops_registered(self, registered):
        """The 12 distinct operator names land in the registry."""
        ops = set(registered.list_operators())
        assert ops == {
            "dynamic_per_token_quant_int8",
            "silu_and_mul",
            "gelu_and_mul",
            "rms_norm",
            "rotary_embedding",
            "attention_backend",
            "moe_align_block_size",
            "moe_sum",
            "topk_softmax",
            "invoke_fused_moe_triton_kernel",
            "grouped_topk",
            "w8a8_moe_experts",
        }

    def test_quant_op_has_two_impl_ids(self, registered):
        """dynamic_per_token_quant_int8 registers vllm and triton variants."""
        ids = {
            impl.impl_id
            for impl in registered.get_implementations("dynamic_per_token_quant_int8")
        }
        assert ids == {"default.flagos", "default.flagos_triton"}

    def test_impl_metadata(self, registered):
        """Kind, priority and availability wiring are as documented."""
        from vllm_fl.dispatch.types import BackendImplKind, BackendPriority

        top = registered.get_implementation(
            "dynamic_per_token_quant_int8", "default.flagos"
        )
        assert top.kind == BackendImplKind.DEFAULT
        assert top.priority == BackendPriority.DEFAULT + 10
        assert top.vendor is None
        assert top.is_available() is True

        triton = registered.get_implementation(
            "dynamic_per_token_quant_int8", "default.flagos_triton"
        )
        assert triton.priority == BackendPriority.DEFAULT

        silu = registered.get_implementation("silu_and_mul", "default.flagos")
        assert silu.kind == BackendImplKind.DEFAULT
        assert silu.priority == BackendPriority.DEFAULT
        assert silu.is_available() is True

    def test_bound_fn_calls_through(self, registered, stubs, monkeypatch):
        """The bound wrapper delegates to the backend method."""
        import flaggems_vllm

        recorded = {}

        def fake_silu(x1, x2):
            recorded["shapes"] = (tuple(x1.shape), tuple(x2.shape))
            return x1 * x2

        monkeypatch.setattr(flaggems_vllm, "silu_and_mul", fake_silu)

        silu = registered.get_implementation("silu_and_mul", "default.flagos")
        x = torch.randn(2, 8)
        out = silu.fn(None, x)
        assert out.shape == (2, 4)
        # The kernel receives the two halves of the last dimension.
        assert recorded["shapes"] == ((2, 4), (2, 4))

    def test_filtering_use_flaggems_zero(self, stubs, monkeypatch):
        """USE_FLAGGEMS=0 filters every op out of the registration."""
        from vllm_fl.dispatch.backends.flaggems.register_ops import register_builtins
        from vllm_fl.dispatch.registry import OpRegistry

        monkeypatch.setenv("USE_FLAGGEMS", "0")
        registry = OpRegistry()
        register_builtins(registry)
        assert registry.list_operators() == []

    def test_filtering_whitelist(self, stubs, monkeypatch):
        """A whitelist keeps only the listed operator names."""
        from vllm_fl.dispatch.backends.flaggems.register_ops import register_builtins
        from vllm_fl.dispatch.registry import OpRegistry

        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "silu_and_mul,rms_norm")
        registry = OpRegistry()
        register_builtins(registry)
        assert set(registry.list_operators()) == {"silu_and_mul", "rms_norm"}

    def test_filtering_blacklist(self, stubs, monkeypatch):
        """A blacklist drops only the listed operator names."""
        from vllm_fl.dispatch.backends.flaggems.register_ops import register_builtins
        from vllm_fl.dispatch.registry import OpRegistry

        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "silu_and_mul")
        registry = OpRegistry()
        register_builtins(registry)
        ops = set(registry.list_operators())
        assert "silu_and_mul" not in ops
        assert "rms_norm" in ops
        assert len(ops) == 11

    def test_bind_is_available_wraps_transparently(self, stubs):
        """_bind_is_available delegates and exposes the probe attribute."""
        from vllm_fl.dispatch.backends.flaggems.register_ops import _bind_is_available

        calls = []

        def fn(a, b=0):
            calls.append((a, b))
            return a + b

        def probe():
            return True

        wrapped = _bind_is_available(fn, probe)
        assert wrapped(1, b=2) == 3
        assert calls == [(1, 2)]
        assert wrapped._is_available is probe
        assert wrapped.__name__ == "fn"
