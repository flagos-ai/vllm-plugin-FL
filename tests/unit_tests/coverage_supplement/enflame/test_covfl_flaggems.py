"""Coverage-focused tests for the FlagGems dispatch backend (``vllm_fl``).

These tests are deliberately self-contained and hermetic:

* the real sources under ``vllm_fl/dispatch/backends/flaggems`` are executed
  from disk (so coverage attributes to the real files),
* every heavy third-party dependency (``vllm``, ``flaggems_vllm``, ``flag_gems``,
  ``triton``) is replaced by a tiny in-process stub inside an isolated
  ``sys.modules`` sandbox,
* the actual FlagGems kernels are ``unittest.mock`` objects, so nothing is ever
  compiled, launched or executed on a device.

No network access, no subprocesses, no device allocation.
"""

import contextlib
import importlib.util
import logging
import os
import sys
import types
import typing
from pathlib import Path
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

T = typing.TypeVar("T")
T_MLA = typing.TypeVar("T_MLA")

_FLAGGEMS_REL = Path("vllm_fl") / "dispatch" / "backends" / "flaggems"


def _locate_flaggems_dir():
    candidates = []
    env_root = os.environ.get("VLLM_FL_SRC")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(
        Path(r"C:\Users\21649\AppData\Local\Temp\opencode\enflame-src-030b")
    )
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parent)
    for root in candidates:
        for candidate in (root / _FLAGGEMS_REL, root):
            if (candidate / "impl" / "attention.py").is_file():
                return candidate.resolve()
    return None


FLAGGEMS_DIR = _locate_flaggems_dir()
if FLAGGEMS_DIR is None:
    pytest.skip(
        "cannot locate vllm_fl/dispatch/backends/flaggems; set VLLM_FL_SRC",
        allow_module_level=True,
    )

TRITON_ATTN_PATH = "stub.vllm.TritonAttentionBackend"
FLAGGEMS_ATTN_PATH = (
    "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
)


# ---------------------------------------------------------------------------
# stubbing helpers
# ---------------------------------------------------------------------------


def _module(name, attrs=None, is_package=False, path=None):
    """Create (or replace) a module inside ``sys.modules`` and link the parent."""
    module = types.ModuleType(name)
    module.__package__ = name
    if is_package:
        module.__path__ = list(path or [])
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    sys.modules[name] = module
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, module)
    return module


class _TLValue:
    """Permissive ``tl.*`` value supporting all trait ops used by kernels."""

    def __getattr__(self, _name):
        return _TLValue()

    def __call__(self, *_args, **_kwargs):
        return _TLValue()

    def __getitem__(self, _item):
        return _TLValue()

    def __iter__(self):
        return iter([_TLValue()])

    def __bool__(self):
        return True

    def __gt__(self, _other):
        return _TLValue()

    def __ge__(self, _other):
        return _TLValue()

    def __lt__(self, _other):
        return _TLValue()

    def __le__(self, _other):
        return _TLValue()

    def __add__(self, _other):
        return _TLValue()

    __radd__ = __add__

    def __sub__(self, _other):
        return _TLValue()

    __rsub__ = __sub__

    def __mul__(self, _other):
        return _TLValue()

    __rmul__ = __mul__

    def __truediv__(self, _other):
        return _TLValue()

    __rtruediv__ = __truediv__

    def __neg__(self):
        return _TLValue()

    def __and__(self, _other):
        return _TLValue()

    def __or__(self, _other):
        return _TLValue()

    def __xor__(self, _other):
        return _TLValue()

    def __invert__(self):
        return _TLValue()


def _declaring(*args, **_kwargs):
    if len(args) == 1 and not _kwargs and callable(args[0]):
        return args[0]

    def decorator(fn):
        return fn

    return decorator


def _raise_assertion():
    raise AssertionError("DCP group is not initialised in tests")


_OWNED_PREFIXES = ("vllm", "vllm_fl", "flaggems_vllm", "flag_gems", "triton")


def _is_owned(name):
    return any(
        name == prefix or name.startswith(prefix + ".") for prefix in _OWNED_PREFIXES
    )


class _MultipleOf:
    def __init__(self, value):
        self.value = int(value)

    def __eq__(self, other):
        return isinstance(other, _MultipleOf) and other.value == self.value

    def __hash__(self):
        return hash(("MultipleOf", self.value))

    def __repr__(self):  # pragma: no cover - debug helper
        return f"MultipleOf({self.value})"


class _BackendEnumMember:
    def __init__(self, path):
        self._path = path

    def get_path(self):
        return self._path


class _StubEnv:
    """Builds, holds and tears down the stubbed import environment."""

    def __init__(self, root):
        self.root = Path(root)
        self.impl_dir = self.root / "impl"
        self.vllm_fl_dir = self.root.parents[2]
        self.saved = None
        self.mods = types.SimpleNamespace()
        self.stubs = types.SimpleNamespace()

    # -- lifecycle ---------------------------------------------------------
    def setup(self):
        self.saved = dict(sys.modules)
        self._build_third_party_stubs()
        self._build_package_namespace()
        self._inject_fake_impl_modules()
        self._load_real_modules()

    def teardown(self):
        saved = self.saved or {}
        # Only undo the sandbox names we own, so modules imported by other
        # test files (or by pytest plugins) are left untouched.
        for name in list(sys.modules):
            if _is_owned(name) and name not in saved:
                sys.modules.pop(name, None)
        for name, module in saved.items():
            if _is_owned(name):
                sys.modules[name] = module

    # -- third party -------------------------------------------------------
    def _build_third_party_stubs(self):
        attn_type = types.SimpleNamespace(
            DECODER="DECODER",
            ENCODER="ENCODER",
            ENCODER_ONLY="ENCODER_ONLY",
            ENCODER_DECODER="ENCODER_DECODER",
            DECODER_CROSS_ATTN="DECODER_CROSS_ATTN",
        )

        class _AttentionBackend:
            accept_output_buffer = False
            forward_includes_kv_cache_update = True
            supported_dtypes = []

        class _AttentionImpl:
            can_return_lse_for_decode = False

        class _AttentionMetadataBuilder(typing.Generic[T]):
            _cudagraph_support = None

            def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
                self.kv_cache_spec = kv_cache_spec
                self.layer_names = list(layer_names)
                self.vllm_config = vllm_config
                self.device = device

        class _MLACommonBackend:
            pass

        class _MLACommonMetadata:
            pass

        class _MLACommonImpl(typing.Generic[T_MLA]):
            def __init__(
                self,
                num_heads,
                head_size,
                scale,
                num_kv_heads,
                alibi_slopes,
                sliding_window,
                kv_cache_dtype,
                logits_soft_cap,
                attn_type,
                kv_sharing_target_layer_name,
                **mla_args,
            ):
                self.num_heads = num_heads
                self.head_size = head_size
                self.scale = scale
                self.num_kv_heads = num_kv_heads
                self.alibi_slopes = alibi_slopes
                self.sliding_window = sliding_window
                self.kv_cache_dtype = kv_cache_dtype
                self.logits_soft_cap = logits_soft_cap
                self.attn_type = attn_type
                self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
                self.kv_lora_rank = mla_args.get("kv_lora_rank", 512)
                self._pad_v = bool(mla_args.get("_pad_v", False))

        class _AttentionLayer:
            pass

        class _AttentionCGSupport:
            ALWAYS = "ALWAYS"
            UNIFORM_BATCH = "UNIFORM_BATCH"
            NEVER = "NEVER"

        class _VllmConfig:
            pass

        class _AttentionSpec:
            pass

        class _CommonAttentionMetadata:
            pass

        triton_member = _BackendEnumMember(TRITON_ATTN_PATH)

        class _AttentionBackendEnum:
            TRITON_ATTN = triton_member

        registry_calls = []

        def _register_backend(**kwargs):
            registry_calls.append(kwargs)

        flash_attn = mock.MagicMock(name="flash_attn_varlen_func")
        flash_mla = mock.MagicMock(name="flash_mla")
        gems_silu = mock.MagicMock(name="silu_and_mul")
        gems_gelu = mock.MagicMock(name="gelu_and_mul")
        scaled_int8 = mock.MagicMock(name="scaled_int8_quant")
        reshape_and_cache = mock.MagicMock(name="triton_reshape_and_cache_flash")
        cp_lse = mock.MagicMock(name="cp_lse_ag_out_rs")
        merge_states = mock.MagicMock(name="merge_attn_states")

        device_stub = types.SimpleNamespace(
            device=lambda _dev: contextlib.nullcontext()
        )

        # vllm -----------------------------------------------------------------
        _module("vllm", is_package=True)
        _module("vllm.envs", {"VLLM_BATCH_INVARIANT": False})
        _module(
            "vllm.logger",
            {"init_logger": lambda name=None: logging.getLogger(name or "vllm")},
        )

        config_mod = _module("vllm.config", is_package=True)
        config_mod.VllmConfig = _VllmConfig
        config_mod.get_current_vllm_config = lambda: types.SimpleNamespace(
            model_config=None, cache_config=None
        )
        config_mod.get_layers_from_vllm_config = lambda vllm_config, cls=None: {}
        _module("vllm.config.cache", {"CacheDType": str})

        _module("vllm.distributed", is_package=True)
        _module(
            "vllm.distributed.parallel_state",
            {"get_dcp_group": _raise_assertion},
        )

        _module("vllm.model_executor", is_package=True)
        _module("vllm.model_executor.layers", is_package=True)
        _module("vllm.model_executor.layers.attention", is_package=True)
        _module(
            "vllm.model_executor.layers.attention.attention",
            {"Attention": type("Attention", (), {})},
        )
        _module(
            "vllm.model_executor.layers.attention.mla_attention",
            {
                "MLACommonBackend": _MLACommonBackend,
                "MLACommonImpl": _MLACommonImpl,
                "MLACommonMetadata": _MLACommonMetadata,
            },
        )

        _module("vllm.platforms", is_package=True)
        _module(
            "vllm.platforms.interface",
            {"DeviceCapability": type("DeviceCapability", (), {})},
        )
        _module(
            "vllm.platforms",
            {
                "current_platform": types.SimpleNamespace(
                    device_type="musa", vendor_name=None
                )
            },
        )

        _module("vllm.utils", is_package=True)
        _module(
            "vllm.utils.math_utils",
            {
                "cdiv": lambda a, b: -(-int(a) // int(b)),
                "round_up": lambda a, b: -(-int(a) // int(b)) * int(b),
            },
        )
        _module(
            "vllm.utils.torch_utils",
            {
                "is_quantized_kv_cache": lambda dtype: isinstance(dtype, str)
                and dtype.startswith("fp8"),
                "direct_register_custom_op": lambda *a, **k: None,
            },
        )

        _module("vllm.v1", is_package=True)
        _module("vllm.v1.attention", is_package=True)
        _module(
            "vllm.v1.attention.backend",
            {
                "AttentionBackend": _AttentionBackend,
                "AttentionCGSupport": _AttentionCGSupport,
                "AttentionImpl": _AttentionImpl,
                "AttentionMetadataBuilder": _AttentionMetadataBuilder,
                "AttentionType": attn_type,
                "MultipleOf": _MultipleOf,
                "AttentionLayer": _AttentionLayer,
            },
        )
        _module("vllm.v1.attention.backends", is_package=True)
        _module(
            "vllm.v1.attention.backends.utils",
            {
                "CommonAttentionMetadata": _CommonAttentionMetadata,
                "get_dcp_local_seq_lens": lambda lens, world, rank, interleave: lens,
            },
        )
        _module(
            "vllm.v1.attention.backends.registry",
            {
                "AttentionBackendEnum": _AttentionBackendEnum,
                "register_backend": _register_backend,
            },
        )
        _module("vllm.v1.attention.ops", is_package=True)
        _module(
            "vllm.v1.attention.ops.dcp",
            {"cp_lse_ag_out_rs": cp_lse},
        )
        _module(
            "vllm.v1.attention.ops.merge_attn_states",
            {"merge_attn_states": merge_states},
        )
        _module(
            "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
            {"triton_reshape_and_cache_flash": reshape_and_cache},
        )
        _module("vllm.v1.kv_cache_interface", {"AttentionSpec": _AttentionSpec})

        # flaggems_vllm --------------------------------------------------------
        _module("flaggems_vllm", is_package=True)
        fv = sys.modules["flaggems_vllm"]
        fv.flash_attn_varlen_func = flash_attn
        fv.flash_mla = flash_mla
        fv.silu_and_mul = gems_silu
        fv.gelu_and_mul = gems_gelu
        _module("flaggems_vllm.ops", is_package=True)
        _module(
            "flaggems_vllm.ops.scaled_int8_quant",
            {"scaled_int8_quant": scaled_int8},
        )

        # triton ---------------------------------------------------------------
        triton_mod = _module("triton", is_package=True)
        triton_mod.jit = _declaring
        triton_mod.next_power_of_2 = lambda n: 1 << max(0, int(n) - 1).bit_length()
        tl_mod = _module("triton.language")
        tl_mod.__getattr__ = lambda name: _TLValue()
        triton_mod.language = tl_mod

        # flag_gems ------------------------------------------------------------
        flag_gems_mod = _module("flag_gems", is_package=True)
        flag_gems_mod.vendor_name = "stub"
        _module("flag_gems.runtime", {"torch_device_fn": device_stub})
        _module("flag_gems.utils", {"libentry": _declaring})
        _module("flag_gems.config", {"use_c_extension": lambda: False})
        _module("flag_gems.modules", is_package=True)
        _module(
            "flag_gems.modules.normalization",
            {
                "gems_rms_forward": mock.MagicMock(name="gems_rms_forward"),
                "use_c_extension": lambda: False,
            },
        )

        self.stubs.flash_attn = flash_attn
        self.stubs.flash_mla = flash_mla
        self.stubs.gems_silu = gems_silu
        self.stubs.gems_gelu = gems_gelu
        self.stubs.scaled_int8_quant = scaled_int8
        self.stubs.reshape_and_cache = reshape_and_cache
        self.stubs.cp_lse = cp_lse
        self.stubs.merge_states = merge_states
        self.stubs.attention_type = attn_type
        self.stubs.multiple_of = _MultipleOf
        self.stubs.registry_calls = registry_calls
        self.stubs.triton_attn = triton_member
        self.stubs.torch_device_fn = device_stub

    # -- package namespace --------------------------------------------------
    def _build_package_namespace(self):
        backends = self.root.parent
        dispatch = backends.parent
        _module("vllm_fl", is_package=True, path=[str(self.vllm_fl_dir)])
        _module("vllm_fl.dispatch", is_package=True, path=[str(dispatch)])
        _module("vllm_fl.dispatch.backends", is_package=True, path=[str(backends)])
        _module(
            "vllm_fl.dispatch.backends.flaggems",
            is_package=True,
            path=[str(self.root)],
        )
        _module(
            "vllm_fl.dispatch.backends.flaggems.impl",
            is_package=True,
            path=[str(self.impl_dir)],
        )
        _module("vllm_fl.dispatch.backends.vendor", is_package=True, path=[])
        _module("vllm_fl.dispatch.backends.vendor.iluvatar", is_package=True, path=[])
        _module(
            "vllm_fl.dispatch.backends.vendor.iluvatar.activation",
            {"_iluvatar_gems_silu_and_mul": mock.MagicMock(name="iluvatar_silu")},
        )
        _module("vllm_fl.quantization", is_package=True, path=[])
        _module("vllm_fl.quantization.w8a8", is_package=True, path=[])

    def _inject_fake_impl_modules(self):
        fake_modules = {}
        fake_funcs = {}

        for name, func_names in {
            "normalization": ["rms_norm_flaggems"],
            "rotary": ["rotary_embedding_flaggems"],
            "fused_moe": [
                "moe_align_block_size_flaggems",
                "moe_sum_flaggems",
                "topk_softmax_flaggems",
                "invoke_fused_moe_triton_kernel_flaggems",
                "grouped_topk_flaggems",
            ],
            "musa_ops": ["silu_and_mul", "rotary_embedding", "rms_norm"],
        }.items():
            funcs = {fn: mock.MagicMock(name=fn) for fn in func_names}
            module = _module(
                "vllm_fl.dispatch.backends.flaggems.impl." + name, funcs
            )
            fake_modules[name] = module
            fake_funcs[name] = types.SimpleNamespace(**funcs)

        fused_experts = mock.MagicMock(name="_flaggems_fused_experts_impl")
        _module(
            "vllm_fl.quantization.w8a8.moe_experts",
            {"_flaggems_fused_experts_impl": fused_experts},
        )

        self.stubs.fake_modules = fake_modules
        self.stubs.fake_funcs = fake_funcs
        self.stubs.fused_experts = fused_experts

    # -- real sources -------------------------------------------------------
    def _load(self, name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        parent_name, _, leaf = name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, leaf, module)
        spec.loader.exec_module(module)
        return module

    def _load_real_modules(self):
        prefix = "vllm_fl.dispatch.backends.flaggems.impl."
        self.mods.base = self._load(
            "vllm_fl.dispatch.backends.base", self.root.parent / "base.py"
        )
        self.mods.quantization = self._load(
            prefix + "quantization", self.impl_dir / "quantization.py"
        )
        self.mods.activation = self._load(
            prefix + "activation", self.impl_dir / "activation.py"
        )
        self.mods.w8a8_moe = self._load(
            prefix + "w8a8_moe", self.impl_dir / "w8a8_moe.py"
        )
        self.mods.custom_attention = self._load(
            prefix + "custom_attention", self.impl_dir / "custom_attention.py"
        )
        self.mods.attention = self._load(
            prefix + "attention", self.impl_dir / "attention.py"
        )
        self.mods.mla = self._load(prefix + "mla", self.impl_dir / "mla.py")
        self.mods.backend = self._load(
            "vllm_fl.dispatch.backends.flaggems.flaggems",
            self.root / "flaggems.py",
        )
        self.mods.FlagGemsBackend = self.mods.backend.FlagGemsBackend


def _exec_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def env():
    stub = _StubEnv(FLAGGEMS_DIR)
    stub.setup()
    try:
        yield stub
    finally:
        stub.teardown()


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------


def _make_vllm_config(
    hybrid=False,
    mamba_ssm_dtype="auto",
    mamba_dtype="auto",
    full_cudagraph=False,
    max_cudagraph_size=0,
    max_num_seqs=8,
):
    model_config = types.SimpleNamespace(
        is_hybrid=hybrid,
        get_num_attention_heads=lambda parallel_config: 8,
        get_num_kv_heads=lambda parallel_config: 8,
        get_head_size=lambda: 64,
    )
    cache_config = types.SimpleNamespace(
        mamba_ssm_cache_dtype=mamba_ssm_dtype,
        mamba_cache_dtype=mamba_dtype,
    )
    compilation_config = types.SimpleNamespace(
        cudagraph_mode=types.SimpleNamespace(
            has_full_cudagraphs=lambda: full_cudagraph
        ),
        max_cudagraph_capture_size=max_cudagraph_size,
    )
    parallel_config = types.SimpleNamespace(cp_kv_cache_interleave_size=1)
    scheduler_config = types.SimpleNamespace(max_num_seqs=max_num_seqs)
    return types.SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
    )


def _make_builder(env, monkeypatch, vllm_config=None, dcp_group=None, **spec_kwargs):
    mods = env.mods
    vllm_config = vllm_config or _make_vllm_config()
    if dcp_group is None:
        monkeypatch.setattr(mods.attention, "get_dcp_group", _raise_assertion)
        monkeypatch.setattr(
            sys.modules["vllm.distributed.parallel_state"],
            "get_dcp_group",
            _raise_assertion,
        )
    else:
        getter = lambda: dcp_group  # noqa: E731
        monkeypatch.setattr(mods.attention, "get_dcp_group", getter)
        # AttentionFLMetadataBuilder re-imports get_dcp_group from the vLLM
        # parallel_state module inside __init__, so patch that one too.
        monkeypatch.setattr(
            sys.modules["vllm.distributed.parallel_state"], "get_dcp_group", getter
        )
    spec = types.SimpleNamespace(
        dtype=spec_kwargs.pop("dtype", torch.float16),
        block_size=spec_kwargs.pop("block_size", 16),
        **spec_kwargs,
    )
    builder = mods.attention.AttentionFLMetadataBuilder(
        spec, ["layer0"], vllm_config, torch.device("cpu")
    )
    return builder, vllm_config, spec


def _make_common_attn(
    num_reqs=1,
    num_actual_tokens=5,
    max_query_len=5,
    max_seq_len=5,
    causal=True,
):
    starts = [0]
    per_req = max_query_len
    for _ in range(num_reqs):
        starts.append(starts[-1] + per_req)
    return types.SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        seq_lens=torch.tensor([max_seq_len] * num_reqs, dtype=torch.int32),
        block_table_tensor=torch.zeros(num_reqs, 4, dtype=torch.int32),
        slot_mapping=torch.tensor(list(range(num_actual_tokens)), dtype=torch.int32),
        causal=causal,
    )


def _make_impl(env, attn_type=None, **overrides):
    kwargs = dict(
        num_heads=8,
        head_size=64,
        scale=0.125,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    if attn_type is not None:
        kwargs["attn_type"] = attn_type
    kwargs.update(overrides)
    impl = env.mods.attention.AttentionFLImpl(**kwargs)
    # AttentionFLImpl.__init__ never initialises this (the vLLM base class
    # normally does), so tests set it explicitly.
    impl.dcp_world_size = 1
    return impl


def _make_attn_metadata(**overrides):
    base = dict(
        num_actual_tokens=5,
        max_query_len=5,
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        max_seq_len=5,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        causal=True,
        max_dcp_context_kv_len=0,
        dcp_context_kv_lens=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _make_layer():
    return types.SimpleNamespace(
        _q_scale=torch.ones(1),
        _k_scale=torch.ones(1),
        _v_scale=torch.ones(1),
    )


def _make_qkv(num_tokens=5, num_heads=8, head_size=64):
    return (
        torch.zeros(num_tokens, num_heads, head_size),
        torch.zeros(num_tokens, num_heads, head_size),
        torch.zeros(num_tokens, num_heads, head_size),
    )


def _kv_cache(num_blocks=4, num_kv_heads=8, block_size=16, head_size=64):
    return torch.zeros(num_blocks, num_kv_heads, block_size, 2 * head_size)


# ---------------------------------------------------------------------------
# attention backend: registration / capability surface
# ---------------------------------------------------------------------------


def test_attention_backend_static_metadata(env):
    mods = env.mods
    backend = mods.attention.AttentionFLBackend
    assert backend.accept_output_buffer is True
    assert backend.forward_includes_kv_cache_update is False
    assert backend.supported_dtypes == [torch.float16, torch.bfloat16]
    assert backend.get_name() == "CUSTOM"
    assert backend.get_impl_cls() is mods.attention.AttentionFLImpl
    assert backend.get_builder_cls() is mods.attention.AttentionFLMetadataBuilder
    assert backend.supports_sink() is False

    attn_type = mods.attention.AttentionType
    for supported in (
        attn_type.DECODER,
        attn_type.ENCODER,
        attn_type.ENCODER_ONLY,
        attn_type.ENCODER_DECODER,
    ):
        assert backend.supports_attn_type(supported) is True
    assert backend.supports_attn_type("bogus") is False


def test_get_supported_kernel_block_sizes_default(env, monkeypatch):
    mods = env.mods
    monkeypatch.setattr(
        mods.attention, "get_current_vllm_config", lambda: _make_vllm_config()
    )
    sizes = mods.attention.AttentionFLBackend.get_supported_kernel_block_sizes()
    assert sizes == [mods.attention.MultipleOf(16)]


@pytest.mark.parametrize(
    "mamba_ssm_dtype, mamba_dtype",
    [("float32", "auto"), ("auto", "float32")],
)
def test_get_supported_kernel_block_sizes_hybrid(env, monkeypatch, mamba_ssm_dtype, mamba_dtype):
    mods = env.mods
    config = _make_vllm_config(
        hybrid=True, mamba_ssm_dtype=mamba_ssm_dtype, mamba_dtype=mamba_dtype
    )
    monkeypatch.setattr(mods.attention, "get_current_vllm_config", lambda: config)
    sizes = mods.attention.AttentionFLBackend.get_supported_kernel_block_sizes()
    assert sizes == [16, 32, 64]


def test_supports_kv_cache_dtype(env):
    backend = env.mods.attention.AttentionFLBackend
    assert backend.supports_kv_cache_dtype(None) is True
    assert backend.supports_kv_cache_dtype("auto") is True
    assert backend.supports_kv_cache_dtype("fp8_e4m3") is False


def test_supports_head_size(env):
    backend = env.mods.attention.AttentionFLBackend
    assert backend.supports_head_size(8) is True
    assert backend.supports_head_size(128) is True
    assert backend.supports_head_size(256) is True
    assert backend.supports_head_size(257) is False
    assert backend.supports_head_size(12) is False


def test_supports_combination(env):
    backend = env.mods.attention.AttentionFLBackend
    kwargs = dict(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype=None,
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=None,
    )
    assert backend.supports_combination(**kwargs) is None
    assert (
        backend.supports_combination(**dict(kwargs, has_sink=True))
        == "not support sink"
    )
    assert (
        backend.supports_combination(**dict(kwargs, use_mm_prefix=True))
        == "FlagGems FA2 does not support multimodal prefix attention"
    )


def test_get_sliding_window_configs(env, monkeypatch):
    mods = env.mods
    impl_a = object.__new__(mods.attention.AttentionFLImpl)
    impl_a.sliding_window = (-1, -1)
    impl_b = object.__new__(mods.attention.AttentionFLImpl)
    impl_b.sliding_window = (-1, -1)
    layers = {
        "a": types.SimpleNamespace(impl=impl_a),
        "b": types.SimpleNamespace(impl=impl_b),
    }
    monkeypatch.setattr(
        mods.attention,
        "get_layers_from_vllm_config",
        lambda vllm_config, cls=None: layers,
    )
    assert mods.attention._get_sliding_window_configs(object()) == {(-1, -1)}


# ---------------------------------------------------------------------------
# attention metadata builder
# ---------------------------------------------------------------------------


def test_metadata_builder_init_and_build(env, monkeypatch):
    builder, _config, _spec = _make_builder(env, monkeypatch)
    assert builder.num_heads_q == 8
    assert builder.num_heads_kv == 8
    assert builder.headdim == 64
    assert builder.block_size == 16
    assert builder.kv_cache_dtype == torch.float16
    assert builder.dcp_world_size == 1
    assert builder.dcp_rank == 0
    assert builder.max_num_splits == 0
    assert builder.aot_schedule is False
    assert builder.aot_sliding_window is None
    assert builder.use_full_cuda_graph is False
    assert builder.max_cudagraph_size == 0

    common = _make_common_attn()
    metadata = builder.build(0, common)
    assert metadata.num_actual_tokens == 5
    assert metadata.max_query_len == 5
    assert metadata.max_seq_len == 5
    assert metadata.use_cascade is False
    assert metadata.common_prefix_len == 0
    assert metadata.max_num_splits == 0
    assert metadata.causal is True
    assert metadata.cu_prefix_query_lens is None
    assert metadata.prefix_kv_lens is None
    assert metadata.suffix_kv_lens is None
    assert metadata.scheduler_metadata is None
    assert metadata.prefix_scheduler_metadata is None
    assert metadata.dcp_context_kv_lens is None
    assert metadata.max_dcp_context_kv_len == 0
    # aot_sliding_window is populated on the first build even when aot is off.
    assert builder.aot_sliding_window == (-1, -1)

    fast = builder.build(0, common, fast_build=True)
    assert fast.use_cascade is False
    assert fast.scheduler_metadata is None


def test_metadata_builder_aot_sliding_window(env, monkeypatch):
    mods = env.mods
    common = _make_common_attn()

    builder, _config, _spec = _make_builder(env, monkeypatch)
    builder.aot_schedule = True
    monkeypatch.setattr(
        mods.attention, "_get_sliding_window_configs", lambda cfg: {(-3, 0)}
    )
    builder.build(0, common)
    assert builder.aot_sliding_window == (-3, 0)

    builder_none, _c, _s = _make_builder(env, monkeypatch)
    builder_none.aot_schedule = True
    monkeypatch.setattr(
        mods.attention, "_get_sliding_window_configs", lambda cfg: {None}
    )
    builder_none.build(0, common)
    assert builder_none.aot_sliding_window == (-1, -1)

    builder_multi, _c2, _s2 = _make_builder(env, monkeypatch)
    builder_multi.aot_schedule = True
    monkeypatch.setattr(
        mods.attention,
        "_get_sliding_window_configs",
        lambda cfg: {(-1, -1), (4, 0)},
    )
    builder_multi.build(0, common)
    assert builder_multi.aot_schedule is False
    assert builder_multi.aot_sliding_window == (-1, -1)


def test_metadata_builder_build_cascade(env, monkeypatch):
    builder, _config, _spec = _make_builder(env, monkeypatch)
    metadata = builder.build(32, _make_common_attn())
    assert metadata.use_cascade is True
    assert metadata.common_prefix_len == 32
    assert metadata.cu_prefix_query_lens is not None
    assert metadata.prefix_kv_lens is not None
    assert metadata.suffix_kv_lens is not None
    assert metadata.scheduler_metadata is None


def test_metadata_builder_build_dcp(env, monkeypatch):
    mods = env.mods
    group = types.SimpleNamespace(world_size=2, rank_in_group=1)
    builder, _config, _spec = _make_builder(env, monkeypatch, dcp_group=group)
    assert builder.dcp_world_size == 2
    assert builder.dcp_rank == 1
    monkeypatch.setattr(
        mods.attention,
        "get_dcp_local_seq_lens",
        lambda lens, world, rank, interleave: lens,
    )
    metadata = builder.build(0, _make_common_attn(max_seq_len=5))
    assert metadata.dcp_context_kv_lens is not None
    # ceil(5 / (2 * 1)) * 1 == 3
    assert metadata.max_dcp_context_kv_len == 3
    assert metadata.scheduler_metadata is None
    assert metadata.use_cascade is False


def test_metadata_builder_full_cudagraph_without_aot(env, monkeypatch):
    config = _make_vllm_config(full_cudagraph=True, max_cudagraph_size=16)
    builder, _config, _spec = _make_builder(env, monkeypatch, vllm_config=config)
    assert builder.use_full_cuda_graph is True
    assert builder.max_num_splits == 0
    metadata = builder.build(0, _make_common_attn(num_actual_tokens=5))
    assert metadata.max_num_splits == 0


def test_metadata_builder_use_cascade_delegates(env, monkeypatch):
    builder, _config, _spec = _make_builder(env, monkeypatch)
    spy = mock.MagicMock(return_value=True)
    monkeypatch.setattr(env.mods.attention, "use_cascade_attention", spy)
    assert builder.use_cascade_attention(common_prefix_len=300) is True
    spy.assert_called_once_with(common_prefix_len=300)


# ---------------------------------------------------------------------------
# attention impl: cache split / init / kv cache update
# ---------------------------------------------------------------------------


def test_split_kv_cache_shapes(env):
    k_cache, v_cache = env.mods.attention._split_kv_cache(
        torch.zeros(4, 8, 16, 128), 64
    )
    assert tuple(k_cache.shape) == (4, 16, 8, 64)
    assert tuple(v_cache.shape) == (4, 16, 8, 64)


@pytest.mark.parametrize(
    "kv_cache",
    [torch.zeros(4, 8, 16, 127), torch.zeros(4, 8, 128)],
)
def test_split_kv_cache_rejects_bad_layout(env, kv_cache):
    with pytest.raises(ValueError):
        env.mods.attention._split_kv_cache(kv_cache, 64)


def test_impl_init_defaults(env):
    impl = _make_impl(env)
    assert impl.num_heads == 8
    assert impl.head_size == 64
    assert impl.scale == 0.125
    assert impl.num_kv_heads == 8
    assert impl.alibi_slopes is None
    assert impl.sliding_window == (-1, -1)
    assert impl.kv_cache_dtype == "auto"
    assert impl.logits_soft_cap == 0
    assert impl.kv_sharing_target_layer_name is None
    assert impl.num_queries_per_kv == 1
    assert impl.attn_type == env.mods.attention.AttentionType.DECODER
    assert impl.vllm_flash_attn_version == 2
    assert impl.batch_invariant_enabled is False
    assert impl.supports_quant_query_input is False
    assert impl.can_return_lse_for_decode is True


def test_impl_init_variants(env):
    mods = env.mods
    impl = _make_impl(
        env,
        alibi_slopes=[0.1, 0.2],
        sliding_window=4,
        logits_soft_cap=30.0,
        kv_sharing_target_layer_name="layer.kv",
    )
    assert tuple(impl.alibi_slopes.shape) == (2,)
    assert impl.sliding_window == (3, 0)
    assert impl.logits_soft_cap == 30.0
    assert impl.kv_sharing_target_layer_name == "layer.kv"

    encoder = _make_impl(
        env,
        attn_type=mods.attention.AttentionType.ENCODER_ONLY,
        sliding_window=4,
    )
    assert encoder.sliding_window == (3, 3)


def test_impl_init_rejects_quantized_kv_cache(env):
    with pytest.raises(NotImplementedError):
        _make_impl(env, kv_cache_dtype="fp8_e4m3")


def test_do_kv_cache_update_encoder_is_noop(env, monkeypatch):
    mods = env.mods
    writer = mock.MagicMock()
    monkeypatch.setattr(
        sys.modules["vllm.v1.attention.ops.triton_reshape_and_cache_flash"],
        "triton_reshape_and_cache_flash",
        writer,
    )
    layer = _make_layer()
    for attn_type in (
        mods.attention.AttentionType.ENCODER_ONLY,
        mods.attention.AttentionType.ENCODER,
    ):
        impl = _make_impl(env, attn_type=attn_type)
        assert (
            impl.do_kv_cache_update(
                layer,
                torch.zeros(5, 8, 64),
                torch.zeros(5, 8, 64),
                _kv_cache(),
                torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
            )
            is None
        )
    writer.assert_not_called()


def test_do_kv_cache_update_decoder_writes(env, monkeypatch):
    mods = env.mods
    writer = mock.MagicMock()
    monkeypatch.setattr(
        sys.modules["vllm.v1.attention.ops.triton_reshape_and_cache_flash"],
        "triton_reshape_and_cache_flash",
        writer,
    )
    impl = _make_impl(env)
    key = torch.zeros(5, 8, 64)
    value = torch.zeros(5, 8, 64)
    kv_cache = _kv_cache()
    slot_mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    impl.do_kv_cache_update(_make_layer(), key, value, kv_cache, slot_mapping)
    writer.assert_called_once()
    args = writer.call_args.args
    assert len(args) == 8
    assert args[0] is key
    assert args[1] is value
    assert tuple(args[2].shape) == (4, 16, 8, 64)
    assert args[4] is slot_mapping
    assert args[5] == "auto"


# ---------------------------------------------------------------------------
# attention impl: forward
# ---------------------------------------------------------------------------


def test_forward_requires_output(env):
    impl = _make_impl(env)
    query, key, value = _make_qkv()
    with pytest.raises(AssertionError):
        impl.forward(
            _make_layer(), query, key, value, _kv_cache(), _make_attn_metadata(),
            output=None,
        )


@pytest.mark.parametrize(
    "output_kwargs",
    [{"output_scale": torch.ones(1)}, {"output_block_scale": torch.ones(1)}],
)
def test_forward_rejects_fused_output_quant(env, output_kwargs):
    impl = _make_impl(env)
    query, key, value = _make_qkv()
    with pytest.raises(NotImplementedError):
        impl.forward(
            _make_layer(),
            query,
            key,
            value,
            _kv_cache(),
            _make_attn_metadata(),
            output=torch.zeros(5, 8, 64),
            **output_kwargs,
        )


def test_forward_profiling_fills_zero(env):
    impl = _make_impl(env)
    query, key, value = _make_qkv()
    output = mock.MagicMock()
    result = impl.forward(
        _make_layer(), query, key, value, _kv_cache(), None, output=output
    )
    output.fill_.assert_called_once_with(0)
    assert result is output.fill_.return_value


def test_forward_encoder_calls_flash(env, monkeypatch):
    mods = env.mods
    flash = mock.MagicMock(return_value=None)
    monkeypatch.setattr(mods.attention, "flash_attn_varlen_func", flash)
    impl = _make_impl(env, attn_type=mods.attention.AttentionType.ENCODER_ONLY)
    query, key, value = _make_qkv()
    output = torch.zeros(5, 8, 64)
    result = impl.forward(
        _make_layer(), query, key, value, _kv_cache(), _make_attn_metadata(),
        output=output,
    )
    assert tuple(result.shape) == (5, 8, 64)
    kwargs = flash.call_args.kwargs
    assert kwargs["causal"] is False
    assert kwargs["fa_version"] == 2
    assert kwargs["cu_seqlens_k"] is kwargs["cu_seqlens_q"]
    assert kwargs["window_size"] == (-1, -1)
    assert "num_splits" not in kwargs


def test_forward_encoder_rejects_fp8_cache(env, monkeypatch):
    mods = env.mods
    monkeypatch.setattr(
        mods.attention, "flash_attn_varlen_func", mock.MagicMock(return_value=None)
    )
    impl = _make_impl(env, attn_type=mods.attention.AttentionType.ENCODER_ONLY)
    impl.kv_cache_dtype = "fp8"
    query, key, value = _make_qkv()
    with pytest.raises(NotImplementedError):
        impl.forward(
            _make_layer(), query, key, value, _kv_cache(), _make_attn_metadata(),
            output=torch.zeros(5, 8, 64),
        )


@pytest.mark.parametrize("causal", [True, False])
def test_forward_decoder_calls_flash(env, monkeypatch, causal):
    mods = env.mods
    flash = mock.MagicMock(return_value=None)
    monkeypatch.setattr(mods.attention, "flash_attn_varlen_func", flash)
    impl = _make_impl(env)
    query, key, value = _make_qkv()
    output = torch.zeros(5, 8, 64)
    result = impl.forward(
        _make_layer(),
        query,
        key,
        value,
        _kv_cache(),
        _make_attn_metadata(causal=causal),
        output=output,
    )
    assert result is output
    kwargs = flash.call_args.kwargs
    assert kwargs["causal"] is causal
    assert kwargs["num_splits"] == 0
    assert kwargs["s_aux"] is None
    assert kwargs["softcap"] == 0
    assert kwargs["softmax_scale"] == 0.125
    assert kwargs["fa_version"] == 2
    assert kwargs["scheduler_metadata"] is None
    assert kwargs["q"].shape[0] == 5


def test_forward_cascade_dispatches_cascade_helper(env, monkeypatch):
    mods = env.mods
    flash = mock.MagicMock(return_value=None)
    cascade = mock.MagicMock(return_value=None)
    monkeypatch.setattr(mods.attention, "flash_attn_varlen_func", flash)
    monkeypatch.setattr(mods.attention, "cascade_attention", cascade)
    impl = _make_impl(env)
    query, key, value = _make_qkv()
    output = torch.zeros(5, 8, 64)
    metadata = _make_attn_metadata(
        use_cascade=True,
        common_prefix_len=32,
        cu_prefix_query_lens=torch.tensor([0, 5], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([32], dtype=torch.int32),
        suffix_kv_lens=torch.tensor([5], dtype=torch.int32),
    )
    result = impl.forward(
        _make_layer(), query, key, value, _kv_cache(), metadata, output=output
    )
    assert result is output
    cascade.assert_called_once()
    flash.assert_not_called()


def test_forward_dcp_merges_context_and_query(env, monkeypatch):
    mods = env.mods
    group = types.SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        all_gather=lambda tensor, dim: tensor,
    )
    monkeypatch.setattr(mods.attention, "get_dcp_group", lambda: group)

    context_out = torch.zeros(5, 8, 64)
    query_out = torch.zeros(5, 8, 64)
    context_lse = torch.zeros(8, 5)
    query_lse = torch.zeros(8, 5)
    flash = mock.MagicMock(
        side_effect=[(context_out, context_lse), (query_out, query_lse)]
    )
    monkeypatch.setattr(mods.attention, "flash_attn_varlen_func", flash)
    cp_lse = mock.MagicMock(return_value=(context_out, torch.zeros(5, 8)))
    monkeypatch.setattr(mods.attention, "cp_lse_ag_out_rs", cp_lse)
    merge = mock.MagicMock()
    monkeypatch.setattr(mods.attention, "merge_attn_states", merge)

    impl = _make_impl(env)
    impl.dcp_world_size = 2
    query, key, value = _make_qkv()
    output = torch.zeros(5, 8, 64)
    metadata = _make_attn_metadata(
        dcp_context_kv_lens=torch.tensor([5], dtype=torch.int32),
        max_dcp_context_kv_len=3,
    )
    result = impl.forward(
        _make_layer(), query, key, value, _kv_cache(), metadata, output=output
    )
    assert result is output
    assert flash.call_count == 2
    first_kwargs = flash.call_args_list[0].kwargs
    assert first_kwargs["causal"] is False
    assert first_kwargs["return_softmax_lse"] is True
    assert first_kwargs["seqused_k"] is metadata.dcp_context_kv_lens
    cp_lse.assert_called_once()
    merge.assert_called_once()


# ---------------------------------------------------------------------------
# cascade heuristics + cascade attention helper
# ---------------------------------------------------------------------------


def _cascade_kwargs(**overrides):
    kwargs = dict(
        common_prefix_len=300,
        query_lens=np.ones(8, dtype=np.int64),
        num_query_heads=16,
        num_kv_heads=8,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=8,
        dcp_world_size=1,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "overrides",
    [
        {"common_prefix_len": 100},
        {"use_alibi": True},
        {"use_sliding_window": True},
        {"use_local_attention": True},
        {"query_lens": np.ones(3, dtype=np.int64)},
        {"dcp_world_size": 2},
    ],
)
def test_use_cascade_attention_early_false(env, overrides):
    assert env.mods.attention.use_cascade_attention(**_cascade_kwargs(**overrides)) is False


def test_use_cascade_attention_true_without_flash_decoding(env):
    kwargs = _cascade_kwargs(query_lens=np.array([3, 4, 5, 6, 7, 8, 9, 10]))
    assert env.mods.attention.use_cascade_attention(**kwargs) is True


def test_use_cascade_attention_flash_decoding_perf_model(env):
    fn = env.mods.attention.use_cascade_attention
    # cascade_time (64) < flash_decoding_time (320)
    assert (
        fn(
            common_prefix_len=512,
            query_lens=np.ones(10, dtype=np.int64),
            num_query_heads=16,
            num_kv_heads=8,
            use_alibi=False,
            use_sliding_window=False,
            use_local_attention=False,
            num_sms=1,
            dcp_world_size=1,
        )
        is True
    )
    # cascade_time (2) >= flash_decoding_time (1)
    assert (
        fn(
            common_prefix_len=256,
            query_lens=np.ones(8, dtype=np.int64),
            num_query_heads=8,
            num_kv_heads=1,
            use_alibi=False,
            use_sliding_window=False,
            use_local_attention=False,
            num_sms=64,
            dcp_world_size=1,
        )
        is False
    )


def _cascade_call_kwargs(**overrides):
    kwargs = dict(
        output=torch.zeros(5, 8, 64),
        query=torch.zeros(5, 8, 64),
        key_cache=torch.zeros(32, 16, 8, 64),
        value_cache=torch.zeros(32, 16, 8, 64),
        cu_query_lens=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
        max_query_len=1,
        cu_prefix_query_lens=torch.tensor([0, 5], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([32], dtype=torch.int32),
        suffix_kv_lens=torch.tensor([5], dtype=torch.int32),
        max_kv_len=64,
        softmax_scale=0.1,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        block_table=torch.zeros(5, 4, dtype=torch.int32),
        common_prefix_len=32,
        fa_version=2,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("with_descale", [False, True])
def test_cascade_attention_merges_prefix_and_suffix(env, monkeypatch, with_descale):
    mods = env.mods
    flash = mock.MagicMock(
        side_effect=[
            (torch.zeros(5, 8, 64), torch.zeros(8, 5)),
            (torch.zeros(5, 8, 64), torch.zeros(8, 5)),
        ]
    )
    monkeypatch.setattr(mods.attention, "flash_attn_varlen_func", flash)
    merge = mock.MagicMock()
    monkeypatch.setattr(mods.attention, "merge_attn_states", merge)

    kwargs = _cascade_call_kwargs()
    if with_descale:
        kwargs["q_descale"] = torch.ones(1, 8)
        kwargs["k_descale"] = torch.ones(1, 8)
        kwargs["v_descale"] = torch.ones(1, 8)
    assert mods.attention.cascade_attention(**kwargs) is None
    assert flash.call_count == 2
    prefix_kwargs = flash.call_args_list[0].kwargs
    assert prefix_kwargs["causal"] is False
    assert prefix_kwargs["return_softmax_lse"] is True
    suffix_kwargs = flash.call_args_list[1].kwargs
    assert suffix_kwargs["causal"] is True
    if with_descale:
        assert tuple(prefix_kwargs["q_descale"].shape) == (1, 8)
        assert tuple(suffix_kwargs["q_descale"].shape) == (5, 8)
    else:
        assert prefix_kwargs["q_descale"] is None
    merge.assert_called_once()


@pytest.mark.parametrize(
    "overrides",
    [
        {"alibi_slopes": torch.zeros(1)},
        {"sliding_window": (1, 0)},
        {"common_prefix_len": 33},
    ],
)
def test_cascade_attention_validations(env, overrides):
    with pytest.raises(AssertionError):
        env.mods.attention.cascade_attention(**_cascade_call_kwargs(**overrides))


def test_attention_module_requires_flaggems_vllm(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    with pytest.raises(ImportError):
        _exec_file_module("_cov_missing_attention", env.impl_dir / "attention.py")


# ---------------------------------------------------------------------------
# FlagGemsBackend registration / delegation
# ---------------------------------------------------------------------------


def test_backend_name(env):
    assert env.mods.FlagGemsBackend().name == "flagos"


def test_backend_is_available_caches_success(env, monkeypatch):
    backend = env.mods.FlagGemsBackend()
    cls = env.mods.FlagGemsBackend
    old = cls._available
    cls._available = None
    try:
        assert backend.is_available() is True
        assert cls._available is True
        # second call is served from the cache, even if the import would fail
        monkeypatch.setitem(sys.modules, "flag_gems", None)
        assert backend.is_available() is True
    finally:
        cls._available = old


def test_backend_is_available_reports_import_error(env, monkeypatch):
    backend = env.mods.FlagGemsBackend()
    cls = env.mods.FlagGemsBackend
    old = cls._available
    cls._available = None
    try:
        monkeypatch.setitem(sys.modules, "flag_gems", None)
        assert backend.is_available() is False
        assert cls._available is False
    finally:
        cls._available = old


def test_backend_quantization_delegation(env, monkeypatch):
    mods = env.mods
    backend = mods.FlagGemsBackend()
    vllm_impl = mock.MagicMock(return_value=("out", "scale"))
    triton_impl = mock.MagicMock(return_value=(1, 2))
    monkeypatch.setattr(
        mods.quantization, "dynamic_per_token_quant_int8_flaggems_vllm", vllm_impl
    )
    monkeypatch.setattr(
        mods.quantization, "dynamic_per_token_quant_int8_flaggems_triton", triton_impl
    )
    x = torch.zeros(2, 4)
    assert backend.dynamic_per_token_quant_int8(x) == ("out", "scale")
    vllm_impl.assert_called_once_with(x)
    assert backend.dynamic_per_token_quant_int8_triton(x) == (1, 2)
    triton_impl.assert_called_once_with(x)


def test_backend_activation_delegation(env, monkeypatch):
    mods = env.mods
    backend = mods.FlagGemsBackend()
    silu = mock.MagicMock(return_value="silu")
    gelu = mock.MagicMock(return_value="gelu")
    monkeypatch.setattr(mods.activation, "silu_and_mul_flaggems", silu)
    monkeypatch.setattr(mods.activation, "gelu_and_mul_flaggems", gelu)
    x = torch.zeros(2, 4)
    assert backend.silu_and_mul(None, x) == "silu"
    silu.assert_called_once_with(None, x)
    assert backend.gelu_and_mul(None, x) == "gelu"
    gelu.assert_called_once_with(None, x)


def test_backend_normalization_and_rotary_delegation(env, monkeypatch):
    mods = env.mods
    backend = mods.FlagGemsBackend()
    norm = mock.MagicMock(return_value="norm")
    rope = mock.MagicMock(return_value=("q", "k"))
    monkeypatch.setattr(env.stubs.fake_modules["normalization"], "rms_norm_flaggems", norm)
    monkeypatch.setattr(
        env.stubs.fake_modules["rotary"], "rotary_embedding_flaggems", rope
    )
    x = torch.zeros(2, 4)
    residual = torch.zeros(2, 4)
    assert backend.rms_norm(None, x, residual) == "norm"
    norm.assert_called_once_with(None, x, residual)
    assert backend.rotary_embedding(
        None, x, x, x, x, torch.zeros(2, dtype=torch.int64), rotary_interleaved=True, inplace=False
    ) == ("q", "k")
    kwargs = rope.call_args.kwargs
    assert kwargs["rotary_interleaved"] is True
    assert kwargs["inplace"] is False


def test_backend_attention_backend_requires_cuda(env, monkeypatch):
    backend = env.mods.FlagGemsBackend()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        backend.attention_backend()


def test_backend_attention_backend_rejects_mla(env, monkeypatch):
    backend = env.mods.FlagGemsBackend()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(NotImplementedError):
        backend.attention_backend(use_mla=True)


def test_backend_attention_backend_rejects_sparse_without_mla(env, monkeypatch):
    backend = env.mods.FlagGemsBackend()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError):
        backend.attention_backend(use_mla=False, use_sparse=True)


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_backend_attention_backend_flag_env_selects_flaggems(env, monkeypatch, value):
    backend = env.mods.FlagGemsBackend()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", value)
    assert backend.attention_backend() == FLAGGEMS_ATTN_PATH


def test_backend_attention_backend_defaults_to_triton(env, monkeypatch):
    backend = env.mods.FlagGemsBackend()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    assert backend.attention_backend() == TRITON_ATTN_PATH


def test_backend_fused_moe_delegations(env, monkeypatch):
    backend = env.mods.FlagGemsBackend()
    mod = env.stubs.fake_modules["fused_moe"]

    align = mock.MagicMock(return_value=(1, 2, 3))
    monkeypatch.setattr(mod, "moe_align_block_size_flaggems", align)
    topk_ids = torch.zeros(4, 2, dtype=torch.int32)
    assert backend.moe_align_block_size(topk_ids, 16, 4) == (1, 2, 3)
    assert align.call_args.args == (topk_ids, 16, 4, None, False, False)

    total = mock.MagicMock()
    monkeypatch.setattr(mod, "moe_sum_flaggems", total)
    inp = torch.zeros(4, 2, 8)
    out = torch.zeros(4, 8)
    assert backend.moe_sum(inp, out) is None
    total.assert_called_once_with(inp, out)

    topk = mock.MagicMock(return_value=("weights", "indices"))
    monkeypatch.setattr(mod, "topk_softmax_flaggems", topk)
    weights = torch.zeros(4, 2)
    indices = torch.zeros(4, 2, dtype=torch.int32)
    token_expert = torch.zeros(4, 2, dtype=torch.int32)
    gating = torch.zeros(4, 4)
    assert backend.topk_softmax(weights, indices, token_expert, gating, True) == (
        "weights",
        "indices",
    )
    assert topk.call_args.args == (weights, indices, token_expert, gating, True)

    invoke = mock.MagicMock()
    monkeypatch.setattr(mod, "invoke_fused_moe_triton_kernel_flaggems", invoke)
    bias = torch.zeros(4, 8)
    tensors = [torch.zeros(4, 8) for _ in range(3)]
    backend.invoke_fused_moe_triton_kernel(
        *tensors,
        None,
        None,
        torch.zeros(4, 2),
        torch.zeros(8, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        False,
        2,
        None,
        "fp16",
        False,
        True,
        False,
        False,
        True,
        block_shape=(1, 8),
        B_bias=bias,
    )
    invoke.assert_called_once()
    assert invoke.call_args.kwargs == {"block_shape": (1, 8), "B_bias": bias}

    grouped = mock.MagicMock(return_value="topk")
    monkeypatch.setattr(mod, "grouped_topk_flaggems", grouped)
    scores = torch.zeros(4, 8)
    assert backend.grouped_topk(scores, 2, 1, 2, True, 1.0, None, 0) == "topk"
    assert grouped.call_args.args == (scores, 2, 1, 2, True, 1.0, None, 0)


# ---------------------------------------------------------------------------
# MLA backend
# ---------------------------------------------------------------------------


def _mla_kwargs(**overrides):
    kwargs = dict(
        num_heads=8,
        head_size=576,
        scale=0.1,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="DECODER",
        kv_sharing_target_layer_name=None,
    )
    kwargs.update(overrides)
    return kwargs


def _make_mla_impl(env, **overrides):
    kwargs = _mla_kwargs(**overrides)
    return env.mods.mla.MLAFLImpl(**kwargs)


def test_mla_backend_accessors(env):
    mods = env.mods
    assert mods.mla.MLAFLBackend.get_name() == "MLAFL"
    assert mods.mla.MLAFLBackend.get_impl_cls() is mods.mla.MLAFLImpl


@pytest.mark.parametrize(
    "overrides",
    [
        {"alibi_slopes": [0.1]},
        {"sliding_window": 4},
        {"logits_soft_cap": 30.0},
        {"attn_type": "ENCODER_ONLY"},
        {"kv_cache_dtype": "fp8"},
    ],
)
def test_mla_impl_init_rejects_unsupported(env, overrides):
    with pytest.raises(NotImplementedError):
        _make_mla_impl(env, **overrides)


def test_mla_impl_init_ok(env):
    impl = _make_mla_impl(
        env,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        _pad_v=True,
    )
    assert impl.kv_cache_dtype == "auto"
    assert impl.kv_lora_rank == 512
    assert impl._pad_v is True
    assert impl.num_heads == 8


def test_mla_flash_attn_diff_headdims_no_pad(env, monkeypatch):
    mods = env.mods
    flash = mock.MagicMock(return_value=torch.zeros(5, 8, 64))
    monkeypatch.setattr(mods.mla, "flash_attn_varlen_func", flash)
    impl = _make_mla_impl(env)
    query = torch.zeros(5, 8, 128)
    key = torch.zeros(5, 8, 128)
    value = torch.zeros(5, 8, 64)
    result = impl._flash_attn_varlen_diff_headdims(
        query,
        key,
        value,
        return_softmax_lse=False,
        softmax_scale=0.1,
        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
        max_seqlen_q=5,
    )
    assert result is flash.return_value
    kwargs = flash.call_args.kwargs
    assert kwargs["v"] is value
    assert kwargs["return_softmax_lse"] is False
    assert kwargs["softmax_scale"] == 0.1


def test_mla_flash_attn_diff_headdims_pads_v(env, monkeypatch):
    mods = env.mods
    flash = mock.MagicMock(return_value=torch.zeros(5, 8, 128))
    monkeypatch.setattr(mods.mla, "flash_attn_varlen_func", flash)
    impl = _make_mla_impl(env, _pad_v=True)
    query = torch.zeros(5, 8, 128)
    key = torch.zeros(5, 8, 128)
    value = torch.zeros(5, 8, 64)
    impl._flash_attn_varlen_diff_headdims(query, key, value)
    kwargs = flash.call_args.kwargs
    assert kwargs["v"] is not value
    assert tuple(kwargs["v"].shape) == (5, 8, 128)


def test_mla_flash_attn_diff_headdims_unpacks_lse(env, monkeypatch):
    mods = env.mods
    attn_out = torch.zeros(5, 8, 64)
    lse = torch.zeros(8, 5)
    flash = mock.MagicMock(return_value=(attn_out, lse))
    monkeypatch.setattr(mods.mla, "flash_attn_varlen_func", flash)
    impl = _make_mla_impl(env)
    query = torch.zeros(5, 8, 128)
    key = torch.zeros(5, 8, 128)
    value = torch.zeros(5, 8, 64)
    out_with_lse, returned_lse = impl._flash_attn_varlen_diff_headdims(
        query, key, value, return_softmax_lse=True
    )
    assert out_with_lse is attn_out
    assert returned_lse is lse

    out_only = impl._flash_attn_varlen_diff_headdims(
        query, key, value, return_softmax_lse=False
    )
    assert out_only is attn_out


def _mla_decode_metadata():
    return types.SimpleNamespace(
        decode=types.SimpleNamespace(
            block_table=torch.zeros(2, 4, dtype=torch.int32),
            seq_lens=torch.tensor([5, 5], dtype=torch.int32),
        )
    )


def test_mla_forward_decode_basic(env, monkeypatch):
    mods = env.mods
    flash_mla = mock.MagicMock(return_value=torch.zeros(2, 8, 512))
    monkeypatch.setattr(mods.mla, "flash_mla", flash_mla)
    impl = _make_mla_impl(env)
    q = torch.zeros(2, 8, 576)
    kv_cache = torch.zeros(16, 16, 576)
    out, lse = impl._forward_decode(q, kv_cache, _mla_decode_metadata(), None)
    assert out is flash_mla.return_value
    assert tuple(lse.shape) == (2, 8)
    args = flash_mla.call_args.args
    assert args[4] == 16  # PAGE_SIZE == kv_cache.size(1)
    assert args[5] == 2  # batch size


def test_mla_forward_decode_tuple_q(env, monkeypatch):
    mods = env.mods
    flash_mla = mock.MagicMock(return_value=torch.zeros(2, 8, 512))
    monkeypatch.setattr(mods.mla, "flash_mla", flash_mla)
    impl = _make_mla_impl(env)
    q_nope = torch.zeros(2, 8, 512)
    q_pe = torch.zeros(2, 8, 64)
    out, _lse = impl._forward_decode(
        (q_nope, q_pe), torch.zeros(16, 16, 576), _mla_decode_metadata(), None
    )
    assert out is flash_mla.return_value
    assert tuple(flash_mla.call_args.args[0].shape) == (2, 8, 576)


def test_mla_forward_decode_rejects_fp8(env):
    impl = object.__new__(env.mods.mla.MLAFLImpl)
    impl.kv_cache_dtype = "fp8"
    with pytest.raises(NotImplementedError):
        impl._forward_decode(
            torch.zeros(2, 8, 576),
            torch.zeros(16, 16, 576),
            _mla_decode_metadata(),
            None,
        )


def test_mla_forward_decode_assertions(env, monkeypatch):
    mods = env.mods
    impl = _make_mla_impl(env)
    monkeypatch.setattr(mods.mla, "flash_mla", mock.MagicMock())
    with pytest.raises(AssertionError):
        impl._forward_decode(
            torch.zeros(2, 8, 576), torch.zeros(0, 16, 576), _mla_decode_metadata(), None
        )
    empty_metadata = types.SimpleNamespace(decode=None)
    with pytest.raises(AssertionError):
        impl._forward_decode(
            torch.zeros(2, 8, 576), torch.zeros(16, 16, 576), empty_metadata, None
        )


def test_mla_module_requires_flaggems_vllm(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    with pytest.raises(ImportError):
        _exec_file_module("_cov_missing_mla", env.impl_dir / "mla.py")


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------


def test_quantization_flaggems_vllm_delegates(env, monkeypatch):
    mods = env.mods
    scaled = mock.MagicMock(
        return_value=(torch.zeros(2, 4, dtype=torch.int8), torch.zeros(2, 1), None)
    )
    monkeypatch.setattr(
        sys.modules["flaggems_vllm.ops.scaled_int8_quant"],
        "scaled_int8_quant",
        scaled,
    )
    x = torch.zeros(2, 4)
    out, scale = mods.quantization.dynamic_per_token_quant_int8_flaggems_vllm(x)
    assert out is scaled.return_value[0]
    assert scale is scaled.return_value[1]
    assert scaled.call_args.args[0] is x
    assert scaled.call_args.kwargs == {
        "scale": None,
        "azp": None,
        "symmetric": True,
    }


def test_quantization_flaggems_vllm_missing_dependency(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "flaggems_vllm.ops.scaled_int8_quant", None)
    with pytest.raises(ImportError):
        env.mods.quantization.dynamic_per_token_quant_int8_flaggems_vllm(
            torch.zeros(2, 4)
        )


def test_quantization_triton_rejects_bad_input(env):
    fn = env.mods.quantization.dynamic_per_token_quant_int8_flaggems_triton
    with pytest.raises(ValueError, match="2D"):
        fn(torch.zeros(4))
    with pytest.raises(TypeError, match="floating point"):
        fn(torch.zeros(2, 4, dtype=torch.int8))
    with pytest.raises(ValueError, match="hidden_size"):
        fn(torch.zeros(2, 0))


def test_quantization_triton_launches_kernel(env, monkeypatch):
    mods = env.mods
    kernel = mock.MagicMock()
    monkeypatch.setattr(
        mods.quantization, "_dynamic_per_token_quant_int8_kernel", kernel
    )
    monkeypatch.setattr(
        mods.quantization,
        "torch_device_fn",
        types.SimpleNamespace(device=lambda _dev: contextlib.nullcontext()),
    )
    x = torch.zeros(3, 7)
    out, scale = mods.quantization.dynamic_per_token_quant_int8_flaggems_triton(x)
    assert out.dtype == torch.int8
    assert tuple(out.shape) == (3, 7)
    assert scale.dtype == torch.float32
    assert tuple(scale.shape) == (3, 1)
    kernel.__getitem__.assert_called_once_with((3,))
    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    assert launch.call_args.kwargs == {"hidden_size": 7, "BLOCK_SIZE": 8}


def test_quantization_kernel_bodies_execute_under_stub_triton(env):
    """The hermetic triton stub makes @triton.jit/@libentry identity, so the
    kernel bodies can be walked with mocked ``tl`` operations (no compilation).
    """
    mods = env.mods
    assert mods.quantization._round_half_to_even(1.5) is not None
    mods.quantization._dynamic_per_token_quant_int8_kernel(
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
        hidden_size=8,
        BLOCK_SIZE=16,
    )


# ---------------------------------------------------------------------------
# activation
# ---------------------------------------------------------------------------


def _set_platform(monkeypatch, env, device_type="cuda", vendor_name=None):
    monkeypatch.setattr(
        env.mods.activation,
        "current_platform",
        types.SimpleNamespace(device_type=device_type, vendor_name=vendor_name),
    )


def test_silu_and_mul_uses_flaggems(env, monkeypatch):
    mods = env.mods
    _set_platform(monkeypatch, env)
    gems = mock.MagicMock(return_value=torch.zeros(2, 4))
    monkeypatch.setattr(sys.modules["flaggems_vllm"], "silu_and_mul", gems)
    x = torch.zeros(2, 8)
    result = mods.activation.silu_and_mul_flaggems(None, x)
    assert result is gems.return_value
    first, second = gems.call_args.args
    assert tuple(first.shape) == (2, 4)
    assert tuple(second.shape) == (2, 4)


def test_silu_and_mul_iluvatar_branch(env, monkeypatch):
    mods = env.mods
    _set_platform(monkeypatch, env, vendor_name="iluvatar")
    iluvatar = mock.MagicMock(return_value="iluvatar-out")
    monkeypatch.setattr(
        sys.modules["vllm_fl.dispatch.backends.vendor.iluvatar.activation"],
        "_iluvatar_gems_silu_and_mul",
        iluvatar,
    )
    x = torch.zeros(2, 8)
    assert mods.activation.silu_and_mul_flaggems(None, x) == "iluvatar-out"
    iluvatar.assert_called_once_with(x)


def _patch_compiling(monkeypatch, compiling):
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is None:
        compiler_mod = types.SimpleNamespace(is_compiling=lambda: False)
        monkeypatch.setattr(torch, "compiler", compiler_mod, raising=False)
    monkeypatch.setattr(compiler_mod, "is_compiling", lambda: compiling)


def test_silu_and_mul_musa_compiled_path(env, monkeypatch):
    mods = env.mods
    _set_platform(monkeypatch, env, device_type="musa")
    musa = mock.MagicMock(return_value="musa-out")
    monkeypatch.setattr(mods.activation, "musa_silu_and_mul", musa)
    _patch_compiling(monkeypatch, True)
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)
    x = types.SimpleNamespace(device=types.SimpleNamespace(type="musa"))
    assert mods.activation.silu_and_mul_flaggems(None, x) == "musa-out"
    musa.assert_called_once_with(x)


def test_silu_and_mul_musa_eager_falls_back(env, monkeypatch):
    mods = env.mods
    _set_platform(monkeypatch, env, device_type="musa")
    musa = mock.MagicMock()
    monkeypatch.setattr(mods.activation, "musa_silu_and_mul", musa)
    gems = mock.MagicMock(return_value="gems")
    monkeypatch.setattr(sys.modules["flaggems_vllm"], "silu_and_mul", gems)
    x = mock.MagicMock(shape=(2, 8), device=types.SimpleNamespace(type="musa"))
    assert mods.activation.silu_and_mul_flaggems(None, x) == "gems"
    musa.assert_not_called()


def test_silu_and_mul_missing_flaggems(env, monkeypatch):
    _set_platform(monkeypatch, env)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    with pytest.raises(ImportError):
        env.mods.activation.silu_and_mul_flaggems(None, torch.zeros(2, 4))


@pytest.mark.parametrize(
    "obj, expected",
    [(None, "none"), (types.SimpleNamespace(approximate="tanh"), "tanh")],
)
def test_gelu_and_mul_approximate(env, monkeypatch, obj, expected):
    gems = mock.MagicMock(return_value="gelu")
    monkeypatch.setattr(sys.modules["flaggems_vllm"], "gelu_and_mul", gems)
    x = torch.zeros(4, 10)
    assert env.mods.activation.gelu_and_mul_flaggems(obj, x) == "gelu"
    first, second, approximate = gems.call_args.args
    assert tuple(first.shape) == (4, 5)
    assert tuple(second.shape) == (4, 5)
    assert approximate == expected


# ---------------------------------------------------------------------------
# W8A8 MoE provider
# ---------------------------------------------------------------------------


def _w8a8_kwargs(**overrides):
    kwargs = dict(
        hidden_states=torch.zeros(2, 8),
        w1=torch.zeros(4, 8),
        w2=torch.zeros(4, 8),
        topk_weights=torch.ones(2, 2),
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        w1_scale=torch.ones(4),
        w2_scale=torch.ones(4),
        w1_bias=None,
        w2_bias=None,
        expert_map=None,
        apply_router_weight_on_input=False,
        activation="silu",
        global_num_experts=4,
        clamp_limit=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_w8a8_moe_rejects_non_silu(env):
    with pytest.raises(NotImplementedError):
        env.mods.w8a8_moe.w8a8_moe_experts(**_w8a8_kwargs(activation="gelu"))


def test_w8a8_moe_rejects_clamp_limit(env):
    with pytest.raises(NotImplementedError):
        env.mods.w8a8_moe.w8a8_moe_experts(**_w8a8_kwargs(clamp_limit=10.0))


@pytest.mark.parametrize("clamp_limit", [None, 0])
def test_w8a8_moe_delegates_to_fused_experts(env, monkeypatch, clamp_limit):
    fused = mock.MagicMock(return_value="moe-out")
    monkeypatch.setattr(
        sys.modules["vllm_fl.quantization.w8a8.moe_experts"],
        "_flaggems_fused_experts_impl",
        fused,
    )
    kwargs = _w8a8_kwargs(clamp_limit=clamp_limit)
    assert env.mods.w8a8_moe.w8a8_moe_experts(**kwargs) == "moe-out"
    call_kwargs = fused.call_args.kwargs
    assert call_kwargs["activation"] == "silu"
    assert call_kwargs["inplace"] is False
    assert call_kwargs["use_fp8_w8a8"] is False
    assert call_kwargs["use_int8_w8a8"] is True
    assert call_kwargs["use_int8_w8a16"] is False
    assert call_kwargs["use_int4_w4a16"] is False
    assert call_kwargs["per_channel_quant"] is True
    assert call_kwargs["a1_scale"] is None
    assert call_kwargs["a2_scale"] is None
    assert call_kwargs["block_shape"] is None


def test_w8a8_moe_missing_impl(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_fl.quantization.w8a8.moe_experts", None)
    with pytest.raises(ImportError):
        env.mods.w8a8_moe.w8a8_moe_experts(**_w8a8_kwargs())


# ---------------------------------------------------------------------------
# custom attention registration
# ---------------------------------------------------------------------------


def test_register_attention_registers_custom_backend(env):
    calls = env.stubs.registry_calls
    calls.clear()
    env.mods.custom_attention.register_attention()
    assert len(calls) == 1
    call = calls[0]
    assert call["backend"] is env.stubs.triton_attn
    assert call["class_path"] == FLAGGEMS_ATTN_PATH
    assert call["is_mamba"] is False
