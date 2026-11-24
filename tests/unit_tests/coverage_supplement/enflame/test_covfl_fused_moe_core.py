"""Coverage tests for the FL fused-MoE core operators.

Targets (executed from disk so coverage attributes to the real files):

* vllm_fl/ops/fused_moe/fused_moe.py
* vllm_fl/ops/fused_moe/activation.py
* vllm_fl/ops/fused_moe/layer.py
* vllm_fl/dispatch/backends/flaggems/impl/fused_moe.py

All heavy third-party modules (``vllm``, ``flag_gems``, ``flaggems_vllm``) are
replaced by tiny in-process stubs.  Kernel entry points are ``unittest.mock``
objects and tensors are small CPU tensors, so no GCU/NPU kernel ever runs.
No network, no subprocess, no device allocation.
"""

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

_SRC_REL_FILES = {
    "fused_moe": Path("vllm_fl") / "ops" / "fused_moe" / "fused_moe.py",
    "activation": Path("vllm_fl") / "ops" / "fused_moe" / "activation.py",
    "layer": Path("vllm_fl") / "ops" / "fused_moe" / "layer.py",
    "flaggems_fused_moe": Path("vllm_fl")
    / "dispatch"
    / "backends"
    / "flaggems"
    / "impl"
    / "fused_moe.py",
}


def _locate_src():
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
        if all((root / rel).is_file() for rel in _SRC_REL_FILES.values()):
            return root.resolve()
    return None


SRC = _locate_src()
if SRC is None:
    pytest.skip(
        "cannot locate vllm_fl fused_moe sources; set VLLM_FL_SRC",
        allow_module_level=True,
    )


_OWNED_PREFIXES = ("vllm", "vllm_fl", "flaggems_vllm", "flag_gems")


def _is_owned(name):
    return any(
        name == prefix or name.startswith(prefix + ".") for prefix in _OWNED_PREFIXES
    )


def _module(name, attrs=None, is_package=False, path=None):
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


class _FakeTensor:
    def __init__(self, dev_type, contiguous=True, numel=4):
        self.device = types.SimpleNamespace(type=dev_type)
        self._contig = contiguous
        self._numel = numel

    def is_contiguous(self):
        return self._contig

    def numel(self):
        return self._numel


class _Activation:
    def __init__(self, value, gated):
        self.value = value
        self.is_gated = gated

    def __repr__(self):
        return f"Activation({self.value})"


def _make_default_quant_config():
    return types.SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_act_token_quant=False,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        w1_bias=None,
        w2_bias=None,
    )


def _setup_dispatch_stub():
    cached_impls = {}
    resolve_impls = {}

    class CachedOp:
        def __init__(self, op_name):
            self.op_name = op_name

        def __call__(self, *args, **kwargs):
            fn = cached_impls.get(self.op_name)
            if fn is None:
                raise NotImplementedError(
                    f"no stub implementation registered for {self.op_name}"
                )
            return fn(*args, **kwargs)

    def resolve_op(op_name):
        fn = resolve_impls.get(op_name)
        if fn is None:
            fn = mock.MagicMock(name=f"resolve_op[{op_name}]")
            resolve_impls[op_name] = fn
        return fn

    def call_op(op_name, *args, **kwargs):
        return resolve_op(op_name)(*args, **kwargs)

    return _module(
        "vllm_fl.dispatch",
        {
            "CachedOp": CachedOp,
            "resolve_op": resolve_op,
            "call_op": call_op,
            "warmup_cached_ops": lambda *a, **k: None,
            "_CACHED_IMPLS": cached_impls,
            "_RESOLVE_IMPLS": resolve_impls,
        },
    )


class _Sandbox:
    def __init__(self, root):
        self.root = Path(root)
        self.saved = None
        self.stubs = types.SimpleNamespace()
        self.mods = types.SimpleNamespace()

    def setup(self):
        self.saved = dict(sys.modules)
        self._build_third_party()
        self._build_vllm_fl_namespace()
        self._load_targets()

    def teardown(self):
        saved = self.saved or {}
        for name in list(sys.modules):
            if _is_owned(name) and name not in saved:
                sys.modules.pop(name, None)
        for name, module in saved.items():
            if _is_owned(name):
                sys.modules[name] = module

    def _build_third_party(self):
        logger = lambda name=None: logging.getLogger(name or "vllm")
        _module("vllm", is_package=True)
        _module("vllm.logger", {"init_logger": logger})

        platform = types.SimpleNamespace(
            vendor_name="enflame",
            device_type="gcu",
            is_cuda=lambda: False,
            is_cpu=lambda: False,
            is_out_of_tree=lambda: True,
            is_rocm=lambda: False,
            is_xpu=lambda: False,
            is_tpu=lambda: False,
        )
        _module("vllm.platforms", {"current_platform": platform})

        tl = types.SimpleNamespace(
            bfloat16=object(), float16=object(), float32=object()
        )
        triton = types.SimpleNamespace(cdiv=lambda a, b: -(-int(a) // int(b)))
        _module("vllm.triton_utils", {"tl": tl, "triton": triton})

        _module("vllm.utils", is_package=True)
        _module(
            "vllm.utils.math_utils",
            {"round_up": lambda a, b: -(-int(a) // int(b)) * int(b)},
        )

        quant_config = _make_default_quant_config()

        def _get_config_dtype_str(**_kwargs):
            return "stub_dtype"

        _module("vllm.model_executor", is_package=True)
        _module("vllm.model_executor.layers", is_package=True)
        fused_pkg = _module(
            "vllm.model_executor.layers.fused_moe",
            {"FusedMoEFactory": object(), "FusedMoE": object()},
            is_package=True,
        )
        _module(
            "vllm.model_executor.layers.fused_moe.config",
            {
                "FUSED_MOE_UNQUANTIZED_CONFIG": quant_config,
                "FusedMoEQuantConfig": type("FusedMoEQuantConfig", (), {}),
                "FusedMoEConfig": type("FusedMoEConfig", (), {}),
                "_get_config_dtype_str": _get_config_dtype_str,
            },
        )
        _module(
            "vllm.model_executor.layers.fused_moe.fused_moe",
            {
                "_get_config_quant_dtype": lambda **k: None,
                "try_get_optimal_moe_config": lambda *a, **k: {"BLOCK_SIZE_M": 16},
            },
        )

        def _quantize(A=None, **kwargs):
            return A, None

        _module(
            "vllm.model_executor.layers.fused_moe.utils",
            {"moe_kernel_quantize_input": _quantize},
        )

        silu = mock.MagicMock(name="MoEActivation.SILU")
        silu.value = "silu"
        silu.is_gated = True
        gelu = _Activation("gelu", True)
        swigluoai = _Activation("swigluoai", True)
        _module(
            "vllm.model_executor.layers.fused_moe.activation",
            {
                "ApplyMoEActivationConfig": type(
                    "ApplyMoEActivationConfig",
                    (),
                    {"__init__": lambda self, clamp_limit=None: setattr(
                        self, "clamp_limit", clamp_limit
                    )},
                ),
                "MoEActivation": types.SimpleNamespace(
                    SILU=silu, GELU=gelu, SWIGLUOAI=swigluoai
                ),
                "apply_moe_activation": mock.MagicMock(
                    name="upstream_apply_moe_activation"
                ),
            },
        )

        _module("vllm.model_executor.layers.fused_moe.runner", is_package=True)
        _module(
            "vllm.model_executor.layers.fused_moe.runner.moe_runner",
            {"MoERunner": type("MoERunner", (), {})},
        )

        class _UnquantizedFusedMoEMethod:
            def __init__(self, moe):
                self.moe = moe
                self.moe_kernel = None

        _module(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
            {"UnquantizedFusedMoEMethod": _UnquantizedFusedMoEMethod},
        )

        flag_gems = _module("flag_gems", is_package=True)
        flag_gems.vendor_name = "stub"
        flag_gems.moe_sum = mock.MagicMock(name="flag_gems.moe_sum")
        flag_gems.moe_align_block_size_triton = mock.MagicMock(
            name="flag_gems.moe_align_block_size_triton"
        )

        fv = _module("flaggems_vllm", is_package=True)
        fv.moe_align_block_size_triton = mock.MagicMock(
            name="flaggems_vllm.moe_align_block_size_triton"
        )
        fv.topk_softmax = mock.MagicMock(name="flaggems_vllm.topk_softmax")
        fv.invoke_fused_moe_triton_kernel = mock.MagicMock(
            name="flaggems_vllm.invoke_fused_moe_triton_kernel"
        )
        fv.grouped_topk = mock.MagicMock(name="flaggems_vllm.grouped_topk")
        fv.moe_sum = mock.MagicMock(name="flaggems_vllm.moe_sum")
        _module("flaggems_vllm.ops", is_package=True)
        _module(
            "flaggems_vllm.ops.moe_sum",
            {
                "_check_moe_sum_inputs": mock.MagicMock(
                    name="flaggems_vllm._check_moe_sum_inputs"
                )
            },
        )

        self.stubs.platform = platform
        self.stubs.fused_pkg = fused_pkg
        self.stubs.unquantized_base = _UnquantizedFusedMoEMethod
        self.stubs.flag_gems = flag_gems
        self.stubs.flaggems_vllm = fv
        self.stubs.moe_enum_silu = silu
        self.stubs.moe_enum_gelu = gelu

    def _build_vllm_fl_namespace(self):
        vllm_fl_dir = self.root / "vllm_fl"
        _module("vllm_fl", is_package=True, path=[str(vllm_fl_dir)])
        dispatch = _setup_dispatch_stub()
        _module("vllm_fl.ops", is_package=True, path=[str(vllm_fl_dir / "ops")])
        fused_moe_pkg = _module(
            "vllm_fl.ops.fused_moe",
            is_package=True,
            path=[str(vllm_fl_dir / "ops" / "fused_moe")],
        )
        fused_moe_utils = _module(
            "vllm_fl.ops.fused_moe.fused_moe_utils",
            {
                "select_unquantized_moe_backend_oot": mock.MagicMock(
                    name="select_unquantized_moe_backend_oot",
                    return_value=("backend-sentinel", None),
                )
            },
        )
        router = _module(
            "vllm_fl.ops.fused_moe.router",
            {"replace_router_with_fl": mock.MagicMock(name="replace_router_with_fl")},
        )
        _module(
            "vllm_fl.dispatch.backends", is_package=True, path=[]
        )
        _module(
            "vllm_fl.dispatch.backends.flaggems", is_package=True, path=[]
        )
        _module(
            "vllm_fl.dispatch.backends.flaggems.impl", is_package=True, path=[]
        )
        self.stubs.dispatch = dispatch
        self.stubs.fused_moe_utils = fused_moe_utils
        self.stubs.router = router
        self.stubs.fused_moe_pkg = fused_moe_pkg

    def _load(self, name, rel):
        spec = importlib.util.spec_from_file_location(name, str(self.root / rel))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        parent_name, _, leaf = name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, leaf, module)
        spec.loader.exec_module(module)
        return module

    def _load_targets(self):
        self.mods.fused_moe = self._load(
            "vllm_fl.ops.fused_moe.fused_moe", _SRC_REL_FILES["fused_moe"]
        )
        self.mods.activation = self._load(
            "vllm_fl.ops.fused_moe.activation", _SRC_REL_FILES["activation"]
        )
        self.mods.layer = self._load(
            "vllm_fl.ops.fused_moe.layer", _SRC_REL_FILES["layer"]
        )
        self.mods.flaggems_fused_moe = self._load(
            "vllm_fl.dispatch.backends.flaggems.impl.fused_moe",
            _SRC_REL_FILES["flaggems_fused_moe"],
        )


@pytest.fixture(scope="module")
def env():
    sandbox = _Sandbox(SRC)
    sandbox.setup()
    try:
        yield sandbox
    finally:
        sandbox.teardown()


def _silu_impl(_none, x):
    return x.new_empty((x.shape[0], x.shape[1] // 2))


def _gelu_impl(_none, x):
    return x.new_empty((x.shape[0], x.shape[1] // 2))


def _install_core_impls(monkeypatch, env, invoke=None, moe_sum=None):
    impls = env.stubs.dispatch._CACHED_IMPLS
    align = mock.MagicMock(
        name="moe_align_block_size",
        return_value=(
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        ),
    )
    invoke = invoke or mock.MagicMock(name="invoke_fused_moe_triton_kernel")
    moe_sum = moe_sum or mock.MagicMock(name="moe_sum")
    monkeypatch.setitem(impls, "moe_align_block_size", align)
    monkeypatch.setitem(impls, "invoke_fused_moe_triton_kernel", invoke)
    monkeypatch.setitem(impls, "silu_and_mul", _silu_impl)
    monkeypatch.setitem(impls, "gelu_and_mul", _gelu_impl)
    monkeypatch.setitem(impls, "moe_sum", moe_sum)
    return align, invoke, moe_sum


def _moe_inputs(num_tokens=4, E=2, N=8, H=4, topk=2, dtype=torch.float32):
    hidden = torch.randn(num_tokens, H, dtype=dtype)
    w1 = torch.randn(E, N, H, dtype=dtype)
    w2 = torch.randn(E, H, N, dtype=dtype)
    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32)
    topk_weights = torch.full((num_tokens, topk), 1.0 / topk, dtype=torch.float32)
    return hidden, w1, w2, topk_weights, topk_ids


def test_impl_silu_path(env, monkeypatch):
    impls = env.stubs.dispatch._CACHED_IMPLS
    align, invoke, moe_sum = _install_core_impls(monkeypatch, env)
    monkeypatch.setitem(impls, "silu_and_mul", mock.MagicMock(wraps=_silu_impl))
    hidden, w1, w2, tw, ti = _moe_inputs()
    out = env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="silu"
    )
    assert out.shape == hidden.shape
    assert invoke.call_count == 2
    assert moe_sum.call_count == 1
    env.stubs.dispatch._CACHED_IMPLS["silu_and_mul"].assert_called()


def test_impl_gelu_and_mul_path(env, monkeypatch):
    impls = env.stubs.dispatch._CACHED_IMPLS
    align, invoke, moe_sum = _install_core_impls(monkeypatch, env)
    gelu = mock.MagicMock(wraps=_gelu_impl)
    monkeypatch.setitem(impls, "gelu_and_mul", gelu)
    hidden, w1, w2, tw, ti = _moe_inputs()
    out = env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="gelu"
    )
    assert out.shape == hidden.shape
    gelu.assert_called()
    assert invoke.call_count == 2


def test_impl_silu_no_mul_path(env, monkeypatch):
    _install_core_impls(monkeypatch, env)
    hidden, w1, w2, tw, ti = _moe_inputs()
    out = env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="silu_no_mul"
    )
    assert out.shape == hidden.shape


def test_impl_gelu_no_mul_path(env, monkeypatch):
    _install_core_impls(monkeypatch, env)
    hidden, w1, w2, tw, ti = _moe_inputs()
    out = env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="gelu_no_mul"
    )
    assert out.shape == hidden.shape


def test_impl_unsupported_activation_raises(env, monkeypatch):
    _install_core_impls(monkeypatch, env)
    hidden, w1, w2, tw, ti = _moe_inputs()
    with pytest.raises(ValueError):
        env.mods.fused_moe.fused_experts_impl(
            hidden, w1, w2, tw, ti, activation="nope"
        )


def test_impl_activation_enum_value_is_normalized(env, monkeypatch):
    _install_core_impls(monkeypatch, env)
    hidden, w1, w2, tw, ti = _moe_inputs()
    activation = types.SimpleNamespace(value="silu")
    out = env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation=activation
    )
    assert out.shape == hidden.shape


@pytest.mark.parametrize(
    "dtype, expected_name",
    [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
    ],
)
def test_impl_compute_type_selection(env, monkeypatch, dtype, expected_name):
    _, invoke, _ = _install_core_impls(monkeypatch, env)
    hidden, w1, w2, tw, ti = _moe_inputs(dtype=dtype)
    env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="silu"
    )
    expected = getattr(env.mods.fused_moe.tl, expected_name)
    assert invoke.call_args_list[0].kwargs["compute_type"] is expected


def test_impl_inplace_true_reuses_hidden(env, monkeypatch):
    _install_core_impls(monkeypatch, env)
    hidden, w1, w2, tw, ti = _moe_inputs()
    out = env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="silu", inplace=True
    )
    assert out is hidden


def test_impl_per_channel_and_quant_flags(env, monkeypatch):
    _, invoke, _ = _install_core_impls(monkeypatch, env)
    hidden, w1, w2, tw, ti = _moe_inputs()
    env.mods.fused_moe.fused_experts_impl(
        hidden,
        w1,
        w2,
        tw,
        ti,
        activation="silu",
        use_fp8_w8a8=True,
        use_int8_w8a8=True,
        use_int8_w8a16=True,
        per_channel_quant=True,
        block_shape=[128, 128],
    )
    assert invoke.call_args_list[0].kwargs["use_fp8_w8a8"] is True
    assert invoke.call_args_list[0].kwargs["per_channel_quant"] is True
    assert invoke.call_args_list[0].kwargs["block_shape"] == [128, 128]


def test_impl_int4_shape_check_passes(env, monkeypatch):
    _install_core_impls(monkeypatch, env)
    hidden, w1, _w2, tw, ti = _moe_inputs()
    w1 = torch.randn(2, 8, 2)
    w2 = torch.randn(2, 4, 8)
    out = env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="silu", use_int4_w4a16=True
    )
    assert out.shape == hidden.shape


def test_impl_int4_shape_mismatch_raises(env, monkeypatch):
    _install_core_impls(monkeypatch, env)
    hidden, w1, w2, tw, ti = _moe_inputs()
    with pytest.raises(AssertionError):
        env.mods.fused_moe.fused_experts_impl(
            hidden, w1, w2, tw, ti, activation="silu", use_int4_w4a16=True
        )


def test_impl_topk_shape_mismatch_raises(env, monkeypatch):
    _install_core_impls(monkeypatch, env)
    hidden, w1, w2, _tw, _ti = _moe_inputs()
    ti_bad = torch.zeros(4, 1, dtype=torch.int32)
    tw = torch.zeros(4, 2)
    with pytest.raises(AssertionError):
        env.mods.fused_moe.fused_experts_impl(
            hidden, w1, w2, tw, ti_bad, activation="silu"
        )


def test_impl_chunked_last_chunk_adjust(env, monkeypatch):
    # More than one 65536-token chunk exercises the "adjust last chunk" path.
    _, invoke, moe_sum = _install_core_impls(monkeypatch, env)
    num_tokens = 65537
    hidden = torch.empty(num_tokens, 2)
    w1 = torch.zeros(2, 2, 2)
    w2 = torch.zeros(2, 2, 2)
    ti = torch.zeros(num_tokens, 1, dtype=torch.int32)
    tw = torch.zeros(num_tokens, 1)
    out = env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="silu"
    )
    assert out.shape == (num_tokens, 2)
    assert invoke.call_count == 4
    assert moe_sum.call_count == 2


def test_impl_zero_tokens_breaks_loop(env, monkeypatch):
    _, invoke, moe_sum = _install_core_impls(monkeypatch, env)
    hidden = torch.empty(0, 4)
    w1 = torch.zeros(2, 8, 4)
    w2 = torch.zeros(2, 4, 8)
    ti = torch.zeros(0, 2, dtype=torch.int32)
    tw = torch.zeros(0, 2)
    out = env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="silu"
    )
    assert out.shape == (0, 4)
    assert invoke.call_count == 0
    assert moe_sum.call_count == 0


def test_impl_global_num_experts_override(env, monkeypatch):
    align, _invoke, _ = _install_core_impls(monkeypatch, env)
    hidden, w1, w2, tw, ti = _moe_inputs()
    env.mods.fused_moe.fused_experts_impl(
        hidden, w1, w2, tw, ti, activation="silu", global_num_experts=64
    )
    assert align.call_args_list[0].args[2] == 64


def test_fused_experts_wrapper_uses_unquantized_default(env, monkeypatch):
    fake = mock.MagicMock(name="impl")
    monkeypatch.setattr(env.mods.fused_moe, "fused_experts_impl", fake)
    monkeypatch.setattr(env.stubs.platform, "vendor_name", "enflame")
    hidden, w1, w2, tw, ti = _moe_inputs()
    env.mods.fused_moe.fused_experts(hidden, w1, w2, tw, ti)
    kwargs = fake.call_args.kwargs
    assert kwargs["use_fp8_w8a8"] is False
    assert kwargs["per_act_token_quant"] is False
    assert kwargs["w1_scale"] is None


def test_fused_experts_wrapper_passes_quant_config(env, monkeypatch):
    fake = mock.MagicMock(name="impl")
    monkeypatch.setattr(env.mods.fused_moe, "fused_experts_impl", fake)
    monkeypatch.setattr(env.stubs.platform, "vendor_name", "enflame")
    qc = types.SimpleNamespace(
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_act_token_quant=True,
        w1_scale="w1s",
        w2_scale="w2s",
        w1_zp="w1z",
        w2_zp="w2z",
        a1_scale="a1s",
        a2_scale="a2s",
        block_shape=[1, 2],
        w1_bias="b1",
        w2_bias="b2",
    )
    hidden, w1, w2, tw, ti = _moe_inputs()
    env.mods.fused_moe.fused_experts(hidden, w1, w2, tw, ti, quant_config=qc)
    kwargs = fake.call_args.kwargs
    assert kwargs["use_fp8_w8a8"] is True
    assert kwargs["per_act_token_quant"] is True
    assert kwargs["w1_scale"] == "w1s"
    assert kwargs["w2_zp"] == "w2z"
    assert kwargs["block_shape"] == [1, 2]
    assert kwargs["w1_bias"] == "b1"
    assert kwargs["w2_bias"] == "b2"


def test_fused_experts_wrapper_kunlunxin_resolves_op(env, monkeypatch):
    resolved = mock.MagicMock(name="resolved_fused_experts")
    monkeypatch.setitem(
        env.stubs.dispatch._RESOLVE_IMPLS, "fused_experts", resolved
    )
    monkeypatch.setattr(env.stubs.platform, "vendor_name", "kunlunxin")
    monkeypatch.setattr(
        env.mods.fused_moe,
        "fused_experts_impl",
        mock.MagicMock(side_effect=AssertionError("must not use local impl")),
    )
    hidden, w1, w2, tw, ti = _moe_inputs()
    env.mods.fused_moe.fused_experts(hidden, w1, w2, tw, ti)
    resolved.assert_called_once()


def test_activation_silu_without_config(env, monkeypatch):
    mod = env.mods.activation
    gem = mock.MagicMock(return_value=torch.ones(1, 2))
    monkeypatch.setitem(env.stubs.dispatch._CACHED_IMPLS, "silu_and_mul", gem)
    inp = torch.randn(1, 4)
    out = torch.empty(1, 2)
    result = mod.apply_moe_activation(env.stubs.moe_enum_silu, out, inp)
    assert result is out
    assert torch.equal(out, torch.ones(1, 2))
    gem.assert_called_once()


def test_activation_silu_config_without_clamp(env, monkeypatch):
    mod = env.mods.activation
    gem = mock.MagicMock(return_value=torch.zeros(1, 2))
    monkeypatch.setitem(env.stubs.dispatch._CACHED_IMPLS, "silu_and_mul", gem)
    cfg = mod.ApplyMoEActivationConfig(clamp_limit=None)
    out = torch.empty(1, 2)
    mod.apply_moe_activation(env.stubs.moe_enum_silu, out, torch.randn(1, 4), activation_config=cfg)
    gem.assert_called_once()


def test_activation_gelu(env, monkeypatch):
    mod = env.mods.activation
    gem = mock.MagicMock(return_value=torch.ones(2, 3))
    monkeypatch.setitem(env.stubs.dispatch._CACHED_IMPLS, "gelu_and_mul", gem)
    out = torch.empty(2, 3)
    mod.apply_moe_activation(env.stubs.moe_enum_gelu, out, torch.randn(2, 6))
    gem.assert_called_once()
    assert torch.equal(out, torch.ones(2, 3))


def test_activation_silu_with_clamp_delegates_upstream(env, monkeypatch):
    mod = env.mods.activation
    upstream = mod.upstream_apply_moe_activation
    upstream.reset_mock()
    cfg = mod.ApplyMoEActivationConfig(clamp_limit=1.0)
    out = torch.empty(1, 2)
    inp = torch.randn(1, 4)
    topk = torch.zeros(1, 1, dtype=torch.int32)
    emap = torch.zeros(1, dtype=torch.int32)
    result = mod.apply_moe_activation(
        env.stubs.moe_enum_silu,
        out,
        inp,
        activation_config=cfg,
        topk_ids=topk,
        expert_map=emap,
    )
    assert result is out
    upstream.assert_called_once()
    assert upstream.call_args.kwargs["activation_config"] is cfg
    assert upstream.call_args.kwargs["topk_ids"] is topk
    assert upstream.call_args.kwargs["expert_map"] is emap


def test_activation_non_gated_delegates_upstream(env, monkeypatch):
    mod = env.mods.activation
    upstream = mod.upstream_apply_moe_activation
    upstream.reset_mock()
    activation = _Activation("identity", False)
    out = torch.empty(1, 4)
    mod.apply_moe_activation(activation, out, torch.randn(1, 4))
    upstream.assert_called_once()


def test_activation_assertions(env, monkeypatch):
    mod = env.mods.activation
    gated = env.stubs.moe_enum_gelu
    non_gated = _Activation("identity", False)
    with pytest.raises(AssertionError):
        mod.apply_moe_activation(gated, torch.empty(1, 2), torch.randn(4))
    with pytest.raises(AssertionError):
        mod.apply_moe_activation(gated, torch.empty(2), torch.randn(1, 4))
    with pytest.raises(AssertionError):
        mod.apply_moe_activation(gated, torch.empty(1, 2), torch.randn(1, 3))
    with pytest.raises(AssertionError):
        mod.apply_moe_activation(non_gated, torch.empty(1, 2), torch.randn(1, 3))


def test_layer_factory_replaces_unquantized_method(env, monkeypatch):
    mod = env.mods.layer
    upstream_cls = env.stubs.unquantized_base

    class RoutedExperts:
        def __init__(self):
            self.quant_method = upstream_cls(object())

    class Runner:
        def __init__(self):
            self.routed_experts = RoutedExperts()
            self.moe_config = object()
            self.replaced = None

        def _replace_quant_method(self, replacement):
            self.replaced = replacement

    runner = Runner()
    replacement = object()
    replace_router = mock.MagicMock(name="replace_router_with_fl")
    monkeypatch.setattr(mod, "_OrigFusedMoEFactory", lambda *a, **k: runner)
    monkeypatch.setattr(mod, "UnquantizedFusedMoEMethodFL", lambda _cfg: replacement)
    monkeypatch.setattr(mod, "replace_router_with_fl", replace_router)

    result = mod.FusedMoEFactoryFL()
    assert result is runner
    assert runner.replaced is replacement
    replace_router.assert_called_once()


def test_layer_factory_preserves_quantized_method(env, monkeypatch):
    mod = env.mods.layer

    class RoutedExperts:
        quant_method = object()

    class Runner:
        routed_experts = RoutedExperts()
        moe_config = object()

        def _replace_quant_method(self, _replacement):
            raise AssertionError("must not replace a quantized method")

    runner = Runner()
    replace_router = mock.MagicMock(name="replace_router_with_fl")
    monkeypatch.setattr(mod, "_OrigFusedMoEFactory", lambda *a, **k: runner)
    monkeypatch.setattr(mod, "replace_router_with_fl", replace_router)

    assert mod.FusedMoEFactoryFL() is runner
    replace_router.assert_called_once()


def test_layer_factory_alias(env):
    mod = env.mods.layer
    assert mod.FusedMoEFL is mod.FusedMoEFactoryFL
    assert set(mod.__all__) == {
        "FusedMoEFactoryFL",
        "FusedMoEFL",
        "UnquantizedFusedMoEMethodFL",
    }


def test_layer_falls_back_to_fused_moe_attribute(env, monkeypatch):
    pkg = env.stubs.fused_pkg
    sentinel = object()
    saved_factory = pkg.FusedMoEFactory
    saved_fused = pkg.FusedMoE
    try:
        del pkg.FusedMoEFactory
        pkg.FusedMoE = sentinel
        _module("covfb", is_package=True)
        _module(
            "covfb.fused_moe_utils",
            {"select_unquantized_moe_backend_oot": lambda **k: (None, None)},
        )
        path = SRC / _SRC_REL_FILES["layer"]
        spec = importlib.util.spec_from_file_location("covfb.layer", str(path))
        variant = importlib.util.module_from_spec(spec)
        sys.modules["covfb.layer"] = variant
        spec.loader.exec_module(variant)
        assert variant._OrigFusedMoEFactory is sentinel
    finally:
        pkg.FusedMoEFactory = saved_factory
        pkg.FusedMoE = saved_fused
        sys.modules.pop("covfb.layer", None)
        sys.modules.pop("covfb.fused_moe_utils", None)
        sys.modules.pop("covfb", None)


def test_layer_unquantized_method_init(env, monkeypatch):
    mod = env.mods.layer
    select = mock.MagicMock(return_value=("backend-x", "experts-x"))
    monkeypatch.setattr(mod, "select_unquantized_moe_backend_oot", select)
    moe = object()
    method = mod.UnquantizedFusedMoEMethodFL(moe)
    assert method.moe is moe
    assert method.unquantized_backend == "backend-x"
    assert method.experts_cls == "experts-x"
    select.assert_called_once_with(moe_config=moe)


@pytest.mark.parametrize(
    "moe_kernel, experts_cls, expected",
    [
        (None, None, True),
        (None, types.SimpleNamespace(is_monolithic=lambda: False), False),
        (types.SimpleNamespace(is_monolithic=True), "ignored", True),
        (types.SimpleNamespace(is_monolithic=False), "ignored", False),
    ],
)
def test_layer_is_monolithic(env, moe_kernel, experts_cls, expected):
    mod = env.mods.layer
    obj = object.__new__(mod.UnquantizedFusedMoEMethodFL)
    obj.moe_kernel = moe_kernel
    obj.experts_cls = experts_cls
    assert obj.is_monolithic is expected


def test_flaggems_align_gcu_uses_flag_gems(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    real = torch

    class _TorchProxy:
        int32 = real.int32

        @staticmethod
        def empty(*a, **k):
            k.pop("device", None)
            return real.empty(*a, **k)

    monkeypatch.setattr(mod, "torch", _TorchProxy)
    flag_gems_kernel = env.stubs.flag_gems.moe_align_block_size_triton
    fv_kernel = env.stubs.flaggems_vllm.moe_align_block_size_triton
    flag_gems_kernel.reset_mock()
    fv_kernel.reset_mock()
    topk_ids = _FakeTensor("gcu", numel=4)
    sorted_ids, expert_ids, num_post = mod.moe_align_block_size_flaggems(
        topk_ids, 2, 2
    )
    assert isinstance(sorted_ids, real.Tensor)
    assert isinstance(expert_ids, real.Tensor)
    assert isinstance(num_post, real.Tensor)
    flag_gems_kernel.assert_called_once()
    fv_kernel.assert_not_called()


def test_flaggems_align_non_gcu_pad_and_small(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    flag_gems_kernel = env.stubs.flag_gems.moe_align_block_size_triton
    fv_kernel = env.stubs.flaggems_vllm.moe_align_block_size_triton
    flag_gems_kernel.reset_mock()
    fv_kernel.reset_mock()
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    sorted_ids, expert_ids, num_post = mod.moe_align_block_size_flaggems(
        topk_ids, 2, 8, pad_sorted_ids=True
    )
    assert sorted_ids.numel() == 8
    assert expert_ids.dtype == torch.int32
    fv_kernel.assert_called_once()
    flag_gems_kernel.assert_not_called()


def test_flaggems_align_expert_map_indexing(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    env.stubs.flaggems_vllm.moe_align_block_size_triton.reset_mock()
    topk_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32)
    expert_map = torch.arange(2, dtype=torch.int32)
    sorted_ids, expert_ids, num_post = mod.moe_align_block_size_flaggems(
        topk_ids, 2, 2, expert_map=expert_map
    )
    assert expert_ids.dtype == torch.int32
    assert expert_ids.shape == (0,)


def test_flaggems_topk_softmax_npu_normalizes(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    kernel = env.stubs.flaggems_vllm.topk_softmax
    kernel.reset_mock()
    kernel.side_effect = None
    weights = torch.tensor([[0.2, 0.3], [0.5, 0.5]])
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    token_expert = torch.zeros(2, 2, dtype=torch.int32)
    gating = _FakeTensor("npu")
    out_w, out_i = mod.topk_softmax_flaggems(
        weights, indices, token_expert, gating, renormalize=True
    )
    assert out_i is indices
    assert torch.allclose(out_w.sum(dim=-1), torch.ones(2))
    assert kernel.call_args.args[-1] is False


def test_flaggems_topk_softmax_typeerror_fallback(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    calls = {"n": 0}

    def kernel(*args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("no renormalize kwarg")

    monkeypatch.setattr(env.stubs.flaggems_vllm, "topk_softmax", kernel)
    weights = torch.tensor([[0.2, 0.3], [0.4, 0.6]])
    indices = torch.zeros(2, 2, dtype=torch.int32)
    token_expert = torch.zeros(2, 2, dtype=torch.int32)
    gating = torch.randn(2, 2)
    out_w, out_i = mod.topk_softmax_flaggems(
        weights, indices, token_expert, gating, renormalize=True
    )
    assert calls["n"] == 2
    assert torch.allclose(out_w.sum(dim=-1), torch.ones(2))


def test_flaggems_topk_softmax_non_npu_renormalize(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    kernel = mock.MagicMock(name="topk")
    monkeypatch.setattr(env.stubs.flaggems_vllm, "topk_softmax", kernel)
    weights = torch.tensor([[0.2, 0.3]])
    indices = torch.zeros(1, 2, dtype=torch.int32)
    token_expert = torch.zeros(1, 2, dtype=torch.int32)
    gating = torch.randn(1, 2)
    mod.topk_softmax_flaggems(weights, indices, token_expert, gating, renormalize=True)
    assert kernel.call_args.args[-1] is True


def test_flaggems_topk_softmax_no_renormalize(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    kernel = mock.MagicMock(name="topk")
    monkeypatch.setattr(env.stubs.flaggems_vllm, "topk_softmax", kernel)
    weights = torch.tensor([[0.2, 0.3]])
    indices = torch.zeros(1, 2, dtype=torch.int32)
    token_expert = torch.zeros(1, 2, dtype=torch.int32)
    gating = torch.randn(1, 2)
    mod.topk_softmax_flaggems(weights, indices, token_expert, gating, renormalize=False)
    assert kernel.call_args.args[-1] is False


def test_flaggems_invoke_delegates(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    kernel = mock.MagicMock(name="invoke")
    monkeypatch.setattr(env.stubs.flaggems_vllm, "invoke_fused_moe_triton_kernel", kernel)
    args = [torch.zeros(2, 2) for _ in range(3)]
    mod.invoke_fused_moe_triton_kernel_flaggems(
        *args,
        None,
        None,
        torch.zeros(2),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        False,
        2,
        {"BLOCK_SIZE_M": 16},
        "float16",
        False,
        False,
        False,
        False,
        True,
        block_shape=[1, 2],
        B_bias="bias",
    )
    kernel.assert_called_once()
    assert kernel.call_args.kwargs == {"block_shape": [1, 2], "B_bias": "bias"}


def test_flaggems_grouped_topk_delegates(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    kernel = mock.MagicMock(return_value="topk-result", name="grouped_topk")
    monkeypatch.setattr(env.stubs.flaggems_vllm, "grouped_topk", kernel)
    scores = torch.randn(2, 8)
    result = mod.grouped_topk_flaggems(scores, 2, 1, 2, True, 1.0, None, 0)
    assert result == "topk-result"
    assert kernel.call_args.args == (scores, 2, 1, 2, True, 1.0, None, 0)


def test_flaggems_moe_sum_gcu_contiguous(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    check = sys.modules["flaggems_vllm.ops.moe_sum"]._check_moe_sum_inputs
    check.reset_mock()
    flag_gems_sum = env.stubs.flag_gems.moe_sum
    fv_sum = env.stubs.flaggems_vllm.moe_sum
    flag_gems_sum.reset_mock()
    fv_sum.reset_mock()
    inp = _FakeTensor("gcu", contiguous=True)
    out = _FakeTensor("gcu", contiguous=True)
    mod.moe_sum_flaggems(inp, out)
    check.assert_called_once_with(inp, out)
    flag_gems_sum.assert_called_once_with(inp, out)
    fv_sum.assert_not_called()


def test_flaggems_moe_sum_gcu_non_contiguous_falls_through(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    fv_sum = env.stubs.flaggems_vllm.moe_sum
    flag_gems_sum = env.stubs.flag_gems.moe_sum
    fv_sum.reset_mock()
    flag_gems_sum.reset_mock()
    fv_sum.side_effect = None
    inp = _FakeTensor("gcu", contiguous=False)
    out = _FakeTensor("gcu", contiguous=True)
    mod.moe_sum_flaggems(inp, out)
    fv_sum.assert_called_once_with(inp, out)
    flag_gems_sum.assert_not_called()


def test_flaggems_moe_sum_success_path(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    fv_sum = env.stubs.flaggems_vllm.moe_sum
    flag_gems_sum = env.stubs.flag_gems.moe_sum
    fv_sum.reset_mock()
    flag_gems_sum.reset_mock()
    fv_sum.side_effect = None
    inp = torch.zeros(2, 2, 4)
    out = torch.zeros(2, 4)
    mod.moe_sum_flaggems(inp, out)
    fv_sum.assert_called_once_with(inp, out)
    flag_gems_sum.assert_not_called()


def test_flaggems_moe_sum_metax_fallback(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    fv_sum = env.stubs.flaggems_vllm.moe_sum
    flag_gems_sum = env.stubs.flag_gems.moe_sum
    fv_sum.reset_mock()
    flag_gems_sum.reset_mock()
    fv_sum.side_effect = RuntimeError("no maca kernel")
    monkeypatch.setattr(env.stubs.flag_gems, "vendor_name", "metax")
    inp = torch.zeros(2, 2, 4)
    out = torch.zeros(2, 4)
    mod.moe_sum_flaggems(inp, out)
    flag_gems_sum.assert_called_once_with(inp, out)


def test_flaggems_moe_sum_non_metax_reraises(env, monkeypatch):
    mod = env.mods.flaggems_fused_moe
    fv_sum = env.stubs.flaggems_vllm.moe_sum
    flag_gems_sum = env.stubs.flag_gems.moe_sum
    fv_sum.reset_mock()
    flag_gems_sum.reset_mock()
    fv_sum.side_effect = RuntimeError("boom")
    monkeypatch.setattr(env.stubs.flag_gems, "vendor_name", "stub")
    inp = torch.zeros(2, 2, 4)
    out = torch.zeros(2, 4)
    with pytest.raises(RuntimeError):
        mod.moe_sum_flaggems(inp, out)
    flag_gems_sum.assert_not_called()
