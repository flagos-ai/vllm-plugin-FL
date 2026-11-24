# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G54 residual coverage (814b) for three modules.

  * vllm_fl/__init__.py -- triton import shim, ARM/CPU platform probe,
    lazy ``distributed`` attribute, FlagCX/FA/custom-op/flash-attn patches,
    vendor dispatch in ``register`` and ``register_model``.
  * vllm_fl/quantization/modelslim_w8a8.py -- static/dynamic linear scheme,
    MoE offset parameters, npu_quantize/quant_matmul and the checkpoint
    prefix/quant-kind resolution.
  * vllm_fl/dispatch/backends/vendor/ascend/worker/ascend_v2_graph.py --
    uniform decode validation, metadata bridge, manager init/capture,
    replay metadata staging, statistics and release.

CPU-only: torch_npu, torch.npu, vLLM collaborators and the graph runtime are
recording fakes. The package ``__init__`` is executed from its source file
with a synthetic parent package so importing all of ``vllm_fl`` never runs.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import pathlib
import platform
import sys
import types
from types import SimpleNamespace

import pytest
import torch

_MISSING = object()


def _find_repo():
    import os

    env = os.environ.get("FL_REPO_ROOT") or os.environ.get("FL_WORKSPACE")
    cands = [pathlib.Path(env)] if env else []
    here = pathlib.Path(__file__).resolve()
    cands.extend([here.parent, *here.parents, pathlib.Path.cwd()])
    cands.extend(pathlib.Path(p) for p in sys.path if p)
    for base in cands:
        if (base / "vllm_fl" / "dispatch" / "backends").is_dir():
            return base
    return None


def _mod(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


class _Ctx:
    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


# ===========================================================================
# vllm_fl/__init__.py
# ===========================================================================

_REPO = _find_repo()


def _load_pkg_init():
    if _REPO is None:
        return None, RuntimeError("cannot locate vllm_fl tree")
    pkg = types.ModuleType("g54pkg")
    pkg.__path__ = [str(_REPO / "vllm_fl")]
    sys.modules["g54pkg"] = pkg
    ver = types.ModuleType("g54pkg.version")
    ver.git_version = "deadbeef"
    sys.modules["g54pkg.version"] = ver
    pkg.version = ver

    utils = types.ModuleType("vllm_fl.utils")
    utils.get_op_config = lambda: None

    class _DI:
        vendor_name = "nvidia"

    utils.DeviceInfo = _DI
    had = "vllm_fl.utils" in sys.modules
    saved = sys.modules.get("vllm_fl.utils")
    sys.modules["vllm_fl.utils"] = utils
    try:
        spec = importlib.util.spec_from_file_location(
            "g54pkg.init", str(_REPO / "vllm_fl" / "__init__.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["g54pkg.init"] = mod
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - env guard
        return None, exc
    finally:
        if had:
            sys.modules["vllm_fl.utils"] = saved
        else:
            sys.modules.pop("vllm_fl.utils", None)
    return mod, None


try:
    initmod, _INIT_ERROR = _load_pkg_init()
except Exception as exc:  # pragma: no cover - env guard
    initmod, _INIT_ERROR = None, exc


@pytest.fixture()
def init():
    if initmod is None:
        pytest.skip(f"vllm_fl.__init__ unavailable: {_INIT_ERROR}")
    return initmod


def _fake_utils(monkeypatch, vendor="nvidia", **extra):
    utils = types.ModuleType("vllm_fl.utils")

    class _DI:
        vendor_name = vendor

    utils.DeviceInfo = _DI
    utils.get_op_config = extra.pop("get_op_config", lambda: None)
    utils.is_oot_enabled = extra.pop("is_oot_enabled", lambda: False)
    for key, value in extra.items():
        setattr(utils, key, value)
    monkeypatch.setitem(sys.modules, "vllm_fl.utils", utils)
    return utils


def _fake_logger(monkeypatch, init):
    logs = []

    class _L:
        def __getattr__(self, name):
            return lambda *a, **k: logs.append((name, a))

    monkeypatch.setattr(init, "logger", _L())
    return logs




def test_init_flag_gems_triton_compat(init, monkeypatch):
    assert init._patch_flag_gems_triton_import_compat() is None

    triton = types.ModuleType("triton")
    triton.__path__ = []
    tl = types.ModuleType("triton.language")
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)
    monkeypatch.delitem(sys.modules, "triton.knobs", raising=False)
    init._patch_flag_gems_triton_import_compat()
    assert tl.map_elementwise.__name__ == "map_elementwise"
    with pytest.raises(NotImplementedError):
        tl.map_elementwise(1)
    assert triton.knobs.autotuning.adjust_block_size is True

    sentinel = lambda *a, **k: "keep"
    tl.map_elementwise = sentinel
    init._patch_flag_gems_triton_import_compat()
    assert tl.map_elementwise is sentinel

    fake_importlib = SimpleNamespace(
        import_module=lambda name: (_ for _ in ()).throw(
            ModuleNotFoundError("boom", name="other.module")
        )
    )
    monkeypatch.setattr(init, "importlib", fake_importlib)
    with pytest.raises(ModuleNotFoundError):
        init._patch_flag_gems_triton_import_compat()


def test_init_patch_flash_attn(init, monkeypatch):
    main = "vllm.vllm_flash_attn"
    child = f"{main}.flash_attn_interface"
    saved_main = sys.modules.pop(main, _MISSING)
    saved_child = sys.modules.pop(child, _MISSING)
    try:
        existing = types.ModuleType(main)
        sys.modules[main] = existing
        init._patch_flash_attn_import()
        assert sys.modules[main] is existing

        sys.modules.pop(main, None)
        sys.modules.pop(child, None)
        real_import = builtins.__import__

        def _raise(name, *args, **kwargs):
            if name == main:
                raise ImportError("no C ext")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raise)
        init._patch_flash_attn_import()
        stub = sys.modules[main]
        assert stub.FA2_AVAILABLE is False
        assert stub.FA3_AVAILABLE is False
        assert stub.fa_version_unsupported_reason()
        assert stub.is_fa_version_supported() is False
        assert stub.flash_attn_varlen_func is None
    finally:
        sys.modules.pop(main, None)
        sys.modules.pop(child, None)
        if saved_main is not _MISSING:
            sys.modules[main] = saved_main
        if saved_child is not _MISSING:
            sys.modules[child] = saved_child


def test_init_register_flagcx_connector(init, monkeypatch):
    factory = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.factory"
    )
    calls = []

    class _Factory:
        _registry = {}

        @classmethod
        def register_connector(cls, alias, module, klass):
            calls.append((alias, module, klass))
            cls._registry[alias] = klass

    factory.KVConnectorFactory = _Factory
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.factory",
        factory,
    )
    init._register_flagcx_connector()
    assert [c[0] for c in calls] == ["FlagCXConnector", "FlagcxConnector"]
    assert calls[0] == (
        "FlagCXConnector",
        "vllm_fl.distributed.kv_transfer.flagcx_connector",
        "FlagCXConnector",
    )

    calls.clear()
    _Factory._registry = {"FlagCXConnector": "x"}
    init._register_flagcx_connector()
    assert [c[0] for c in calls] == ["FlagcxConnector"]




def test_init_vendor_device_variants(init, monkeypatch):
    metax_calls = []
    monkeypatch.setattr(
        init, "_install_metax_compat", lambda: metax_calls.append(True)
    )

    musa = _mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.musa.patch",
        patch_accelerator_missing_attrs=lambda: True,
    )
    _fake_utils(monkeypatch, "mthreads")
    init._init_vendor_device()
    assert metax_calls == [True]

    musa.patch_accelerator_missing_attrs = lambda: False
    with pytest.raises(RuntimeError):
        init._init_vendor_device()

    kl = []
    _mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils",
        _patch_xpu_get_device=lambda: kl.append(True),
    )
    _fake_utils(monkeypatch, "kunlunxin")
    init._init_vendor_device()
    assert kl == [True]

    amd = []
    _mod(
        monkeypatch,
        "vllm_fl.patches.flagtree_target_info",
        apply_flagtree_target_info=lambda: amd.append(True),
    )
    _fake_utils(monkeypatch, "amd")
    init._init_vendor_device()
    assert amd == [True]


def test_init_install_metax_compat(init, monkeypatch):
    logs = _fake_logger(monkeypatch, init)
    installed = []
    shims = _mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.shims",
        install=lambda: installed.append(True),
        metax_active=lambda: False,
    )
    init._install_metax_compat()
    assert installed == []

    shims.metax_active = lambda: True
    init._install_metax_compat()
    assert installed == [True]

    incomplete = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.metax.shims"
    )
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.metax.shims", incomplete
    )
    init._install_metax_compat()
    assert any(name == "warning" for name, _ in logs)


def test_init_install_kunlunxin_fla_alias(init, monkeypatch):
    logs = _fake_logger(monkeypatch, init)
    calls = []
    _mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils",
        _fix_platform_vars=lambda: calls.append("fix"),
    )
    _mod(
        monkeypatch,
        "vllm_fl.patches.fla_module_relocation",
        install=lambda: calls.append("install"),
    )
    init._install_kunlunxin_fla_alias()
    assert calls == ["install", "fix"]

    calls.clear()
    assert init._kunlunxin_fla_alias_lock.acquire(blocking=False)
    try:
        init._install_kunlunxin_fla_alias()
    finally:
        init._kunlunxin_fla_alias_lock.release()
    assert calls == []

    incomplete = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils"
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils",
        incomplete,
    )
    init._install_kunlunxin_fla_alias()
    assert any(name == "debug" for name, _ in logs)


def test_init_register_variants(init, monkeypatch):
    inits = []
    monkeypatch.setattr(
        init, "_init_vendor_device", lambda: inits.append(True)
    )

    monkeypatch.setattr(init, "_arm_cpu_platform", lambda: "CPU")
    assert init.register() == "CPU"
    assert inits == [True]

    monkeypatch.setattr(init, "_arm_cpu_platform", lambda: None)
    _fake_utils(monkeypatch, "nvidia")
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    assert init.register() == "vllm_fl.nvidia_platform.NvidiaPlatformFL"
    import os

    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

    patched = []
    for name in (
        "_patch_custom_ops",
        "_patch_flash_attn_import",
    ):
        monkeypatch.setattr(
            init, name, lambda *a, _n=name, **k: patched.append(_n)
        )
    monkeypatch.setattr(
        init, "_install_kunlunxin_fla_alias", lambda: patched.append("kl")
    )
    _fake_utils(monkeypatch, "kunlunxin")
    assert init.register() == "vllm_fl.platform.PlatformFL"
    assert patched == ["_patch_custom_ops", "_patch_flash_attn_import", "kl"]
    patched.clear()

    _fake_utils(monkeypatch, "mthreads")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    assert init.register() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_init_register_quant_linear_and_router(init, monkeypatch):
    platform_mod = types.ModuleType("vllm.platforms")
    platform_mod.current_platform = SimpleNamespace(device_type="musa")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform_mod)
    lin = _mod(
        monkeypatch,
        "vllm_fl.quantization.quant_linear",
        add_oot_quant_kernel=lambda: lin_calls.append(True),
    )
    lin_calls = []
    init.register_quant_linear()
    init.register_router()
    assert lin_calls == []

    platform_mod.current_platform = SimpleNamespace(device_type="cuda")
    init.register_quant_linear()
    assert lin_calls == [True]

    router_calls = []
    _mod(
        monkeypatch,
        "vllm_fl.ops.fused_moe.router",
        replace_router_with_fl=lambda: router_calls.append(True),
    )
    _fake_utils(monkeypatch, "nvidia", is_oot_enabled=lambda: False)
    init.register_router()
    assert router_calls == []

    _fake_utils(monkeypatch, "nvidia", is_oot_enabled=lambda: True)
    init.register_router()
    assert router_calls == [True]


def test_init_register_gdn_packed_decode_patch(init, monkeypatch):
    monkeypatch.setattr(
        init,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: (_ for _ in ()).throw(
                ImportError("missing")
            )
        ),
    )
    assert init._register_gdn_packed_decode_patch() is False

    monkeypatch.setattr(init, "importlib", importlib)
    _mod(
        monkeypatch,
        "vllm_fl.patches.gdn_packed_decode",
        patch_vllm_packed_gdn_beta=lambda: True,
    )
    assert init._register_gdn_packed_decode_patch() is True


def test_init_register_model_arm_and_ascend(init, monkeypatch):
    logs = _fake_logger(monkeypatch, init)
    platform_mod = types.ModuleType("vllm.platforms")
    platform_mod.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform_mod)
    monkeypatch.setattr(init, "_arm_cpu_platform", lambda: "CPU")
    _mod(monkeypatch, "vllm_fl.patches.arm_cpu_gdn")
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.patches.arm_cpu_gdn",
        types.ModuleType("vllm_fl.patches.arm_cpu_gdn"),
    )
    sys.modules["vllm_fl.patches.arm_cpu_gdn"].apply_arm_cpu_gdn_state_indices_patch = (
        lambda: None
    )

    real_import = builtins.__import__

    def _no_flag_gems(name, *args, **kwargs):
        if name == "flag_gems":
            raise ModuleNotFoundError("no flag_gems", name="flag_gems")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_flag_gems)
    init.register_model()
    assert any(name == "warning" for name, _ in logs)

    monkeypatch.setattr(builtins, "__import__", real_import)
    installed = []
    _mod(monkeypatch, "flag_gems", vendor_name="not-arm")
    init.register_model()
    assert installed == []

    sys.modules["flag_gems"].vendor_name = "arm"
    _mod(
        monkeypatch,
        "vllm_fl.quantization.arm_cpu_w4a8",
        install_arm_cpu_packed_w4a8=lambda: installed.append(True),
    )
    init.register_model()
    assert installed == [True]

    platform_mod.current_platform = SimpleNamespace(device_type="npu")
    monkeypatch.setattr(init, "_arm_cpu_platform", lambda: None)
    seen = []
    for name in (
        "_register_flagcx_connector",
        "register_quant_linear",
        "register_router",
        "_register_gdn_packed_decode_patch",
    ):
        monkeypatch.setattr(
            init, name, lambda *a, _n=name, **k: seen.append(_n)
        )
    _fake_utils(monkeypatch, "ascend")
    models = _mod(
        monkeypatch, "vllm.model_executor.models", ModelRegistry=object()
    )
    models.ModelRegistry = SimpleNamespace(
        register_model=lambda name, path: seen.append((name, path))
    )
    _mod(monkeypatch, "vllm_fl.quantization.modelslim_w8a8")
    init.register_model()
    assert ("DeepseekV4ForCausalLM", "vllm_fl.models.deepseek_v4:DeepseekV4ForCausalLM") in seen
    assert ("GlmMoeDsaForCausalLM", "vllm_fl.models.glm_moe_dsa:GlmMoeDsaForCausalLM") in seen


# ===========================================================================
# vllm_fl/quantization/modelslim_w8a8.py
# ===========================================================================

_MS_CACHE = {}


@pytest.fixture()
def msmod():
    if not _MS_CACHE:
        try:
            from vllm_fl.quantization import modelslim_w8a8 as module
        except Exception as exc:  # pragma: no cover - env guard
            _MS_CACHE.update(module=None, error=exc)
        else:
            _MS_CACHE.update(module=module, error=None)
    if _MS_CACHE["module"] is None:
        pytest.skip(f"modelslim_w8a8 unavailable: {_MS_CACHE['error']}")
    return _MS_CACHE["module"]


def _fake_npu(monkeypatch, quant=None, matmul=None):
    mod = types.ModuleType("torch_npu")
    mod.npu_quantize = quant or (lambda *a, **k: "Q")
    mod.npu_quant_matmul = matmul or (
        lambda *a, **k: torch.zeros(2, 4)
    )
    monkeypatch.setitem(sys.modules, "torch_npu", mod)
    return mod






def test_ms_quantize_and_matmul(msmod, monkeypatch):
    seen = {}

    def _quant(*args):
        seen["quant"] = args
        return "Q"

    def _mm(*args, **kwargs):
        seen["mm"] = (args, kwargs)
        return "M"

    _fake_npu(monkeypatch, _quant, _mm)
    x = torch.zeros(2, 8)
    scale = torch.ones(1, 8)
    offset = torch.zeros(1, 8)
    assert msmod._ModelSlimW8A8StaticLinearScheme._quantize(x, scale, offset) == "Q"
    assert seen["quant"][3] is torch.qint8
    assert seen["quant"][4] == -1
    assert seen["quant"][5] is False

    out = msmod._ModelSlimW8A8StaticLinearScheme._quant_matmul(
        x, torch.zeros(8, 4, dtype=torch.int8), torch.ones(4), "B", torch.bfloat16
    )
    assert out == "M"
    assert seen["mm"][1]["bias"] == "B"
    assert seen["mm"][1]["output_dtype"] is torch.bfloat16








def test_ms_moe_scheme_create_weights(msmod, monkeypatch):
    monkeypatch.setattr(
        msmod.CompressedTensorsW8A8Int8MoEMethod,
        "create_weights",
        lambda self, *a, **k: None,
    )
    method = object.__new__(msmod._ModelSlimW8A8MoEMethod)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    assert tuple(layer.w13_weight_offset.shape) == (2, 6, 1)
    assert tuple(layer.w2_weight_offset.shape) == (2, 4, 1)
    assert layer.w13_weight_offset.quant_method == "channel"


def test_ms_config_classmethods_and_root_prefixes(msmod):
    cfg_cls = msmod.ModelSlimW8A8Config
    assert cfg_cls.get_name() == "fl_modelslim_w8a8"
    assert cfg_cls.get_supported_act_dtypes() == [
        torch.bfloat16,
        torch.float16,
    ]
    assert cfg_cls.get_min_capability() == 0
    assert cfg_cls.get_config_filenames() == ["quant_model_description.json"]
    assert cfg_cls._root_prefixes("model.x") == ["model.x", "x"]
    assert cfg_cls._root_prefixes("y") == ["y", "model.y"]
    built = cfg_cls.from_config({"a": 1})
    assert built.description == {"a": 1}
    assert built.weight_quant.num_bits == 8
    assert built.input_quant.dynamic is True




def test_ms_quant_kind_variants(msmod):
    cfg = object.__new__(msmod.ModelSlimW8A8Config)
    for kind in ("FLOAT", "W8A8", "W8A8_DYNAMIC"):
        cfg.description = {"p.weight": kind}
        assert cfg._quant_kind("p") == kind
    cfg.description = {}
    assert cfg._quant_kind("p") is None
    cfg.description = {"p.weight": "W4A16"}
    with pytest.raises(NotImplementedError):
        cfg._quant_kind("p")
    cfg.description = {"p.down_proj.weight": "W8A8", "p.w2.weight": "W8A8_DYNAMIC"}
    with pytest.raises(ValueError):
        cfg._quant_kind("p.down_proj")
    cfg.description = {"p.w1.weight": "W8A8"}
    with pytest.raises(ValueError):
        cfg._quant_kind("p.gate_up_proj")
    cfg.description = {"p.weight": "W8A8_DYNAMIC"}
    assert cfg._is_dynamic_w8a8("p") is True
    assert cfg._is_static_w8a8("p") is False
    cfg.description = {"p.weight": "W8A8"}
    assert cfg._is_static_w8a8("p") is True


def test_ms_get_quant_method_dispatch(msmod, monkeypatch):
    class _Linear:
        pass

    class _Experts:
        moe_config = object()

    monkeypatch.setattr(msmod, "LinearBase", _Linear)
    monkeypatch.setattr(msmod, "RoutedExperts", _Experts)
    monkeypatch.setattr(msmod, "UnquantizedLinearMethod", lambda: "UQL")
    monkeypatch.setattr(
        msmod, "CompressedTensorsLinearMethod", lambda cfg: ("CT", cfg)
    )
    monkeypatch.setattr(
        msmod, "UnquantizedFusedMoEMethod", lambda mc: ("UQE", mc)
    )
    monkeypatch.setattr(
        msmod, "_ModelSlimW8A8StaticLinearScheme", lambda: "STATIC"
    )
    monkeypatch.setattr(
        msmod, "_ModelSlimW8A8LinearScheme", lambda **k: ("DYN", k)
    )
    monkeypatch.setattr(
        msmod, "_ModelSlimW8A8MoEMethod", lambda *a, **k: ("MOE", a, k)
    )

    cfg = object.__new__(msmod.ModelSlimW8A8Config)
    cfg.weight_quant = "WQ"
    cfg.input_quant = "IQ"

    for kind, expected in (
        ("W8A8", ("CT", cfg)),
        ("W8A8_DYNAMIC", ("CT", cfg)),
        ("FLOAT", "UQL"),
    ):
        cfg.description = {"x.weight": kind}
        layer = _Linear()
        assert cfg.get_quant_method(layer, "x") == expected
        if kind == "W8A8":
            assert layer.scheme == "STATIC"
        elif kind == "W8A8_DYNAMIC":
            assert layer.scheme[0] == "DYN"

    cfg.description = {}
    assert cfg.get_quant_method(_Linear(), "x") == "UQL"

    cfg.description = {"x.weight": "W8A8_DYNAMIC"}
    result = cfg.get_quant_method(_Experts(), "x")
    assert result[0] == "MOE"
    cfg.description = {"x.weight": "FLOAT"}
    assert cfg.get_quant_method(_Experts(), "x")[0] == "UQE"
    cfg.description = {"x.weight": "W8A8"}
    with pytest.raises(NotImplementedError):
        cfg.get_quant_method(_Experts(), "x")
    assert cfg.get_quant_method(object(), "x") is None


# ===========================================================================
# .../ascend/worker/ascend_v2_graph.py
# ===========================================================================

_GM_CACHE = {}


@pytest.fixture()
def graph():
    if not _GM_CACHE:
        try:
            from vllm_fl.dispatch.backends.vendor.ascend.worker import (
                ascend_v2_graph as module,
            )
        except Exception as exc:  # pragma: no cover - env guard
            _GM_CACHE.update(module=None, error=exc)
        else:
            _GM_CACHE.update(module=module, error=None)
    if _GM_CACHE["module"] is None:
        pytest.skip(f"ascend_v2_graph unavailable: {_GM_CACHE['error']}")
    return _GM_CACHE["module"]


def _decode_batch(**over):
    base = dict(
        decode_graph_eligible=True,
        num_reqs=2,
        num_scheduled_tokens=[1, 1],
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_graph_validate_uniform_decode(graph):
    with pytest.raises(ValueError):
        graph._validate_uniform_decode_batch(None)
    with pytest.raises(ValueError):
        graph._validate_uniform_decode_batch(_decode_batch(decode_graph_eligible=False))
    with pytest.raises(ValueError):
        graph._validate_uniform_decode_batch(_decode_batch(num_reqs=0))
    with pytest.raises(ValueError):
        graph._validate_uniform_decode_batch(
            _decode_batch(num_scheduled_tokens=[1, 2])
        )
    assert graph._validate_uniform_decode_batch(_decode_batch()) is None


def test_graph_install_metadata_bridge(graph, monkeypatch):
    class _State:
        def prepare_attn(self, *args, **kwargs):
            return {"args": args, "kwargs": kwargs}

    state = _State()
    state._ascend_graph_prepare_attn_original = "already"
    graph.install_graph_metadata_bridge(state, lambda: None)
    assert state.prepare_attn({"a": 1}) == {
        "args": ({"a": 1},),
        "kwargs": {},
    }

    state = _State()
    staged = []
    manager = object.__new__(graph.AscendModelGraphManager)
    manager.stage_replay_metadata = lambda md: staged.append(md)
    graph.install_graph_metadata_bridge(state, lambda: manager)
    assert hasattr(state, "_ascend_graph_prepare_attn_original")
    orig = state.prepare_attn
    batch = _decode_batch()
    out = orig(batch, graph.CUDAGraphMode.NONE)
    assert staged == [] and out["args"] == (batch, graph.CUDAGraphMode.NONE)
    out = orig(batch, graph.CUDAGraphMode.FULL)
    assert staged == [out]
    with pytest.raises(ValueError):
        orig(_decode_batch(decode_graph_eligible=False), graph.CUDAGraphMode.FULL)

    other = _State()
    graph.install_graph_metadata_bridge(other, lambda: object())
    with pytest.raises(RuntimeError):
        other.prepare_attn(batch, graph.CUDAGraphMode.FULL)




def _capture_runner(graph):
    runtime = SimpleNamespace(
        capture_scope=lambda ctx: _Ctx(ctx), bind_attention_groups=lambda g: None
    )
    self = SimpleNamespace(
        _capture_descs={},
        _max_full_descs_to_capture=None,
        _npu_capture_stream=None,
        device=SimpleNamespace(type="npu"),
        vllm_config=object(),
        pool="POOL",
        _capture_has_lora=False,
        _capture_mem_samples=None,
        _capturing=False,
        graphs={},
        capture_count=0,
        runtime=runtime,
        use_breakable_cg=False,
        cudagraph_mode=SimpleNamespace(has_piecewise_cudagraphs=lambda: False),
    )
    return self


def _patch_npu(monkeypatch, graph):
    class _Graph:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    npu = SimpleNamespace(
        Stream=lambda device=None: SimpleNamespace(
            wait_stream=lambda other: None
        ),
        current_stream=lambda device=None: SimpleNamespace(
            wait_stream=lambda other: None
        ),
        stream=lambda s: _Ctx(s),
        synchronize=lambda device=None: None,
        NPUGraph=lambda: _Graph(),
        graph=lambda g, pool=None, stream=None: _Ctx(g),
        mem_get_info=lambda device=None: (1000, 2000),
    )
    monkeypatch.setattr(graph.torch, "npu", npu, raising=False)
    monkeypatch.setattr(graph, "capture_pool", lambda *a, **k: _Ctx("pool"))
    monkeypatch.setattr(
        graph, "set_current_vllm_config", lambda cfg: _Ctx(cfg)
    )
    monkeypatch.setattr(graph, "is_global_first_rank", lambda: False)
    monkeypatch.setattr(
        graph,
        "get_offloader",
        lambda: SimpleNamespace(
            sync_prev_onload=lambda: None, join_after_forward=lambda: None
        ),
    )
    monkeypatch.setattr(
        graph,
        "compilation_counter",
        SimpleNamespace(num_cudagraph_captured=0),
    )
    monkeypatch.setattr(
        graph,
        "logger",
        SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None),
    )








