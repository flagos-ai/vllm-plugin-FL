# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual CPU contracts for PlatformFL branches not covered by w4a.

Only real imports plus monkeypatch/doubles are used; no AST splicing, no git
metadata, no accelerator initialisation. The vendor-heavy modules are replaced
by lightweight package doubles so importing a leaf never executes a real vendor
``__init__``.
"""

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_fl.platform as platform_mod
from vllm_fl.platform import PlatformFL

_counter = {"n": 0}


def _sub(**attrs):
    _counter["n"] += 1
    return type(f"ResidualPlatform{_counter['n']}", (PlatformFL,), attrs)


def _modes():
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode

    return CUDAGraphMode, CompilationMode


def _ensure_parents(monkeypatch, full):
    parts = full.split(".")
    for i in range(1, len(parts)):
        pname = ".".join(parts[:i])
        if pname in sys.modules:
            continue
        pkg = ModuleType(pname)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, pname, pkg)
        if i > 1:
            parent = sys.modules[".".join(parts[: i - 1])]
            monkeypatch.setattr(parent, parts[i - 1], pkg, raising=False)


def _install(monkeypatch, full, **attrs):
    _ensure_parents(monkeypatch, full)
    module = ModuleType(full)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, full, module)
    parent_name, _, child = full.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child, module, raising=False)
    return module


def _cfg(**overrides):
    cfg = SimpleNamespace(
        use_v2_model_runner=False,
        speculative_config=None,
        model_config=SimpleNamespace(
            is_diffusion=False,
            is_hybrid=False,
            use_mla=False,
            is_mm_prefix_lm=False,
            is_moe=True,
            model="probe-model",
            verify_with_parallel_config=Mock(),
            disable_cascade_attn=False,
        ),
        parallel_config=SimpleNamespace(
            worker_cls=None,
            all2all_backend="allgather_reducescatter",
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            disable_custom_all_reduce=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            world_size_across_dp=1,
        ),
        scheduler_config=SimpleNamespace(
            is_multimodal_model=False,
            disable_chunked_mm_input=False,
            async_scheduling=False,
        ),
        cache_config=SimpleNamespace(block_size=None, mamba_cache_mode=None),
        compilation_config=SimpleNamespace(
            mode=0,
            cudagraph_mode=0,
            compile_sizes=None,
            backend="eager",
            splitting_ops=["vllm::unified_attention"],
            use_inductor_graph_partition=False,
            inductor_compile_config={},
            pass_config=SimpleNamespace(fuse_rope_kvcache_cat_mla=True),
            set_splitting_ops_for_v1=Mock(),
        ),
        attention_config=SimpleNamespace(
            use_cudnn_prefill=True,
            use_trtllm_ragged_deepseek_prefill=True,
            use_trtllm_attention=True,
            disable_flashinfer_prefill=False,
        ),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# --------------------------------------------------------------------------- #
# TXDA: eager-only guards (lines 552-561)
# --------------------------------------------------------------------------- #


def test_txda_rejects_full_cudagraph_mode():
    cuda_mode, comp_mode = _modes()
    cfg = _cfg()
    cfg.compilation_config.mode = comp_mode.NONE
    cfg.compilation_config.cudagraph_mode = cuda_mode.PIECEWISE
    cls = _sub(device_type="txda", vendor_name="tsingmicro", dist_backend="tccl")
    with pytest.raises(ValueError, match="graph mode is not supported"):
        cls.check_and_update_config(cfg)


def test_txda_rejects_compilation_mode():
    cuda_mode, comp_mode = _modes()
    cfg = _cfg()
    cfg.compilation_config.mode = comp_mode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = cuda_mode.NONE
    cls = _sub(device_type="txda", vendor_name="tsingmicro", dist_backend="tccl")
    with pytest.raises(ValueError, match="eager execution only"):
        cls.check_and_update_config(cfg)


# --------------------------------------------------------------------------- #
# MUSA: piecewise collective splitting edge branches (lines 692-731)
# --------------------------------------------------------------------------- #


def _musa_graph(monkeypatch, append_result):
    append = Mock(return_value=append_result)
    limit = Mock(return_value=None)
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.musa.graph_config",
        limit_cudagraph_capture_sizes=limit,
        append_collective_splitting_ops=append,
    )
    return append, limit


def test_musa_piecewise_non_moe_uses_single_dp(monkeypatch):
    cuda_mode, comp_mode = _modes()
    append, limit = _musa_graph(monkeypatch, append_result=True)
    cfg = _cfg()
    cfg.model_config.is_moe = False
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.parallel_config.data_parallel_size = 4
    cfg.compilation_config.mode = comp_mode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = cuda_mode.PIECEWISE
    cfg.compilation_config.splitting_ops = ["vllm::unified_attention"]
    cls = _sub(device_type="musa", vendor_name="mthreads", dist_backend="mccl")
    cls.check_and_update_config(cfg)
    cfg.compilation_config.set_splitting_ops_for_v1.assert_called_once_with(
        all2all_backend=cfg.parallel_config.all2all_backend,
        data_parallel_size=1,
    )
    assert "vllm::all_reduce" in cfg.compilation_config.splitting_ops
    append.assert_called_once_with(cfg.compilation_config.splitting_ops)
    limit.assert_called_once_with(cfg.compilation_config)


def test_musa_piecewise_all_reduce_present_and_collective_noop(monkeypatch):
    cuda_mode, comp_mode = _modes()
    append, _limit = _musa_graph(monkeypatch, append_result=False)
    cfg = _cfg()
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.compilation_config.mode = comp_mode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = cuda_mode.PIECEWISE
    existing = ["vllm::unified_attention", "vllm::all_reduce", "_c10d_functional::all_reduce"]
    cfg.compilation_config.splitting_ops = list(existing)
    cls = _sub(device_type="musa", vendor_name="mthreads", dist_backend="mccl")
    cls.check_and_update_config(cfg)
    assert cfg.compilation_config.splitting_ops == existing
    append.assert_called_once_with(existing)


# --------------------------------------------------------------------------- #
# NPU / cache / gcu residual branches
# --------------------------------------------------------------------------- #


def test_npu_model_config_none_skips_glm_fallback(monkeypatch):
    refresh = Mock()
    ascend_compile = Mock()
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patch",
        refresh_block_size=refresh,
    )
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.compilation",
        _configure_ascend_compilation=ascend_compile,
    )
    cfg = _cfg()
    cfg.model_config = None
    cls = _sub(device_type="npu", vendor_name="ascend")
    cls.check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 128
    refresh.assert_called_once_with(cfg)
    ascend_compile.assert_called_once_with(cfg)


def test_cache_config_none_is_ignored():
    cuda_mode, comp_mode = _modes()
    cfg = _cfg()
    cfg.cache_config = None
    cfg.compilation_config.cudagraph_mode = cuda_mode.NONE
    cfg.compilation_config.mode = comp_mode.NONE
    cls = _sub(device_type="cuda", vendor_name="nvidia")
    cls.check_and_update_config(cfg)
    assert cfg.compilation_config.compile_sizes == []


def test_gcu_eager_mode_leaves_inductor_config(monkeypatch):
    cuda_mode, comp_mode = _modes()
    patch_mm = Mock()
    _install(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=patch_mm
    )
    cfg = _cfg()
    cfg.compilation_config.backend = "inductor"
    cfg.compilation_config.mode = comp_mode.NONE
    cfg.compilation_config.cudagraph_mode = cuda_mode.NONE
    cfg.compilation_config.inductor_compile_config = {}
    cls = _sub(device_type="gcu", vendor_name="gcu")
    cls.check_and_update_config(cfg)
    patch_mm.assert_called_once_with()
    assert cfg.compilation_config.inductor_compile_config == {}


# --------------------------------------------------------------------------- #
# apply_config_platform_defaults residual branches (lines 312-357)
# --------------------------------------------------------------------------- #


def test_apply_config_defaults_non_metax_returns_early(monkeypatch):
    ensure = Mock()
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.patches",
        ensure_patches=ensure,
    )
    cfg = _cfg()
    cls = _sub(device_type="cuda", vendor_name="nvidia")
    cls.apply_config_platform_defaults(cfg)
    ensure.assert_not_called()


def test_apply_config_defaults_empty_capture_sizes(monkeypatch):
    cuda_mode, comp_mode = _modes()
    ensure = Mock(return_value=set())
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.patches",
        ensure_patches=ensure,
    )
    compilation = SimpleNamespace(
        mode=comp_mode.VLLM_COMPILE,
        cudagraph_mode=None,
        cudagraph_capture_sizes=[],
        cudagraph_mm_encoder=True,
        pass_config=SimpleNamespace(fuse_rope_kvcache_cat_mla=True),
    )
    cfg = SimpleNamespace(compilation_config=compilation)
    cls = _sub(device_type="cuda", vendor_name="metax")
    cls.apply_config_platform_defaults(cfg)
    ensure.assert_called_once_with()
    assert compilation.cudagraph_mode == cuda_mode.PIECEWISE
    assert compilation.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert compilation.cudagraph_mm_encoder is False
    assert compilation.pass_config.fuse_rope_kvcache_cat_mla is False


# --------------------------------------------------------------------------- #
# _vllm_c_is_registered_provider residual branches (lines 1183-1208)
# --------------------------------------------------------------------------- #


def _install_ir(monkeypatch, config_cls, ir_op=None, ir_ops=None):
    op_module = ir_op or ModuleType("vllm.ir.op")
    if not hasattr(op_module, "IrOp"):
        op_module.IrOp = type("IrOp", (), {})
    ops_module = ir_ops or ModuleType("vllm.ir.ops")
    ir_pkg = ModuleType("vllm.ir")
    ir_pkg.__path__ = []
    ir_pkg.op = op_module
    ir_pkg.ops = ops_module
    _install(monkeypatch, "vllm.config.kernel", IrOpPriorityConfig=config_cls)
    monkeypatch.setitem(sys.modules, "vllm.ir", ir_pkg)
    monkeypatch.setitem(sys.modules, "vllm.ir.op", op_module)
    monkeypatch.setitem(sys.modules, "vllm.ir.ops", ops_module)
    import vllm as vllm_pkg

    monkeypatch.setattr(vllm_pkg, "ir", ir_pkg, raising=False)


def test_vllm_c_registered_provider_non_dataclass_config(monkeypatch):
    class NotDataclass:
        pass

    _install_ir(monkeypatch, NotDataclass)
    assert PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_registered_provider_empty_config(monkeypatch):
    @dataclass
    class EmptyConfig:
        pass

    _install_ir(monkeypatch, EmptyConfig)
    assert PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_registered_provider_impls_lookup_error(monkeypatch):
    op_module = ModuleType("vllm.ir.op")

    class IrOp:
        @property
        def impls(self):
            raise RuntimeError("cannot enumerate impls")

    op_module.IrOp = IrOp
    ops_module = ModuleType("vllm.ir.ops")
    ops_module.rms_norm = IrOp()

    @dataclass
    class Config:
        rms_norm: object = None

    _install_ir(monkeypatch, Config, ir_op=op_module, ir_ops=ops_module)
    assert PlatformFL._vllm_c_is_registered_provider() is False


# --------------------------------------------------------------------------- #
# pre_register residual: breakable graph already "0"
# --------------------------------------------------------------------------- #


def test_pre_register_npu_breakable_already_zero(monkeypatch):
    import os

    _install(monkeypatch, "vllm_fl.dispatch.backends.vendor.ascend")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    cls = _sub(device_name="npu", vendor_name="mthreads")
    cls.pre_register_and_update()
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
