# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual real-import contracts for PlatformFL.check_and_update_config branches.

w4a/w16 already exercise the happy paths and most vendor guards.  This module
targets the remaining branch arms: the enflame (non-GCU device) patch hook,
metax with ``model_config is None``, the MUSA piecewise collective block when
``model_config`` is None / ``use_inductor_graph_partition`` is set / the dist
backend is not MCCL, and the "already aligned" arms of the compile-size and MLA
block-size rewrites.

Only the real ``vllm_fl.platform`` module is imported; vendor-heavy leaf modules
are replaced by lightweight doubles via monkeypatch so importing one never runs
a real vendor ``__init__``.  No accelerator, git metadata or process group is
touched.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.platform import PlatformFL

_counter = {"n": 0}


def _sub(**attrs):
    _counter["n"] += 1
    return type(f"Plat3_{_counter['n']}", (PlatformFL,), attrs)


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
# enflame vendor on a non-GCU device still installs the ViT attention patch
# --------------------------------------------------------------------------- #


def test_enflame_patch_installed_when_device_type_not_gcu(monkeypatch):
    patch_mm = Mock()
    _install(monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=patch_mm)
    cfg = _cfg()
    # device_type is "cuda" here: the GCU-only guards must be skipped but the
    # vendor-level patch hook still runs for the "enflame" vendor.
    _sub(device_type="cuda", vendor_name="enflame").check_and_update_config(cfg)
    patch_mm.assert_called_once_with()
    assert cfg.parallel_config.disable_custom_all_reduce is True
    assert cfg.cache_config.block_size == 16


# --------------------------------------------------------------------------- #
# metax: model_config may be None, attention flags still rewritten
# --------------------------------------------------------------------------- #


def test_metax_none_model_config_skips_cascade_but_rewrites_attention(monkeypatch):
    _install(monkeypatch, "vllm_fl.dispatch.backends.vendor.metax.patches", ensure_patches=Mock(return_value=set()))
    cfg = _cfg(model_config=None)
    _sub(device_type="cuda", vendor_name="metax").check_and_update_config(cfg)
    attn = cfg.attention_config
    assert attn.use_cudnn_prefill is False
    assert attn.use_trtllm_ragged_deepseek_prefill is False
    assert attn.use_trtllm_attention is False
    assert attn.disable_flashinfer_prefill is True
    assert cfg.cache_config.block_size == 16


# --------------------------------------------------------------------------- #
# MUSA piecewise collective splitting: residual arms
# --------------------------------------------------------------------------- #


def _musa_graph(monkeypatch, append_result=True):
    append = Mock(return_value=append_result)
    limit = Mock(return_value=None)
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.musa.graph_config",
        limit_cudagraph_capture_sizes=limit,
        append_collective_splitting_ops=append,
    )
    return append, limit


def test_musa_piecewise_none_model_config_uses_data_parallel_size(monkeypatch):
    cuda_mode, comp_mode = _modes()
    append, limit = _musa_graph(monkeypatch)
    cfg = _cfg(model_config=None)
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.parallel_config.data_parallel_size = 4
    cfg.compilation_config.mode = comp_mode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = cuda_mode.PIECEWISE
    _sub(device_type="musa", vendor_name="mthreads", dist_backend="mccl").check_and_update_config(cfg)
    cfg.compilation_config.set_splitting_ops_for_v1.assert_called_once_with(
        all2all_backend=cfg.parallel_config.all2all_backend,
        data_parallel_size=4,
    )
    assert "vllm::all_reduce" in cfg.compilation_config.splitting_ops
    append.assert_called_once_with(cfg.compilation_config.splitting_ops)
    # The generic thinning runs for any capture mode other than NONE.
    limit.assert_called_once_with(cfg.compilation_config)


def test_musa_piecewise_skipped_when_inductor_partition_enabled(monkeypatch):
    cuda_mode, comp_mode = _modes()
    append, _limit = _musa_graph(monkeypatch)
    cfg = _cfg()
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.compilation_config.mode = comp_mode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = cuda_mode.PIECEWISE
    cfg.compilation_config.use_inductor_graph_partition = True
    _sub(device_type="musa", vendor_name="mthreads", dist_backend="mccl").check_and_update_config(cfg)
    cfg.compilation_config.set_splitting_ops_for_v1.assert_not_called()
    append.assert_not_called()
    assert cfg.compilation_config.splitting_ops == ["vllm::unified_attention"]


def test_musa_piecewise_skipped_when_backend_not_mccl(monkeypatch):
    cuda_mode, comp_mode = _modes()
    append, _limit = _musa_graph(monkeypatch)
    cfg = _cfg()
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.compilation_config.mode = comp_mode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = cuda_mode.PIECEWISE
    _sub(device_type="musa", vendor_name="mthreads", dist_backend="nccl").check_and_update_config(cfg)
    cfg.compilation_config.set_splitting_ops_for_v1.assert_not_called()
    append.assert_not_called()


# --------------------------------------------------------------------------- #
# Already-satisfied rewrites are left untouched
# --------------------------------------------------------------------------- #


def test_existing_compile_sizes_preserved():
    _cuda_mode, _comp = _modes()
    cfg = _cfg()
    cfg.compilation_config.compile_sizes = [2, 4]
    _sub(device_type="cuda", vendor_name="nvidia").check_and_update_config(cfg)
    assert cfg.compilation_config.compile_sizes == [2, 4]


def test_mla_block_size_already_multiple_of_64_untouched():
    cfg = _cfg()
    cfg.model_config.use_mla = True
    cfg.cache_config.block_size = 128
    _sub(device_type="cuda", vendor_name="nvidia").check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 128


# --------------------------------------------------------------------------- #
# GCU: DFlash with neither V2-only condition is accepted
# --------------------------------------------------------------------------- #


def test_gcu_dflash_without_v2_only_conditions_proceeds(monkeypatch):
    patch_mm = Mock()
    _install(monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=patch_mm)
    cfg = _cfg()
    cfg.speculative_config = SimpleNamespace(method="dflash")
    cfg._dflash_needs_multi_kv_group = Mock(return_value=False)
    cfg._is_dflash_candidate_draft = Mock(return_value=False)
    _sub(device_type="gcu", vendor_name="gcu").check_and_update_config(cfg)
    patch_mm.assert_called_once_with()
    # No V2-only feature was collected, so the eager path is configured.
    assert cfg.parallel_config.disable_custom_all_reduce is True
    assert cfg.cache_config.block_size == 16
