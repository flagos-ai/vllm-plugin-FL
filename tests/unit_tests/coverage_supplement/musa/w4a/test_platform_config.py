# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for PlatformFL.check_and_update_config per-vendor branches.

``vllm.config`` enums and every heavy vendor patch module are replaced by
lightweight doubles so the real control flow runs without an accelerator.
"""

import os
import sys
from enum import IntEnum
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.platform import PlatformFL


class CompilationMode(IntEnum):
    NONE = 0
    STOCK_TORCH_COMPILE = 1
    DYNAMO_AS_IS = 2
    VLLM_COMPILE = 3


class CUDAGraphMode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_AND_PIECEWISE = 3

    def has_full_cudagraphs(self):
        return self in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE)

    def has_piecewise_cudagraphs(self):
        return self in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE)


def _install_fake_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child, module, raising=False)
    return module


@pytest.fixture(autouse=True)
def fake_vllm_enums(monkeypatch):
    _install_fake_module(monkeypatch, "vllm.config", CUDAGraphMode=CUDAGraphMode)
    _install_fake_module(
        monkeypatch, "vllm.config.compilation", CompilationMode=CompilationMode
    )


@pytest.fixture
def vendor_modules(monkeypatch):
    mods = {
        "patch_mm": Mock(),
        "prepare_glm_dsa": Mock(return_value=False),
        "refresh_block_size": Mock(),
        "ascend_compilation": Mock(),
        "iluvatar_compilation": Mock(),
        "limit_sizes": Mock(return_value=None),
        "append_collective": Mock(return_value=True),
        "ensure_patches": Mock(return_value=set()),
    }
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=mods["patch_mm"]
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.ascend_glm_dsa",
        prepare_glm_dsa_dense_fallback=mods["prepare_glm_dsa"],
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patch",
        refresh_block_size=mods["refresh_block_size"],
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.compilation",
        _configure_ascend_compilation=mods["ascend_compilation"],
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.iluvatar.compilation",
        configure_compilation=mods["iluvatar_compilation"],
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.musa.graph_config",
        limit_cudagraph_capture_sizes=mods["limit_sizes"],
        append_collective_splitting_ops=mods["append_collective"],
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.patches",
        ensure_patches=mods["ensure_patches"],
    )
    return mods


def _platform(device_type, vendor_name, dist_backend="nccl"):
    return type(
        f"CfgPlatform_{device_type}_{vendor_name}",
        (PlatformFL,),
        {
            "device_type": device_type,
            "vendor_name": vendor_name,
            "device_name": device_type,
            "dist_backend": dist_backend,
        },
    )


def _cfg():
    return SimpleNamespace(
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
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
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


def test_gcu_rejects_upstream_v2_runner(vendor_modules):
    cfg = _cfg()
    cfg.use_v2_model_runner = True
    with pytest.raises(ValueError, match="requires ModelRunnerFL"):
        _platform("gcu", "enflame").check_and_update_config(cfg)


@pytest.mark.parametrize(
    "setup,needle",
    [
        (lambda c: setattr(c, "speculative_config", SimpleNamespace(method="dspark")), "DSpark"),
        (
            lambda c: (
                setattr(c, "speculative_config", SimpleNamespace(method="dflash")),
                setattr(c, "_dflash_needs_multi_kv_group", Mock(return_value=True)),
                setattr(c, "_is_dflash_candidate_draft", Mock(return_value=False)),
            ),
            "mixed-KV DFlash",
        ),
        (
            lambda c: (
                setattr(c, "speculative_config", SimpleNamespace(method="dflash")),
                setattr(c, "_dflash_needs_multi_kv_group", Mock(return_value=False)),
                setattr(c, "_is_dflash_candidate_draft", Mock(return_value=True)),
            ),
            "DFlash candidate head",
        ),
        (lambda c: setattr(c.model_config, "is_diffusion", True), "diffusion"),
    ],
)
def test_gcu_rejects_v2_only_features(vendor_modules, setup, needle):
    cfg = _cfg()
    setup(cfg)
    with pytest.raises(ValueError, match=needle):
        _platform("gcu", "gcu").check_and_update_config(cfg)


def test_gcu_rejects_speculative_hybrid_mamba_align(vendor_modules):
    cfg = _cfg()
    cfg.speculative_config = SimpleNamespace(method="plain")
    cfg.model_config.is_hybrid = True
    cfg.cache_config.mamba_cache_mode = "align"
    with pytest.raises(ValueError, match="mamba_cache_mode='align'"):
        _platform("gcu", "enflame").check_and_update_config(cfg)


def test_gcu_happy_path_patches_and_configures(vendor_modules):
    cfg = _cfg()
    _platform("gcu", "gcu").check_and_update_config(cfg)
    vendor_modules["patch_mm"].assert_called_once_with()
    assert cfg.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"
    assert cfg.parallel_config.disable_custom_all_reduce is True
    assert cfg.cache_config.block_size == 16
    assert cfg.compilation_config.compile_sizes == []


def test_gcu_inductor_combo_kernels_disabled(vendor_modules):
    cfg = _cfg()
    cfg.compilation_config.backend = "inductor"
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    _platform("gcu", "enflame").check_and_update_config(cfg)
    assert cfg.compilation_config.inductor_compile_config["combo_kernels"] is False
    assert (
        cfg.compilation_config.inductor_compile_config["benchmark_combo_kernel"]
        is False
    )


def test_txda_rejects_v2_runner(vendor_modules):
    cfg = _cfg()
    cfg.use_v2_model_runner = True
    with pytest.raises(ValueError, match="requires ModelRunnerFL"):
        _platform("txda", "tsingmicro").check_and_update_config(cfg)


def test_txda_rejects_speculative(vendor_modules):
    cfg = _cfg()
    cfg.speculative_config = SimpleNamespace()
    with pytest.raises(ValueError, match="Remove --speculative-config"):
        _platform("txda", "tsingmicro").check_and_update_config(cfg)


@pytest.mark.parametrize(
    "field", ["prefill_context_parallel_size", "decode_context_parallel_size"]
)
def test_txda_rejects_context_parallelism(vendor_modules, field):
    cfg = _cfg()
    setattr(cfg.parallel_config, field, 2)
    with pytest.raises(ValueError, match="does not support context parallelism"):
        _platform("txda", "tsingmicro").check_and_update_config(cfg)


def test_txda_multi_device_requires_flagcx(vendor_modules):
    cfg = _cfg()
    cfg.parallel_config.world_size_across_dp = None
    cfg.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="requires FlagCX"):
        _platform("txda", "tsingmicro", dist_backend="tccl").check_and_update_config(
            cfg
        )


def test_txda_world_size_default_all_ones(vendor_modules):
    cfg = _cfg()
    cfg.parallel_config.world_size_across_dp = None
    _platform("txda", "tsingmicro", dist_backend="tccl").check_and_update_config(cfg)
    assert cfg.parallel_config.disable_custom_all_reduce is True
    assert cfg.cache_config.block_size == 16


def test_txda_accepts_flagcx_multi_device(vendor_modules):
    cfg = _cfg()
    cfg.parallel_config.world_size_across_dp = 8
    _platform("txda", "tsingmicro", dist_backend="flagcx").check_and_update_config(cfg)
    assert cfg.parallel_config.disable_custom_all_reduce is True


def test_npu_applies_dense_glm_fallback_and_revalidates(vendor_modules):
    vendor_modules["prepare_glm_dsa"].return_value = True
    cfg = _cfg()
    _platform("npu", "ascend").check_and_update_config(cfg)
    vendor_modules["prepare_glm_dsa"].assert_called_once_with(cfg.model_config)
    cfg.model_config.verify_with_parallel_config.assert_called_once_with(
        cfg.parallel_config
    )
    assert cfg.cache_config.block_size == 128
    vendor_modules["refresh_block_size"].assert_called_once_with(cfg)
    vendor_modules["ascend_compilation"].assert_called_once_with(cfg)


def test_npu_without_dense_fallback(vendor_modules):
    cfg = _cfg()
    _platform("npu", "ascend").check_and_update_config(cfg)
    cfg.model_config.verify_with_parallel_config.assert_not_called()
    assert cfg.cache_config.block_size == 128


def test_iluvatar_configures_compilation(vendor_modules):
    cfg = _cfg()
    _platform("cuda", "iluvatar").check_and_update_config(cfg)
    vendor_modules["iluvatar_compilation"].assert_called_once_with(
        cfg.compilation_config, cfg.model_config
    )
    assert cfg.parallel_config.worker_cls.endswith("IluvatarWorkerFL")


def test_musa_block_size_and_inductor_flags(vendor_modules):
    cfg = _cfg()
    _platform("musa", "mthreads").check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 64
    assert cfg.compilation_config.inductor_compile_config["combo_kernels"] is False
    assert (
        cfg.compilation_config.inductor_compile_config["benchmark_combo_kernel"]
        is False
    )
    vendor_modules["limit_sizes"].assert_not_called()


def test_musa_full_graph_downgraded_to_piecewise(vendor_modules):
    cfg = _cfg()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    vendor_modules["limit_sizes"].return_value = [1, 2]
    _platform("musa", "mthreads").check_and_update_config(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    vendor_modules["limit_sizes"].assert_called_once_with(cfg.compilation_config)


def test_musa_thinned_sizes_logged(vendor_modules):
    cfg = _cfg()
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    vendor_modules["limit_sizes"].return_value = [1, 4, 8]
    _platform("musa", "mthreads").check_and_update_config(cfg)
    vendor_modules["limit_sizes"].assert_called_once()


def test_musa_piecewise_collective_splitting(vendor_modules):
    cfg = _cfg()
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.parallel_config.all2all_backend = "allgather_reducescatter"
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    _platform("musa", "mthreads", dist_backend="mccl").check_and_update_config(cfg)
    cfg.compilation_config.set_splitting_ops_for_v1.assert_called_once()
    assert "vllm::all_reduce" in cfg.compilation_config.splitting_ops
    vendor_modules["append_collective"].assert_called_once_with(
        cfg.compilation_config.splitting_ops
    )


def test_musa_collective_splitting_skipped_for_single_tp(vendor_modules):
    cfg = _cfg()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    _platform("musa", "mthreads", dist_backend="mccl").check_and_update_config(cfg)
    cfg.compilation_config.set_splitting_ops_for_v1.assert_not_called()


def test_metax_patches_and_attention_flags(vendor_modules):
    vendor_modules["ensure_patches"].return_value = {"leftover"}
    cfg = _cfg()
    _platform("cuda", "metax").check_and_update_config(cfg)
    assert cfg.model_config.disable_cascade_attn is True
    assert cfg.attention_config.use_cudnn_prefill is False
    assert cfg.attention_config.use_trtllm_ragged_deepseek_prefill is False
    assert cfg.attention_config.use_trtllm_attention is False
    assert cfg.attention_config.disable_flashinfer_prefill is True


def test_metax_without_attention_config(vendor_modules):
    cfg = _cfg()
    cfg.attention_config = None
    _platform("cuda", "metax").check_and_update_config(cfg)
    assert cfg.model_config.disable_cascade_attn is True


def test_kunlunxin_disables_async_and_compilation(vendor_modules):
    cfg = _cfg()
    cfg.scheduler_config.async_scheduling = None
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    _platform("cuda", "kunlunxin").check_and_update_config(cfg)
    assert cfg.scheduler_config.async_scheduling is False
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert cfg.compilation_config.mode == CompilationMode.NONE


def test_kunlunxin_explicit_false_async_not_overwritten(vendor_modules):
    cfg = _cfg()
    cfg.scheduler_config.async_scheduling = False
    _platform("cuda", "kunlunxin").check_and_update_config(cfg)
    assert cfg.scheduler_config.async_scheduling is False


def test_kunlunxin_compilation_left_alone_when_eager(vendor_modules):
    cfg = _cfg()
    _platform("cuda", "kunlunxin").check_and_update_config(cfg)


def test_wide_ep_disables_graphs(vendor_modules):
    cfg = _cfg()
    cfg.parallel_config.all2all_backend = "deepep_high_throughput"
    cfg.parallel_config.data_parallel_size = 2
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    _platform("cuda", "nvidia").check_and_update_config(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_mla_forces_block_size_64(vendor_modules):
    cfg = _cfg()
    cfg.model_config.use_mla = True
    cfg.cache_config.block_size = 32
    _platform("cuda", "nvidia").check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 64


def test_mm_prefix_lm_forces_disable_chunked_input(vendor_modules):
    cfg = _cfg()
    cfg.model_config.is_mm_prefix_lm = True
    cfg.scheduler_config.is_multimodal_model = True
    _platform("cuda", "nvidia").check_and_update_config(cfg)
    assert cfg.scheduler_config.disable_chunked_mm_input is True


def test_default_cache_block_size_and_worker(vendor_modules):
    cfg = _cfg()
    _platform("cuda", "nvidia").check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 16
    assert cfg.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"


def test_apply_config_platform_defaults_ignored_for_other_vendors(vendor_modules):
    cfg = _cfg()
    _platform("cuda", "nvidia").apply_config_platform_defaults(cfg)
    vendor_modules["ensure_patches"].assert_not_called()


def test_apply_config_platform_defaults_no_compilation_config(vendor_modules):
    cfg = _cfg()
    cfg.compilation_config = None
    _platform("cuda", "metax").apply_config_platform_defaults(cfg)
    vendor_modules["ensure_patches"].assert_called_once_with()


def test_apply_config_platform_defaults_respects_eager_mode(vendor_modules):
    cfg = _cfg()
    cfg.compilation_config.mode = CompilationMode.NONE
    _platform("cuda", "metax").apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_apply_config_platform_defaults_respects_explicit_graph_mode(vendor_modules):
    cfg = _cfg()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    _platform("cuda", "metax").apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.FULL


def test_apply_config_platform_defaults_install_metax_plan(vendor_modules):
    cfg = _cfg()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = None
    cfg.compilation_config.cudagraph_mm_encoder = True
    cfg.compilation_config.cudagraph_capture_sizes = None
    _platform("cuda", "metax").apply_config_platform_defaults(cfg)
    comp = cfg.compilation_config
    assert comp.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert comp.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert comp.cudagraph_mm_encoder is False
    assert comp.pass_config.fuse_rope_kvcache_cat_mla is False


def test_apply_config_platform_defaults_keeps_existing_capture_sizes(vendor_modules):
    cfg = _cfg()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = None
    cfg.compilation_config.cudagraph_capture_sizes = [2, 4]
    cfg.compilation_config.pass_config = None
    _platform("cuda", "metax").apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == [2, 4]
