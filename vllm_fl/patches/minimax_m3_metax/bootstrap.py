# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Ordered initialization and effective-policy assertions for the sealed M3 deployment."""

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from vllm_fl.minimax_m3_config import load, thaw

_INITIALIZED = False


def _identity(fn):
    while hasattr(fn, "fn"):
        fn = fn.fn
    try:
        p = inspect.getsourcefile(fn)
    except TypeError:
        p = None
    return dict(
        module=getattr(fn, "__module__", None),
        qualname=getattr(fn, "__qualname__", type(fn).__qualname__),
        source=p,
        sha256=(
            hashlib.sha256(Path(p).read_bytes()).hexdigest()
            if p and Path(p).is_file()
            else None
        ),
    )


def initialize():
    global _INITIALIZED
    if _INITIALIZED:
        return
    profile = load()
    from .startup import configure_libraries

    from vllm_fl.dispatch.backends.flaggems.impl.minimax_m3.linear import (
        linear_with_config,
    )
    from vllm_fl.dispatch.backends.flaggems.impl.minimax_m3.numerics import gemma
    from vllm_fl.dispatch.backends.vendor.metax.impl.minimax_m3 import (
        attention_launch, moe_gemv,
    )

    required_parameters = {
        linear_with_config: {"config", "kind", "order", "group_m"},
        gemma: {"x", "weight", "eps", "residual"},
        attention_launch.index_score: {"config"},
        attention_launch.prefill: {"config"},
        attention_launch.decode: {"config", "decode_query_len"},
        moe_gemv.expert_gemv: {"top_k", "weighted", "config"},
    }
    for fn, parameters in required_parameters.items():
        if not parameters <= set(inspect.signature(fn).parameters):
            raise RuntimeError(
                f"Incompatible M3 adapter: {fn.__module__}.{fn.__name__}"
            )
    configure_libraries()
    import torch
    import triton
    import triton.language

    from vllm.platforms import current_platform

    assert getattr(current_platform, "vendor_name", None) == "metax"

    assert torch.__version__ == "2.10.0+metax3.8.1.0"
    assert importlib.metadata.version("vllm").split("+", 1)[0] == "0.24.0"
    assert importlib.metadata.version("flagtree") == "0.6.1+metax3.6"
    assert (
        Path(triton.__file__)
        .resolve()
        .is_relative_to(Path(os.environ["FLAGTREE_ROOT"]).resolve())
    )
    from vllm_fl.models.minimax_m3_registration import register_minimax_m3_support

    register_minimax_m3_support()
    # Correctness-sensitive registration must be fixed before WorkerFL enables ATen.
    from vllm_fl.dispatch.config import get_effective_config, get_flagos_blacklist
    from vllm_fl.utils import (
        get_flag_gems_whitelist_blacklist,
        get_oot_blacklist,
        is_oot_enabled,
    )

    assert is_oot_enabled() is True
    assert {k: get_effective_config()[k] for k in profile.dispatch} == thaw(
        profile.dispatch
    )
    assert get_flag_gems_whitelist_blacklist() == (
        None,
        list(profile.dispatch["flagos_blacklist"]),
    )
    assert get_flagos_blacklist() == list(profile.dispatch["flagos_blacklist"])
    assert get_oot_blacklist() == list(profile.dispatch["oot_blacklist"])
    from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle

    from vllm_fl.ops.minimax_m3.aten import MiniMaxRegistrar

    assert callable(MiniMaxRegistrar) and callable(moe_align_block_size_no_tle)
    assert profile.kernels.moe_alignment_tle is False
    from vllm_fl.attention.minimax_m3.bridge import initialize_consumer_contract

    initialize_consumer_contract()
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )

    assert (
        FusedTopKBiasRouter._compute_routing.__module__
        == "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
    )
    from vllm import _custom_ops

    assert _custom_ops.moe_sum.__module__ == "vllm._custom_ops"
    import vllm.model_executor.layers.fused_moe as native_moe
    import vllm.model_executor.layers.fused_moe.oracle.unquantized as native_oracle

    assert (
        native_moe.FusedMoE.__module__ == "vllm.model_executor.layers.fused_moe.layer"
    )
    assert (
        native_oracle.select_unquantized_moe_backend.__module__
        == "vllm.model_executor.layers.fused_moe.oracle.unquantized"
    )
    assert not any(
        n == "vllm_metax" or n.startswith("vllm_metax.") for n in sys.modules
    )
    from vllm_fl.dispatch import get_default_manager
    from .sampling import install as install_sampling

    # Vendor initialization installs the native fallback first.
    get_default_manager().ensure_initialized()
    install_sampling()
    _INITIALIZED = True
    audit_effective("import")


def audit_effective(stage, runner=None):
    import flag_gems
    import torch
    import triton

    from vllm_fl.attention.minimax_m3 import operators
    from vllm_fl.dispatch import CachedOp, get_default_manager, get_policy
    from vllm_fl.ops.minimax_m3 import experts, moe, router, routing, sampling, torch_ops
    from vllm_fl.utils import get_flag_gems_whitelist_blacklist, get_oot_blacklist

    mgr = get_default_manager()
    mgr.ensure_initialized()
    bindings = {}
    for module in (operators, torch_ops, experts, moe, routing, router, sampling):
        for obj in vars(module).values():
            if isinstance(obj, CachedOp):
                impl = obj.bind() if stage == "import" else obj.inspect_binding()
                assert not obj.allows_error_fallback
                bindings[obj.op_name] = {
                    "impl_id": impl.impl_id,
                    "kind": str(impl.kind),
                    "function": _identity(impl.fn),
                    "fallback_on": "never",
                }
    impl = mgr._resolve_impl("attention_backend")
    bindings["attention_backend"] = {
        "impl_id": impl.impl_id,
        "kind": str(impl.kind),
        "function": _identity(impl.fn),
    }
    expected = {
        "m3_sampling_stable_sort": ("vendor", "vendor.metax.m3.sampling_stable_sort"),
        "m3_index_score": ("vendor", "vendor.metax.m3.index_score"),
        "m3_sparse_attn": ("vendor", "vendor.metax.m3.sparse_attn"),
        "m3_sparse_decode": ("vendor", "vendor.metax.m3.sparse_decode"),
        "m3_router_bf16_fp32": ("flagos", "m3.flaggems_vllm.router_bf16_fp32"),
        "m3_expert_gemv": ("vendor", "vendor.metax.m3.expert_gemv"),
    }
    for name, (kind, impl_id) in expected.items():
        assert bindings[name]["kind"] == kind
        assert bindings[name]["impl_id"] == impl_id
    for name in (
        "index_topk", "index_score_native", "index_decode_native",
        "sparse_attn_native", "sparse_decode_native",
    ):
        assert bindings["m3_" + name]["kind"] == "vendor"
    from flag_gems.ops.gemma_rms_norm import gemma_rms_norm_kernel
    from flag_gems.ops.linear import linear_kernel
    from flag_gems.ops.linear_primitives import linear_gemv_untransposed_kernel
    from flaggems_vllm.ops.fp32_router_gemm import fp32_router_gemm

    library_apis = {
        name: _identity(fn)
        for name, fn in (
            ("gemma_rms_norm_kernel", gemma_rms_norm_kernel),
            ("linear_kernel", linear_kernel),
            ("linear_gemv_untransposed_kernel", linear_gemv_untransposed_kernel),
            ("fp32_router_gemm", fp32_router_gemm),
        )
    }
    registrar = getattr(flag_gems, "current_work_registrar", None)
    aten_bindings = {}
    if registrar is not None and hasattr(registrar, "selected_functions"):
        aten_bindings = {
            key: _identity(fn)
            for key, fn in registrar.selected_functions.items()
            if key in {"linear", "masked_fill_.Scalar", "masked_fill_.Tensor"}
        }
    if stage == "after_capture":
        from vllm_fl.ops.minimax_m3.aten import MiniMaxRegistrar
        from vllm_fl.ops.minimax_m3.linear import linear

        generic = importlib.import_module("flag_gems.ops.masked_fill").masked_fill_
        assert isinstance(registrar, MiniMaxRegistrar)
        assert "sort.values_stable" not in registrar.selected_functions
        import vllm.v1.sample.ops.topk_topp_sampler as sampling_ops

        assert sampling_ops.apply_top_k_top_p is sampling.apply_top_k_top_p
        assert registrar.selected_functions["linear"] is linear
        assert all(
            registrar.selected_functions[key] is generic
            for key in ("masked_fill_.Scalar", "masked_fill_.Tensor")
        )
        registered = set(flag_gems.aten_lib._op_impls)
        assert {
            "aten/masked_fill_.Scalar/CUDA",
            "aten/masked_fill_.Tensor/CUDA",
        } <= registered
        assert (
            not {"aten/masked_fill.Scalar/CUDA", "aten/masked_fill.Tensor/CUDA"}
            & registered
        )
    bad = [
        k
        for k in os.environ
        if k.startswith("M3_")
        or k
        in (
            "VLLM_FL_FLAGOS_BLACKLIST",
            "VLLM_FL_FLAGOS_WHITELIST",
            "VLLM_FL_OOT_BLACKLIST",
            "VLLM_FL_OOT_WHITELIST",
        )
    ]
    assert not bad, ("Legacy project env survived", bad)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
    original_model_modules = [
        name
        for name in sys.modules
        if name == "vllm.models.minimax_m3"
        or name.startswith("vllm.models.minimax_m3.")
    ]
    record = dict(
        stage=stage,
        pid=os.getpid(),
        rank=rank,
        profile_sha256=load().fingerprint,
        framework=asdict(load().framework),
        kernels=asdict(load().kernels),
        dispatch=thaw(load().dispatch),
        whitelist_blacklist=get_flag_gems_whitelist_blacklist(),
        oot_blacklist=get_oot_blacklist(),
        bindings=bindings,
        aten_bindings=aten_bindings,
        library_apis=library_apis,
        policy=str(get_policy()),
        triton_module=triton.__file__,
        native_router_unchanged=True,
        native_moe_sum_unchanged=True,
        original_model_loaded=bool(original_model_modules),
        effective_env={
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                (
                    "VLLM_",
                    "FLAG",
                    "GEMS",
                    "USE_FLAG",
                    "TRITON_",
                    "TORCHINDUCTOR_",
                    "MCCL_",
                    "NCCL_",
                    "GLOO_",
                    "MACA_",
                    "PYTHONPATH",
                )
            )
        },
    )
    if runner is not None:
        record["runner"] = type(runner).__module__ + "." + type(runner).__name__
        scheduler = runner.scheduler_config.get_scheduler_cls()
        record["scheduling"] = {
            "async_scheduling": runner.use_async_scheduling,
            "scheduler": scheduler.__module__ + "." + scheduler.__name__,
        }
        assert runner.use_async_scheduling == load().service.get("async_scheduling", False)
        if runner.use_async_scheduling:
            assert scheduler.__name__ == "AsyncScheduler"
        import vllm.v1.sample.ops.topk_topp_sampler as sampling_ops

        record["sampling_apis"] = {
            name: _identity(getattr(sampling_ops, name))
            for name in (
                "apply_top_k_top_p", "random_sample", "sample_with_exponential_noise"
            )
        }
    root = Path(os.environ["VLLM_CACHE_ROOT"]).parent / "audit"
    root.mkdir(parents=True, exist_ok=True)
    (
        root / (stage + "-rank" + str(rank) + "-pid" + str(os.getpid()) + ".json")
    ).write_text(json.dumps(record, indent=2) + "\n")
    print(
        "M3_EFFECTIVE_CONFIGURATION "
        + json.dumps(
            dict(
                stage=stage,
                rank=rank,
                profile=load().fingerprint,
                bindings=len(bindings),
            )
        ),
        flush=True,
    )
    return record


def audit_memory_budget(worker, returned_bytes):
    """Record already-computed public budget fields without GPU operations."""
    import torch

    record = {
        "rank": torch.distributed.get_rank(),
        "profile_sha256": load().fingerprint,
        "requested_memory": worker.requested_memory,
        "model_memory_usage": worker.model_runner.model_memory_usage,
        "peak_activation_memory": worker.peak_activation_memory,
        "non_torch_memory": worker.non_torch_memory,
        "cudagraph_memory_estimate": worker.cudagraph_memory_estimate,
        "available_kv_cache_memory_bytes": worker.available_kv_cache_memory_bytes,
        "returned_bytes": returned_bytes,
        "initial_free_memory": worker.init_snapshot.free_memory,
        "total_memory": worker.init_snapshot.total_memory,
    }
    root = Path(os.environ["VLLM_CACHE_ROOT"]).parent / "audit"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"memory-rank{record['rank']}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print("M3_MEMORY_BUDGET " + json.dumps(record), flush=True)
