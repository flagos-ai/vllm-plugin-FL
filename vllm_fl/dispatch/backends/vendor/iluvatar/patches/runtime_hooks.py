# SPDX-License-Identifier: Apache-2.0
"""Declarative thin-backend runtime hooks assembling shared `vllm_fl` engines for Iluvatar BI-V150 (`ivcore11`).

Keeps hardware-specific code strictly confined to configuration constants and
FlagGems `_iluvatar` kernel handles while delegating FX graph fusion, load-time
`QuaRot-R2` / `0-ulp` dyadic quantization, 12-in-1 step preparation, HBM block
coalescing, and chunked sort-free sampling to vendor-neutral `vllm_fl` modules.
"""

from __future__ import annotations

from functools import wraps
import logging
import sys
from typing import Any

import torch

from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_attention import prepare_attention_kernels
from flag_gems.runtime.backend._iluvatar.fused.topp_sampling import (
    fl_apply_temperature,
    fl_fast_topp_gumbel_sample,
    prewarm_topp_sampler,
)
from flag_gems.runtime.backend._iluvatar.ops.quant_gemm import prewarm_quant_layer
from vllm_fl.compilation import QuantFusionConfig, QuantFusionPass, RopeKvFusionPass, register_post_grad_passes
from vllm_fl.quantization.quant_transform import (
    QuantTransformConfig,
    apply_online_quant_transform,
    has_qk_norm,
    resolve_attn_geometry,
)
from vllm_fl.worker import (
    StepPrepConfig,
    install_locality_free_block_queue,
    install_sort_free_sampler_hooks,
    install_step_prep,
    install_step_prep_on_runner_module,
)

from .kv_cache_k8v4 import (
    enable_auto_iluvatar_kv_cache_config,
    install_kv_cache_k8v4_hooks,
    install_reshape_kv_hook_on_module,
)

logger = logging.getLogger(__name__)
_HOOKS_INSTALLED: bool = False
# Post-grad FX fusion policy for Iluvatar BI-V150 (`ivcore11`): enable Hadamard
# rotation on RMSNorm and SwiGLU paths, interleaved SwiGLU fusion, and fused attention output quant.
_ILUVATAR_FUSION_CFG = QuantFusionConfig(
    norm_do_hadamard=True,
    down_do_hadamard=True,
    down_num_scale_groups=1,
    fuse_interleaved_swiglu=True,
    fuse_attn_out_quant=True,
)
# Stage Prefill-over-Decode (POD) sub-batch horizons during step preparation.
_ILUVATAR_STEP_PREP_CFG = StepPrepConfig(enable_pod_horizon=True)


def _iluvatar_extra_prewarm(unique_cfgs: Any, attn_sig: Any, sampler_sig: Any) -> None:
    """Pre-compile and warm up GEMM, K8V4 Paged Attention, and Top-P kernels prior to graph capture."""
    for cfg, (c_dev, c_dt) in unique_cfgs.items():
        prewarm_quant_layer(cfg, c_dev, c_dt)
    if attn_sig is not None:
        prepare_attention_kernels(attn_sig[0], num_heads=attn_sig[1], num_kv_heads=attn_sig[2], head_dim=attn_sig[3], use_k8v4=True)
    if sampler_sig is not None:
        prewarm_topp_sampler(*sampler_sig)


_ILUVATAR_QUANT_CFG = QuantTransformConfig(
    norm_do_hadamard=True, down_do_hadamard=True, down_num_scale_groups=1,
    untied_lm_head_w8a8=True, alpha_min=0.85, extra_prewarm_fn=_iluvatar_extra_prewarm,
)


def _install_sampler() -> None:
    """Bind Iluvatar FlagGems sort-free Top-P / Gumbel-Max and temperature kernels to `Sampler`."""
    install_sort_free_sampler_hooks(fl_fast_topp_gumbel_sample, apply_temp_fn=fl_apply_temperature, prewarm_fn=prewarm_topp_sampler)


def _install_platform_and_loader_hooks() -> None:
    """Idempotently wrap `PlatformFL.check_and_update_config` and `BaseModelLoader.load_model`.

    Lifecycle across processes:
    - `PlatformFL.check_and_update_config` runs before `EngineCore` spawning (and
      again inside workers), upgrading `'auto'` KV cache to `'int8_per_token_head'`
      (`DG-R2-K8V4`), registering `QuantFusionPass` + `RopeKvFusionPass` (with a
      cloudpickle-safe string op spec), and tagging `additional_config["_fl_compile_tag"]`
      so TorchInductor cache keys stay isolated.
    - `BaseModelLoader.load_model` runs in each GPU worker immediately after raw
      checkpoint weights finish loading, transforming weights into `QuantWeightTensor`s
      and freeing BF16 storage before KV-cache memory profiling runs.
    """
    import vllm.model_executor.model_loader.base_loader as base_loader
    from vllm_fl.platform import PlatformFL

    orig_desc = PlatformFL.__dict__.get("check_and_update_config")
    orig_check = orig_desc.__func__ if isinstance(orig_desc, classmethod) else orig_desc
    if orig_check is not None and not getattr(orig_check, "_fl_iluvatar_patched", False):

        @wraps(orig_check)
        def _checked_config(cls: Any, vllm_config: Any) -> None:
            orig_check(cls, vllm_config)
            if vllm_config is None or getattr(cls, "vendor_name", "") != "iluvatar":
                return
            enable_auto_iluvatar_kv_cache_config(vllm_config)
            register_post_grad_passes(
                vllm_config,
                QuantFusionPass(_ILUVATAR_FUSION_CFG),
                RopeKvFusionPass(
                    rope_kv_op="vllm::fused_rope_and_cache_k8v4.default",
                    qk_norm_rope_kv_op="vllm::fused_qk_norm_rope_and_cache_k8v4.default",
                ),
            )
            if (add_cfg := getattr(vllm_config, "additional_config", None)) is None:
                vllm_config.additional_config = {"_fl_compile_tag": "iluvatar_quant_k8v4_v7"}
            elif isinstance(add_cfg, dict):
                add_cfg["_fl_compile_tag"] = "iluvatar_quant_k8v4_v7"

        setattr(_checked_config, "_fl_iluvatar_patched", True)
        PlatformFL.check_and_update_config = classmethod(_checked_config)

    loader_cls = base_loader.BaseModelLoader
    if not getattr(loader_cls.load_model, "_fl_iluvatar_patched", False):
        orig_load = loader_cls.load_model

        @wraps(orig_load)
        def _load_and_quantize(self: Any, vllm_config: Any, model_config: Any, prefix: str = "") -> torch.nn.Module:
            load_dummy = (
                type(self).__name__ == "DummyModelLoader"
                or str(getattr(getattr(self, "load_config", None), "load_format", "")).lower() == "dummy"
                or str(getattr(getattr(vllm_config, "load_config", None), "load_format", "")).lower() == "dummy"
            )
            model = orig_load(self, vllm_config, model_config, prefix=prefix)
            eff_mc = model_config or getattr(vllm_config, "model_config", None)
            if not load_dummy and not getattr(eff_mc, "quantization", None):
                apply_online_quant_transform(model, eff_mc, _ILUVATAR_QUANT_CFG)
                # Prewarm the 5-in-1 QK-Norm+RoPE+K8V4 kernel variant when the model uses per-head QK-Norm.
                layers = getattr(getattr(model, "model", model), "layers", None)
                attn0 = getattr(layers[0], "self_attn", None) if layers else None
                if attn0 is not None and has_qk_norm(attn0) and (g0 := resolve_attn_geometry(attn0)) is not None:
                    p0 = next(model.parameters(), None)
                    if p0 is not None and p0.is_cuda:
                        prepare_attention_kernels(p0.device, num_heads=g0[0], num_kv_heads=g0[1], head_dim=g0[2], use_k8v4=True, has_qk_norm=True)
            return model

        setattr(_load_and_quantize, "_fl_iluvatar_patched", True)
        loader_cls.load_model = _load_and_quantize


def install_worker_runtime_hooks() -> None:
    """Bind step-prep, locality KV queue, KV-view, and sampler hooks on worker modules."""
    install_kv_cache_k8v4_hooks()
    install_locality_free_block_queue()
    _install_sampler()
    install_step_prep(_ILUVATAR_STEP_PREP_CFG, import_runners=True)
    for mod_name in ("vllm.v1.worker.gpu_model_runner", "vllm_fl.worker.model_runner"):
        install_reshape_kv_hook_on_module(sys.modules.get(mod_name))


def install_iluvatar_runtime_hooks() -> None:
    """Install all explicit Iluvatar runtime hooks idempotently."""
    global _HOOKS_INSTALLED
    if not _HOOKS_INSTALLED:
        _HOOKS_INSTALLED = True
        _install_platform_and_loader_hooks()
        install_kv_cache_k8v4_hooks()
        install_locality_free_block_queue()
        install_step_prep(_ILUVATAR_STEP_PREP_CFG, import_runners=False)
        try:
            _install_sampler()
        except Exception:
            logger.debug("Deferred sampler hook installation until worker init.")
    for mod_name in ("vllm.v1.worker.gpu_model_runner", "vllm_fl.worker.model_runner"):
        if (mod := sys.modules.get(mod_name)) is not None:
            install_reshape_kv_hook_on_module(mod)
            install_step_prep_on_runner_module(mod)
