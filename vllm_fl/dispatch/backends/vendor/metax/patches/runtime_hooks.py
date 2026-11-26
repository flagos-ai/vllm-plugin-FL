# SPDX-License-Identifier: Apache-2.0
"""Thin runtime hooks assembling common `vllm_fl` engines for MetaX C500 (`xcore1000`)."""

from __future__ import annotations

from functools import wraps
import logging
import sys
from typing import Any

import torch

from flag_gems.utils.tile_solver import get_device_topology
from vllm_fl.compilation import QuantFusionConfig, QuantFusionPass, RopeKvFusionPass, register_post_grad_passes
from vllm_fl.quantization.quant_transform import QuantTransformConfig, apply_online_quant_transform
from vllm_fl.worker import (
    StepPrepConfig,
    install_locality_free_block_queue,
    install_sort_free_sampler_hooks,
    install_step_prep,
    install_step_prep_on_runner_module,
)

logger = logging.getLogger(__name__)
_HOOKS_INSTALLED: bool = False
_SAMPLER_INSTALLED: bool = False

# Post-grad FX fusion policy tuned for MetaX C500 (`xcore1000` homogeneous W8A8):
# - `norm_do_hadamard=True`: Fuses block-64 orthogonal Hadamard rotation directly inside
#   `rmsnorm_quant` / `fused_add_rmsnorm_quant` (`+0.16%` decode latency) so `qkv_proj` and
#   `gate_up_proj` inputs match offline-rotated weights for maximal INT8 accuracy.
# - `down_do_hadamard=True, down_num_scale_groups=0`: `SwiGLU` (`up * silu(gate)`)
#   quadratically amplifies activation spikes entering `down_proj`; `0` auto-derives the
#   scale-group count from hardware Wave64 chunk width (`warp_size * 32`) to bound per-group
#   crest factor without cross-CTA synchronization or accumulator spills.
# - `fuse_interleaved_swiglu=False`: Keeps contiguous `[gate, up]` halves so `silu_mul_quant`
#   streams directly into independent Hadamard scale groups without weight-channel interleaving.
# - `fuse_attn_out_quant=True`: Fuses dynamic per-token INT8 quantization onto attention
#   output before `o_proj`.
_METAX_FUSION_CFG = QuantFusionConfig(
    norm_do_hadamard=True,
    down_do_hadamard=True,
    down_num_scale_groups=0,
    fuse_interleaved_swiglu=False,
    fuse_attn_out_quant=True,
)
# Disable speculative CPU/GPU horizon staging on MACA and align pure-decode batch
# padding to FULL CUDAGraph capture buckets (`1, 2, 4, 8, 16, ..., 64`).
_METAX_STEP_PREP_CFG = StepPrepConfig(
    enable_pod_horizon=False,
    decode_pad_buckets=(1, 2, 4, *range(8, 65, 8)),
)


def _metax_extra_prewarm(unique_cfgs: Any, attn_sig: Any, sampler_sig: Any) -> None:
    """Eagerly populate `FastIntervalLauncher` handles and persistent workspaces before step 1."""
    import flag_gems
    from flag_gems.runtime.backend._metax.ops.quant_gemm import prewarm_quant_layer

    max_dec_m = int(_METAX_STEP_PREP_CFG.decode_pad_buckets[-1])
    l2_bytes = get_device_topology().l2_cache_bytes
    warmed_norms: set[tuple[torch.device, torch.dtype, int, bool]] = set()
    infer_dt = next(iter(unique_cfgs.values()))[1] if unique_cfgs else (
        sampler_sig[1] if sampler_sig is not None else torch.bfloat16
    )
    with torch.inference_mode():
        for qcfg, (dev, dt) in unique_cfgs.items():
            norm_key = (dev, dt, int(qcfg.shape_k), bool(qcfg.do_hadamard))
            do_norm = (
                qcfg.num_scale_groups == 1
                and qcfg.shape_n * qcfg.shape_k <= 24 * l2_bytes
                and norm_key not in warmed_norms
            )
            if do_norm:
                warmed_norms.add(norm_key)
            prewarm_quant_layer(qcfg, dev, dtype=dt, max_decode_m=max_dec_m, warm_norm=do_norm)
        if attn_sig is not None:
            from ..impl.attention.flash_attn import prewarm_metax_rope_kv_workspaces

            prewarm_metax_rope_kv_workspaces(attn_sig[0], infer_dt, attn_sig[1], attn_sig[2], attn_sig[3])
        if sampler_sig is not None:
            s_dev, s_dt, vocab_sz = sampler_sig
            logits_w = torch.zeros((max_dec_m, vocab_sz), device=s_dev, dtype=s_dt)
            for b in (1, max_dec_m):
                flag_gems.argmax(logits_w[:b], dim=-1)
                flag_gems.argmax(logits_w[:b].float(), dim=-1)
            del logits_w
            from flag_gems.runtime.backend._metax.fused import prewarm_topp_sampler

            prewarm_topp_sampler(s_dev, s_dt, vocab_sz)


# Offline-at-load W8A8 weight transform policy paired with `_METAX_FUSION_CFG`:
# - `norm_do_hadamard=True`: Absorbs orthogonal block-64 `xi` sign-flipped Hadamard rotation
#   offline into `ln1`/`ln2` weights and downstream `qkv_proj`/`gate_up_proj` weights, matched by
#   online `do_hadamard=True` inside `rmsnorm_quant`/`fused_add_rmsnorm_quant`.
# - `enable_quarot_r2=True`: Fuses an orthogonal per-head `QuaRot-R2` rotation offline into
#   `v_proj` (output) and `o_proj` (input) weights, dispersing multi-head value/attention
#   outliers across head dimensions at zero runtime cost before `o_proj` W8A8 quantization.
# - `enable_qk_dyadic=True, enable_up_down_dyadic=True`: Applies exact power-of-two (`2^k`)
#   channel equilibration across NeoX-coupled `q_proj`/`k_proj` heads and SwiGLU-gated
#   `up_proj`/`down_proj` channels at zero runtime cost (`down_do_hadamard=True` +
#   auto-derived `down_num_scale_groups=0` still suppresses remaining SwiGLU spikes).
# - `untied_lm_head_w8a8=True, lm_head_do_hadamard=True`: Converts untied `lm_head` to
#   homogeneous W8A8 with chunked offline block-64 Hadamard rotation absorbed into
#   `model.norm.weight` and paired runtime activation rotation before vocabulary GEMM.
# - `alpha_min=0.85, gamma_post_rmsnorm=0.0, gamma_other=0.25`: Tightens SW-MGOC clipping
#   search floor to `0.85` on Hadamard-smoothed projections (`qkv_proj`, `gate_up_proj`,
#   `lm_head`) while `gamma_other=0.25` preserves static weight-outlier damping on
#   non-RMSNorm projections (`o_proj`, `down_proj`, `lm_head`).
_METAX_QUANT_CFG = QuantTransformConfig(
    norm_do_hadamard=True,
    down_do_hadamard=True,
    down_num_scale_groups=0,
    enable_qk_dyadic=True,
    enable_quarot_r2=True,
    enable_up_down_dyadic=True,
    untied_lm_head_w8a8=True,
    lm_head_do_hadamard=True,
    alpha_min=0.85,
    gamma_post_rmsnorm=0.0,
    gamma_other=0.25,
    extra_prewarm_fn=_metax_extra_prewarm,
)


def _install_sampler() -> None:
    """Install MetaX sort-free Gumbel Top-P / Top-K sampler hooks once per process."""
    global _SAMPLER_INSTALLED
    if _SAMPLER_INSTALLED:
        return
    from flag_gems.runtime.backend._metax.fused import (
        apply_top_k_top_p_metax,
        fl_fast_topp_gumbel_sample,
        prewarm_topp_sampler,
        topk_topp_forward_metax,
    )

    install_sort_free_sampler_hooks(
        fl_fast_topp_gumbel_sample,
        prewarm_fn=prewarm_topp_sampler,
        apply_topk_topp_fn=apply_top_k_top_p_metax,
        topk_topp_forward_fn=topk_topp_forward_metax,
    )
    _SAMPLER_INSTALLED = True


def _configure_metax_compilation(vllm_config: Any) -> None:
    """Configure CUDAGraph regime, unsplit FX graph, and post-grad fusion passes for C500."""
    if vllm_config is None:
        return
    from vllm.config import CUDAGraphMode

    if (cc := getattr(vllm_config, "compilation_config", None)) is not None:
        # Never run `PIECEWISE` CUDAGraphs at `M=2048` on MetaX MACA: `mcGraphLaunch`
        # injects a hidden stream-synchronization barrier at every subgraph boundary
        # (85 subgraphs/step across 42 layers), destroying async launch overlap across
        # attention layers. Restrict CUDAGraphs to `FULL_DECODE_ONLY` (`M <= 64`) and
        # run Prefill / Mixed Prefill+Decode (`M=2048`) in zero-sync Eager mode.
        if cc.cudagraph_mode is not None and cc.cudagraph_mode.has_piecewise_cudagraphs():
            cc.cudagraph_mode = (
                CUDAGraphMode.FULL_DECODE_ONLY
                if cc.cudagraph_mode.has_full_cudagraphs()
                else CUDAGraphMode.NONE
            )
        # Clear `splitting_ops` so TorchInductor compiles a single unsplit post-grad FX
        # graph across attention, allowing `QuantFusionPass` and `RopeKvFusionPass` to
        # fuse QK-RoPE + KV-cache writes and `o_proj`/RMSNorm/SwiGLU INT8 quant across boundaries.
        cc.splitting_ops = []
    register_post_grad_passes(vllm_config, QuantFusionPass(_METAX_FUSION_CFG), RopeKvFusionPass())
    # Version the Inductor cache key so fused MetaX graphs never collide with vanilla cached artifacts.
    if (add_cfg := getattr(vllm_config, "additional_config", None)) is None:
        vllm_config.additional_config = {"_fl_compile_tag": "metax_quant_rope_v4"}
    elif isinstance(add_cfg, dict):
        add_cfg["_fl_compile_tag"] = "metax_quant_rope_v4"


def _install_platform_and_loader_hooks() -> None:
    """Wire compilation config and online W8A8 weight transform into vLLM startup lifecycle.

    Hooking both `PlatformFL.check_and_update_config` (scheduler/engine process) and
    `BaseModelLoader.load_model` (GPU worker process) ensures compilation flags, post-grad
    fusion passes, and online W8A8 weight conversion activate automatically under standard
    `VLLM_PLUGINS=fl` discovery without requiring custom environment variables.
    """
    import vllm.model_executor.model_loader.base_loader as base_loader
    from vllm_fl.platform import PlatformFL

    orig_desc = PlatformFL.__dict__.get("check_and_update_config")
    orig_check = orig_desc.__func__ if isinstance(orig_desc, classmethod) else orig_desc
    if orig_check is not None and not getattr(orig_check, "_fl_metax_patched", False):

        @wraps(orig_check)
        def _checked_config(cls: Any, vllm_config: Any) -> None:
            orig_check(cls, vllm_config)
            if vllm_config is not None and getattr(cls, "vendor_name", "") == "metax":
                _configure_metax_compilation(vllm_config)

        setattr(_checked_config, "_fl_metax_patched", True)
        PlatformFL.check_and_update_config = classmethod(_checked_config)

    loader_cls = base_loader.BaseModelLoader
    if not getattr(loader_cls.load_model, "_fl_metax_patched", False):
        orig_load = loader_cls.load_model

        @wraps(orig_load)
        def _load_and_quantize(self: Any, vllm_config: Any, model_config: Any, prefix: str = "") -> torch.nn.Module:
            # Re-assert compilation passes in worker processes spawned before config serialization.
            _configure_metax_compilation(vllm_config)
            try:
                _install_sampler()
            except Exception:
                logger.debug("Deferred MetaX sampler hook installation.")
            load_dummy = (
                type(self).__name__ == "DummyModelLoader"
                or str(getattr(getattr(self, "load_config", None), "load_format", "")).lower() == "dummy"
                or str(getattr(getattr(vllm_config, "load_config", None), "load_format", "")).lower() == "dummy"
            )
            model = orig_load(self, vllm_config, model_config, prefix=prefix)
            eff_mc = model_config or getattr(vllm_config, "model_config", None)
            # Apply online SmoothQuant + QuaRot-R2 + W8A8 conversion only to unquantized real
            # weights (skipping profiling dummy loads and pre-quantized checkpoints);
            # `apply_online_quant_transform` reclaims replaced BF16 storage immediately.
            if not load_dummy and not getattr(eff_mc, "quantization", None):
                apply_online_quant_transform(model, eff_mc, _METAX_QUANT_CFG)
            return model

        setattr(_load_and_quantize, "_fl_metax_patched", True)
        loader_cls.load_model = _load_and_quantize


def install_metax_runtime_hooks() -> None:
    """Install all explicit MetaX runtime hooks idempotently across main and worker processes."""
    global _HOOKS_INSTALLED
    if not _HOOKS_INSTALLED:
        _HOOKS_INSTALLED = True
        _install_platform_and_loader_hooks()
        # Replace FIFO free-block ordering when APC is off so fresh allocations keep contiguous page IDs.
        install_locality_free_block_queue()
        install_step_prep(_METAX_STEP_PREP_CFG, import_runners=True)
        try:
            _install_sampler()
        except Exception:
            logger.debug("Deferred MetaX sampler hook installation until worker init.")
        from ..impl.attention.flash_attn import _install_adaptive_cudagraph_regime_hook

        _install_adaptive_cudagraph_regime_hook()
    else:
        # Patch BOTH upstream `GPUModelRunner` and `vllm_fl`'s `ModelRunnerFL` (`WorkerFL` instantiates
        # `ModelRunnerFL`, which overrides `_prepare_inputs` and `_get_slot_mappings` in `vllm serve`).
        for mod_name in ("vllm.v1.worker.gpu_model_runner", "vllm_fl.worker.model_runner"):
            if (mod := sys.modules.get(mod_name)) is not None:
                install_step_prep_on_runner_module(mod)


__all__ = ["install_metax_runtime_hooks"]
