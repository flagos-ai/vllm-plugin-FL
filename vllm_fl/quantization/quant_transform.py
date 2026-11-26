# SPDX-License-Identifier: Apache-2.0
"""Vendor-neutral calibration-free online W8A8 quantization and algebraic gauge engine.

Applies three exact / orthogonal load-time equivalence transformations in FP32
before packing model weights into `QuantWeightTensor` instances:
1. `QuaRot-R2` GQA Head-Wise Orthogonal Absorption: absorbs randomized Walsh-Hadamard
   matrices $Q_h = H_d \\operatorname{diag}(\\xi_h) / \\sqrt{d}$ ($Q_h Q_h^T = I_d$)
   across each KV head's $(W_V, b_V)$ and its $G = H_q / H_{kv}$ paired $W_O$ slices,
   achieving outlier-free $V$ / Attention-Out distributions at zero runtime cost.
2. `0-ulp` IEEE-754 Dyadic Gauge Scaling: scales paired $(W_Q, W_K)$ RoPE planes
   and SwiGLU $(W_{up}, W_{down})$ channels by integer powers of two $2^k$ via
   `torch.ldexp`, shifting only IEEE-754 exponent bits while preserving mantissa
   bits (`0-ulp`) and exact mathematical equivalence.
3. Deterministic Rademacher Sign Absorption: folds $\\xi \\in \\{\\pm 1\\}^K$ sign
   vectors into preceding `RMSNorm` weights (and `up_proj` rows), turning deterministic
   runtime block-64 FWHT into randomized Hadamard preconditioning with zero overhead.

Physical weight memory layouts (row-major, preshuffled MMA tiles, or 64-column
SwiGLU `[Gate, Up]` interleaving) are delegated to `resolve_quant_weight_layout`
so this module remains strictly hardware-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
import logging
import time
from typing import Any

import torch
import torch.nn as nn

import flag_gems  # noqa: F401 - ensures active vendor layout resolvers are registered
from flag_gems.utils.quant_config import (
    QuantLayerConfig,
    QuantWeightTensor,
    quantize_linear_weight,
    resolve_quant_weight_layout,
)
from flag_gems.utils.quant_precondition import fwht_lastdim_fp32, make_deterministic_rademacher
from vllm_fl.quantization.online_linear_walker import (
    DecoderLayerLinearView,
    iter_causal_decoder_layers,
    resolve_untied_lm_head,
)

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.fl.quant_transform")
except Exception:
    logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantTransformConfig:
    """Backend-agnostic configuration for `apply_online_quant_transform`.

    Attributes:
        norm_do_hadamard: Enable block-64 randomized Walsh-Hadamard preconditioning
            on `RMSNorm`-fed projections (`qkv_proj` and `gate_up_proj`).
        down_do_hadamard: Enable block-64 randomized Walsh-Hadamard preconditioning
            on `down_proj` inputs.
        down_num_scale_groups: Number of per-row dynamic INT8 activation scale
            groups for `down_proj` (e.g., 2 or 4 for wide intermediate dimensions).
        enable_qk_dyadic: Enable `0-ulp` SO(2)-commuting power-of-two gauge
            balancing between `W_Q` and `W_K`.
        enable_quarot_r2: Enable GQA head-wise orthogonal `QuaRot-R2` rotation
            across `(W_V, b_V)` and `W_O` (eliminating runtime Hadamard on `o_proj`).
        enable_up_down_dyadic: Enable `0-ulp` power-of-two channel balancing
            between `W_up` and `W_down`.
        untied_lm_head_w8a8: Quantize untied `ParallelLMHead` weights to `W8A8`
            (with Rademacher signs absorbed into `model.norm.weight`) when eligible.
        lm_head_do_hadamard: Enable block-64 Hadamard smoothing on `W8A8` `lm_head`.
        alpha_min: Lower bound for clipping-search weight scale optimization.
        gamma_post_rmsnorm: Weight quantization smoothing exponent for post-norm layers.
        gamma_other: Weight quantization smoothing exponent for non-norm layers.
        extra_prewarm_fn: Optional vendor callback invoked after quantization to
            pre-populate kernel dispatch caches before CUDA Graph capture.
    """

    norm_do_hadamard: bool = True
    down_do_hadamard: bool = True
    down_num_scale_groups: int = 1
    enable_qk_dyadic: bool = True
    enable_quarot_r2: bool = True
    enable_up_down_dyadic: bool = True
    untied_lm_head_w8a8: bool = True
    lm_head_do_hadamard: bool = True
    alpha_min: float = 0.85
    gamma_post_rmsnorm: float = 0.0
    gamma_other: float = 0.0
    extra_prewarm_fn: Callable[..., None] | None = None


def compute_up_down_dyadic_scales(
    w_gate_f32: torch.Tensor, w_up_f32: torch.Tensor, w_down_f32: torch.Tensor,
    alpha: float = 0.5, max_exp: int = 2,
) -> torch.Tensor:
    """Compute exact power-of-two SwiGLU channel gauge scales $s_j = 2^{e_j}$ ($e_j \\in [0, \\text{max\\_exp}]$).

    In SwiGLU, `silu(gate_j) * up_j` feeds column `j` of `down_proj`. Scaling only
    the linear `up_proj` row by $s_j^{-1} = 2^{-e_j}$ and `down_proj` column `j`
    by $s_j = 2^{e_j}$ preserves exact equality:
        $\\operatorname{SiLU}(g_j) \\cdot (u_j / 2^{e_j}) \\cdot (2^{e_j} w_{\\text{down}, j}) = \\operatorname{SiLU}(g_j) u_j w_{\\text{down}, j}$
    while shifting only IEEE-754 exponent bits (`0-ulp` mantissa preservation) and
    suppressing heavy-tailed SwiGLU activation spikes before `down_proj` W8A8 GEMM.
    """
    act_sig = w_gate_f32.pow(2).mean(dim=1).sqrt().clamp_min(1e-12) * w_up_f32.pow(2).mean(dim=1).sqrt().clamp_min(1e-12)
    act_ratio = act_sig / act_sig.median().clamp_min(1e-12)
    dn_rms = w_down_f32.pow(2).mean(dim=0).sqrt().clamp_min(1e-12)
    dn_ratio = dn_rms / dn_rms.median().clamp_min(1e-12)
    raw_exp = torch.round(alpha * torch.log2((act_ratio / dn_ratio.clamp_min(1e-6)).clamp_min(1e-6)))
    exp = torch.where((act_ratio >= 2.0) & (dn_ratio <= 2.5), raw_exp.clamp(0, max_exp), torch.zeros_like(raw_exp))
    return torch.ldexp(torch.ones_like(exp), exp.to(torch.int32))


def apply_quarot_r2_heads_fp32(
    wv_f32: torch.Tensor, wo_f32: torch.Tensor, num_kv_heads: int, gqa_ratio: int,
    head_dim: int, layer_idx: int, device: torch.device | str,
    bv_f32: torch.Tensor | None = None, kv_head_base: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply GQA-aware `QuaRot-R2` head-wise orthogonal rotation across `(W_V, W_O, b_V)`.

    For each KV head `kv_h`, constructs a deterministic Rademacher-Hadamard
    orthogonal operator $Q_h = \\frac{1}{\\sqrt{d}} H_d \\operatorname{diag}(\\xi_h)$
    satisfying $Q_h Q_h^T = I_d$, and transforms:
      - $W_{V, h}' = Q_h W_{V, h}$  (and $b_{V, h}' = Q_h b_{V, h}$ if present)
      - $W_{O, :, h, g}' = W_{O, :, h, g} Q_h^T$ for all $g \\in [0, \\text{gqa\\_ratio})$
    Because softmax attention forms a scalar-weighted sum $\\sum_t \\alpha_t v_{t, h}$
    within each head, $Q_h$ commutes through attention and cancels identically
    against $Q_h^T$ inside `o_proj`, requiring zero runtime Hadamard kernels while
    simultaneously smoothing $V$ for sub-8-bit KV cache quantization (e.g. K8V4)
    and attention outputs for `o_proj` W8A8 quantization.
    """
    kv_size, q_size, hidden_size = num_kv_heads * head_dim, num_kv_heads * head_dim * gqa_ratio, wv_f32.shape[1]
    wv_v = wv_f32.to(torch.float32).clone().view(num_kv_heads, head_dim, hidden_size)
    wo_v = wo_f32.to(torch.float32).clone().view(hidden_size, num_kv_heads, gqa_ratio, head_dim)
    bv_v = bv_f32.to(torch.float32).clone().view(num_kv_heads, head_dim) if bv_f32 is not None else None
    for kv_h in range(num_kv_heads):
        xi_v = make_deterministic_rademacher(head_dim, 10000 + layer_idx * 17 + kv_head_base + kv_h, device)
        wv_v[kv_h] = fwht_lastdim_fp32((wv_v[kv_h] * xi_v[:, None]).t().contiguous(), head_dim).t()
        wo_v[:, kv_h] = fwht_lastdim_fp32(wo_v[:, kv_h] * xi_v[None, None, :], head_dim)
        if bv_v is not None:
            bv_v[kv_h] = fwht_lastdim_fp32(bv_v[kv_h] * xi_v, head_dim)
    return (
        wv_v.reshape(kv_size, hidden_size).contiguous(),
        wo_v.reshape(hidden_size, q_size).contiguous(),
        bv_v.reshape(kv_size).contiguous() if bv_v is not None else None,
    )


def _resolve_attn_geometry(attn_mod: nn.Module) -> tuple[int, int, int, bool] | None:
    """Resolve `(num_heads, num_kv_heads, head_dim, is_neox_style)` from an attention module.

    Returns `None` if geometry attributes are missing, GQA divisibility fails,
    `head_dim` is odd, or partial RoPE (`rotary_dim != head_dim`) is used.
    """
    inner = getattr(attn_mod, "attn", None)
    nh = getattr(attn_mod, "num_heads", None) or getattr(inner, "num_heads", None)
    nkv = getattr(attn_mod, "num_kv_heads", None) or getattr(inner, "num_kv_heads", None)
    hd = getattr(attn_mod, "head_dim", None) or getattr(inner, "head_size", None)
    if not (isinstance(nh, int) and isinstance(nkv, int) and isinstance(hd, int)):
        return None
    if nh <= 0 or nkv <= 0 or hd <= 0 or (nh % nkv != 0) or (hd % 2 != 0):
        return None
    rope = getattr(attn_mod, "rotary_emb", None)
    if rope is not None and getattr(rope, "rotary_dim", hd) != hd:
        return None
    return nh, nkv, hd, (bool(getattr(rope, "is_neox_style", True)) if rope is not None else True)


def _has_qk_norm(attn_mod: nn.Module) -> bool:
    """Return True if `attn_mod` applies per-head QK-Norm (scale-invariant, incompatible with QK dyadic gauge)."""
    if bool(getattr(attn_mod, "qk_norm", False)):
        return True
    return any(
        getattr(attn_mod, attr, None) is not None
        for attr in ("q_norm", "k_norm", "q_layernorm", "k_layernorm")
    )


def _is_compatible_rmsnorm(ln: Any, dim: int) -> bool:
    """Return True if `ln` is a bias-free 1D `RMSNorm` / `RMSNormFL` of width `dim` suitable for sign absorption."""
    if ln is None or type(ln).__name__ not in ("RMSNorm", "RMSNormFL"):
        return False
    if getattr(ln, "bias", None) is not None or getattr(ln, "variance_size_override", None) is not None:
        return False
    w = getattr(ln, "weight", None)
    return isinstance(w, torch.Tensor) and w.ndim == 1 and w.shape == (dim,)


def _is_eligible_view(view: DecoderLayerLinearView) -> bool:
    """Check whether a decoder block satisfies all Pre-LN SwiGLU geometry and norm invariants for online W8A8."""
    if not view.is_fused_gate_up:
        return False
    if any(getattr(view.block, k, None) is not None for k in ("pre_feedforward_layernorm", "post_feedforward_layernorm")):
        return False
    gate_up_proj, act_fn = view.gate_up_mods[0][1], getattr(view.mlp, "act_fn", None)
    if act_fn is None or type(act_fn).__name__ not in ("SiluAndMul", "SiluAndMulFL") or getattr(gate_up_proj, "bias", None) is not None:
        return False
    all_projs = [m for _, m in view.qkv_mods] + [view.o_proj, gate_up_proj, view.down_proj]
    if any(isinstance(getattr(p, "weight", None), QuantWeightTensor) for p in all_projs) or (geom := _resolve_attn_geometry(view.self_attn)) is None:
        return False
    nh, nkv, hd, _ = geom
    qs, kvs, hs, inter = nh * hd, nkv * hd, view.o_proj.weight.shape[0], view.down_proj.weight.shape[1]
    if (gate_up_proj.weight.shape[0] != 2 * inter) or view.o_proj.weight.shape != (hs, qs):
        return False
    if view.is_fused_qkv and view.qkv_mods[0][1].weight.shape != (qs + 2 * kvs, hs):
        return False
    return all(_is_compatible_rmsnorm(ln, hs) for ln in (view.input_layernorm, view.post_attention_layernorm))


def _compute_qk_dyadic_exponents(
    wq_f32: torch.Tensor, wk_f32: torch.Tensor, ln_w: torch.Tensor, orig_dtype: torch.dtype,
    nh: int, nkv: int, hd: int, is_neox: bool,
) -> torch.Tensor:
    """Compute `0-ulp` SO(2) pair-symmetric RoPE K-plane dyadic exponents `exp_k in {0, 1, 2}`.

    RoPE mixes 2D coordinate pairs $(x_i, x_{i + d/2})$ (NeoX) or $(x_{2i}, x_{2i+1})$
    (interleaved) via $2\\times 2$ planar rotations $R(\\theta) \\in SO(2)$. Assigning
    a shared integer exponent $e_p \\in \\{0, 1, 2\\}$ to both coordinates of plane $p$
    makes the diagonal scaling scalar on each 2D subspace, so $2^{e_p}$ commutes
    strictly with $R(\\theta)$:
        $(R(\\theta) (2^{e_p} q_p))^T (R(\\theta) (2^{-e_p} k_p)) = q_p^T k_p$.
    Exponents are clamped to `{0, 1, 2}` and guarded against `orig_dtype` overflow
    and subnormal underflow (`finfo.max` / `finfo.tiny`) to guarantee `0-ulp`
    mantissa preservation while compressing outlier Key planes before INT8 KV storage.
    """
    gqa, half, hs = nh // nkv, hd // 2, wq_f32.shape[1]
    g_abs = ln_w.detach().to(torch.float32).abs().clamp_min(1e-6)[None, :]
    wq_g, wk_g = wq_f32 * g_abs, wk_f32 * g_abs
    l2_q = (wq_g * wq_g).sum(dim=1).view(nkv, gqa, hd)
    amax_q = wq_f32.abs().amax(dim=1).view(nkv, gqa, hd).amax(dim=1)
    wk2 = wk_g * wk_g
    l4_k = (wk2 * wk2).sum(dim=1).view(nkv, hd)
    wk_abs = wk_f32.abs()
    amin_k = torch.where(wk_abs > 0.0, wk_abs, torch.full_like(wk_abs, float("inf"))).amin(dim=1).view(nkv, hd)

    qp_l2 = (l2_q[:, :, :half] + l2_q[:, :, half:]) if is_neox else (l2_q[:, :, 0::2] + l2_q[:, :, 1::2])
    kp_l4 = (l4_k[:, :half] + l4_k[:, half:]) if is_neox else (l4_k[:, 0::2] + l4_k[:, 1::2])
    mu_k = (kp_l4 * (1.0 / (2.0 * hs))).pow(0.25).clamp_min(1e-8)
    sig_q_rms, sig_q_max = qp_l2.mean(dim=1).sqrt().clamp_min(1e-8), qp_l2.sqrt().amax(dim=1).clamp_min(1e-8)
    q_rat = sig_q_rms / sig_q_rms.median(dim=-1, keepdim=True).values.clamp_min(1e-8)
    q_max_rat = sig_q_max / sig_q_max.median(dim=-1, keepdim=True).values.clamp_min(1e-8)
    rat_k = mu_k / mu_k.median(dim=-1, keepdim=True).values.clamp_min(1e-8)

    raw_ek = torch.floor(torch.log2(torch.clamp(rat_k / 1.65, min=0.5)) + 1.0)
    raw_ek = torch.where((rat_k >= 1.65) & (q_rat <= 1.50) & (q_max_rat <= 1.50 * 1.25), raw_ek, torch.zeros_like(raw_ek))
    raw_ek = torch.where((raw_ek >= 2.0) & (q_rat > 1.50 * 0.75), torch.ones_like(raw_ek), raw_ek)
    exp_kh = torch.clamp(raw_ek, 0, 2).to(torch.int32)
    exp_k = torch.cat([exp_kh, exp_kh], dim=1) if is_neox else exp_kh.repeat_interleave(2, dim=1)

    # Guard against BF16/FP16 overflow on Q and subnormal underflow on K, then take
    # the per-plane minimum so both coordinates of each 2D RoPE plane share one exponent.
    finfo, ones = torch.finfo(orig_dtype), torch.ones_like(amax_q)
    safe = (amax_q * torch.ldexp(ones, exp_k) < 0.25 * finfo.max) & ((amin_k == float("inf")) | (amin_k * torch.ldexp(ones, -exp_k) > 4.0 * finfo.tiny))
    exp_k = torch.where(safe, exp_k, torch.zeros_like(exp_k))
    if is_neox:
        ps = torch.minimum(exp_k[:, :half], exp_k[:, half:])
        return torch.cat([ps, ps], dim=1)
    return torch.minimum(exp_k[:, 0::2], exp_k[:, 1::2]).repeat_interleave(2, dim=1)


def resolve_valid_scale_groups(
    shape_k: int, target_nsg: int, hadamard_bs: int = 64, out_features: int = 0
) -> int:
    """Resolve a valid activation scale-group count dividing `shape_k` (and `max(64, hadamard_bs)`)."""
    if target_nsg == 1 or target_nsg < 0 or (out_features > 0 and out_features % 128 != 0):
        return 1
    from flag_gems.utils.tile_solver import get_device_topology

    # Auto-derive group count (`target_nsg == 0`) from hardware warp vector chunk width,
    # falling back to the closest valid divisor in [2, 8] dividing `shape_k` by `g * unit`.
    vec_chunk = max(512, get_device_topology().warp_size * 32)
    unit = max(64, hadamard_bs)
    if target_nsg == 0:
        if shape_k < 2 * vec_chunk:
            return 1
        auto_g = max(2, min(8, round(shape_k / vec_chunk)))
        if shape_k % (auto_g * unit) == 0:
            return auto_g
    elif shape_k % (target_nsg * unit) == 0:
        return int(target_nsg)
    best_g, best_dist = 1, 10**9
    for g in range(2, 9):
        if shape_k % (g * unit) == 0:
            dist = abs((shape_k // g) - vec_chunk)
            if dist < best_dist:
                best_dist, best_g = dist, g
    return best_g


def _bind_quant_param(proj: nn.Module, qw: QuantWeightTensor, untie: bool = False) -> int:
    """Replace `proj.weight` with quantized `qw` and return net HBM bytes freed."""
    orig_p = proj._parameters["weight"]
    orig_bytes = 0 if untie else (orig_p.numel() * orig_p.element_size())
    proj._parameters["weight"] = nn.Parameter(qw, requires_grad=False)
    del orig_p
    return max(0, orig_bytes - (qw.qweight_nk.numel() + qw.w_scale.numel() * 4))


@torch.no_grad()
def apply_online_quant_transform(
    model: nn.Module, model_config: Any = None, cfg: QuantTransformConfig | None = None
) -> dict[str, Any]:
    """Apply load-time orthogonal/dyadic gauge transforms and pack weights into `QuantWeightTensor`s.

    Idempotently walks all eligible Pre-LN decoder layers and any untied
    `ParallelLMHead`, performing:
    - `0-ulp` QK dyadic gauge scaling across `(W_Q, W_K)` (and biases if present);
    - `QuaRot-R2` head-wise orthogonal absorption across `(W_V, b_V)` and `W_O`;
    - Deterministic Rademacher sign absorption (`xi_in`, `xi_mlp`, `xi_dn`, `xi_lm`)
      into `input_layernorm`, `post_attention_layernorm`, `up_proj`, and `model.norm`;
    - `0-ulp` SwiGLU Up-Down dyadic scaling across `(W_up, W_down)`;
    - Hardware-layout-aware weight packing via `resolve_quant_weight_layout` +
      `quantize_linear_weight`, followed by immediate HBM reclamation and optional
      `extra_prewarm_fn` kernel-cache warmup prior to CUDA Graph capture.
    """
    cfg = cfg or QuantTransformConfig()
    stats = {"blocks": 0, "linear_modules": 0, "lm_head": 0, "saved_bytes": 0, "elapsed_s": 0.0}
    if getattr(model, "_fl_online_quant_applied", False):
        return stats
    if model_config is not None and any(getattr(model_config, k, False) for k in ("is_attention_free", "use_mla", "is_hybrid")):
        return stats

    t0 = time.perf_counter()
    unique_cfgs: dict[QuantLayerConfig, tuple[torch.device, torch.dtype]] = {}
    attn_prewarm_sig: tuple[torch.device, int, int, int] | None = None
    sampler_prewarm_sig: tuple[torch.device, torch.dtype, int] | None = None
    norm_had_bs, down_had_bs, alpha_min = (64 if cfg.norm_do_hadamard else 0), (64 if cfg.down_do_hadamard else 0), float(cfg.alpha_min)
    gamma_post_norm, gamma_other = float(cfg.gamma_post_rmsnorm), float(cfg.gamma_other)

    def _mk_cfg(n_out: int, k_in: int, qm: str, had_bs: int, nsg: int, silu_ep: bool, has_b: bool) -> QuantLayerConfig:
        """Build a `QuantLayerConfig` with backend-resolved physical weight layout."""
        return QuantLayerConfig(n_out, k_in, qm, had_bs, nsg, resolve_quant_weight_layout(n_out, k_in, qm, silu_ep), has_b)

    for i, view in enumerate(v for v in iter_causal_decoder_layers(model, require_unquantized=True, div_n=64, div_k=64) if _is_eligible_view(v)):
        ln1, ln2, attn = view.input_layernorm, view.post_attention_layernorm, view.self_attn
        o_proj, gate_up_proj, down_proj = view.o_proj, view.gate_up_mods[0][1], view.down_proj
        qkv_proj = view.qkv_mods[0][1] if view.is_fused_qkv else None
        nh, nkv, hd, is_neox = _resolve_attn_geometry(attn)  # type: ignore[misc]
        qs, kvs, gqa = nh * hd, nkv * hd, nh // nkv
        hs, inter, dev, orig_dtype = o_proj.weight.shape[0], down_proj.weight.shape[1], o_proj.weight.device, o_proj.weight.dtype
        ref_proj = qkv_proj if qkv_proj is not None else attn.q_proj
        kv_rep = max(1, int(getattr(ref_proj, "num_kv_head_replicas", 1) or 1))
        kv_head_base = (int(getattr(ref_proj, "tp_rank", 0) or 0) // kv_rep) * nkv
        rope_mod = getattr(attn, "rotary_emb", None)
        cs_buf = getattr(rope_mod, "cos_sin_cache", None) if rope_mod is not None else None
        if isinstance(cs_buf, torch.Tensor) and (cs_buf.device != dev or cs_buf.dtype != orig_dtype):
            rope_mod.cos_sin_cache = cs_buf.to(device=dev, dtype=orig_dtype).contiguous()

        if qkv_proj is not None:
            wq_f32, wk_f32, wv_f32 = (
                qkv_proj.weight.data[:qs].to(torch.float32),
                qkv_proj.weight.data[qs : qs + kvs].to(torch.float32),
                qkv_proj.weight.data[qs + kvs :].to(torch.float32),
            )
        else:
            wq_f32, wk_f32, wv_f32 = attn.q_proj.weight.data.to(torch.float32), attn.k_proj.weight.data.to(torch.float32), attn.v_proj.weight.data.to(torch.float32)

        # 1) `0-ulp` SO(2)-commuting QK dyadic gauge: scale Q up by 2^e and K down by 2^-e.
        if cfg.enable_qk_dyadic and kv_rep == 1 and not _has_qk_norm(attn):
            exp_k = _compute_qk_dyadic_exponents(wq_f32, wk_f32, ln1.weight.data, orig_dtype, nh, nkv, hd, is_neox)
            if bool((exp_k != 0).any().item()):
                ones_k = torch.ones((nkv, hd), dtype=torch.float32, device=dev)
                sq = torch.ldexp(ones_k, exp_k)[:, None, :].expand(nkv, gqa, hd).reshape(qs)
                sk = torch.ldexp(ones_k, -exp_k).reshape(kvs)
                wq_f32.mul_(sq[:, None])
                wk_f32.mul_(sk[:, None])
                if qkv_proj is not None and qkv_proj.bias is not None:
                    qkv_proj.bias.data[:qs].mul_(sq.to(qkv_proj.bias.dtype))
                    qkv_proj.bias.data[qs : qs + kvs].mul_(sk.to(qkv_proj.bias.dtype))
                elif qkv_proj is None:
                    if attn.q_proj.bias is not None:
                        attn.q_proj.bias.data.mul_(sq.to(attn.q_proj.bias.dtype))
                    if attn.k_proj.bias is not None:
                        attn.k_proj.bias.data.mul_(sk.to(attn.k_proj.bias.dtype))

        # 2) `QuaRot-R2` orthogonal head rotation across (W_V, b_V) and W_O.
        use_r2 = cfg.enable_quarot_r2 and (hd in (64, 128, 256))
        wo_f32 = o_proj.weight.data.to(torch.float32)
        if use_r2:
            v_bias = qkv_proj.bias.data[qs + kvs :] if (qkv_proj is not None and qkv_proj.bias is not None) else (attn.v_proj.bias.data if qkv_proj is None and attn.v_proj.bias is not None else None)
            wv_f32, wo_f32, bv_rot = apply_quarot_r2_heads_fp32(
                wv_f32, wo_f32, nkv, gqa, hd, i, dev, v_bias.to(torch.float32) if v_bias is not None else None, kv_head_base=kv_head_base
            )
            if v_bias is not None and bv_rot is not None:
                v_bias.copy_(bv_rot.to(v_bias.dtype))

        # 3) Absorb Rademacher signs `xi_in` into `input_layernorm` and `qkv_proj` columns.
        xi_in = make_deterministic_rademacher(hs, 20000 + i, dev) if cfg.norm_do_hadamard else None
        if xi_in is not None:
            ln1.weight.data.mul_(xi_in.to(ln1.weight.dtype))
        if qkv_proj is not None:
            wqkv_f32 = torch.cat([wq_f32, wk_f32, wv_f32], dim=0)
            del wq_f32, wk_f32, wv_f32
            cfg_qkv = _mk_cfg(qs + 2 * kvs, hs, "w8a8", norm_had_bs, 1, False, qkv_proj.bias is not None)
            stats["saved_bytes"] += _bind_quant_param(qkv_proj, quantize_linear_weight(wqkv_f32, cfg_qkv, alpha_min=alpha_min, gamma=gamma_post_norm, col_scale=xi_in))
            del wqkv_f32
            unique_cfgs.setdefault(cfg_qkv, (dev, orig_dtype))
            stats["linear_modules"] += 1
        else:
            for sub_p, sub_w in ((attn.q_proj, wq_f32), (attn.k_proj, wk_f32), (attn.v_proj, wv_f32)):
                sub_cfg = _mk_cfg(sub_w.shape[0], hs, "w8a8", norm_had_bs, 1, False, sub_p.bias is not None)
                stats["saved_bytes"] += _bind_quant_param(sub_p, quantize_linear_weight(sub_w, sub_cfg, alpha_min=alpha_min, gamma=gamma_post_norm, col_scale=xi_in))
                unique_cfgs.setdefault(sub_cfg, (dev, orig_dtype))
                stats["linear_modules"] += 1
            del wq_f32, wk_f32, wv_f32

        # When `use_r2` is active, `o_proj` activations are already orthogonally rotated
        # by `QuaRot-R2`, so runtime Hadamard block size on `o_proj` is set to 0.
        cfg_o = _mk_cfg(hs, qs, "w8a8", 0 if use_r2 else norm_had_bs, 1, False, o_proj.bias is not None)
        stats["saved_bytes"] += _bind_quant_param(o_proj, quantize_linear_weight(wo_f32, cfg_o, alpha_min=alpha_min, gamma=gamma_other))
        del wo_f32
        unique_cfgs.setdefault(cfg_o, (dev, orig_dtype))

        # 4) MLP `Up-Down` 0-ulp dyadic gauge + Rademacher absorption (`xi_mlp`, `xi_dn`).
        # `silu_ep=True` on `gate_up_proj` signals `resolve_quant_weight_layout` to apply
        # 64-column `[Gate, Up]` interleaving when supported by the backend's GEMM epilogue.
        wg_f32, wu_f32, wd_f32 = gate_up_proj.weight.data[:inter].to(torch.float32), gate_up_proj.weight.data[inter:].to(torch.float32), down_proj.weight.data.to(torch.float32)
        s_ud = compute_up_down_dyadic_scales(wg_f32, wu_f32, wd_f32) if cfg.enable_up_down_dyadic else torch.ones((inter,), dtype=torch.float32, device=dev)
        xi_mlp = make_deterministic_rademacher(hs, 30000 + i, dev) if cfg.norm_do_hadamard else None
        xi_dn = make_deterministic_rademacher(inter, 40000 + i + int(getattr(down_proj, "tp_rank", 0) or 0) * 100000, dev) if cfg.down_do_hadamard else torch.ones((inter,), dtype=torch.float32, device=dev)
        if xi_mlp is not None:
            ln2.weight.data.mul_(xi_mlp.to(ln2.weight.dtype))

        wu_f32.mul_((xi_dn / s_ud)[:, None])
        wgu_f32 = torch.cat([wg_f32, wu_f32], dim=0)
        del wg_f32, wu_f32
        cfg_gu = _mk_cfg(2 * inter, hs, "w8a8", norm_had_bs, 1, True, False)
        stats["saved_bytes"] += _bind_quant_param(gate_up_proj, quantize_linear_weight(wgu_f32, cfg_gu, alpha_min=alpha_min, gamma=gamma_post_norm, col_scale=xi_mlp))
        del wgu_f32
        unique_cfgs.setdefault(cfg_gu, (dev, orig_dtype))

        dn_nsg = resolve_valid_scale_groups(inter, cfg.down_num_scale_groups, down_had_bs, hs)
        cfg_dn = _mk_cfg(hs, inter, "w8a8", down_had_bs, dn_nsg, False, down_proj.bias is not None)
        stats["saved_bytes"] += _bind_quant_param(down_proj, quantize_linear_weight(wd_f32, cfg_dn, alpha_min=alpha_min, gamma=gamma_other, col_scale=s_ud * xi_dn))
        del wd_f32
        unique_cfgs.setdefault(cfg_dn, (dev, orig_dtype))

        stats["blocks"] += 1
        stats["linear_modules"] += 3
        if attn_prewarm_sig is None:
            attn_prewarm_sig = (dev, nh, nkv, hd)

    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod

    # 5) Untied `ParallelLMHead` quantization: promote to W8A8 (with `xi_lm` folded
    # into final `model.norm.weight`) when preceded by a compatible 1D final RMSNorm,
    # or fall back to weight-only W8A16 otherwise.
    untied_lm = resolve_untied_lm_head(model, model_config, require_parallel_lm_head=True, div_n=64, div_k=64)
    lm_head = untied_lm if untied_lm is not None else getattr(model, "lm_head", None)
    if (
        untied_lm is not None and isinstance(lm_head, ParallelLMHead)
        and type(getattr(lm_head, "quant_method", None)) is UnquantizedEmbeddingMethod
        and isinstance(getattr(lm_head, "weight", None), torch.Tensor)
        and not isinstance(lm_head.weight, QuantWeightTensor)
        and lm_head.weight.ndim == 2 and lm_head.weight.is_cuda
        and lm_head.weight.dtype in (torch.bfloat16, torch.float16)
        and (lm_head.weight.shape[0] % 64 == 0) and (lm_head.weight.shape[1] % 64 == 0)
    ):
        w_lm, dev_lm, dt_lm = lm_head.weight, lm_head.weight.device, lm_head.weight.dtype
        ln_final = getattr(getattr(model, "model", None), "norm", None)
        ln_fw = getattr(ln_final, "weight", None)
        lm_w8a8 = bool(cfg.untied_lm_head_w8a8 and _is_compatible_rmsnorm(ln_final, int(w_lm.shape[1])) and ln_fw.dtype in (torch.bfloat16, torch.float16))
        lm_had = bool(lm_w8a8 and cfg.lm_head_do_hadamard)
        xi_lm = make_deterministic_rademacher(int(w_lm.shape[1]), 50000, dev_lm) if lm_had else None
        if xi_lm is not None and ln_fw is not None:
            ln_fw.data.mul_(xi_lm.to(ln_fw.dtype))
        cfg_lm = _mk_cfg(w_lm.shape[0], w_lm.shape[1], "w8a8" if lm_w8a8 else "w8a16", 64 if lm_had else 0, 1, False, lm_head.bias is not None)
        qw_lm = quantize_linear_weight(w_lm.data, cfg_lm, alpha_min=alpha_min, gamma=gamma_other, col_scale=xi_lm)
        stats["saved_bytes"] += _bind_quant_param(lm_head, qw_lm, untie=False)
        unique_cfgs.setdefault(cfg_lm, (dev_lm, dt_lm))
        sampler_prewarm_sig, stats["lm_head"] = (dev_lm, dt_lm, int(w_lm.shape[0])), 1
    elif sampler_prewarm_sig is None:
        ref_emb = getattr(lm_head, "weight", None)
        if not isinstance(ref_emb, torch.Tensor):
            ref_emb = getattr(getattr(getattr(model, "model", None), "embed_tokens", None), "weight", None)
        if isinstance(ref_emb, torch.Tensor) and ref_emb.ndim == 2 and ref_emb.is_cuda:
            sampler_prewarm_sig = (ref_emb.device, ref_emb.dtype, int(ref_emb.shape[0]))

    setattr(model, "_fl_online_quant_applied", True)
    if stats["linear_modules"] > 0 or stats["lm_head"] > 0:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        if cfg.extra_prewarm_fn is not None:
            cfg.extra_prewarm_fn(unique_cfgs, attn_prewarm_sig, sampler_prewarm_sig)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        stats["elapsed_s"] = time.perf_counter() - t0
        logger.info(
            "[FL-OnlineQuant] Quantized %d trunk linears across %d blocks + %d lm_head (%d layer configs) in %.2f s (freed %.2f GiB).",
            stats["linear_modules"], stats["blocks"], stats["lm_head"], len(unique_cfgs), stats["elapsed_s"], stats["saved_bytes"] / (1024**3),
        )
    return stats


has_qk_norm = _has_qk_norm
resolve_attn_geometry = _resolve_attn_geometry


__all__ = [
    "QuantTransformConfig",
    "apply_online_quant_transform",
    "apply_quarot_r2_heads_fp32",
    "compute_up_down_dyadic_scales",
    "has_qk_norm",
    "resolve_attn_geometry",
    "resolve_valid_scale_groups",
]
