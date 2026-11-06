# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 checkpoint mappings and MTP quantization metadata.

These helpers describe checkpoint layout. Numerical backend selection stays in
vLLM and the operator providers; importing a packed mapping does not load a
quantization backend.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

from .mtp_config import remap_compressed_mtp_quant_config

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


def hyv4_packed_modules_mapping() -> dict[str, list[str]]:
    """Return independently owned packed projection names for each model."""
    return {
        "gate_up_proj": ["gate_proj", "up_proj"],
        # MLA runs both latent down-projections as one GEMM.
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        # The indexer fuses wk and weights_proj into one GEMM.
        "wk_weights_proj": ["wk", "weights_proj"],
    }


_MTP_SCALE_ONLY_PARAM_SUBSTRINGS = (
    "_scale",
    "_weight_scale",
    "_weight_scale_inv",
    "_input_scale",
)

# Names of the quant-config attribute that stores the unquantized-module
# prefix list, in the order we look them up. Different QuantizationConfig
# subclasses expose the list under different names:
#   * ``ignored_layers`` -- Fp8Config, GPTQ / AWQ configs.
#   * ``exclude_modules`` -- ModelOpt* configs (Fp8, NvFp4, MxFp8, Mixed).
_MTP_QUANT_EXCLUSION_ATTRS = ("ignored_layers", "exclude_modules")


def _get_spec_layer_idx_from_weight_name(
    config: PreTrainedConfig, weight_name: str
) -> int | None:
    """Return the MTP layer index a checkpoint weight belongs to, or None.

    Compatible with``num_nextn_predict_layers`` of 1 and 2 for a single MTP
    head, matching how the HY V4 checkpoints are exported.
    """
    if config.num_nextn_predict_layers < 1:
        return None
    layer_idx = config.num_hidden_layers
    num_mtp_layers = max(config.num_nextn_predict_layers - 1, 1)
    for i in range(num_mtp_layers):
        if weight_name.startswith(f"model.layers.{layer_idx + i}."):
            return layer_idx + i
    return None


def _should_skip_missing_mtp_scale_param(
    quant_config: QuantizationConfig | None,
    name: str,
) -> bool:
    """Whether an unmatched scale parameter can be silently ignored."""
    return quant_config is None and any(
        scale_name in name for scale_name in _MTP_SCALE_ONLY_PARAM_SUBSTRINGS
    )


def _resolve_fused_expert_param(
    param_base: str,
    ckpt_suffix: str,
    params_dict: dict,
) -> str | None:
    """Map a fused expert checkpoint suffix onto a draft parameter name.

    Fused checkpoints pack every expert into one tensor and name the companion
    scales by suffixing the projection (``experts.gate_up_proj_scale_inv``),
    while the draft model owns ``experts.routed_experts.w13_weight`` plus a
    separate ``..._scale_inv`` parameter. Dropping the suffix would push the
    scale into the weight parameter and leave the scale at its sentinel init.

    Args:
        param_base: Draft parameter holding the packed weight.
        ckpt_suffix: Checkpoint text following the projection name.
        params_dict: The draft model's named parameters.

    Returns:
        The matching draft parameter name, or None when there is none.

    """
    if not ckpt_suffix:
        return param_base if param_base in params_dict else None
    tail = ckpt_suffix.lstrip(".").removeprefix("weight_")
    for candidate in (
        f"{param_base}_{tail}",
        f"{param_base}_scale_inv",
        f"{param_base}_scale",
    ):
        if candidate in params_dict:
            return candidate
    return None


def _prepare_mtp_fp8_expert_scale(
    quant_config: QuantizationConfig | None,
    name: str,
    loaded_weight: torch.Tensor,
) -> tuple[str, torch.Tensor]:
    """Normalize block scales while preserving packed MXFP4 expert bytes."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if (
        isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size is not None
        and name.endswith(".scale")
    ):
        name = name[: -len(".scale")] + ".weight_scale_inv"
        if loaded_weight.dtype == torch.uint8:
            if getattr(quant_config, "is_scale_e8m0", False):
                loaded_weight = loaded_weight.view(torch.float8_e8m0fnu)
            elif ".mlp.experts." not in name:
                # Linear scales are FP32 parameters. Decode UE8M0 exponents;
                # a numeric uint8-to-float cast would load the encoded bytes.
                loaded_weight = loaded_weight.view(torch.float8_e8m0fnu).float()
    return name, loaded_weight


def _create_mtp_quant_config(
    hf_config: PreTrainedConfig,
    backbone_quant_config: QuantizationConfig | None = None,
) -> QuantizationConfig | None:
    """Create the quantization config for the MTP layers.

    The MTP quantization algorithm is given by the ``mtp_quant_algo`` field in
    ``config.json``, independently of the backbone's quantization. Supported
    values are ``"FP8"``, ``"NONE"`` (inherit the backbone) and ``"BF16"`` /
    ``"FP16"`` (unquantized). A missing or ``"NONE"`` value falls back to the
    backbone config.

    Args:
        hf_config: The draft model's HF config.
        backbone_quant_config: The target model's quantization config.

    Returns:
        The quantization config to use for the MTP layers, or None when the MTP
        layers are unquantized.

    """
    mtp_quant_algo = getattr(hf_config, "mtp_quant_algo", None)

    if mtp_quant_algo is None or mtp_quant_algo.upper() == "NONE":
        logger.info_once(
            "No explicit MTP quant algo (mtp_quant_algo=%s). "
            "Using the backbone's quant config.",
            mtp_quant_algo,
        )
        return backbone_quant_config

    mtp_quant_algo = mtp_quant_algo.upper()

    if mtp_quant_algo in ("BF16", "FP16"):
        logger.info_once(
            "MTP layers are unquantized (mtp_quant_algo=%s).", mtp_quant_algo
        )
        return None

    if mtp_quant_algo == "FP8":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        logger.info_once("MTP uses FP8 quantization (mtp_quant_algo=FP8).")
        hf_quant_config = getattr(hf_config, "quantization_config", None) or {}
        weight_block_size = hf_quant_config.get("weight_block_size")
        activation_scheme = hf_quant_config.get("activation_scheme", "dynamic")
        if weight_block_size is not None:
            activation_scheme = "dynamic"
        ignored_layers = hf_quant_config.get("ignored_layers") or hf_quant_config.get(
            "modules_to_not_convert"
        )

        fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers or [],
            weight_block_size=weight_block_size,
        )
        if hf_quant_config.get("scale_fmt") == "ue8m0":
            # Fp8LinearMethod reads this through getattr, so it is not a
            # declared Fp8Config field.
            setattr(fp8_config, "is_scale_e8m0", True)  # noqa: B010
        return fp8_config

    logger.warning(
        "Unknown mtp_quant_algo=%s, falling back to the backbone config.",
        mtp_quant_algo,
    )
    return backbone_quant_config


def _remap_mtp_quant_exclusions(
    quant_config: QuantizationConfig | None,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
) -> QuantizationConfig | None:
    """Translate checkpoint-named MTP quant exclusions to draft prefixes.

    ``modules_to_not_convert`` / ``exclude_modules`` name MTP modules the way
    the checkpoint does (``model.mtp_layers.0.self_attn.linear_gate``), while
    the draft model builds them under ``model.layers.<num_hidden_layers + i>``.
    ``is_layer_skipped`` compares prefixes for exact equality, so without this
    translation an excluded MTP module gets a quant method even though its
    checkpoint weight is BF16 with no ``weight_scale`` / ``weight_scale_inv``
    companion: the scale then keeps its ``finfo(float32).min`` sentinel and the
    layer silently produces garbage.

    The list lives under a different attribute per config class, hence the
    lookup over ``_MTP_QUANT_EXCLUSION_ATTRS``.

    Args:
        quant_config: The MTP quantization config to widen.
        mtp_start_layer_idx: Index of the first MTP layer in the draft model.
        num_mtp_layers: Number of MTP layers.

    Returns:
        A shallow copy with the translated exclusions, or the input unchanged.

    """
    if quant_config is None:
        return None

    if quant_config.get_name() == "compressed-tensors":
        return remap_compressed_mtp_quant_config(
            quant_config, mtp_start_layer_idx, num_mtp_layers
        )

    # Different QuantizationConfig subclasses expose the exclusion list under
    # different attribute names; only the one that is actually populated on
    # the backbone config governs which draft modules stay unquantized. For
    # ModelOpt checkpoints (``quant_method: modelopt`` and friends) this is
    # ``exclude_modules``; for the legacy fp8 path it is ``ignored_layers``.
    exclusion_attr: str | None = None
    patterns: list[str] | None = None
    for attr in _MTP_QUANT_EXCLUSION_ATTRS:
        value = getattr(quant_config, attr, None)
        if value:
            exclusion_attr = attr
            patterns = list(value)
            break
    if not patterns or exclusion_attr is None:
        return quant_config

    extra: list[str] = []
    for offset in range(num_mtp_layers):
        source_prefix = f"model.mtp_layers.{offset}."
        draft_prefix = f"model.layers.{mtp_start_layer_idx + offset}."
        for pattern in patterns:
            if pattern.startswith(source_prefix):
                extra.append(draft_prefix + pattern[len(source_prefix) :])
            # ModelOpt exclusion patterns may end with a ``*`` wildcard; the
            # remap must preserve that suffix so ``is_layer_skipped`` still
            # matches the draft module prefixes.
            elif pattern.rstrip("*").startswith(source_prefix):
                stripped = pattern.rstrip("*")
                trailing_stars = pattern[len(stripped) :]
                extra.append(
                    draft_prefix + stripped[len(source_prefix) :] + trailing_stars
                )
    if not extra:
        return quant_config

    # The backbone shares this object with the target model; copy before
    # widening the exclusion list. The attribute is only present on the
    # concrete configs that support exclusions, hence the getattr/setattr pair.
    quant_config = copy.copy(quant_config)
    setattr(quant_config, exclusion_attr, patterns + extra)  # noqa: B010
    logger.info_once(
        "HYV4 MTP: mapped %d checkpoint-named quant exclusions (%s) onto the "
        "draft module prefixes: %s",
        len(extra),
        exclusion_attr,
        ", ".join(extra),
    )
    return quant_config
