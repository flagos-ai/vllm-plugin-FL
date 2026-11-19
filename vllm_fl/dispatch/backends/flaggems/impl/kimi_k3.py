"""Thin library adapters preserving the accepted K3 numeric contracts."""
from typing import Optional, Union
import torch
from vllm.platforms import current_platform
from flag_gems.modules.normalization import gems_rms_forward
from .kimi_k3_routing import grouped_topk

def rms_norm_flaggems(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using FlagGems.

    Args:
        obj: The calling obj (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    if residual is not None and getattr(current_platform, "vendor_name", None) == "metax":
        raise NotImplementedError(
            "MetaX fused residual RMSNorm is pending strict validation against "
            "the pinned vLLM rounding/mutation contract"
        )
    # Get weight and epsilon from obj
    weight = obj.weight
    epsilon = obj.variance_epsilon

    return gems_rms_forward(x, residual, weight, epsilon)
