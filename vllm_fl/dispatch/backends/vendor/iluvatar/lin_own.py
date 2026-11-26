"""Verify the FlagGems linear implementation without redirecting embedding calls."""
import logging
import torch
from flag_gems.fused.minicpm.lin_own import self_check

def patch_lm_head_linear_for_iluvatar():
    err = self_check(torch.device("cuda", torch.cuda.current_device()))
    if not err <= 2e-2:
        raise RuntimeError(f"E050 FlagGems linear self-check failed: {err}")
    logging.getLogger(__name__).info("BI-V150 lm_head linear: self-check passed (%.2e); E050 original FlagGems linear route", err)
