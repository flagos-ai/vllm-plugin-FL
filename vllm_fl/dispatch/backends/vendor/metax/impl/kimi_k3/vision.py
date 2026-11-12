"""Preserved complex64 vision rotation and bounded temporary-row policy."""
import torch
from vllm.model_executor.models.kimi_k25_vit import _apply_rope_input_validation

def _apply_rope_block(xq, xk, freqs_cis):
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rope(xq, xk, freqs_cis):
    """Preserve the original complex64 rotation while bounding temporary rows."""
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)
    if xq.ndim != 3 or xq.shape[0] < 16384:
        return _apply_rope_block(xq, xk, freqs_cis)
    out_q, out_k = torch.empty_like(xq), torch.empty_like(xk)
    freqs_real = torch.view_as_real(freqs_cis)
    for start in range(0, xq.shape[0], 8192):
        stop = min(start + 8192, xq.shape[0])
        q, k = _apply_rope_block(
            xq[start:stop],
            xk[start:stop],
            torch.view_as_complex(freqs_real[start:stop]),
        )
        out_q[start:stop].copy_(q)
        out_k[start:stop].copy_(k)
        del q, k
    return out_q, out_k
