"""Register FlagGems implementations for C++ ops absent from the empty wheel.

vLLM's empty (``VLLM_TARGET_DEVICE=empty``) build ships no ``vllm._C`` / ``_moe_C``
extensions, but native layers still do ``torch.ops._C.silu_and_mul`` etc. at
construction time.  FlagGems' ``patch_empty_vllm()`` defines the schemas and
binds Triton implementations; run it before the model is built.

Kill switch: ``FL_METAX_EMPTY_VLLM_OPS=0``.
"""

import os

import torch

from ._fl_metax_gate import metax_active

_SWITCH = "FL_METAX_EMPTY_VLLM_OPS"
_CACHE_LIB = None


def _concat_mla_q(ql_nope, q_pe, q_out) -> None:
    """vLLM contract: write into the first ``ql_nope.size(1)`` heads of q_out.

    ``q_out`` comes from the workspace and may be head-padded; FlagGems-vllm's
    strict ``concat_mla_q`` rejects that layout, so copy directly.
    """
    heads = ql_nope.shape[1]
    nope = ql_nope.shape[-1]
    rope = q_pe.shape[-1]
    q_out[:, :heads, :nope].copy_(ql_nope)
    if rope:
        q_out[:, :heads, nope : nope + rope].copy_(q_pe)


def patch_empty_ops() -> bool:
    if not metax_active():
        return False
    value = os.environ.get(_SWITCH, "1").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    import flag_gems

    flag_gems.patch_empty_vllm()

    # Empty wheel also lacks the MLA query-pack op used by sparse MLA.
    global _CACHE_LIB
    _CACHE_LIB = torch.library.Library("_C_cache_ops", "FRAGMENT")
    if not hasattr(torch.ops._C_cache_ops, "concat_mla_q"):
        _CACHE_LIB.define(
            "concat_mla_q(Tensor ql_nope, Tensor q_pe, Tensor! q_out) -> ()"
        )
    _CACHE_LIB.impl("concat_mla_q", _concat_mla_q, "CUDA")
    return True


patch_empty_ops()
