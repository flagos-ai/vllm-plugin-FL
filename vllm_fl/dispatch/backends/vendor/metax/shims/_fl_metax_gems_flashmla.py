"""Wire FlagGems-vllm Triton FlashMLA into vLLM's empty build on MetaX.

The stock ``vllm.v1.attention.ops.flashmla`` only exposes the CUDA extensions
(``vllm._flashmla_C`` / ``_flashmla_extension_C``), which the empty wheel does
not build.  FlagGems-vllm ships functionally equivalent Triton kernels with the
same public signatures (sparse prefill + paged decode + a dummy scheduler).
Bind them into every module that imported the names by value.

Kill switch: ``FL_METAX_GEMS_FLASHMLA=0``.
"""

import importlib
import os

from ._fl_metax_gate import metax_active

_SWITCH = "FL_METAX_GEMS_FLASHMLA"

_TARGETS = (
    "vllm.v1.attention.ops.flashmla",
    "vllm.v1.attention.backends.mla.flashmla_sparse",
    "vllm.models.hy_v4.nvidia.flashmla_sparse",
    "vllm.v1.attention.backends.mla.sparse_swa",
)


def _supported() -> tuple[bool, str | None]:
    return True, None


def patch() -> bool:
    if not metax_active():
        return False
    value = os.environ.get(_SWITCH, "1").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False

    from flaggems_vllm.ops.flash_mla_with_kvcache import (
        FlashMLASchedMeta,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
    from flaggems_vllm.ops.flashmla_sparse import (
        flash_mla_sparse_fwd as _gems_sparse_fwd,
    )

    def flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
        out=None,
        **kwargs,
    ):
        """vLLM API adds ``out`` (in-place output); FlagGems does not."""
        output, max_logits, lse = _gems_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        if out is not None:
            out.copy_(output)
            output = out
        return output, max_logits, lse

    attrs = {
        "flash_mla_sparse_fwd": flash_mla_sparse_fwd,
        "flash_mla_with_kvcache": flash_mla_with_kvcache,
        "get_mla_metadata": get_mla_metadata,
        "FlashMLASchedMeta": FlashMLASchedMeta,
        "is_flashmla_sparse_supported": _supported,
        "is_flashmla_dense_supported": _supported,
    }
    patched = []
    for name in _TARGETS:
        try:
            mod = importlib.import_module(name)
        except Exception:  # pragma: no cover - optional modules
            continue
        for attr, impl in attrs.items():
            setattr(mod, attr, impl)
        patched.append(name)
    if patched:
        print("[vllm_fl][metax] FlashMLA -> FlagGems Triton:", ", ".join(patched))
    return bool(patched)


patch()
