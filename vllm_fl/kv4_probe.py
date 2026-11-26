"""KV page-size accounting for the Iluvatar BI-V150 K4V2 cache format (132 B/token-head)."""
import logging
logger = logging.getLogger(__name__)
INT4_BYTES_PER_TOKEN_HEAD = 132
INT8_BYTES_PER_TOKEN_HEAD = 264
PHYS_DIV = 2
_PATCHED = False

def int4_active():
    return True

def probe_active():
    return False

def patch_page_budget() -> None:
    """vLLM's attention specs budget the int4 page for the per-token-head cache dtype (head size 128)."""
    global _PATCHED
    if _PATCHED or not int4_active():
        return
    from vllm.v1 import kv_cache_interface as kci

    def _mine(spec) -> bool:
        return spec.kv_quant_mode == kci.KVQuantMode.INT8_PER_TOKEN_HEAD and spec.head_size == 128

    for cls in (kci.AttentionSpec, kci.FullAttentionSpec):
        orig_real = cls.__dict__.get("real_page_size_bytes")
        if orig_real is not None:
            def real(self, _o=orig_real):
                if _mine(self):
                    return self.block_size * self.num_kv_heads * INT4_BYTES_PER_TOKEN_HEAD
                return _o.fget(self)
            setattr(cls, "real_page_size_bytes", property(real))
    orig_page = kci.AttentionSpec.__dict__["page_size_bytes"]

    def page(self, _o=orig_page):
        if _mine(self):
            real = self.real_page_size_bytes
            if self.page_size_padded is not None:
                assert self.page_size_padded >= real
                return self.page_size_padded
            return real
        return _o.fget(self)
    kci.AttentionSpec.page_size_bytes = property(page)
    _PATCHED = True
    logger.warning("E038/E040 int4 KV: page budget %d B per (token, head) per layer (int4 K/V codes + 4 B scales)",
                   INT4_BYTES_PER_TOKEN_HEAD)

