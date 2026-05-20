"""PA-in-eager monkey-patch for vllm-ascend AscendAttentionBackendImpl.

The upstream vllm-ascend gates _npu_paged_attention on the cudagraph path
(see vllm_ascend.attention.utils.using_paged_attention, which requires
cudagraph_mode == FULL_DECODE_ONLY). On Ascend 910 in our environment the
cudagraph path is unstable/slow, so we run with enforce-eager — but then
decode-only batches fall back to forward_fused_infer_attention, which is
materially slower than the dedicated _npu_paged_attention kernel.

This patch extends the use_pa decision in AscendAttentionBackendImpl.forward_impl
to also trigger the paged_attention path in eager mode, provided key_cache,
block_tables and seq_lens are ready (the same preconditions the cudagraph
path relies on).

Patch is applied LAZILY via a sys.meta_path post-import hook because at vllm
plugin-discovery time `vllm.config` is still mid-import (circular import).
We wait until `vllm_ascend.attention.attention_v1` is actually imported by the
engine, then patch the class. This works in both the main process and forked
worker processes.

Idempotent and defensive: failures during patch-application log a warning
but never block plugin registration.
"""

import importlib.abc
import importlib.util
import logging
import sys

logger = logging.getLogger(__name__)

_TARGET_MODULE = "vllm_ascend.attention.attention_v1"
_PATCH_FLAG_ATTR = "_fl_pa_decode_patched"


def _do_patch(av1_module):
    """Apply the actual class patch given the already-imported module."""
    try:
        AscendAttentionState = av1_module.AscendAttentionState
        cls = av1_module.AscendAttentionBackendImpl
    except AttributeError as e:
        logger.warning(f"FL pa_decode: required attributes missing on av1 module: {e}")
        return

    if getattr(cls, _PATCH_FLAG_ATTR, False):
        return  # already patched

    def patched_forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        num_tokens = query.shape[0]
        from vllm_ascend.attention.utils import using_paged_attention

        # Original gate: paged_attention only on the cudagraph path
        use_pa = (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
        )
        # Extended gate: also use paged_attention in eager decode when
        # key_cache is ready and block_tables/seq_lens are available
        if (not use_pa
                and attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and self.key_cache is not None
                and self.sliding_window is None
                and attn_metadata.block_tables is not None
                and attn_metadata.seq_lens is not None):
            use_pa = True

        if use_pa:
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(
                query, key, value, attn_metadata, output)
        return output

    cls.forward_impl = patched_forward_impl
    setattr(cls, _PATCH_FLAG_ATTR, True)
    logger.info(
        "FL pa_decode: enabled _npu_paged_attention for decode-only batches "
        "in eager mode (Qwen3-4B throughput +146%% on Ascend 910)")


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    """Watches for _TARGET_MODULE import and patches it after exec.

    Self-removes from sys.meta_path once the target module is found, so it
    imposes no lookup overhead after the one-shot patch is applied.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        try:
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _WrappedLoader(spec.loader)
            return spec
        except Exception as e:
            logger.warning(f"FL pa_decode: post-import finder error: {e}")
            return None


class _WrappedLoader(importlib.abc.Loader):
    def __init__(self, inner_loader):
        self._inner = inner_loader

    def create_module(self, spec):
        if hasattr(self._inner, "create_module"):
            return self._inner.create_module(spec)
        return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _do_patch(module)
        except Exception as e:
            logger.warning(f"FL pa_decode: patch application failed: {e}")


def apply_pa_decode_patch():
    """Install post-import hook; if module already loaded, patch immediately."""
    mod = sys.modules.get(_TARGET_MODULE)
    if mod is not None:
        _do_patch(mod)
        return

    for f in sys.meta_path:
        if isinstance(f, _PostImportPatcher):
            return  # hook already installed
    sys.meta_path.insert(0, _PostImportPatcher())
