# SPDX-License-Identifier: Apache-2.0
"""GLM-5 (GlmMoeDsa) specific patches for vLLM 0.13.0 compatibility.

All monkey-patches required to run GLM-5 FP8 on the current environment
(transformers 4.57.6, CUDA 13.1, no deep_gemm JIT) are collected here.
"""

import logging

logger = logging.getLogger(__name__)


def patch_tokenizer_compat():
    """Patch transformers tokenizer loading for 5.x compat on 4.57.6.

    GLM-5's tokenizer uses transformers 5.x naming (TokenizersBackend) and
    special_tokens format (list instead of dict). This patches both issues
    so the tokenizer loads correctly on transformers 4.57.6.
    """
    try:
        import transformers.models.auto.tokenization_auto as ta

        if not getattr(ta, "_fl_patched", False):
            _orig = ta.tokenizer_class_from_name

            def _patched(class_name):
                result = _orig(class_name)
                if result is None and "TokenizersBackend" in class_name:
                    from transformers import PreTrainedTokenizerFast
                    return PreTrainedTokenizerFast
                return result

            ta.tokenizer_class_from_name = _patched
            ta._fl_patched = True
    except Exception:
        pass

    try:
        import transformers.tokenization_utils_base as tub

        if not getattr(tub.SpecialTokensMixin, "_fl_patched_special", False):
            _orig_set = tub.SpecialTokensMixin._set_model_specific_special_tokens

            def _patched_set(self, special_tokens=None):
                if isinstance(special_tokens, list):
                    special_tokens = {t: t for t in special_tokens}
                return _orig_set(self, special_tokens=special_tokens)

            tub.SpecialTokensMixin._set_model_specific_special_tokens = _patched_set
            tub.SpecialTokensMixin._fl_patched_special = True
    except Exception:
        pass


def patch_deep_gemm_fallback():
    """Patch vllm.utils.deep_gemm to use torch fallback implementations.

    deep_gemm's JIT kernels may fail on certain CUDA versions (e.g. 13.1).
    This replaces the DSA Indexer's fp8_mqa_logits / fp8_paged_mqa_logits
    with pure-torch reference implementations so GLM-5 FP8 can run without
    a working deep_gemm JIT compiler.
    """
    import torch
    import vllm.utils.deep_gemm as dg_mod

    def _torch_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
        """Torch fallback for fp8_mqa_logits (prefill DSA Indexer).

        q: [M, H, D] fp8  |  kv: (k_fp8 [N,D], k_scale [N]) | weights: [M,H]
        Returns: [M, N] float32
        """
        k_fp8, k_scale = kv
        q_f = q.float()
        k_f = k_fp8.float() * k_scale.view(-1, 1).float()
        N = k_f.shape[0]
        # score: [H, M, N]
        score = torch.einsum("mhd,nd->hmn", q_f, k_f)
        # weighted sum over heads -> [M, N]
        logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        # mask
        idx = torch.arange(N, device=q.device).unsqueeze(0)
        mask = (idx >= cu_seqlen_ks.unsqueeze(1)) & (idx < cu_seqlen_ke.unsqueeze(1))
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def _torch_fp8_paged_mqa_logits(
        q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
        schedule_metadata, max_model_len, clean_logits=True,
    ):
        """Torch fallback for fp8_paged_mqa_logits (decode DSA Indexer).

        q_fp8: [B, next_n, H, D] fp8
        kv_cache_fp8: [num_blocks, block_size, 1, D+4] uint8
          (last 4 bytes = float scale per position)
        weights: [B*next_n, H] float32
        context_lens: [B] int32
        block_tables: [B, max_blocks] int32
        Returns: [B*next_n, max_model_len] float32
        """
        B, next_n, H, D = q_fp8.shape
        block_size = kv_cache_fp8.shape[1]
        kv_D = kv_cache_fp8.shape[-1]  # D+4
        feat_D = kv_D - 4  # actual feature dim

        # Vectorized: gather all blocks for all batches at once
        max_blocks = block_tables.shape[1]
        # Flatten block_tables to gather all physical blocks
        flat_blocks = block_tables.reshape(-1)  # [B * max_blocks]
        # Gather all block data: [B * max_blocks, block_size, 1, D+4]
        all_blk_data = kv_cache_fp8[flat_blocks].squeeze(2)  # [B*max_blocks, block_size, D+4]
        # Reshape to [B, max_blocks * block_size, D+4]
        all_blk_data = all_blk_data.reshape(B, max_blocks * block_size, kv_D)
        # Split FP8 features and scales
        k_fp8_raw = all_blk_data[:, :, :feat_D].contiguous().view(
            torch.float8_e4m3fn)  # [B, total_pos, feat_D]
        scale_bytes = all_blk_data[:, :, feat_D:feat_D + 4].contiguous()
        k_scale = scale_bytes.view(torch.float32).squeeze(-1)  # [B, total_pos]
        # Dequantize: [B, total_pos, feat_D]
        k_f = k_fp8_raw.float() * k_scale.unsqueeze(-1)

        # q: [B, next_n, H, D] -> [B*next_n, H, D]
        q_f = q_fp8.float().view(B * next_n, H, D)

        # Compute scores for all batches vectorized
        # k_f: [B, total_pos, D] -> expand for next_n
        # score = einsum('bnhd,bpd->bnhp', q, k)
        q_4d = q_f.view(B, next_n, H, D)
        # [B, next_n, H, total_pos]
        score = torch.einsum('bnhd,bpd->bnhp', q_4d, k_f)
        # Apply ReLU and weight: weights [B*next_n, H] -> [B, next_n, H, 1]
        w_4d = weights.view(B, next_n, H, 1)
        # Weighted sum over heads: [B, next_n, total_pos]
        vals = (score.relu() * w_4d).sum(dim=2)

        # Build output logits with -inf masking
        logits = torch.full(
            (B * next_n, max_model_len), float("-inf"),
            dtype=torch.float32, device=q_fp8.device,
        )
        total_pos = max_blocks * block_size
        # Create position mask: [B, total_pos]
        pos_idx = torch.arange(total_pos, device=q_fp8.device).unsqueeze(0)
        mask = pos_idx < context_lens.unsqueeze(1)  # [B, total_pos]
        # Expand mask for next_n: [B, next_n, total_pos]
        mask = mask.unsqueeze(1).expand_as(vals)
        # Apply mask
        vals = vals.masked_fill(~mask, float("-inf"))
        # Write to logits: only up to total_pos columns
        out_cols = min(total_pos, max_model_len)
        logits_view = logits.view(B, next_n, max_model_len)
        logits_view[:, :, :out_cols] = vals[:, :, :out_cols]
        return logits

    def _torch_get_paged_mqa_logits_metadata(context_lens, block_size, num_sms):
        """No-op metadata for torch fallback (not needed)."""
        return torch.empty(0, dtype=torch.int32, device=context_lens.device)

    # Force _lazy_init to run first so we can override its results
    dg_mod._lazy_init()

    # Patch the module-level impl functions
    dg_mod._fp8_mqa_logits_impl = _torch_fp8_mqa_logits
    dg_mod._fp8_paged_mqa_logits_impl = _torch_fp8_paged_mqa_logits
    dg_mod._get_paged_mqa_logits_metadata_impl = _torch_get_paged_mqa_logits_metadata

    # Replace _lazy_init with no-op to prevent deep_gemm from
    # overwriting our patches on subsequent calls
    dg_mod._lazy_init = lambda: None
    logger.info("Patched vllm.utils.deep_gemm with torch fallback for "
                "fp8_mqa_logits / fp8_paged_mqa_logits")


def patch_is_deepseek_mla():
    """Patch ModelConfig.is_deepseek_mla to recognise glm_moe_dsa as MLA."""
    from vllm.config.model import ModelConfig
    _orig_is_mla = ModelConfig.is_deepseek_mla.fget

    @property
    def _patched_is_mla(self):
        if (
            hasattr(self.hf_text_config, "model_type")
            and self.hf_text_config.model_type == "glm_moe_dsa"
            and getattr(self.hf_text_config, "kv_lora_rank", None)
            is not None
        ):
            return True
        return _orig_is_mla(self)

    ModelConfig.is_deepseek_mla = _patched_is_mla


def apply_platform_patches():
    """All GLM-5 patches needed at platform registration time."""
    patch_tokenizer_compat()
    patch_deep_gemm_fallback()


def apply_model_patches():
    """All GLM-5 patches needed at model registration time."""
    patch_is_deepseek_mla()
