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


def patch_fp8_mqa_logits_dim():
    """Fix k_scale dim mismatch for deep_gemm fp8_mqa_logits.

    vLLM 0.13.0 passes k_scale as [N, 1] but deep_gemm 2.3.0 expects [N].
    Upstream fix: https://github.com/vllm-project/vllm/pull/32652
    We wrap fp8_mqa_logits to flatten k_scale before calling the native impl.
    """
    import vllm.utils.deep_gemm as dg_mod

    dg_mod._lazy_init()
    _orig_impl = dg_mod._fp8_mqa_logits_impl
    if _orig_impl is None:
        return

    def _fixed_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
        k_fp8, k_scale = kv
        return _orig_impl(
            q, (k_fp8, k_scale.flatten()), weights,
            cu_seqlen_ks, cu_seqlen_ke,
        )

    dg_mod._fp8_mqa_logits_impl = _fixed_fp8_mqa_logits
    dg_mod._lazy_init = lambda: None
    logger.info("Patched fp8_mqa_logits: flatten k_scale [N,1] -> [N] "
                "for deep_gemm 2.3.0 compat")


def patch_indexer_schedule_metadata():
    """Fix schedule_metadata not computed when VLLM_USE_DEEP_GEMM=0.

    In vLLM 0.13.0, the indexer metadata builder gates schedule_metadata
    computation behind ``is_deep_gemm_supported()`` which checks
    ``VLLM_USE_DEEP_GEMM``. But the DSA kernel (fp8_paged_mqa_logits)
    only checks ``has_deep_gemm()`` — so when VLLM_USE_DEEP_GEMM=0 and
    deep_gemm is installed, the kernel runs with uninitialised metadata,
    causing CUDA_ERROR_ILLEGAL_ADDRESS.

    Fix: patch the builder's ``build`` method to always compute
    schedule_metadata when ``has_deep_gemm()`` is True.
    """
    from vllm.utils.import_utils import has_deep_gemm
    if not has_deep_gemm():
        return

    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

    _orig_build = DeepseekV32IndexerMetadataBuilder.build

    def _patched_build(self, common_prefix_len, common_attn_metadata,
                       fast_build=False):
        result = _orig_build(self, common_prefix_len,
                             common_attn_metadata, fast_build)
        if (result.decode is not None
                and result.decode.schedule_metadata is not None):
            seq_lens = common_attn_metadata.seq_lens[:result.num_decodes]
            self.scheduler_metadata_buffer[:] = (
                get_paged_mqa_logits_metadata(
                    seq_lens, self.kv_cache_spec.block_size, self.num_sms
                )
            )
        return result

    DeepseekV32IndexerMetadataBuilder.build = _patched_build
    logger.info("Patched indexer: schedule_metadata always computed "
                "when deep_gemm is available")


def apply_platform_patches():
    """All GLM-5 patches needed at platform registration time."""
    patch_tokenizer_compat()
    patch_fp8_mqa_logits_dim()


def patch_indexer_rope_reshape():
    """Fix RoPE output shape in Indexer.forward for DSA models.

    vLLM 0.13.0 uses squeeze(0) / squeeze((0, 2)) on RoPE outputs, which
    can fail when the RoPE implementation introduces extra leading dims.
    Replace squeeze with explicit reshape for robustness.
    """
    import torch
    from vllm.model_executor.models.deepseek_v2 import (
        Indexer,
        per_token_group_quant_fp8,
    )

    def _patched_forward(self, hidden_states, qr, positions, rotary_emb):
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        k, _ = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Use reshape instead of squeeze to handle extra leading dims
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # quant q (k quant is fused with cache insertion)
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights, _ = self.weights_proj(hidden_states)
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale
            * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    Indexer.forward = _patched_forward
    logger.info("Patched Indexer.forward: reshape RoPE outputs to ensure "
                "correct dims")


def apply_model_patches():
    """All GLM-5 patches needed at model registration time."""
    patch_is_deepseek_mla()
    patch_indexer_schedule_metadata()
    patch_indexer_rope_reshape()
