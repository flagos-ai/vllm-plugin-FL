from vllm.logger import init_logger

logger = init_logger(__name__)

_KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE = 512

# Query-axis tile for the TXDA ViT attention. Bounds the score matrix to
# CHUNK x seq; 1024 keeps peak memory near 1 GiB at seq=65536 (16 heads,
# bf16) while still using large GEMMs.
_TXDA_VIT_ATTENTION_CHUNK = 1024


def _patch_hygon_sdpa_contiguous():
    """Use contiguous BHSD inputs and D72 padding for Hygon ViT Torch SDPA."""
    from vllm.v1.attention.ops import vit_attn_wrappers as vit_mod

    if getattr(vit_mod.apply_sdpa, "_fl_hygon_sdpa_layout", None) == "bhsd-pad-d72":
        return

    def _hygon_apply_sdpa(q, k, v, scale=None, enable_gqa=False):
        q, k, v = (
            vit_mod.einops.rearrange(x, "b s h d -> b h s d").contiguous()
            for x in (q, k, v)
        )
        pad_d72 = q.shape[-1] == k.shape[-1] == v.shape[-1] == 72
        if pad_d72:
            q, k, v = (vit_mod.F.pad(x, (0, 56)).contiguous() for x in (q, k, v))
            if scale is None:
                scale = 72**-0.5
        output = vit_mod.F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
        )
        if pad_d72:
            output = output[..., :72]
        return vit_mod.einops.rearrange(output, "b h s d -> b s h d ")

    _hygon_apply_sdpa._fl_hygon_sdpa_layout = "bhsd-pad-d72"
    vit_mod.apply_sdpa = _hygon_apply_sdpa
    logger.info_once(
        "Using contiguous BHSD Torch SDPA inputs with D72 padding for Hygon MM encoder."
    )


def _gcu_forward_mm_attention(
    self,
    query,
    key,
    value,
    cu_seqlens=None,
    max_seqlen=None,
    sequence_lengths=None,
):
    """Use Enflame's non-paged FlashAttention for the vision encoder."""
    import torch

    from vllm.utils.gpu_sync_debug import gpu_sync_allowed
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if self.attn_backend != AttentionBackendEnum.FLASH_ATTN:
        return self._fl_gcu_original_forward_native(
            query, key, value, cu_seqlens, max_seqlen, sequence_lengths
        )

    from flash_attn.vllm_flash_attn import flash_attn_varlen_func

    assert (cu_seqlens is None) == (max_seqlen is None), (
        "cu_seqlens and max_seqlen should be both set or both None."
    )
    batch_size, q_len = query.shape[:2]
    kv_len = key.size(1)
    is_reshaped = query.dim() != 4
    query, key, value = self.view_qkv_to_4d(
        query, key, value, batch_size, q_len, kv_len
    )
    if cu_seqlens is None:
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * q_len,
            q_len,
            dtype=torch.int32,
            device=query.device,
        )
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * kv_len,
            kv_len,
            dtype=torch.int32,
            device=key.device,
        )
        max_seqlen_q, max_seqlen_k = q_len, kv_len
    else:
        cu_seqlens_q = cu_seqlens_k = cu_seqlens
        with gpu_sync_allowed():
            max_seqlen_q = max_seqlen.item()
        max_seqlen_k = max_seqlen_q

    output = flash_attn_varlen_func(
        q=query.flatten(0, 1),
        k=key.flatten(0, 1),
        v=value.flatten(0, 1),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        causal=False,
        softmax_scale=self.scale,
        # The installed Enflame provider's FA2 path supports non-paged
        # FP16/BF16 vision attention without CUDA capability detection.
        fa_version=2,
    )
    output = output.reshape(batch_size, q_len, self.num_heads, self.head_size)
    return output.reshape(batch_size, q_len, -1) if is_reshaped else output


def _txda_forward_sdpa(self, query, key, value, cu_seqlens=None):
    """Decompose the ViT SDPA into matmul + softmax.

    ``torch.nn.functional.scaled_dot_product_attention`` has no fused kernel on
    TXDA and falls back to a slow path: one call at the ViT shape
    (heads=16, head_dim=72, seq=4096) measured ~320 ms, and the Qwen3.5 vision
    encoder runs 27 such layers per image. The encoder then dominates model
    profiling and startup appears to hang.

    The ViT head_dim here (72) is not a power of two, which is also why
    FlagGems' ``flash_attn_varlen_func`` degrades ~70x on it (2724 ms) while
    staying fast at 64/128. Explicit matmul + softmax uses the pointwise and
    GEMM kernels that FlagGems implements well and measured 2.9 ms at seq=512
    and 95 ms at seq=4096 -- a 30-70x improvement, with max|diff| vs SDPA of
    0.03 on bf16 (within tolerance).

    ``cu_seqlens`` is ignored on purpose. vLLM's own SDPA path
    (``vit_attn_wrappers.apply_sdpa``) also ignores it and runs a dense
    attention over the padded batch, so honouring it here would change
    behaviour relative to the backend being replaced. It is carried only for
    the FlashAttention/FlashInfer backends.
    """
    import torch

    bsz, q_len = query.size()[:2]
    kv_len = key.size(1)
    is_reshaped = query.dim() != 4

    # ``view_qkv_to_4d`` returns (batch, seq, heads, head_dim); the matmuls
    # below need the head axis next to the batch axis.
    query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    scale = self.scale if self.scale is not None else query.shape[-1] ** -0.5
    if self.num_heads > self.num_kv_heads:
        # Grouped-query attention: repeat each KV head across its query group.
        repeat = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    # Chunk the query axis so the score matrix stays bounded. The ViT runs at
    # very large sequences during profiling (Qwen3.5's preprocessor reports a
    # 65536-token image), where a full S x S score tensor would need tens of
    # GB; chunking keeps peak score memory at CHUNK x S.
    chunk = _TXDA_VIT_ATTENTION_CHUNK
    query_len = query.shape[2]
    if query_len > chunk:
        key_t = key.transpose(-2, -1)
        parts = []
        for start in range(0, query_len, chunk):
            q_chunk = query[:, :, start : start + chunk, :]
            scores = torch.matmul(q_chunk, key_t) * scale
            probs = torch.softmax(scores, dim=-1)
            parts.append(torch.matmul(probs, value))
        output = torch.cat(parts, dim=2)
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, value)

    # The matmuls ran over (batch, heads, seq, head_dim). ``is_reshaped`` means
    # the caller passed (batch, seq, hidden) and expects that layout back;
    # a 4D caller passed (batch, seq, heads, head_dim) and expects BSHD too.
    output = output.transpose(1, 2)
    if is_reshaped:
        output = output.reshape(bsz, q_len, -1)
    return output


def patch_mm_encoder_attention():
    """
    Install vendor multimodal encoder attention adapters.

    Enflame uses its native varlen provider directly because vLLM's CUDA
    FlashAttention imports and capability selection do not apply to GCU.
    """
    import vllm.model_executor.layers.attention.mm_encoder_attention as mm_mod
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if getattr(current_platform, "vendor_name", None) == "hygon":
        _patch_hygon_sdpa_contiguous()

    if getattr(current_platform, "vendor_name", None) in ("gcu", "enflame"):
        if not hasattr(mm_mod.MMEncoderAttention, "_fl_gcu_original_forward_native"):
            mm_mod.MMEncoderAttention._fl_gcu_original_forward_native = (
                mm_mod.MMEncoderAttention.forward_native
            )
        # CustomOp caches forward_oot before MMEncoderAttention selects its
        # backend. Its original OOT method calls self.forward_native, so patch
        # both methods to cover the very first encoder layer as well.
        mm_mod.MMEncoderAttention.forward_native = _gcu_forward_mm_attention
        mm_mod.MMEncoderAttention.forward_oot = _gcu_forward_mm_attention
        return

    if getattr(current_platform, "device_type", None) == "txda":
        if not hasattr(mm_mod.MMEncoderAttention, "_fl_txda_original_forward_sdpa"):
            mm_mod.MMEncoderAttention._fl_txda_original_forward_sdpa = (
                mm_mod.MMEncoderAttention._forward_sdpa
            )
        mm_mod.MMEncoderAttention._forward_sdpa = _txda_forward_sdpa
        logger.info(
            "Installed TXDA ViT attention (matmul+softmax instead of SDPA)"
        )
        return

    if getattr(current_platform, "vendor_name", None) == "kunlunxin":
        import torch
        import torch.nn.functional as F

        def _apply_chunked_sdpa(q, k, v, scale, enable_gqa):
            """Run exact SDPA while bounding the materialized score matrix."""
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            batch_outputs = []
            for batch_idx in range(q.size(0)):
                q_batch = q[batch_idx : batch_idx + 1]
                k_batch = k[batch_idx : batch_idx + 1]
                v_batch = v[batch_idx : batch_idx + 1]
                query_outputs = []
                for start in range(
                    0, q_batch.size(2), _KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE
                ):
                    query_outputs.append(
                        F.scaled_dot_product_attention(
                            q_batch[
                                :,
                                :,
                                start : start + _KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE,
                                :,
                            ],
                            k_batch,
                            v_batch,
                            dropout_p=0.0,
                            scale=scale,
                            enable_gqa=enable_gqa,
                        )
                    )
                batch_outputs.append(torch.cat(query_outputs, dim=2))
            return torch.cat(batch_outputs, dim=0).permute(0, 2, 1, 3)

        def _kunlunxin_forward_sdpa(self, query, key, value, cu_seqlens=None):
            bsz, q_len = query.size()[:2]
            kv_len = key.size(1)
            is_reshaped = query.dim() != 4
            query, key, value = self.view_qkv_to_4d(
                query, key, value, bsz, q_len, kv_len
            )

            if cu_seqlens is None:
                output = _apply_chunked_sdpa(
                    query,
                    key,
                    value,
                    self.scale,
                    self.num_heads > self.num_kv_heads,
                )
            else:
                lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                outputs = []
                for q_part, k_part, v_part in zip(
                    torch.split(query, lengths, dim=1),
                    torch.split(key, lengths, dim=1),
                    torch.split(value, lengths, dim=1),
                ):
                    outputs.append(
                        _apply_chunked_sdpa(
                            q_part,
                            k_part,
                            v_part,
                            self.scale,
                            self.num_heads > self.num_kv_heads,
                        )
                    )
                output = torch.cat(outputs, dim=1)

            if is_reshaped:
                output = output.reshape(bsz, q_len, -1)
            return output

        mm_mod.MMEncoderAttention._forward_sdpa = _kunlunxin_forward_sdpa
        logger.info_once(
            "Using query-chunked Torch SDPA for Kunlunxin MM encoder attention "
            "(chunk_size=%d).",
            _KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE,
        )

    # PlatformFL is registered as an OOT platform, so CustomOp would otherwise
    # bind MMEncoderAttention to forward_oot -> forward_native (Torch SDPA).
    # NVIDIA must keep vLLM's CUDA dispatch so the selected FLASH_ATTN backend
    # is actually used during multimodal profiling and inference.
    if current_platform.is_cuda():
        mm_mod.MMEncoderAttention.forward_oot = mm_mod.MMEncoderAttention.forward_cuda

    def _patched_maybe_get_vit_flash_attn_backend(attn_backend):
        if attn_backend == AttentionBackendEnum.FLASH_ATTN:
            try:
                from vllm.vllm_flash_attn import flash_attn_varlen_func

                logger.info_once("Using vllm.vllm_flash_attn for vit attention")
            except (ImportError, ModuleNotFoundError):
                from flash_attn import flash_attn_varlen_func

                logger.info_once("Using flash_attn for vit attention")
            return flash_attn_varlen_func
        elif attn_backend == AttentionBackendEnum.ROCM_AITER_FA:
            from aiter import flash_attn_varlen_func

            return flash_attn_varlen_func
        else:
            return None

    mm_mod.maybe_get_vit_flash_attn_backend = _patched_maybe_get_vit_flash_attn_backend
