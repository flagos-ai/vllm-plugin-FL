from vllm.logger import init_logger

logger = init_logger(__name__)

_KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE = 512


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
