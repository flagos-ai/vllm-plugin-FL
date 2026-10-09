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


def patch_mm_encoder_attention():
    """
    Patch vllm.attention.layers.mm_encoder_attention.maybe_get_vit_flash_attn_backend
    to support OOT platforms.

    The original implementation imports flash_attn_varlen_func from fa_utils,
    which may not have it defined for OOT platforms. This patch changes the
    FLASH_ATTN branch to import directly from vllm.vllm_flash_attn with a
    fallback to flash_attn.
    """
    import vllm.model_executor.layers.attention.mm_encoder_attention as mm_mod
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if getattr(current_platform, "vendor_name", None) == "hygon":
        _patch_hygon_sdpa_contiguous()

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
