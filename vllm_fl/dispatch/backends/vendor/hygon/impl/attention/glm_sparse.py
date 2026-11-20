"""BF16-cache GLM sparse MLA using the installed FlagGems Triton kernel.

No AITER persistent scheduler metadata is fabricated. Request/block mapping
comes from vLLM's BF16 sparse path; the actual attention runs in FlagGems.
Opt-in until full-model and multi-node validation is complete.
"""
import functools
import hashlib
import os
from pathlib import Path

import torch

from vllm.v1.attention.backend import AttentionCGSupport
from .glm_fp8_cache import write_fp8_cache, sparse_fp8_mla, qualified_interface
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadataBuilder,
)


@functools.lru_cache(maxsize=1)
def _native_sparse_entry():
    """Freeze the experimentally qualified artifact; never catch kernel errors."""
    import flash_mla.cuda as binary
    import flash_mla.flash_mla_interface as interface
    expected = (
        (binary.__file__, "4af1d9c8eef36210a8ffedfcb836d9dd9412520edfa9948b05c35817dd1f3ebd"),
        (interface.__file__, "0c7e927224f28c87439b4493a8a6764e69221aef4277a6350c047a04d4648945"),
    )
    for filename, digest in expected:
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unqualified FlashMLA artifact: {filename}")
    print("GLM_MLA_ROUTE native_flashmla experimental gfx936 heads4->8 "
          f"binary={binary.__file__} sha256={expected[0][1]}", flush=True)
    return interface.flash_mla_sparse_fwd


@torch.library.custom_op("vllm_fl::hygon_glm_native_sparse_mla", mutates_args=())
def hygon_glm_native_sparse_mla(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float,
) -> torch.Tensor:
    """Opt-in BF16/576/4-head experimental route, preserving caller indices."""
    if (q.ndim != 3 or q.shape[1:] != (4, 576)
            or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16
            or kv.shape[-1] != 576 or not kv.is_contiguous()
            or indices.dtype != torch.int32 or indices.shape[0] != q.shape[0]
            or indices.numel() != q.shape[0] * 2048
            or not 1 <= q.shape[0] <= 8192 or scale != .0625
            or not q.is_cuda or kv.device != q.device or indices.device != q.device):
        raise ValueError("FlashMLA experimental scope requires BF16/576, heads4, topk2048, scale .0625, SQ1..8192")
    props = torch.cuda.get_device_properties(q.device)
    if not str(getattr(props, "gcnArchName", "")).startswith("gfx936"):
        raise ValueError("FlashMLA experimental artifact requires gfx936")
    native = _native_sparse_entry()
    padded = q.new_zeros((q.shape[0], 8, 576))
    padded[:, :4].copy_(q)
    # Vendor wrapper sorts in-place; a private copy preserves the pure-op ABI
    # and prevents mutation of shared top-k buffers during capture/replay.
    private_indices = indices.reshape(q.shape[0], 1, 2048).clone()
    result = native(padded, kv.reshape(-1, 1, 576), private_indices, scale)[0]
    return result[:, :4].contiguous()


@hygon_glm_native_sparse_mla.register_fake
def _native_fake(q, kv, indices, scale):
    return q.new_empty((q.shape[0], q.shape[1], 512))


@torch.library.custom_op("vllm_fl::hygon_glm_sparse_mla", mutates_args=())
def hygon_glm_sparse_mla(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float,
) -> torch.Tensor:
    from flag_gems.fused.flashmla_sparse import flash_mla_sparse_fwd

    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("Hygon GLM sparse MLA requires BF16 query and KV cache")
    if q.shape[-1] != 576 or kv.shape[-1] != 576:
        raise ValueError("Hygon GLM sparse MLA requires 576-dimensional Q/K")
    heads = q.shape[1]
    if not 1 <= heads <= 128:
        raise ValueError("Hygon GLM sparse MLA supports 1..128 local heads")
    padded_heads = 64 if heads <= 64 else 128
    padded = q.new_zeros((q.shape[0], padded_heads, 576))
    padded[:, :heads].copy_(q)
    output = flash_mla_sparse_fwd(
        padded, kv.reshape(-1, 1, 576),
        indices.reshape(q.shape[0], 1, -1).contiguous(), scale,
    )[0]
    # No valid-count truncation: invalid slots can appear inside a top-k row.
    return output[:, :heads].contiguous()


@hygon_glm_sparse_mla.register_fake
def _fake(q, kv, indices, scale):
    return q.new_empty((q.shape[0], q.shape[1], 512))


@torch.library.custom_op("vllm_fl::hygon_glm_cache_write", mutates_args=("cache",))
def hygon_glm_cache_write(
    kv: torch.Tensor, rope: torch.Tensor, cache: torch.Tensor,
    slots: torch.Tensor, scale: torch.Tensor,
) -> None:
    from flag_gems.fused.concat_and_cache_mla import concat_and_cache_mla
    if kv.dtype != torch.bfloat16 or rope.dtype != kv.dtype or cache.dtype != kv.dtype:
        raise ValueError("Hygon GLM cache write requires BF16 tensors")
    concat_and_cache_mla(kv, rope, cache, slots, "auto", scale)


@hygon_glm_cache_write.register_fake
def _cache_fake(kv, rope, cache, slots, scale):
    return None


class HygonGLMSparseMetadataBuilder(FlashMLASparseMetadataBuilder):
    # Only single-token decode Graph is currently an intended validation target.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if (vllm_config.model_config.hf_config.model_type != "glm_moe_dsa"
                or vllm_config.model_config.dtype != torch.bfloat16
                or vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16", "fp8_ds_mla")):
            raise ValueError("Hygon GLM sparse backend requires GLM DSA with BF16 or packed FP8 KV")
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self.num_heads = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config)
        self.topk_tokens = self.model_config.hf_config.index_topk
        kernel = os.environ.get("VLLM_FL_HYGON_GLM_MLA_KERNEL", "flaggems")
        if kernel not in ("flaggems", "native_flashmla"):
            raise ValueError(f"Unknown GLM MLA kernel {kernel!r}")
        if kernel == "native_flashmla":
            if (self.num_heads != 4 or self.topk_tokens != 2048
                    or vllm_config.speculative_config is not None):
                raise ValueError("Native FlashMLA requires heads4/topk2048/no speculative decoding")
            _native_sparse_entry()
        if vllm_config.cache_config.cache_dtype == "fp8_ds_mla":
            qualified_interface()
            route=os.environ.get("VLLM_FL_GLM_FP8_MLA_IMPL", "vendor")
            if route not in ("vendor", "triton_grouped"):
                raise ValueError("Unknown FP8 MLA implementation: "+route)
            print(f"GLM_FP8_SELECTED_IMPL {route} gfx936 heads4 cache656", flush=True)
        # Reuse request/page metadata; native FP8 op owns its scheduler.
        self.use_fp8_kv_cache = False
        self.req_id_per_token_buffer = torch.zeros(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32, device=device)


class HygonGLMSparseImpl(FlashMLASparseImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads,
                 alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap,
                 attn_type, kv_sharing_target_layer_name,
                 topk_indices_buffer=None, indexer=None, **mla_args):
        # Our forward concatenates Q directly and reads the packed cache in the
        # vendor kernel. Do not retain the parent's unused BF16 prefill staging
        # buffer (5 * max_model_len * 576 * 2 bytes) or Q workspace view.
        if kv_cache_dtype not in ("auto", "bfloat16", "fp8_ds_mla"):
            raise ValueError("Unsupported Hygon GLM KV cache format")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank = mla_args["kv_lora_rank"]
        self.softmax_scale = scale
        self.topk_indices_buffer = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer)

    def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping,
                           kv_cache_dtype, k_scale):
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype == "fp8_ds_mla":
            write_fp8_cache(kv_c_normed, k_pe.squeeze(1), kv_cache, slot_mapping.flatten())
            return
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("Hygon GLM sparse backend requires BF16 KV")
        hygon_glm_cache_write(kv_c_normed, k_pe.squeeze(1), kv_cache,
                              slot_mapping.flatten(), k_scale)

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        # empty has no native concat_mla_q kernel; torch.cat is capturable.
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert self.topk_indices_buffer is not None
        if self.kv_cache_dtype == "fp8_ds_mla":
            indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token, attn_metadata.block_table,
                self.topk_indices_buffer[:q.shape[0]],
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=self.topk_indices_buffer.shape[1])
            return sparse_fp8_mla(q, kv_c_and_k_pe_cache, indices, self.softmax_scale), None
        return self._forward_bf16_kv(
            q, kv_c_and_k_pe_cache, self.topk_indices_buffer[:q.shape[0]],
            attn_metadata), None

    def _bf16_flash_mla_kernel(self, q, kv_c_and_k_pe_cache, topk_indices,
                             topk_length=None):
        if os.environ.get("VLLM_FL_HYGON_GLM_MLA_KERNEL", "flaggems") == "native_flashmla":
            return hygon_glm_native_sparse_mla(
                q, kv_c_and_k_pe_cache, topk_indices, self.softmax_scale)
        return hygon_glm_sparse_mla(
            q, kv_c_and_k_pe_cache, topk_indices, self.softmax_scale)


class HygonGLMSparseBackend(FlashMLASparseBackend):
    # MLA writes KV via unified_mla_kv_cache_update before attention.forward.
    # The runner must pad slot mappings alongside K/V in PIECEWISE graphs;
    # its -1 padding ensures the writer skips inactive rows.
    forward_includes_kv_cache_update = False

    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8_ds_mla"]

    @staticmethod
    def get_supported_kernel_block_sizes():
        # Unlike native FlashMLA, FlagGems consumes a flat cache. The mapping
        # kernel uses the selected block size. A mixed cache group with the
        # ROCm indexer must also support virtual splitting from 16 down to 1.
        return [1, 16, 64]

    @staticmethod
    def get_name():
        return "HYGON_GEMS_GLM_SPARSE"

    @classmethod
    def supports_compute_capability(cls, capability):
        return True  # Only selected by the explicit Hygon vendor opt-in.

    @staticmethod
    def get_builder_cls():
        return HygonGLMSparseMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return HygonGLMSparseImpl
