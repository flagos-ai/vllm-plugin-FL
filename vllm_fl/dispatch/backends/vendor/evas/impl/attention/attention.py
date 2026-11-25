# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU (evas) GQA attention backend for the FL vLLM plugin.

One flaggems evas pure-PyTorch ``flash_attn_varlen_func`` call per batch —
paged for decode/prefill/mixed (``seqused_k`` = ``seq_lens``, causal offset
Sk - Sq), non-paged for encoder — since neither FlagGems Triton kernels nor
flash_evo/flash-attn work on LPU.  KV is written by ``reshape_and_cache_flash``
in the same forward; sliding window and ``logits_soft_cap`` are forwarded;
``alibi_slopes``, ``sinks``, FP8 KV and any other attention type raise
``NotImplementedError``.

"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.attention.backends.utils import get_kv_cache_layout


class EvasAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16
    ]
    # KV cache update happens inside forward() (reshape_and_cache_flash), so
    # vLLM must NOT route do_kv_cache_update additionally — the inverse of the
    # flaggems AttentionFLBackend wiring bug (writes KV only in
    # do_kv_cache_update but never flips this flag).
    forward_includes_kv_cache_update: bool = True

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["EvasAttentionBackendImpl"]:
        return EvasAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["EvasAttentionMetadataBuilder"]:
        return EvasAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        if include_num_layers_dimension:
            # vllm 0.24: prepend a num_layers dim at logical index 0 (stays
            # outermost); existing dims shift by +1.
            return (0,) + tuple(i + 1 for i in stride_order)
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class EvasAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True
    # Host copies of the three index vectors the varlen placeholder turns into
    # Python lists (see ``build``).  Optional: a metadata object built outside
    # the builder falls back to the device tensors above.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    block_table_cpu: torch.Tensor | None = None


class EvasAttentionMetadataBuilder(AttentionMetadataBuilder[EvasAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> EvasAttentionMetadata:
        # The varlen placeholder converts cu_seqlens_q, seqused_k and the block
        # table to Python lists on the host: it cannot gather KV blocks on the
        # device, because on double-die that advanced-index gather asks for a
        # full-size scratch and OOMs.  Each of those conversions is a
        # device-to-host transfer, and every layer shares this metadata object,
        # so doing them here pays one transfer per forward instead of one per
        # layer.  vLLM already keeps two of the three vectors on the host;
        # ``_seq_lens_cpu`` is None exactly when its host copy is not
        # authoritative (async speculative decoding), and then the device
        # tensor stays in use.  The copies are clones: the host buffers are
        # reused in place by the next step.
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu.clone()
        seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        seq_lens_cpu = None if seq_lens_cpu is None else seq_lens_cpu.clone()
        block_table_cpu = (
            None
            if seq_lens_cpu is None
            else common_attn_metadata.block_table_tensor.to("cpu", copy=True)
        )
        return EvasAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            block_table_cpu=block_table_cpu,
        )


class EvasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # Flash-attn (left, right) window, same convention as vLLM's
        # FlashAttentionImpl: decoder SWA is causal so right is 0, while
        # encoder-only is bidirectional.  (-1, -1) means "no window" and is a
        # no-op in the evas SDPA core (it skips the window mask).
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi is not supported by LPU_ATTN")
        # Soft-cap is a pre-softmax tanh on the scores, ``cap * tanh(s/cap)``.
        # The flaggems evas varlen already implements it (``softcap=``, applied
        # before the alibi/causal/window masks), so this is a pass-through, not
        # a new kernel.  ``0.0`` is the op's "disabled" sentinel.
        # ``0.0`` is the op's "disabled" sentinel, so ``None`` maps to it
        # rather than being falsy-collapsed (a value of ``0`` is not a
        # meaningful cap but must not be silently reinterpreted either).
        self.logits_soft_cap = (0.0 if logits_soft_cap is None
                               else float(logits_soft_cap))
        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError("FP8 KV cache is unsupported in LPU_ATTN")
        if sinks is not None:
            raise NotImplementedError(
                "Attention sinks are not supported by LPU_ATTN"
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,  # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: EvasAttentionMetadata | None,
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for LPU_ATTN"
            )

        # Warm-up path.
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        from flag_gems import flash_attn_varlen_func

        # Host copies when the builder made them (one transfer per forward
        # instead of one per layer), device tensors otherwise.  The varlen
        # placeholder moves these to the host itself, and on the encoder path
        # it indexes them element by element, so a host tensor is free there.
        cu_seqlens_q = (
            attn_metadata.query_start_loc_cpu
            if attn_metadata.query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )

        # Encoder / encoder-only: no KV cache — each seq's KV is itself
        # (cu_seqlens_k == cu_seqlens_q), via a single non-paged varlen call.
        if self.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ):
            result = flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key[:num_actual_tokens],
                v=value[:num_actual_tokens],
                max_seqlen_q=attn_metadata.max_query_len,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=attn_metadata.max_seq_len,
                cu_seqlens_k=cu_seqlens_q,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                window_size=self.sliding_window,
                softcap=self.logits_soft_cap,
            )  # [Tq, H, D]
            output[:num_actual_tokens] = result
            return output

        # Decoder path. key_cache/value_cache are FA-layout
        # [num_blocks, block_size, num_kv_heads, head_size].
        key_cache, value_cache = kv_cache.unbind(0)

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            from flag_gems import reshape_and_cache_flash

            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        seq_lens = (
            attn_metadata.seq_lens_cpu
            if attn_metadata.seq_lens_cpu is not None
            else attn_metadata.seq_lens
        )
        block_table = (
            attn_metadata.block_table_cpu
            if attn_metadata.block_table_cpu is not None
            else attn_metadata.block_table
        )

        # Single paged varlen call (class-A style) for decode + prefill + mixed:
        # the placeholder gathers each seq's KV from the FA-layout paged cache
        # and computes fp32 SDPA. ``seqused_k`` = vLLM ``seq_lens`` (decode:
        # history; prefill: history + current) — positions >= seqused_k are cut
        # by the gather and the causal mask (offset = Sk - Sq).
        result = flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            max_seqlen_q=attn_metadata.max_query_len,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=attn_metadata.max_seq_len,
            seqused_k=seq_lens,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            block_table=block_table,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
        )  # [Tq, H, D]
        output[:num_actual_tokens] = result
        return output
