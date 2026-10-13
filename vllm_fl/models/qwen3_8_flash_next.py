# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next inference model and vLLM integration."""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import islice
from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    GDNAttentionMetadata,
    QwenGatedDeltaNetAttention,
    causal_conv1d_update,
    is_conv_state_dim_first as is_gdn_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMLP,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3_VisionTransformer,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    async_tensor_h2d,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    MambaSpec,
    get_kv_quant_mode,
)

from vllm_fl.dispatch import resolve_op

try:
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )
except ImportError:  # vLLM builds without the legacy fused-MoE helper
    fused_moe_make_expert_params_mapping = None

try:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateCopyFuncsByType,
    )
except ImportError:  # vLLM 0.24 type-only compatibility
    MambaStateCopyFuncsByType = dict

try:
    from vllm.model_executor.models.utils import maybe_fuse_shared_experts
except ImportError:  # vLLM 0.24 predates optional AITER shared-expert fusion

    def maybe_fuse_shared_experts(weights, **_kwargs):
        return weights


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_QSA_CONFIG_FIELDS = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)


class Qwen3_8FlashNextVisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen3_8_flash_next"
    base_config_key = "vision_config"


class Qwen3_8FlashNextTextConfig(Qwen3NextConfig):
    model_type = "qwen3_8_flash_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    supports_qwen38_ngram_context = True

    def __init__(
        self,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        ple_layer_ids: list[int] | None = None,
        ple_embed_dim: int | None = None,
        ple_conv_kernel_size: int = 4,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20_000_000,
        make_ngram_vocab_size_divisible_by: int = 128,
        output_gate_type: str = "sigmoid",
        rope_parameters: dict[str, Any] | None = None,
        layer_types: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if hc_count <= 1:
            raise ValueError(
                f"Qwen3.8-Flash-Next requires hc_count > 1, got {hc_count}."
            )

        if rope_parameters is not None:
            if kwargs.get("rope_scaling") is None:
                kwargs["rope_scaling"] = rope_parameters
            if kwargs.get("rope_theta") is None and "rope_theta" in rope_parameters:
                kwargs["rope_theta"] = rope_parameters["rope_theta"]
            if (
                kwargs.get("partial_rotary_factor") is None
                and "partial_rotary_factor" in rope_parameters
            ):
                kwargs["partial_rotary_factor"] = rope_parameters[
                    "partial_rotary_factor"
                ]

        rope_scaling = kwargs.get("rope_scaling")
        rope_theta = kwargs.get("rope_theta", 10_000.0)
        super().__init__(layer_types=layer_types, **kwargs)

        normalized_rope_parameters = self.rope_parameters
        self.rope_scaling = (
            rope_scaling or rope_parameters or normalized_rope_parameters
        )
        self.rope_parameters = rope_parameters or normalized_rope_parameters
        self.rope_theta = rope_theta

        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.ple_layer_ids = ple_layer_ids or []
        self.ple_embed_dim = (
            self.hidden_size if ple_embed_dim is None else ple_embed_dim
        )
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.output_gate_type = output_gate_type

        self._validate_ple_config()
        self._validate_ple_layer_ids()
        self._validate_qsa_config()

    def _validate_ple_config(self) -> None:
        if self.hc_lowrank <= 0:
            raise ValueError(f"hc_lowrank must be positive, got {self.hc_lowrank}")
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(
                f"heads_per_ngram must be positive, got {self.heads_per_ngram}"
            )
        if self.ple_embed_dim <= 0:
            raise ValueError(
                f"ple_embed_dim must be positive, got {self.ple_embed_dim}"
            )
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ple_embed_dim % ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ple_embed_dim} % {ngram_heads} != 0"
            )
        if self.ple_conv_kernel_size <= 0:
            raise ValueError(
                "ple_conv_kernel_size must be positive, got "
                f"{self.ple_conv_kernel_size}"
            )
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be positive")
        if self.make_ngram_vocab_size_divisible_by <= 0:
            raise ValueError("make_ngram_vocab_size_divisible_by must be positive")

    def _validate_ple_layer_ids(self) -> None:
        invalid = [
            layer_id
            for layer_id in self.ple_layer_ids
            if not 1 <= int(layer_id) <= self.num_hidden_layers
        ]
        if invalid:
            raise ValueError(
                "ple_layer_ids are 1-based and must refer to an existing layer; "
                f"got {invalid} for {self.num_hidden_layers} layers"
            )

    def _validate_qsa_config(self) -> None:
        configured = {name: getattr(self, name, None) for name in _QSA_CONFIG_FIELDS}
        if all(value is None for value in configured.values()):
            return

        missing = [name for name, value in configured.items() if value is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}")

        values = {name: int(cast(int, value)) for name, value in configured.items()}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"QSA config values must be positive: {values}")
        if values["indexer_kv_heads"] != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if values["indexer_budget"] % values["indexer_compress_ratio"] != 0:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        block_topk = values["indexer_budget"] // values["indexer_compress_ratio"]
        if block_topk not in (512, 2048):
            raise ValueError(
                "QSA requires indexer_budget / indexer_compress_ratio "
                f"to be 512 or 2048, got {block_topk}"
            )
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim > values["indexer_head_dim"]:
            raise ValueError(
                "QSA indexer_head_dim must cover the attention rotary "
                f"dimension, got {values['indexer_head_dim']} < {rotary_dim}"
            )

    @property
    def layers_block_type(self) -> list[str]:
        return [
            "attention" if layer_type == "full_attention" else layer_type
            for layer_type in self.layer_types
        ]

    @property
    def short_conv_layer_ids(self) -> list[int]:
        if not self.ple_layer_ids:
            return []
        return sorted({int(layer_id) - 1 for layer_id in self.ple_layer_ids})

    @property
    def short_conv_state_shape(self) -> tuple[int, int] | None:
        if not self.short_conv_layer_ids:
            return None
        ple_state_len = (self.ple_conv_kernel_size - 1) * self.ngram_size
        ple_channels = self.hidden_size * self.hc_count
        return ple_channels, ple_state_len

    @property
    def ngram_context_len(self) -> int:
        if not self.ple_layer_ids:
            return 0
        return max(int(self.ngram_size) - 1, 0)

    @property
    def spec_hidden_size(self) -> int:
        return int(self.hc_count * self.hidden_size)

    @property
    def uses_per_group_attn_metadata(self) -> bool:
        return getattr(self, "indexer_n_heads", None) is not None

    @property
    def spec_decode_returns_tuple(self) -> bool:
        return True


class Qwen3_8FlashNextConfig(PretrainedConfig):
    model_type = "qwen3_8_flash_next"
    sub_configs = {
        "vision_config": Qwen3_8FlashNextVisionConfig,
        "text_config": Qwen3_8FlashNextTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: Qwen3_8FlashNextTextConfig | dict[str, Any] | None = None,
        vision_config: Qwen3_8FlashNextVisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if text_config is not None:
            kwargs.pop("split_ngram_parts", None)

        text_kwargs = (
            dict(kwargs)
            if text_config is None
            and "hidden_size" in kwargs
            and "num_hidden_layers" in kwargs
            else {}
        )

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**text_kwargs)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs)
        # Match the vLLM 0.24 Qwen3.5 config ordering. Passing this through
        # transformers>=5 PretrainedConfig.__init__ causes its compatibility
        # wrapper to serialize already-built sub-configs back into dictionaries.
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters or getattr(
            self.text_config, "rope_parameters", {}
        )


# Checkpoint-facing aliases. Keep these as true aliases rather than inherited
# composite config classes: Transformers 5's config metaclass serializes
# inherited ``sub_configs`` to dictionaries, which breaks ``hf_text_config``.
# The checkpoint-provided ``model_type`` remains on each loaded instance.
Qwen4ExpVisionConfig = Qwen3_8FlashNextVisionConfig
Qwen4ExpTextConfig = Qwen3_8FlashNextTextConfig
Qwen4ExpConfig = Qwen3_8FlashNextConfig



# -----------------------------------------------------------------------------
# Hyperconnection layers
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class HyperConnectionConfig:
    """Configuration shared by all HyperConnection variants."""

    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (
                grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


# ---------------------------------------------------------------------------
# Average-pooling variant
# ---------------------------------------------------------------------------
class HyperConnectionBase(nn.Module):
    """Average-pooling HyperConnection (``hyperconnection_average``).

    Splits the incoming ``[..., HC*HS]`` tensor (HC outer, HS inner) into
    ``HC`` parallel streams, averages them for the block input, and
    broadcasts the block output back to every stream.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        layer_idx: int | None = None,
        role: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.role = role

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    def mix(self, hyper_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Average the HC streams into a single block input."""
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        # [*, HC, HS] — mean over HC (dim=-2).
        unflat = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = unflat.mean(dim=-2)
        return mixed_input, hyper_input

    def combine(
        self, block_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast the block output back to every stream."""
        assert residual.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual_reshaped = residual.unflatten(-1, (self.hc_count, self.hidden_size))
        combined = residual_reshaped + block_output.unsqueeze(-2)
        return combined.flatten(-2)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidualSimple(HyperConnectionBase):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``mix()`` applies GemmaRMSNorm per HC stream and projects through a
    low-rank sigmoid gate to produce a single block input. ``combine()``
    injects the block output back into each stream through a learned
    per-stream injection weight.

    This implementation uses only PyTorch operators. Tensor-parallel
    collectives are supplied by its caller.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        layer_idx: int | None = None,
        role: str | None = None,
        use_mix: bool = True,
        use_combine: bool = True,
    ) -> None:
        super().__init__(config, layer_idx, role)
        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- raw Linear weights (checkpoint-compatible) ----------------------
        if use_mix:
            self.input_mix_weight_down = nn.Linear(
                self.hyper_hidden_size,
                config.hc_lowrank,
                bias=False,
                dtype=config.params_dtype,
            )
            self.input_mix_weight_up = nn.Linear(
                config.hc_lowrank,
                self.hyper_hidden_size,
                bias=False,
                dtype=config.params_dtype,
            )
        if use_combine:
            self.block_inject_weight = nn.Linear(
                self.hyper_hidden_size,
                self.hc_count,
                bias=False,
                dtype=config.params_dtype,
            )

    def _normalize(self, hyper_input: torch.Tensor) -> torch.Tensor:
        if self.config.hc_per_branch_norm:
            return self.hc_norm(hyper_input)
        return self.hc_norm(
            hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        ).flatten(-2)

    def mix(
        self, hyper_input: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Mix: RMSNorm -> low-rank gate -> gated mean."""
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not hasattr(self, "input_mix_weight_down"):
            raise RuntimeError("mix was disabled for this hyper-connection")
        hyper_input_normed = self._normalize(hyper_input)
        # Gate — original mix order: linear+silu then linear+sigmoid.
        gate = F.silu(
            F.linear(hyper_input_normed, self.input_mix_weight_down.weight)
            / self.hc_count
        )
        gate = torch.sigmoid(F.linear(gate, self.input_mix_weight_up.weight)).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        mixed_input = (
            gate * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        return mixed_input.to(hyper_input.dtype), (hyper_input, hyper_input_normed)

    def combine(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not hasattr(self, "block_inject_weight"):
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        # The paired mix keeps its normalized hyper input so combine uses the
        # same HC module's injection weight.
        injection_weight = 2.0 * torch.sigmoid(
            F.linear(hyper_input_normed, self.block_inject_weight.weight)
            / self.hc_count
        )
        output = residual + block_output.unsqueeze(-2) * injection_weight.unsqueeze(-1)
        return output.flatten(-2).to(hyper_input.dtype)



# -----------------------------------------------------------------------------
# PLE checkpoint helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PLEShardOverlap:
    """Source and destination slices for one checkpoint embedding shard."""

    source_start: int
    destination_start: int
    row_count: int


def compute_ple_shard_overlap(
    *,
    checkpoint_start: int,
    checkpoint_rows: int,
    tp_start: int,
    tp_end: int,
) -> PLEShardOverlap | None:
    """Compute the overlap of a checkpoint shard and one TP vocabulary range."""

    if checkpoint_start < 0 or checkpoint_rows < 0:
        raise ValueError("checkpoint shard bounds must be non-negative")
    if tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid TP vocabulary range")
    checkpoint_end = checkpoint_start + checkpoint_rows
    overlap_start = max(checkpoint_start, tp_start)
    overlap_end = min(checkpoint_end, tp_end)
    if overlap_start >= overlap_end:
        return None
    return PLEShardOverlap(
        source_start=overlap_start - checkpoint_start,
        destination_start=overlap_start - tp_start,
        row_count=overlap_end - overlap_start,
    )


def copy_ple_embedding_shard_(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy the overlapping rows of a PLE checkpoint shard into a TP table."""

    if destination.ndim == 0 or loaded_weight.ndim != destination.ndim:
        raise ValueError("destination and loaded weight must have matching ranks")
    if destination.shape[1:] != loaded_weight.shape[1:]:
        raise ValueError(
            "embedding shard dimensions do not match: "
            f"{tuple(destination.shape[1:])} != {tuple(loaded_weight.shape[1:])}"
        )
    if destination.shape[0] < tp_end - tp_start:
        raise ValueError("destination does not cover the requested TP range")
    overlap = compute_ple_shard_overlap(
        checkpoint_start=checkpoint_start,
        checkpoint_rows=loaded_weight.shape[0],
        tp_start=tp_start,
        tp_end=tp_end,
    )
    if overlap is None:
        return 0
    source = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
    target = destination.narrow(0, overlap.destination_start, overlap.row_count)
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    return overlap.row_count

# -----------------------------------------------------------------------------
# QSA cache and metadata
# -----------------------------------------------------------------------------

def canonical_qsa_rope_positions(positions: torch.Tensor) -> torch.Tensor:
    """Return exact per-token positions as ``[tokens, 1, 3]`` int64 rows."""

    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    elif positions.ndim != 2 or positions.shape[0] not in (1, 3):
        raise ValueError("QSA RoPE positions must be [tokens] or [1|3, tokens]")
    if positions.shape[0] == 1:
        positions = positions.expand(3, -1)
    return positions.transpose(0, 1).unsqueeze(1).to(torch.int64)


def _logical_positions(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    num_tokens: int,
    arange: torch.Tensor | None = None,
) -> torch.Tensor:
    if num_tokens == 0:
        return seq_lens.new_empty((0,), dtype=torch.int64)
    if arange is None:
        arange = torch.arange(num_tokens, device=query_start_loc.device)
    else:
        arange = arange[:num_tokens]
    requests = token_to_req[:num_tokens].long()
    query_lens = torch.diff(query_start_loc)
    within_query = arange - query_start_loc.index_select(0, requests)
    return (
        seq_lens.index_select(0, requests).long()
        - query_lens.index_select(0, requests).long()
        + within_query.long()
    )


def _logical_to_physical_qsa_slots(
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("QSA cache block size must be positive")
    if block_table.ndim != 2:
        raise ValueError("QSA block table must be two-dimensional")
    if request_indices.shape != logical_positions.shape:
        request_indices = torch.broadcast_to(request_indices, logical_positions.shape)

    requests = request_indices.to(device=block_table.device, dtype=torch.long)
    positions = logical_positions.to(device=block_table.device, dtype=torch.long)
    valid = (requests >= 0) & (requests < block_table.shape[0]) & (positions >= 0)
    logical_blocks = torch.div(
        positions.clamp_min(0), block_size, rounding_mode="floor"
    )
    valid &= logical_blocks < block_table.shape[1]
    safe_requests = requests.clamp(0, max(block_table.shape[0] - 1, 0))
    safe_blocks = logical_blocks.clamp(0, max(block_table.shape[1] - 1, 0))
    if not all(block_table.shape):
        return torch.full_like(positions, -1)
    physical_blocks = block_table[safe_requests, safe_blocks].long()
    valid &= physical_blocks >= 0
    slots = physical_blocks * block_size + positions.remainder(block_size)
    return torch.where(valid, slots, torch.full_like(slots, -1))


def compressed_qsa_slot_mapping(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build boundary-only slots for an ``MLAAttentionSpec`` QSA cache."""

    if storage_block_size <= 0 or compress_ratio <= 0:
        raise ValueError("QSA block size and compression ratio must be positive")
    compressed_positions = torch.div(
        logical_positions.clamp_min(0), compress_ratio, rounding_mode="floor"
    )
    slots = _logical_to_physical_qsa_slots(
        block_table,
        token_to_req,
        compressed_positions,
        storage_block_size,
    )
    valid = (logical_positions >= 0) & (
        (logical_positions + 1).remainder(compress_ratio) == 0
    )
    slots = torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int64)
    if out is not None:
        out.fill_(-1)
        out[: slots.numel()].copy_(slots)
        return out[: slots.numel()]
    return slots


@dataclass
class QSAForwardMetadata(AttentionMetadata):
    """Common per-forward metadata for one QSA side cache."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    num_actual_tokens: int
    storage_block_size: int
    compress_ratio: int


class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
    """Build QSA metadata from vLLM's cache-group-specific common metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compress_ratio = (
            kv_cache_spec.compress_ratio
            if isinstance(kv_cache_spec, MLAAttentionSpec)
            else 1
        )
        self.storage_block_size = kv_cache_spec.storage_block_size
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self.arange_buffer = torch.arange(max_tokens, dtype=torch.int64, device=device)
        self.slot_mapping_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        self.logical_positions_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QSAForwardMetadata:
        del common_prefix_len, fast_build
        num_tokens = common_attn_metadata.num_actual_tokens
        token_to_req_fn = getattr(common_attn_metadata, "token_to_req_indices", None)
        if token_to_req_fn is not None:
            token_to_req = token_to_req_fn(self.token_to_req_buffer)[:num_tokens]
        else:
            # vLLM 0.24 predates the cached convenience method. Reproduce the
            # same device-side mapping without changing CommonAttentionMetadata.
            num_mapped = int(common_attn_metadata.query_start_loc_cpu[-1])
            query_lens = (
                common_attn_metadata.query_start_loc[1:]
                - common_attn_metadata.query_start_loc[:-1]
            )
            mapped = torch.repeat_interleave(
                torch.arange(
                    query_lens.shape[0],
                    dtype=torch.int32,
                    device=self.token_to_req_buffer.device,
                ),
                query_lens,
                output_size=num_mapped,
            )
            self.token_to_req_buffer[:num_mapped].copy_(mapped)
            if num_mapped < num_tokens:
                self.token_to_req_buffer[num_mapped:num_tokens].zero_()
            token_to_req = self.token_to_req_buffer[:num_tokens]
        num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        logical_positions = self.logical_positions_buffer[:num_tokens]
        logical_positions[:num_mapped_tokens].copy_(
            _logical_positions(
                common_attn_metadata.query_start_loc,
                common_attn_metadata.seq_lens,
                token_to_req[:num_mapped_tokens],
                num_mapped_tokens,
                self.arange_buffer,
            )
        )
        if num_mapped_tokens < num_tokens:
            logical_positions[num_mapped_tokens:].fill_(-1)
        if self.compress_ratio == 1:
            slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
        else:
            slot_mapping = compressed_qsa_slot_mapping(
                common_attn_metadata.block_table_tensor,
                token_to_req,
                logical_positions,
                self.storage_block_size,
                self.compress_ratio,
                self.slot_mapping_buffer,
            )
            slot_mapping.masked_fill_(
                common_attn_metadata.slot_mapping[:num_tokens] < 0, -1
            )
        return QSAForwardMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=slot_mapping,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc,
            token_to_req=token_to_req,
            logical_positions=logical_positions,
            num_actual_tokens=num_tokens,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )


class QSAStateBackend(AttentionBackend):
    """Key-only dummy backend for out-of-band BF16 QSA side-cache operations."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_EXP_QSA_STATE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "QSA state caches run out-of-band and have no attention impl"
        )

    @staticmethod
    def get_builder_cls() -> type[QSAMetadataBuilder]:
        return QSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        if num_kv_heads != 1:
            raise ValueError("QSA side caches require exactly one KV head")
        return (num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        # The cache is num-blocks-first and every QSA kernel consumes the
        # tensor's physical block stride.  This lets vLLM 0.24 pad the raw
        # MRoPE side-cache page to the hybrid allocator's common page size.
        return True

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)


class _QSAStateCache(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(
        self,
        *,
        head_size: int,
        dtype: torch.dtype,
        cache_config: CacheConfig,
        prefix: str,
        vllm_config: VllmConfig,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        if head_size <= 0:
            raise ValueError("QSA cache head size must be positive")
        if compress_ratio <= 0:
            raise ValueError("QSA compression ratio must be positive")
        if cache_config.block_size % compress_ratio:
            raise ValueError(
                "QSA cache block size must be divisible by the compression ratio"
            )
        self.head_size = head_size
        self.dtype = dtype
        self.cache_config = cache_config
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.kv_cache = torch.tensor([])

        static_context = vllm_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def forward(self) -> None: ...

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind storage on both legacy direct-assign and newer hook runtimes."""
        self.kv_cache = kv_cache

    def get_attn_backend(self) -> type[AttentionBackend]:
        return QSAStateBackend


class QSAKeyStateCache(_QSAStateCache):
    """Raw BF16 key, optionally followed by exact int64 MRoPE positions."""

    _BF16_PER_INT64 = 4
    _NUM_ROPE_AXES = 3

    def __init__(self, *, cache_rope_positions: bool = False, **kwargs) -> None:
        key_head_size = int(kwargs.pop("head_size"))
        self.key_head_size = key_head_size
        self.cache_rope_positions = bool(cache_rope_positions)
        self.rope_position_offset = (
            (key_head_size + self._BF16_PER_INT64 - 1) // self._BF16_PER_INT64
        ) * self._BF16_PER_INT64
        storage_head_size = key_head_size
        if self.cache_rope_positions:
            storage_head_size = self.rope_position_offset + (
                self._NUM_ROPE_AXES * self._BF16_PER_INT64
            )
        super().__init__(head_size=storage_head_size, **kwargs)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if kv_cache.ndim != 4 or kv_cache.shape[2] != 1:
            raise ValueError("QSA raw cache must be [blocks, block_size, 1, width]")
        if kv_cache.dtype != torch.bfloat16 or kv_cache.shape[3] != self.head_size:
            raise ValueError("QSA raw cache does not match its packed BF16 cache spec")
        super().bind_kv_cache(kv_cache)
        self.key_cache = kv_cache[..., : self.key_head_size]
        if self.cache_rope_positions:
            position_tail = kv_cache[..., self.rope_position_offset :]
            self.rope_position_cache = position_tail.view(torch.int64)
        else:
            self.rope_position_cache = None

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return FullAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self.dtype,
        )


class QSACompressedKeyCache(_QSAStateCache):
    """Normalized, group-first-RoPE BF16 key at one row per complete group."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
        )



# -----------------------------------------------------------------------------
# PLE short-convolution metadata
# -----------------------------------------------------------------------------

class ShortConvAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "SHORT_CONV_ATTN"

    @staticmethod
    def get_builder_cls() -> type["ShortConvAttentionMetadataBuilder"]:
        return ShortConvAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class ShortConvAttentionMetadata(BaseMambaAttentionMetadata):
    pass


class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    metadata_cls = ShortConvAttentionMetadata


@dataclass
class PleShortConvAttentionMetadata(ShortConvAttentionMetadata):
    # Number of speculative-decode (multi-query / MTP) requests and the total
    # number of tokens they contribute. These are 0 when spec-decode is off.
    num_spec_decodes: int = 0
    num_spec_decode_tokens: int = 0
    num_actual_tokens: int = 0

    # Max query length among spec-decode requests (== num_speculative_tokens + 1).
    # Used as ``max_query_len`` for the varlen spec causal_conv1d_update.
    spec_query_len: int = 1

    # Max query length among the non-spec *prefill* requests, precomputed
    # CPU-side in the builder. The dilated PLE short-conv uses it to size its
    # packing buffer without a device->host sync (``lengths.max().item()``).
    # 0 when there are no prefill requests.
    max_prefill_query_len: int = 0
    query_start_loc: torch.Tensor | None = None

    # ``state_indices_tensor`` keeps the historical (non-spec) layout used by
    # all existing short-conv consumers: the conv-state slot for each regular
    # decode followed by each prefill request. When spec-decode is active this
    # only covers the non-spec requests.
    state_indices_tensor: torch.Tensor | None = None
    has_initial_states_d: torch.Tensor | None = None

    # ``non_spec_query_start_loc`` is the varlen cumulative token offset over
    # the non-spec requests only (decodes then prefills). It equals
    # ``query_start_loc`` when there are no spec-decode requests.
    non_spec_query_start_loc: torch.Tensor | None = None

    # Speculative-decode (MTP) conv metadata. Only column 0 of the block table
    # is needed for the convolution state, so these tensors are 1-D over the
    # spec-decode requests.
    spec_query_start_loc: torch.Tensor | None = None  # [num_spec_decodes + 1]
    spec_state_indices_tensor: torch.Tensor | None = None  # [num_spec_decodes]
    spec_sequence_masks: torch.Tensor | None = None  # [batch]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None


class PleShortConvAttentionBackend(ShortConvAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "PLE_SHORT_CONV_ATTN"

    @staticmethod
    def get_builder_cls() -> type["PleShortConvAttentionMetadataBuilder"]:
        return PleShortConvAttentionMetadataBuilder


class PleShortConvAttentionMetadataBuilder(ShortConvAttentionMetadataBuilder):
    metadata_cls = PleShortConvAttentionMetadata
    # Spec-decode requires a uniform (multi-token) decode batch for full
    # CUDA graph capture, matching the GDN backend.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1
    supports_update_block_table = False

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.num_spec = self.num_spec_tokens
        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_capture_size = self.compilation_config.max_cudagraph_capture_size
        self.decode_cudagraph_max_bs = max_num_seqs
        self.decode_cudagraph_max_tokens = max_num_seqs * (self.num_spec + 1)
        if max_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs, max_capture_size
            )
            self.decode_cudagraph_max_tokens = min(
                self.decode_cudagraph_max_tokens, max_capture_size
            )

        # Persistent buffers reused during full CUDA graph capture and replay.
        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.bool, device=device
        )
        self.spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_tokens,), dtype=torch.int32, device=device
        )
        self.non_spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_tokens,), dtype=torch.int32, device=device
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,), dtype=torch.int32, device=device
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.has_initial_states_d = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.bool, device=device
        )

    def _build_non_spec_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
        **kwargs: Any,
    ) -> PleShortConvAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
            num_accepted_tokens=None,
            **kwargs,
        )
        assert isinstance(metadata, PleShortConvAttentionMetadata)

        state_indices_d = metadata.state_indices_tensor_d
        if state_indices_d is not None and state_indices_d.dim() > 1:
            state_indices_d = state_indices_d[:, 0]
        state_indices_p = metadata.state_indices_tensor_p
        if metadata.num_prefills == 0:
            assert state_indices_d is not None
            # BaseMambaAttentionMetadataBuilder pads decode state indices into
            # a persistent tensor for full CUDA graphs. Keep those rows so the
            # PLE decode receives one cache slot per graph-padded token.
            state_indices_tensor = state_indices_d
        elif metadata.num_decodes == 0:
            assert state_indices_p is not None
            state_indices_tensor = state_indices_p[: metadata.num_prefills]
        else:
            assert state_indices_d is not None
            assert state_indices_p is not None
            state_indices_tensor = torch.cat(
                (state_indices_d, state_indices_p[: metadata.num_prefills])
            )

        has_initial_states_d = None
        if metadata.num_decodes > 0:
            num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
            has_initial_states_d = num_computed_tokens[: metadata.num_decodes] > 0
            if (
                self.use_full_cuda_graph
                and metadata.num_prefills == 0
                and metadata.num_decodes <= self.decode_cudagraph_max_bs
            ):
                assert state_indices_d is not None
                # Prepare tensors for CUDA graph replay. Padded rows have no
                # initial state and use NULL_BLOCK_ID in state_indices_d.
                num_decode_rows = state_indices_d.numel()
                self.has_initial_states_d[: metadata.num_decodes].copy_(
                    has_initial_states_d, non_blocking=True
                )
                self.has_initial_states_d[metadata.num_decodes : num_decode_rows].fill_(
                    False
                )
                has_initial_states_d = self.has_initial_states_d[:num_decode_rows]

        max_prefill_query_len = 0
        if metadata.num_prefills > 0:
            query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
            max_prefill_query_len = int(
                query_lens_cpu[
                    metadata.num_decodes : (
                        metadata.num_decodes + metadata.num_prefills
                    )
                ]
                .max()
                .item()
            )

        return replace(
            metadata,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            spec_query_len=self.num_spec + 1,
            max_prefill_query_len=max_prefill_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            state_indices_tensor=state_indices_tensor,
            has_initial_states_d=has_initial_states_d,
            non_spec_query_start_loc=common_attn_metadata.query_start_loc,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> PleShortConvAttentionMetadata:
        m = common_attn_metadata
        spec_sequence_masks_cpu: torch.Tensor | None = None
        # Detect speculative-decode requests. We use -1 to mark prefill and
        # plain-decode requests, so any value >= 0 is a (multi-query)
        # spec-decode request.
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            candidate_mask = num_decode_draft_tokens_cpu[: m.num_reqs] >= 0
            if bool(candidate_mask.any().item()):
                spec_sequence_masks_cpu = candidate_mask

        if spec_sequence_masks_cpu is None:
            return self._build_non_spec_metadata(
                common_prefix_len,
                common_attn_metadata,
                fast_build,
                num_decode_draft_tokens_cpu,
                **kwargs,
            )

        del common_prefix_len, fast_build, kwargs
        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        query_lens_cpu = torch.diff(query_start_loc_cpu)
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        if query_start_loc.device.type == "cpu":
            spec_sequence_masks = spec_sequence_masks_cpu
        else:
            spec_sequence_masks = async_tensor_h2d(
                spec_sequence_masks_cpu, device=query_start_loc.device
            )

        # For causal_conv1d (non-spec prefill Triton kernel metadata).
        nums_dict = None
        batch_ptr = None
        token_chunk_offset_ptr = None
        has_initial_states_p = None
        has_initial_states_d = None
        num_computed_tokens_p = None
        # Original request indices of the non-spec requests, ordered
        # [decodes, prefills]. Used to gather per-request data consistently.
        non_spec_req_idx_cpu: torch.Tensor | None = None

        query_lens = torch.diff(query_start_loc)
        # Per-request classification by mask, NOT by position. With
        # spec-decode, the front decode group can contain both spec-decode
        # requests and plain non-spec single-token decodes. Spec requests are
        # therefore not guaranteed to occupy the first num_spec_decodes slots.
        non_spec_mask_cpu = ~spec_sequence_masks_cpu
        decode_mask_cpu = non_spec_mask_cpu & (query_lens_cpu == 1)
        prefill_mask_cpu = non_spec_mask_cpu & (query_lens_cpu > 1)

        num_spec_decodes = int(spec_sequence_masks_cpu.sum().item())
        num_decodes = int(decode_mask_cpu.sum().item())
        num_prefills = int(prefill_mask_cpu.sum().item())
        num_decode_tokens = num_decodes
        num_prefill_tokens = int(query_lens_cpu[prefill_mask_cpu].sum().item())
        num_spec_decode_tokens = int(
            query_lens_cpu[spec_sequence_masks_cpu].sum().item()
        )
        # Max prefill query length, CPU-side (no device-to-host sync) for the
        # PLE dilated short-conv packing buffer.
        max_prefill_query_len = (
            int(query_lens_cpu[prefill_mask_cpu].max().item())
            if num_prefills > 0
            else 0
        )

        # Original request indices grouped as
        # [spec | non-spec decode | non-spec prefill]; each group keeps the
        # original (already reordered) relative order via a stable nonzero.
        spec_req_idx_cpu = spec_sequence_masks_cpu.nonzero(as_tuple=True)[0]
        decode_req_idx_cpu = decode_mask_cpu.nonzero(as_tuple=True)[0]
        prefill_req_idx_cpu = prefill_mask_cpu.nonzero(as_tuple=True)[0]
        non_spec_req_idx_cpu = torch.cat((decode_req_idx_cpu, prefill_req_idx_cpu))
        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)
        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)

        if num_decodes == 0 and num_prefills == 0:
            # Pure speculative-decode batch: all real tokens are spec tokens.
            spec_token_indx = torch.arange(
                num_spec_decode_tokens,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            non_spec_token_indx = torch.empty(
                0, dtype=torch.int32, device=query_start_loc.device
            )
            spec_state_indices_tensor = block_table_tensor[spec_req_idx, 0]
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
            non_spec_query_start_loc = None
            non_spec_query_start_loc_cpu = None
        else:
            # Mixed batch: build a per-token group key consistent with the
            # request grouping above (spec=0 | decode=1 | prefill=2) and a
            # stable sort, so tokens of each request stay contiguous and in
            # request order. This yields spec tokens first, then the non-spec
            # [decode, prefill] tokens.
            req_group = torch.full(
                (m.num_reqs,),
                2,
                dtype=torch.int64,
                device=query_start_loc.device,
            )
            req_group[spec_req_idx] = 0
            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1
            token_group = torch.repeat_interleave(req_group, query_lens)
            token_perm = torch.argsort(token_group, stable=True)
            spec_token_indx = token_perm[:num_spec_decode_tokens]
            non_spec_token_indx = token_perm[num_spec_decode_tokens:]

            spec_state_indices_tensor = block_table_tensor[spec_req_idx, 0]
            non_spec_state_indices_tensor = block_table_tensor[non_spec_req_idx, 0]
            spec_query_start_loc = torch.zeros(
                num_spec_decodes + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(query_lens[spec_req_idx], dim=0, out=spec_query_start_loc[1:])
            non_spec_query_start_loc = torch.zeros(
                num_decodes + num_prefills + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(
                query_lens[non_spec_req_idx],
                dim=0,
                out=non_spec_query_start_loc[1:],
            )
            non_spec_query_start_loc_cpu = torch.zeros(
                num_decodes + num_prefills + 1, dtype=torch.int32
            )
            torch.cumsum(
                query_lens_cpu[non_spec_req_idx_cpu],
                dim=0,
                out=non_spec_query_start_loc_cpu[1:],
            )

        assert num_accepted_tokens is not None
        # Accepted-token counts must follow the same request order as the
        # speculative state indices.
        num_accepted_tokens = num_accepted_tokens[
            spec_req_idx_cpu.to(num_accepted_tokens.device)
        ]

        # Compute the conv-state slots for the non-spec decode/prefill split,
        # plus the initial-state masks and Triton causal_conv1d metadata.
        if non_spec_state_indices_tensor is None:
            state_indices_tensor = block_table_tensor[:0, 0]
        else:
            state_indices_tensor = non_spec_state_indices_tensor

        # Build the regular decode/prefill state metadata inherited from the
        # generic short-conv metadata contract.
        query_start_loc_p = None
        query_start_loc_d = None
        state_indices_tensor_p = None
        state_indices_tensor_d = None
        if num_decodes > 0 or num_prefills > 0:
            num_computed_tokens = m.compute_num_computed_tokens()
            if non_spec_req_idx_cpu is not None:
                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)
                num_computed_tokens = num_computed_tokens[non_spec_req_idx]

            state_indices_tensor_d = state_indices_tensor[:num_decodes]
            state_indices_tensor_p = state_indices_tensor[
                num_decodes : num_decodes + num_prefills
            ]
            if num_decodes > 0:
                has_initial_states_d = num_computed_tokens[:num_decodes] > 0
                assert non_spec_query_start_loc is not None
                query_start_loc_d = non_spec_query_start_loc[: num_decodes + 1]
            if num_prefills > 0:
                num_computed_tokens_p = num_computed_tokens[
                    num_decodes : num_decodes + num_prefills
                ]
                has_initial_states_p = num_computed_tokens_p > 0
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                query_start_loc_p = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                query_start_loc_p_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                if query_start_loc.device.type != "cpu":
                    nums_dict, batch_ptr, token_chunk_offset_ptr = (
                        compute_causal_conv1d_metadata(
                            query_start_loc_p_cpu,
                            device=query_start_loc.device,
                        )
                    )

        # Prepare persistent tensors for CUDA graph capture and replay.
        # ``m.num_actual_tokens`` is already padded by the model runner.
        # Request-level buffers use ``m.num_reqs`` while token-level buffers
        # use their independently bounded token count.
        batch_size = m.num_reqs
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and spec_sequence_masks is not None
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_tokens
        ):
            assert spec_state_indices_tensor is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:batch_size].copy_(
                spec_sequence_masks[:batch_size], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]

            assert spec_query_start_loc is not None
            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            assert num_accepted_tokens is not None
            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        return PleShortConvAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_reqs=m.num_reqs,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            spec_query_len=self.num_spec + 1,
            max_prefill_query_len=max_prefill_query_len,
            query_start_loc=query_start_loc,
            state_indices_tensor=state_indices_tensor,
            has_initial_states_p=has_initial_states_p,
            has_initial_states_d=has_initial_states_d,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_query_start_loc=spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            query_start_loc_p=query_start_loc_p,
            query_start_loc_d=query_start_loc_d,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            num_computed_tokens_p=num_computed_tokens_p,
            block_idx_last_scheduled_token=None,
            block_idx_first_scheduled_token_p=None,
            block_idx_last_computed_token=None,
            block_idx_last_scheduled_token_prev_step=None,
            seq_lens=m.seq_lens,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> PleShortConvAttentionMetadata:
        """Build metadata for full CUDA graph capture.

        Currently, only decode is supported for full CUDA graphs with
        short-conv.
        """
        m = common_attn_metadata
        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_tokens
        ), (
            "ShortConv only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture size ({self.decode_cudagraph_max_bs}) and "
            f"number of tokens ({m.num_actual_tokens}) <= "
            f"token capture size ({self.decode_cudagraph_max_tokens})."
        )

        if self.use_spec_decode:
            num_accepted_tokens = torch.diff(m.query_start_loc)
            num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
            return self.build(
                0,
                m,
                num_accepted_tokens=num_accepted_tokens,
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            )
        return self.build(0, m)

# -----------------------------------------------------------------------------
# Packed GDN integration
# -----------------------------------------------------------------------------

# Resolve once outside capture. In-place state updates must not fall back after
# partial execution, nor change implementation between capture and replay.
_packed_decode = resolve_op("gdn_packed_decode")


class Qwen38GatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """Keep upstream projection/prefill/metadata, specialize packed decode only."""

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_gdn_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        _packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        return

# -----------------------------------------------------------------------------
# QSA indexer
# -----------------------------------------------------------------------------

_qsa_compress_groups_with_ratio = resolve_op("qsa_compress_groups_with_ratio")
_qsa_select_paged_tokens = resolve_op("qsa_select_paged_tokens")
_qsa_store_cache_rows = resolve_op("qsa_store_cache_rows")


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact 1D/MRoPE composition to QSA heads."""

    rotary_dim = rotary_emb.rotary_dim
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        sections = rotary_emb.mrope_section
        if rotary_emb.mrope_interleaved:
            axis_cos, axis_sin = cos, sin
            channels = torch.arange(cos.shape[-1], device=cos.device)
            is_height = (channels % 3 == 1) & (channels < sections[1] * 3)
            is_width = (channels % 3 == 2) & (channels < sections[2] * 3)
            cos = torch.where(is_height, axis_cos[1], axis_cos[0])
            cos = torch.where(is_width, axis_cos[2], cos)
            sin = torch.where(is_height, axis_sin[1], axis_sin[0])
            sin = torch.where(is_width, axis_sin[2], sin)
        else:
            cos = torch.cat(
                [axis[index] for index, axis in enumerate(cos.split(sections, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [axis[index] for index, axis in enumerate(sin.split(sections, dim=-1))],
                dim=-1,
            )

    # Cross-vendor fallback: public/native tensor composition lets FlagGems or
    # the vendor PyTorch runtime dispatch every primitive appropriately.
    rotated = rotary_emb.apply_rotary_emb.forward_native(
        tensor[..., :rotary_dim],
        cos,
        sin,
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


class QSAIndexer(nn.Module):
    """Replicated Q/K projection plus paged, weight-free QSA selection.

    ``prefix`` must be the checkpoint's indexer prefix, normally
    ``model.layers.N.self_attn.indexer``.  Consequently the trainable names are
    ``index_qk_proj``, ``q_layernorm`` and ``k_layernorm`` under that prefix.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        layer_id: int,
        rotary_emb: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA currently requires BF16")

        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.rotary_emb = rotary_emb
        self.prefix = prefix

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )

        cache_config = vllm_config.cache_config
        cache_prefix = f"{prefix}." if prefix else ""
        self.raw_key_cache = QSAKeyStateCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            cache_rope_positions=vllm_config.model_config.uses_mrope,
            prefix=f"{cache_prefix}raw_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )
        self.compressed_key_cache = QSACompressedKeyCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            prefix=f"{cache_prefix}compressed_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )

    @property
    def output_width(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project replicated Q/K, normalize+rotate Q, and preserve raw K."""

        qk, _ = self.index_qk_proj(hidden_states)
        q_raw, token_k = qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        q = q_raw.reshape(-1, self.index_n_heads, self.index_head_dim)
        flat_q = q.reshape(-1, self.index_head_dim)
        normalized_q = self.q_layernorm(flat_q)
        q = normalized_q.reshape_as(q)
        q = apply_qsa_rope(self.rotary_emb, positions, q)
        return q, token_k.reshape(-1, 1, self.index_head_dim)

    def normalize_compressed_keys(
        self,
        compressed_keys: torch.Tensor,
        first_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize pooled K and apply the first token's exact group position."""

        keys = compressed_keys.reshape(-1, self.index_head_dim)
        normalized_keys = self.k_layernorm(keys)
        keys = normalized_keys.reshape(-1, 1, self.index_head_dim)
        if getattr(self.rotary_emb, "mrope_section", None):
            positions = first_rope_positions.transpose(0, 1)
        else:
            positions = first_rope_positions[:, 0]
        return apply_qsa_rope(self.rotary_emb, positions, keys)

    def _metadata(
        self,
    ) -> tuple[QSAForwardMetadata, QSAForwardMetadata] | None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return None
        raw = cast(QSAForwardMetadata, metadata[self.raw_key_cache.prefix])
        compressed = cast(
            QSAForwardMetadata, metadata[self.compressed_key_cache.prefix]
        )
        if raw.num_actual_tokens != compressed.num_actual_tokens:
            raise RuntimeError("QSA side-cache metadata token counts disagree")
        if raw.logical_positions.device.type == "cpu" and (
            not torch.equal(raw.logical_positions, compressed.logical_positions)
        ):
            raise RuntimeError("QSA side-cache metadata positions disagree")
        return raw, compressed

    def _update_and_compress(
        self,
        token_k: torch.Tensor,
        positions: torch.Tensor,
        raw_metadata: QSAForwardMetadata,
        compressed_metadata: QSAForwardMetadata,
    ) -> None:
        num_tokens = raw_metadata.num_actual_tokens
        raw_key_cache = self.raw_key_cache.key_cache
        rope_position_cache = self.raw_key_cache.rope_position_cache
        _qsa_store_cache_rows(
            raw_key_cache,
            raw_metadata.slot_mapping,
            token_k[:num_tokens],
        )
        if rope_position_cache is not None:
            position_rows = canonical_qsa_rope_positions(positions)[:num_tokens].to(
                device=rope_position_cache.device
            )
            _qsa_store_cache_rows(
                rope_position_cache,
                raw_metadata.slot_mapping,
                position_rows,
            )
        pooled, first_positions = _qsa_compress_groups_with_ratio(
            raw_key_cache,
            raw_metadata.block_table,
            raw_metadata.token_to_req,
            raw_metadata.logical_positions,
            compressed_metadata.slot_mapping,
            self.compress_ratio,
            rope_position_cache,
        )
        normalized = self.normalize_compressed_keys(pooled, first_positions)
        _qsa_store_cache_rows(
            self.compressed_key_cache.kv_cache,
            compressed_metadata.slot_mapping,
            normalized,
        )

    def _select(
        self,
        q: torch.Tensor,
        metadata: QSAForwardMetadata,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        return _qsa_select_paged_tokens(
            q,
            self.compressed_key_cache.kv_cache,
            metadata.block_table,
            metadata.token_to_req,
            metadata.logical_positions,
            metadata.seq_lens,
            self.token_topk,
            self.compress_ratio,
            out,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-width request-relative token indices padded with ``-1``."""

        metadata = self._metadata()
        if metadata is None:
            result = torch.full(
                (hidden_states.shape[0], self.output_width),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            if out is not None:
                out.copy_(result)
                return out
            return result
        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        q, token_k = self.project_qk(
            hidden_states[:num_tokens], positions[..., :num_tokens]
        )
        self._update_and_compress(
            token_k,
            positions[..., :num_tokens],
            raw_metadata,
            compressed_metadata,
        )
        return self._select(q, compressed_metadata, out)



# -----------------------------------------------------------------------------
# QSA attention
# -----------------------------------------------------------------------------

_qsa_sparse_paged_attention = resolve_op("qsa_sparse_paged_attention")
_qsa_store_cache_rows = resolve_op("qsa_store_cache_rows")


def _unpack_qsa_kv_cache(
    kv_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return paged K/V views in ``(blocks, tokens, heads, dim)`` order.

    vLLM 0.24 stores FlashAttention caches as logical
    ``(blocks, 2, tokens, heads, dim)`` tensors.  This is the only layout
    supported by the QSA store path: unbinding the K/V dimension preserves
    views into the allocator-owned backing storage, including the non-unit
    stride between pages.

    A newer packed ``(blocks, heads, tokens, 2*dim)`` ABI cannot be safely
    adapted here.  Transposing that layout makes the subsequent head/dim
    flatten non-viewable, so a cache store would update a temporary copy
    instead of the allocator backing.  Reject it until the store kernel and
    allocator contract support that layout end to end.
    """
    if kv_cache.ndim == 5:
        if kv_cache.shape[1] != 2 or kv_cache.shape[-1] != head_size:
            raise ValueError(
                "invalid legacy QSA KV cache shape: expected "
                f"(blocks, 2, tokens, heads, {head_size}), got "
                f"{tuple(kv_cache.shape)}"
            )
        key_cache, value_cache = kv_cache.unbind(1)
    elif kv_cache.ndim == 4:
        raise ValueError(
            "QSA does not support the packed 4-D KV cache layout; expected "
            "the allocator-backed vLLM 0.24 5-D layout "
            f"(blocks, 2, tokens, heads, {head_size}), got "
            f"{tuple(kv_cache.shape)}"
        )
    else:
        raise ValueError(
            "invalid QSA KV cache rank: expected the vLLM legacy 5-D or "
            f"packed 4-D layout, got {kv_cache.ndim}-D"
        )
    return (
        canonicalize_singleton_dim_strides(key_cache),
        canonicalize_singleton_dim_strides(value_cache),
    )


class Qwen3_8FlashNextQSAAttentionBackend(AttentionBackend):
    """Main K/V cache owner for the Triton QSA transaction.

    QSA performs cache update and sparse attention in its model custom op, so
    it needs vLLM only for cache allocation and device-side metadata building.
    Owning those interfaces directly avoids coupling the model to a vendor's
    FlashAttention extension or cache-update ABI.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_QSA_FLAGTREE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "QSA executes out-of-band through its model-owned Triton transaction"
        )

    @staticmethod
    def get_builder_cls() -> type[QSAMetadataBuilder]:
        return QSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        # The cache is num-blocks-first and the QSA transaction passes the
        # tensor's physical block stride to every Triton kernel.  Opt in to
        # vLLM 0.24's padded-page view for hybrid cache page-size alignment.
        return True

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen3_8FlashNextQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = False,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen3.8-Flash-Next QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support KV quantization"
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires causal decoder attention"
            )

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support dual-chunk RoPE"
            )
        # Qwen3.8-Flash-Next full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Use the framework's normalization/rotary layers on every vendor.
        # A fused vendor implementation belongs in dispatch, not this model.
        self.use_fused_qk_norm_rope_gate = False

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen3_8FlashNextQSAAttentionBackend
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(QSAForwardMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        key_cache, value_cache = _unpack_qsa_kv_cache(self.kv_cache, self.head_dim)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")

        slot_mapping = main_metadata.slot_mapping[:num_tokens]
        # Portable cache update. ``view`` (rather than reshape) makes a
        # non-viewable allocator layout fail instead of silently copying
        # rows away from the backing cache.
        flat_width = self.num_kv_heads * self.head_dim
        flat_key_cache = key_cache.view(
            key_cache.shape[0], key_cache.shape[1], 1, flat_width
        )
        flat_value_cache = value_cache.view(
            value_cache.shape[0], value_cache.shape[1], 1, flat_width
        )
        _qsa_store_cache_rows(
            flat_key_cache,
            slot_mapping,
            key[:num_tokens].reshape(num_tokens, 1, flat_width),
        )
        _qsa_store_cache_rows(
            flat_value_cache,
            slot_mapping,
            value[:num_tokens].reshape(num_tokens, 1, flat_width),
        )

        output.zero_()
        if num_tokens:
            _qsa_sparse_paged_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                self.topk_indices_buffer[:num_tokens],
                main_metadata.block_table,
                side_metadata.token_to_req[:num_tokens],
                self.scaling,
                output[:num_tokens],
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen3_8_flash_next_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen3_8FlashNextQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen3.8-Flash-Next QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen3_8_flash_next_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, layer_name


direct_register_custom_op(
    op_name="qwen3_8_flash_next_qsa_with_output",
    op_func=qwen3_8_flash_next_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen3_8_flash_next_qsa_with_output_fake,
)



# -----------------------------------------------------------------------------
# PLE layers
# -----------------------------------------------------------------------------

ple_state_gather = resolve_op("ple_state_gather")
ple_state_scatter_ = resolve_op("ple_state_scatter_")

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class Qwen3_8FlashNextPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class Qwen3_8FlashNextNGramEmbedding(nn.Module):
    def __init__(
        self,
        config: Qwen3_8FlashNextTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        seed = int(getattr(config, "seed", 1234))
        base_seed = seed + _PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(self.ngram_size):
            value = base_seed + _SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(self.ngram_heads):
            global_head = ple_dense_layer_id * self.ngram_heads + local_head
            size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((offset + divisor - 1) // divisor) * divisor
        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
        )
        self.register_buffer(
            "positions_buffer",
            torch.arange(max_total_tokens, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "padded_buffer",
            torch.full(
                (max_num_reqs, max_total_tokens),
                self.eos_token_id,
                dtype=torch.int64,
            ),
            persistent=False,
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1).long()
        query_start_loc = query_start_loc.long()
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_tokens > self.positions_buffer.numel():
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.positions_buffer.numel()}"
            )
        if num_reqs > self.padded_buffer.shape[0]:
            raise ValueError(
                f"PLE received {num_reqs} requests, but its workspace supports "
                f"at most {self.padded_buffer.shape[0]}"
            )

        positions = self.positions_buffer[:num_tokens]
        packed = self.padded_buffer[:num_reqs]
        packed.fill_(self.eos_token_id)
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        ngram_ids = torch.cat(id_blocks, dim=-1)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                copy_ple_embedding_shard_(
                    embedding.weight.data,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                    tp_start=embedding.shard_indices.org_vocab_start_index,
                    tp_end=embedding.shard_indices.org_vocab_end_index,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


class Qwen3_8FlashNextPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen3_8FlashNextTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding: nn.Module = Qwen3_8FlashNextNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.scheduler_config.max_num_seqs,
            f"{prefix}.ple_embedding",
        )
        self.key_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen3_8FlashNextPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen3_8FlashNextPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen3_8FlashNextPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        # vLLM 0.24 has no explicit ``num_spec`` parameter; folding it into
        # the effective kernel yields the same state length:
        # (kernel - 1) + num_spec.
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + self.num_spec_tokens + 1,
        )

    def _apply_norm(
        self, norm: Qwen3_8FlashNextPLEGroupedNorm, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        return norm(hidden_states.flatten(-2)).reshape(shape)

    def _short_conv_fallback(self, inputs: torch.Tensor) -> torch.Tensor:
        # Profiling / CUDA graph capture only; conv state is not updated.
        inputs_t = inputs.transpose(0, 1).unsqueeze(0)
        output = self.conv1d(inputs_t)[..., : inputs_t.size(-1)]
        return F.silu(output).squeeze(0).transpose(0, 1)

    def _short_conv_dilated_decode_batched(
        self,
        x_d: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_d: torch.Tensor,
        has_initial_states_d: torch.Tensor | None,
    ) -> torch.Tensor:
        state_indices = state_indices_tensor_d.to(
            device=conv_state.device, dtype=torch.int64
        )
        # These are block-table state indices, so FULL cudagraph padding uses
        # NULL_BLOCK_ID (0). PAD_SLOT_ID (-1) belongs to flattened slot mappings.
        # Remap null rows to slot 0 for a safe gather, then zero their output and
        # skip write-back.
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        if has_initial_states_d is None:
            has_initial_state = valid_state
        else:
            if has_initial_states_d.numel() < state_indices_tensor_d.numel():
                raise ValueError(
                    "has_initial_states_d size mismatch: "
                    f"got {has_initial_states_d.numel()}, "
                    f"need >= {state_indices_tensor_d.numel()}."
                )
            has_initial_state = has_initial_states_d[
                : state_indices_tensor_d.numel()
            ].to(device=conv_state.device, dtype=torch.bool)
            has_initial_state = has_initial_state & valid_state

        cached_state = ple_state_gather(
            conv_state, state_indices, indices_are_safe=True
        )
        state = cached_state[..., : self.conv_state_len].to(x_d.dtype)
        if self.conv_state_len > 0:
            initial_state = torch.where(
                has_initial_state.view(-1, 1, 1),
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, x_d.unsqueeze(-1)), dim=-1)
        else:
            history = x_d.unsqueeze(-1)

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        ).squeeze(-1)
        output = F.silu(conv_output)
        output = output * valid_state.view(-1, 1).to(output.dtype)

        if self.conv_state_len > 0:
            next_state = history[..., -self.conv_state_len :]
            # Padded rows are remapped to the reserved null slot. Preserve its
            # existing value while writing the new states for valid rows.
            existing_base_state = cached_state[..., : self.conv_state_len]
            safe_next_state = torch.where(
                valid_state.view(-1, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            cached_state[..., : self.conv_state_len] = safe_next_state
            ple_state_scatter_(
                conv_state,
                state_indices,
                cached_state,
                write_mask=valid_state,
                indices_are_safe=True,
            )

        return output

    def _short_conv_dilated_prefill_batched(
        self,
        x_p: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_p: torch.Tensor,
        num_prefills: int,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> torch.Tensor:
        # ``non_spec_query_start_loc`` covers the non-spec (decode + prefill)
        # requests and equals ``query_start_loc`` when spec-decode is inactive.
        non_spec_query_start_loc = metadata.non_spec_query_start_loc
        if non_spec_query_start_loc is None:
            raise ValueError("query_start_loc is required for prefill short-conv")
        query_start_loc_p = (
            non_spec_query_start_loc[-num_prefills - 1 :] - num_decode_tokens
        )
        # The metadata builder guarantees that the prefill query offsets start
        # at 0 and end at num_prefill_tokens. Avoid reading those values here,
        # since doing so would force a device-to-host synchronization.
        has_initial_states_p = metadata.has_initial_states_p
        if has_initial_states_p is None:
            raise ValueError("has_initial_states_p is required for prefill short-conv")

        output = torch.empty_like(x_p)
        q_starts = query_start_loc_p.to(torch.int64)
        if state_indices_tensor_p.numel() < num_prefills:
            raise ValueError(
                "state_indices_tensor_p size mismatch: "
                f"got {state_indices_tensor_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if has_initial_states_p.numel() < num_prefills:
            raise ValueError(
                "has_initial_states_p size mismatch: "
                f"got {has_initial_states_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if num_prefills == 0 or x_p.numel() == 0:
            return output
        lengths = q_starts[1:] - q_starts[:-1]
        # Use the CPU-computed packing width from the metadata builder instead
        # of synchronizing on lengths.max().
        max_len = metadata.max_prefill_query_len
        if max_len <= 0:
            return output

        hidden_size = x_p.shape[1]
        positions = torch.arange(
            num_prefill_tokens, device=x_p.device, dtype=torch.int64
        )
        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
        col_indices = positions - q_starts[req_indices]

        packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
        packed_tokens[req_indices, col_indices] = x_p
        packed_tokens = packed_tokens.transpose(1, 2).contiguous()

        state_indices = state_indices_tensor_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.int64
        )
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        has_initial = has_initial_states_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.bool
        )
        if self.conv_state_len > 0:
            if conv_state.shape[0] == 0:
                state = conv_state.new_zeros(
                    (num_prefills, hidden_size, self.conv_state_len),
                    dtype=x_p.dtype,
                )
            else:
                state = ple_state_gather(
                    conv_state, state_indices, indices_are_safe=True
                )[..., : self.conv_state_len].to(x_p.dtype)
            use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
            initial_state = torch.where(
                use_initial_mask,
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, packed_tokens), dim=-1)
        else:
            history = packed_tokens

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
        valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
        valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
            num_prefills, 1
        )
        conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
        output.copy_(conv_output[req_indices, col_indices])

        if self.conv_state_len > 0 and conv_state.shape[0] > 0:
            state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
                num_prefills, 1, 1
            )
            state_offsets = torch.arange(
                self.conv_state_len, device=history.device, dtype=torch.int64
            ).view(1, 1, self.conv_state_len)
            next_state = history.gather(
                dim=2,
                index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
            )
            # Write back without a host synchronization. Valid, non-empty rows
            # receive their new state; padding and zero-length rows keep the
            # current cache value.
            existing_state = ple_state_gather(
                conv_state, state_indices, indices_are_safe=True
            )
            existing_base_state = existing_state[..., : self.conv_state_len]
            update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
            safe_next_state = torch.where(
                update_mask.view(num_prefills, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            existing_state[..., : self.conv_state_len] = safe_next_state
            ple_state_scatter_(
                conv_state,
                state_indices,
                existing_state,
                write_mask=update_mask,
                indices_are_safe=True,
            )
        return output

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        x = inputs[: metadata.num_actual_tokens]

        if has_spec:
            raise NotImplementedError(
                "Qwen3.8 PLE speculative decoding is not supported"
            )
        x_non_spec = x

        # Run regular decode and prefill requests.
        conv_out_non_spec = None
        state_indices_tensor = metadata.state_indices_tensor
        if x_non_spec is not None:
            assert state_indices_tensor is not None
            if has_prefill:
                state_indices_tensor_d, state_indices_tensor_p = torch.split(
                    state_indices_tensor,
                    [num_decodes, num_prefills],
                    dim=0,
                )
                x_d, x_p = torch.split(
                    x_non_spec,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
                non_spec_parts: list[torch.Tensor] = []
                if has_decode:
                    non_spec_parts.append(
                        self._short_conv_dilated_decode_batched(
                            x_d=x_d,
                            conv_state=conv_state,
                            conv_weights=conv_weights,
                            state_indices_tensor_d=state_indices_tensor_d,
                            has_initial_states_d=metadata.has_initial_states_d,
                        )
                    )
                non_spec_parts.append(
                    self._short_conv_dilated_prefill_batched(
                        x_p=x_p,
                        metadata=metadata,
                        conv_state=conv_state,
                        conv_weights=conv_weights,
                        state_indices_tensor_p=state_indices_tensor_p,
                        num_prefills=num_prefills,
                        num_decode_tokens=num_decode_tokens,
                        num_prefill_tokens=num_prefill_tokens,
                    )
                )
                conv_out_non_spec = torch.vstack(non_spec_parts)
            else:
                conv_out_non_spec = self._short_conv_dilated_decode_batched(
                    x_d=x_non_spec,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices_tensor_d=state_indices_tensor[: x_non_spec.size(0)],
                    has_initial_states_d=metadata.has_initial_states_d,
                )

        if conv_out_non_spec is None:
            return x
        return conv_out_non_spec

    def _short_conv(self, inputs: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return self._short_conv_fallback(inputs)

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            raise RuntimeError(
                f"Missing short-conv metadata for layer '{self.prefix}'. "
                "This would bypass conv-state updates and is not allowed."
            )
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        return self._short_conv_dilated_dispatch(
            inputs,
            layer_attn_metadata,
            conv_state,
            conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated_value = gate * value.unsqueeze(-2)
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        conv_output = torch.zeros_like(normalized)
        torch.ops.vllm.qwen3_8_flash_next_ple_short_conv(
            normalized,
            conv_output,
            self.prefix,
        )
        return gated_value.flatten(-2) + conv_output


def qwen3_8_flash_next_ple_short_conv(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    result = layer._short_conv(inputs)
    output[: result.shape[0]].copy_(result)


def qwen3_8_flash_next_ple_short_conv_fake(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen3_8_flash_next_ple_short_conv",
    op_func=qwen3_8_flash_next_ple_short_conv,
    mutates_args=["output"],
    fake_impl=qwen3_8_flash_next_ple_short_conv_fake,
)



# -----------------------------------------------------------------------------
# Model composition and weight loading
# -----------------------------------------------------------------------------

QwenGatedDeltaNetAttention = Qwen38GatedDeltaNetAttention

_HAS_NATIVE_STACKED_WEIGHTS_MAPPER = hasattr(WeightsMapper(), "orig_to_new_stacked")


def _get_qwen35_hf_to_vllm_mapper() -> WeightsMapper:
    """Return the native packed-weight mapper when the vLLM API provides it.

    The Qwen3.8 reference was authored against the post-0.24 mapper API.  The
    FlagOS Day0 base image uses vLLM 0.24, where Qwen3Next/Qwen3.5 still packs
    weights in handwritten loaders and therefore exposes no class mapper.
    """

    mapper = getattr(Qwen3_5Model, "hf_to_vllm_mapper", None)
    if mapper is None:
        mapper = getattr(Qwen3NextModel, "hf_to_vllm_mapper", None)
    return mapper if mapper is not None else WeightsMapper()


_QWEN35_HF_TO_VLLM_MAPPER = _get_qwen35_hf_to_vllm_mapper()


_VLLM024_STACKED_WEIGHT_MAPPINGS = (
    (".q_proj", ".qkv_proj", "q"),
    (".k_proj", ".qkv_proj", "k"),
    (".v_proj", ".qkv_proj", "v"),
    (".mlp.gate_proj", ".mlp.gate_up_proj", 0),
    (".mlp.up_proj", ".mlp.gate_up_proj", 1),
    (".shared_expert.gate_proj", ".shared_expert.gate_up_proj", 0),
    (".shared_expert.up_proj", ".shared_expert.gate_up_proj", 1),
    (".in_proj_qkv", ".in_proj_qkvz", (0, 1, 2)),
    (".in_proj_z", ".in_proj_qkvz", 3),
    (".in_proj_b", ".in_proj_ba", 0),
    (".in_proj_a", ".in_proj_ba", 1),
)


def _map_vllm024_stacked_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Attach packed-linear shard metadata missing from vLLM 0.24's mapper."""

    for name, weight in weights:
        for old, new, shard_id in _VLLM024_STACKED_WEIGHT_MAPPINGS:
            if old in name:
                name = name.replace(old, new, 1)
                weight.shard_id = shard_id
                break
        yield name, weight


def _get_vllm024_expert_mappings(
    model: nn.Module,
    num_experts: int,
    num_redundant_experts: int,
) -> (
    tuple[
        list[tuple[str, str, int, str]],
        list[tuple[str, str, int, str]],
    ]
    | None
):
    """Probe for the vLLM 0.24 fused-MoE mapping ABI.

    vLLM 0.24 exposes ``fused_moe_make_expert_params_mapping`` but does not
    teach ``AutoWeightsLoader`` how to turn a Hugging Face fused 3-D
    ``experts.{gate_up_proj,down_proj}`` tensor into the
    ``experts.routed_experts.{w13,w2}_weight`` parameters.  Newer vLLM builds
    carry that knowledge through the native mapper and never enter this
    compatibility path.  Keeping the probe structural avoids coupling the
    plugin to a version string.

    The first mapping handles ordinary one-expert-at-a-time checkpoints.  The
    second mapping deliberately uses ``gate_up_proj`` for both the gate and up
    names and is reduced to generic names for fused 3-D checkpoints, matching
    the loader in vLLM's Qwen3.5 implementation.
    """

    helper = fused_moe_make_expert_params_mapping
    if not callable(helper):
        return None

    try:
        regular_mapping = list(
            helper(
                model,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
                num_redundant_experts=num_redundant_experts,
            )
        )
        fused_base_mapping = list(
            helper(
                model,
                ckpt_gate_proj_name="gate_up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="gate_up_proj",
                num_experts=1,
            )
        )
    except (AttributeError, TypeError):
        # A future helper with a different ABI must stay on the native path
        # instead of being partially handled by this legacy adapter.
        return None

    fused_mapping: list[tuple[str, str, int, str]] = []
    for param_name, checkpoint_name, _, shard_id in fused_base_mapping:
        parts = checkpoint_name.split(".")
        if len(parts) < 3:
            return None
        fused_mapping.append(
            (
                f"{param_name}weight",
                f"{parts[0]}.{parts[2]}",
                0,
                shard_id,
            )
        )
    if not regular_mapping or not fused_mapping:
        return None
    return regular_mapping, fused_mapping


def _call_vllm024_expert_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    param_name: str,
    shard_id: str,
    expert_id: int,
) -> bool:
    """Call the legacy FusedMoE weight loader with capability probing."""

    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None:
        raise TypeError(f"Fused expert parameter {param_name!r} has no weight_loader")
    try:
        loaded = weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
    except TypeError as exc:
        # Keep support for the older callable form without weakening errors
        # from inside a real loader implementation.
        if "return_success" not in str(exc):
            raise
        weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )
        return True
    return True if loaded is None else bool(loaded)


def _load_vllm024_fused_expert_weight(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    fused_mapping: list[tuple[str, str, int, str]],
    num_experts: int,
) -> tuple[bool, set[str]]:
    """Load one fused HF expert tensor using the vLLM 0.24 parameter ABI."""

    if "mlp.experts.gate_up_proj" not in name and "mlp.experts.down_proj" not in name:
        return False, set()
    if loaded_weight.ndim != 3:
        raise ValueError(
            f"Expected a fused 3-D MoE tensor for {name!r}, got "
            f"shape={tuple(loaded_weight.shape)}"
        )

    loaded_params: set[str] = set()
    for param_name, weight_name, _, shard_id in fused_mapping:
        if weight_name not in name:
            continue
        name_mapped = name.replace(weight_name, param_name, 1)
        param = params_dict.get(name_mapped)
        # A pipeline-parallel rank can legitimately not own this layer.
        if param is None:
            return True, loaded_params

        if "gate_up_proj" in name:
            split_weights = loaded_weight.chunk(2, dim=-2)
            shard_weights = (
                ("w1", split_weights[0]),
                ("w3", split_weights[1]),
            )
        else:
            shard_weights = ((shard_id, loaded_weight),)

        loaded_local_expert = False
        for actual_shard_id, shard_weight in shard_weights:
            for expert_id in range(num_experts):
                if _call_vllm024_expert_weight_loader(
                    param,
                    shard_weight[expert_id],
                    name_mapped,
                    actual_shard_id,
                    expert_id,
                ):
                    loaded_local_expert = True
        if loaded_local_expert:
            loaded_params.add(name_mapped)
        return True, loaded_params
    return True, loaded_params


def _load_vllm024_single_expert_weight(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    expert_mapping: list[tuple[str, str, int, str]],
) -> tuple[bool, set[str]]:
    """Load a non-fused expert tensor when the old ABI supplies one expert."""

    loaded_params: set[str] = set()
    for param_name, weight_name, expert_id, shard_id in expert_mapping:
        if weight_name not in name:
            continue
        name_mapped = name.replace(weight_name, param_name, 1)
        param = params_dict.get(name_mapped)
        if param is None:
            return True, loaded_params
        if _call_vllm024_expert_weight_loader(
            param,
            loaded_weight,
            name_mapped,
            shard_id,
            expert_id,
        ):
            loaded_params.add(name_mapped)
        return True, loaded_params
    return False, loaded_params


class _VLLM024StackedAutoWeightsLoader(AutoWeightsLoader):
    """Teach the vLLM 0.24 auto-loader to forward packed shard identifiers."""

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            shard_id = getattr(weight_data, "shard_id", None)
            if shard_id is None:
                yield from super()._load_param(
                    base_prefix, param, [(weight_name, weight_data)]
                )
                continue

            weight_qualname = self._get_qualname(base_prefix, weight_name)
            if self._can_skip(weight_qualname):
                continue
            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    continue
                raise ValueError(
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into a single parameter {base_prefix!r}"
                )
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                raise TypeError(
                    f"Packed parameter {weight_qualname!r} has no weight_loader"
                )
            weight_loader(param, weight_data, shard_id)
            yield weight_qualname


def without_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Return ``None`` for weights excluded from Qwen3.8-Flash-Next ModelOpt-FP4."""

    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


def _remap_qsa_cache_scale_name(
    name: str,
    qsa_layer_ids: frozenset[int],
) -> str:
    """Map serialized main-cache scales onto the merged QSA owner.

    Regular attention keeps cache scales below its ``attn`` child. QSA owns
    that cache directly, so only QSA layers need the final path component
    moved to the owner's persistent ``_k_scale``/``_v_scale`` buffers.
    """

    scale_suffixes = {
        "k_proj.k_scale": "_k_scale",
        "k_proj.output_scale": "_k_scale",
        "attn.k_scale": "_k_scale",
        "attn._k_scale": "_k_scale",
        "k_scale": "_k_scale",
        "_k_scale": "_k_scale",
        "v_proj.v_scale": "_v_scale",
        "v_proj.output_scale": "_v_scale",
        "attn.v_scale": "_v_scale",
        "attn._v_scale": "_v_scale",
        "v_scale": "_v_scale",
        "_v_scale": "_v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start > 0 and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        mapped_suffix = scale_suffixes.get(suffix)
        if mapped_suffix is not None:
            return f"{name[: marker_start + len(marker)]}{mapped_suffix}"
    return name


_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]


class Qwen3_8FlashNextSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3Next MoE with Qwen3.8-Flash-Next HC validation."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # The current FusedMoEFactory owns its final tensor-parallel
        # reduction. Do not reduce the result a second time in the HC caller.
        self.requires_tp_all_reduce = False


class Qwen3_8FlashNextDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        force_disable_ple: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next HC does not support sequence-parallel MoE"
            )
        self.ple: Qwen3_8FlashNextPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids and not force_disable_ple:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen3_8FlashNextPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            gdn_kwargs = {
                "vllm_config": vllm_config,
                "prefix": f"{prefix}.linear_attn",
                "gqa_interleaved_layout": False,
            }
            if (
                "reduce_results"
                in inspect.signature(QwenGatedDeltaNetAttention.__init__).parameters
            ):
                gdn_kwargs["reduce_results"] = False
            self.linear_attn = QwenGatedDeltaNetAttention(config, **gdn_kwargs)
            # vLLM 0.24 predates the constructor option. HC owns the single TP
            # reduction after each branch, so disable the GDN output linear's
            # internal reduction explicitly to avoid a double all-reduce.
            if (
                "reduce_results"
                not in inspect.signature(QwenGatedDeltaNetAttention.__init__).parameters
            ):
                self.linear_attn.out_proj.reduce_results = False
            self._gdn_requires_output_buffer = (
                "output"
                in inspect.signature(QwenGatedDeltaNetAttention.forward).parameters
            )
        elif layer_type == "full_attention":
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen3_8FlashNextQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe_layer = self.layer_idx not in mlp_only_layers and (
            num_experts > 0 and absolute_layer_id % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen3_8FlashNextSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
            )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidualSimple(
            hc_config,
            layer_idx=self.layer_idx,
            role="attn",
        )
        self.mlp_hyper_connection = GatedResidualSimple(
            hc_config,
            layer_idx=self.layer_idx,
            role="mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del kwargs
        if residual is not None:
            raise ValueError("HC layers do not use a separate residual tensor")
        if self.ple is not None:
            if input_ids is None:
                raise ValueError("PLE requires input_ids")
            if query_start_loc is None:
                raise ValueError("ngram PLE requires query_start_loc")
            if ngram_context is None:
                raise ValueError("ngram PLE requires ngram_context")
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

        mixed, hc_residual = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            if self._gdn_requires_output_buffer:
                # vLLM 0.24's pluggable GDN ABI writes into a caller-provided
                # buffer; newer vLLM returns the projected tensor directly.
                self_attention_output = torch.empty_like(mixed)
                self.linear_attn(mixed, self_attention_output)
            else:
                self_attention_output = self.linear_attn(hidden_states=mixed)
        elif self.layer_type == "full_attention":
            self_attention_output = self.self_attn(
                hidden_states=mixed,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output
        if get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = self.attn_hyper_connection.combine(hidden_states, hc_residual)

        mixed, hc_residual = self.mlp_hyper_connection.mix(hidden_states)
        hidden_states = self.mlp(mixed)
        if get_tensor_model_parallel_world_size() > 1 and getattr(
            self.mlp, "requires_tp_all_reduce", True
        ):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = self.mlp_hyper_connection.combine(hidden_states, hc_residual)
        return hidden_states, None


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "query_start_loc": 0,
        "ngram_context": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3_8FlashNextModel(nn.Module):
    hf_to_vllm_mapper = _QWEN35_HF_TO_VLLM_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self._qsa_layer_ids = frozenset(
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_type == "full_attention"
            and getattr(config, "indexer_n_heads", None) is not None
        )
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen3_8FlashNextDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen3_8FlashNextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size
        )

        self.hyper_connection_mixer: GatedResidualSimple | None
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidualSimple(
                hc_config, use_combine=False, role="final"
            )
        else:
            self.hyper_connection_mixer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None

        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        assert self.hyper_connection_mixer is not None
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (
                _remap_qsa_cache_scale_name(name, self._qsa_layer_ids),
                weight,
            )
            for name, weight in weights
        )
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        # Non-persistent PLE state rebuilt in __init__; skip any ckpt
        # column for them.
        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]
        mapper = self.hf_to_vllm_mapper
        loader_cls = AutoWeightsLoader
        legacy_loaded: set[str] = set()
        if not _HAS_NATIVE_STACKED_WEIGHTS_MAPPER:
            expert_mappings = _get_vllm024_expert_mappings(
                self,
                num_experts=getattr(self.config, "num_experts", 0) or 0,
                num_redundant_experts=self.num_redundant_experts,
            )
            if expert_mappings is not None:
                regular_expert_mapping, fused_expert_mapping = expert_mappings
                params_dict = dict(self.named_parameters())
                legacy_input = weights

                def _legacy_weights() -> Iterable[tuple[str, torch.Tensor]]:
                    for name, loaded_weight in legacy_input:
                        handled, loaded = _load_vllm024_fused_expert_weight(
                            name,
                            loaded_weight,
                            params_dict,
                            fused_expert_mapping,
                            getattr(self.config, "num_experts", 0) or 0,
                        )
                        if handled:
                            legacy_loaded.update(loaded)
                            continue
                        handled, loaded = _load_vllm024_single_expert_weight(
                            name,
                            loaded_weight,
                            params_dict,
                            regular_expert_mapping,
                        )
                        if handled:
                            legacy_loaded.update(loaded)
                            continue
                        yield from _map_vllm024_stacked_weights(
                            ((name, loaded_weight),)
                        )

                weights = _legacy_weights()
            else:
                weights = _map_vllm024_stacked_weights(weights)
            mapper = None
            loader_cls = _VLLM024StackedAutoWeightsLoader
        loader = loader_cls(
            self,
            skip_substrs=skip_substrs,
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(
            weights,
            mapper=mapper,
        )
        loaded.update(legacy_loaded)
        return loaded


class Qwen3_8FlashNextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.8-Flash-Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.model = Qwen3_8FlashNextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        # Set MoE hyperparameters
        if (getattr(config, "num_experts", 0) or 0) > 0:
            QwenNextMixtureOfExperts.set_moe_parameters(self)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Forward kwargs unchanged so the runner's _maybe_add_ngram_kwargs
        # path (query_start_loc / ngram_context) reaches Qwen3_8FlashNextModel.
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_ple_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_text_config
        conv_kernel_size = hf_config.ple_conv_kernel_size
        short_conv_dilation = hf_config.ngram_size
        conv_state_len = (conv_kernel_size - 1) * short_conv_dilation
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        hc_count = hf_config.hc_count
        hc_hidden_size = hf_config.hidden_size * hc_count
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=hc_hidden_size,
            conv_kernel=conv_state_len + num_spec + 1,
        )

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return cls.get_gdn_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs_by_type = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        missing_types = mamba_types - copy_funcs_by_type.keys()
        assert not missing_types, f"missing state copy funcs for {missing_types}"
        return {
            mamba_type: copy_funcs_by_type[mamba_type] for mamba_type in mamba_types
        }

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        """Return all MambaSpecs for this model (GDN layers + PLE layer).

        The PLE layer uses a separate short_conv MambaSpec whose page_size_bytes
        may exceed the GDN spec; callers should take the maximum.
        """
        return (
            MambaSpec(
                shapes=cls.get_gdn_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_gdn_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
            MambaSpec(
                shapes=cls.get_ple_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_ple_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        del mm_features
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["mtp."],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_8FlashNextMixtureOfExperts(MixtureOfExperts):
    """Expose Qwen3.8-Flash-Next routed experts through vLLM's EPLB protocol."""

    language_model: Qwen3_8FlashNextForCausalLM

    def _set_moe_parameters(self) -> None:
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        language_model = getattr(self, "model", None)
        if language_model is None:
            language_model = self.language_model.model
        for layer in language_model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class Qwen3_8FlashNextProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Qwen3_8FlashNextConfig:
        return self.ctx.get_hf_config(Qwen3_8FlashNextConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_8FlashNextProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_8FlashNextForConditionalGeneration(
    Qwen3_5ForConditionalGeneration,
    HasInnerState,
    Qwen3_8FlashNextMixtureOfExperts,
):
    """Qwen3-VL vision tower backed by the Qwen3.8-Flash-Next language model."""

    requires_raw_input_tokens = True

    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping

    def _init_video_pruning(self, multimodal_config) -> None:
        if not hasattr(multimodal_config, "get_video_pruning_spec"):
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            return
        pruning_spec = multimodal_config.get_video_pruning_spec()
        if pruning_spec is None:
            self.video_pruning_method = None
            self.video_pruning_rate = multimodal_config.video_pruning_rate
        else:
            self.video_pruning_method, self.video_pruning_rate = pruning_spec
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen3_8FlashNextConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "Qwen3_8FlashNextForConditionalGeneration requires multimodal_config"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self._init_video_pruning(multimodal_config)
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=config.text_config.rms_norm_eps,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and not isinstance(self.visual, StageMissingLayer)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_8FlashNextForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )
        self._set_moe_parameters()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError(
                "Qwen3.8-Flash-Next language_model_only does not accept "
                "multimodal embeddings"
            )

        is_multimodal = _require_is_multimodal(is_multimodal)
        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = (
                self._compute_deepstack_embeds(
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["visual."] if self.language_model_only else None,
            skip_substrs=["mtp."],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_funcs(mamba_types)

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_specs_from_config(vllm_config)



# Checkpoint architecture aliases use the same concrete implementation.
Qwen4ExpForCausalLM = Qwen3_8FlashNextForCausalLM
Qwen4ExpForConditionalGeneration = Qwen3_8FlashNextForConditionalGeneration

__all__ = [
    "GatedResidualSimple",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen3_8FlashNextConfig",
    "Qwen3_8FlashNextTextConfig",
    "Qwen3_8FlashNextVisionConfig",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
]
