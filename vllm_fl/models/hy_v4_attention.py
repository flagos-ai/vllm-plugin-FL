# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA attention and lightning indexer for HY V4 (NVIDIA).

NOTE: The reference implementation also offers a fused MLA preprocessing
wrapper and an HPC gated-MLA GEMM. Neither is available here, so this port
keeps the eager MLA path with an FP8 indexer cache. Fused variants remain a
follow-up for when those kernels become available.

The per-head learnable sink is supported through `.flashmla_sparse`, which
subclasses the platform's sparse MLA backend to forward ``attn_sink``.
"""

from typing import cast

import regex as re
import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config, PretrainedConfig

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.selector import get_attn_backend

from vllm_fl.patches.hy_v4_runtime import HY4RuntimePlan

logger = init_logger(__name__)

_SPARSE_LAYER_TYPES = ("sparse_attention", "sparse", "deepseek_sparse_attention")
_WEIGHT_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def compute_skip_topk_layers(config: PretrainedConfig) -> set[int]:
    """Return the backbone layers that reuse a previous layer's top-k indices.

    A "shared" indexer layer performs sparse attention with the indices computed
    by the closest preceding "full" indexer layer, so it does not build its own
    indexer and its checkpoint indexer weights must be skipped.

    Args:
        config: The model config.

    Returns:
        The set of layer indices that share another layer's top-k indices.

    Raises:
        ValueError: If ``indexer_types`` has the wrong length or an unknown
            entry, or if ``index_topk_freq`` is not a positive integer.
    """
    if not hasattr(config, "index_topk"):
        return set()

    num_hidden_layers = config.num_hidden_layers
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        if len(indexer_types) != num_hidden_layers:
            raise ValueError(
                "indexer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(indexer_types)}."
            )
        invalid_types = sorted(set(indexer_types) - {"full", "shared"})
        if invalid_types:
            raise ValueError(
                f"indexer_types only supports 'full' and 'shared', got {invalid_types}."
            )
        skip_layers = {i for i, kind in enumerate(indexer_types) if kind == "shared"}
        _validate_topk_producers(config, skip_layers)
        return skip_layers

    freq = getattr(config, "index_topk_freq", 1)
    if not isinstance(freq, int) or freq <= 0:
        raise ValueError(f"index_topk_freq must be a positive integer, got {freq!r}.")
    pattern = getattr(config, "index_topk_pattern", None)
    offset = getattr(config, "index_skip_topk_offset", 2)
    skip_layers: set[int] = set()
    for layer_idx in range(num_hidden_layers):
        if pattern is None:
            if max(layer_idx - offset + 1, 0) % freq != 0:
                skip_layers.add(layer_idx)
        elif 0 <= layer_idx < len(pattern) and pattern[layer_idx] == "S":
            skip_layers.add(layer_idx)
    _validate_topk_producers(config, skip_layers)
    return skip_layers


def _validate_topk_producers(config, skip_layers: set[int]) -> None:
    layer_types = getattr(config, "layer_types", None)
    producer = None
    for i in range(config.num_hidden_layers):
        sparse = layer_types is None or layer_types[i] in _SPARSE_LAYER_TYPES
        if not sparse:
            continue
        if i in skip_layers:
            if producer is None:
                raise ValueError(
                    f"HY4 shared indexer layer {i} has no preceding full sparse producer"
                )
        else:
            producer = i


def validate_hy4_parallel_config(config, pipeline_parallel_size: int) -> None:
    shared = compute_skip_topk_layers(config)
    if pipeline_parallel_size > 1 and shared:
        raise ValueError(
            "HY4 shared top-k indices are not transferred between PP stages; "
            "use pipeline_parallel_size=1 for shared indexers."
        )


def is_skip_topk_indexer_weight(weight_name: str, skip_topk_layers: set[int]) -> bool:
    """Return whether an indexer weight belongs to a top-k sharing layer.

    Args:
        weight_name: Checkpoint weight name.
        skip_topk_layers: Result of `compute_skip_topk_layers`.

    Returns:
        True when the weight is an indexer weight of a layer that has no
        indexer module and therefore must be dropped.
    """
    if ".indexer." not in weight_name or not skip_topk_layers:
        return False
    match = _WEIGHT_LAYER_INDEX_RE.search(weight_name)
    return match is not None and int(match.group(1)) in skip_topk_layers


class Indexer(nn.Module):
    """Lightning indexer selecting the top-k tokens for sparse MLA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
        *,
        runtime_plan: HY4RuntimePlan,
    ):
        super().__init__()
        self.runtime_plan = runtime_plan
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # No tensor parallelism, just replicated.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcast to BF16 while loading to keep the fusion.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        # FP8 naive cache: values in fp8 plus one fp32 scale per
        # ``quant_block_size`` elements.
        assert cache_config is not None, "HYV4 indexer requires cache_config"
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        use_flaggems_indexer = self.runtime_plan.provider == "flaggems"
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )
        if use_flaggems_indexer:
            # PlatformFL is an out-of-tree platform, so CustomOp dispatches
            # to ``forward_oot``.  vLLM's default implementation delegates
            # that method to ``forward_native`` and raises before reaching
            # the Python/FlagGems sparse operator.  Bind only this HY4
            # instance to the CUDA-shaped entry point; its implementation
            # uses the module-local FlagGems replacements installed above.
            self.indexer_op._forward_method = self.indexer_op.forward_cuda

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        hidden_states, q_quant, k, weights = self.prepare_inputs(
            hidden_states, qr, positions, rotary_emb
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the quantized query, key and per-head weights of the indexer."""
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        # Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims.
        q_nope, q_pe = torch.split(
            q, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_nope, k_pe = torch.split(
            k, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # RoPE (NeoX) can introduce extra leading dims, so flatten back to the
        # token-major shapes.
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # Reassemble with the original physical layout: no_pe first, pe last.
        q = torch.cat([q_nope, q_pe], dim=-1)
        # ``k_pe`` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_nope, k_pe.squeeze(-2)], dim=-1)

        # Only q is quantized here; k quantization is fused with cache insertion.
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = self.runtime_plan.query_quantizer(
            q,
            self.quant_block_size,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        return hidden_states, q_fp8, k, weights


class HYV4MLAAttention(nn.Module):
    """Multi-head latent attention with optional sparse lightning indexer.

    Main reference: the DeepSeek-V2 paper and the FlashInfer implementation
    (https://arxiv.org/abs/2405.04434). HY V4 additionally supports an output
    gate (``gated_mla``) and a per-head learnable attention sink.

    The sink is applied by binding the sink-capable backend from
    `.flashmla_sparse`. Construction fails if the selected backend cannot
    consume sinks; the checkpoint's sink semantics are never disabled.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        layer_idx: int = 0,
        *,
        runtime_plan: HY4RuntimePlan,
    ) -> None:
        super().__init__()
        self.runtime_plan = runtime_plan
        self.config = config
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.layer_id = int(prefix.split(".")[-2])
        layer_types = getattr(config, "layer_types", None)
        requested_sparse = (
            hasattr(config, "index_topk")
            and layer_types is not None
            and self.layer_id < len(layer_types)
            and layer_types[self.layer_id] in _SPARSE_LAYER_TYPES
        )
        # Only actual sparse layers may share another layer's top-k indices.
        self.skip_topk = requested_sparse and self.layer_id in compute_skip_topk_layers(
            config
        )
        # The skip pattern only governs backbone layers. MTP/nextn layers
        # (layer_id >= num_hidden_layers) always build a full indexer: they
        # compute indices at draft step 0 and toggle at runtime.
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = (
            num_hidden_layers is not None and self.layer_id >= num_hidden_layers
        )
        self.create_indexer = requested_sparse and (not self.skip_topk or is_mtp_layer)
        self.is_sparse = requested_sparse

        # Do not silently degrade sparse layers into dense attention. Probe the
        # sparse MLA backend directly and fail fast with the real error.
        kv_cache_dtype = cache_config.cache_dtype if cache_config else "auto"
        if self.is_sparse:
            try:
                get_attn_backend(
                    head_size=self.kv_lora_rank + self.qk_rope_head_dim,
                    dtype=torch.get_default_dtype(),
                    kv_cache_dtype=kv_cache_dtype,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=True,
                    num_heads=self.num_local_heads,
                )
            except Exception as exc:
                raise RuntimeError(
                    "HYV4 sparse attention was requested, but no valid sparse MLA "
                    "backend is available for current runtime/config. "
                    "Refusing to fall back to dense attention."
                ) from exc

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.q_a_proj = None
        self.kv_a_proj_with_mqa = None
        if self.q_lora_rank is not None:
            self.q_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
                disable_tp=True,
            )
            self.kv_a_proj_with_mqa = MergedColumnParallelLinear(
                self.hidden_size,
                [self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        self.q_a_layernorm = None
        self.q_b_proj = None
        self.q_proj = None
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        self.indexer_rope_emb: nn.Module | None
        self.indexer: Indexer | None
        if self.create_indexer:
            # The checkpoint stores indexer q_pe/k_pe in interleaved
            # (Megatron/PTM) layout, so the indexer must use interleaved RoPE
            # (is_neox_style=False) like the main attention path. Using NeoX
            # here loses the relative-position dependence and corrupts the DSA
            # top-k selection.
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )
            # The indexer projects its queries from the MLA q_lora activations,
            # so a sparse layer requires a query down-projection.
            assert q_lora_rank is not None, (
                "HYV4 sparse attention requires q_lora_rank to be set"
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
                runtime_plan=runtime_plan,
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        self.gated_mla = bool(getattr(config, "gated_mla", False))
        self.linear_gate: ColumnParallelLinear | None
        if self.gated_mla:
            if config.gating_type == "headwise":
                self.gate_projection_size_per_head = 1
            elif config.gating_type == "elementwise":
                self.gate_projection_size_per_head = self.v_head_dim
            else:
                raise ValueError(f"Unknown gating type: {config.gating_type}")
            self.linear_gate = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.gate_projection_size_per_head,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_gate",
            )
        else:
            self.linear_gate = None
        self.prefix = prefix

        # Per-head learnable attention sink. Created BEFORE ``MLAAttention`` so
        # it can be forwarded as the ``sinks`` impl kwarg. The parameter always
        # holds the local TP shard.
        self.learnable_sink = bool(getattr(config, "learnable_sink", False))
        sinks = None
        sink_backend: type[AttentionBackend] | None = None
        if self.learnable_sink:
            sink_backend = self._resolve_sink_backend(kv_cache_dtype)
            self.learnable_sink_param = nn.Parameter(
                torch.empty(
                    self.num_local_heads,
                    dtype=torch.float32,
                )
            )
            sinks = self.learnable_sink_param
            self._force_sparse_mqa()

        extra_impl_args = {} if sinks is None else {"sinks": sinks}
        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
            attn_backend=sink_backend,
            **extra_impl_args,
        )

    def _resolve_sink_backend(self, kv_cache_dtype: str) -> type[AttentionBackend]:
        """Require a backend that preserves the checkpoint's sink semantics."""
        head_size = self.kv_lora_rank + self.qk_rope_head_dim
        dtype = torch.get_default_dtype()
        try:
            selected_cls = get_attn_backend(
                head_size=head_size,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                use_mla=True,
                use_sparse=self.is_sparse,
                num_heads=self.num_local_heads,
            )
        except Exception as exc:
            raise RuntimeError(
                "HY4 requires attention sink support; backend selection failed "
                f"(device={current_platform.device_type}, dtype={dtype}, "
                f"kv_cache_dtype={kv_cache_dtype}): {exc}"
            ) from exc

        if selected_cls.supports_sink():
            return selected_cls

        from .hy_v4_flashmla_sparse import HYV4FlashMLASparseBackend

        # Mirror how the selector derives the configuration-dependent inputs so
        # this check accepts exactly what the backend would accept at runtime.
        capability = current_platform.get_device_capability()
        if capability is None:
            raise RuntimeError(
                "HY4 requires attention sink support; unknown device capability "
                f"(device={current_platform.device_type}, dtype={dtype}, kv_cache_dtype={kv_cache_dtype})"
            )
        cache_config = get_current_vllm_config().cache_config
        block_size = (
            cache_config.block_size
            if cache_config is not None and cache_config.user_specified_block_size
            else None
        )
        invalid_reasons = HYV4FlashMLASparseBackend.validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=cast(CacheDType, kv_cache_dtype),
            block_size=block_size,
            use_mla=True,
            has_sink=True,
            use_sparse=self.is_sparse,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=capability,
            attn_type=AttentionType.DECODER,
        )
        if invalid_reasons:
            raise RuntimeError(
                f"HY4 requires attention sink support; {selected_cls.get_name()} "
                "has no sinks and HYV4FlashMLASparseBackend rejected "
                f"device={current_platform.device_type}, dtype={dtype}, "
                f"kv_cache_dtype={kv_cache_dtype}: {', '.join(invalid_reasons)}"
            )

        logger.info_once(
            "HYV4 learnable sink enabled: using the sink-capable "
            "FLASHMLA_SPARSE impl instead of %s, which cannot apply sinks.",
            selected_cls.get_name(),
        )
        return HYV4FlashMLASparseBackend

    def _force_sparse_mqa(self) -> None:
        """Keep every token on the sink-capable sparse MQA path.

        ``_resolve_sink_backend`` only binds the backend that serves decode.
        ``MLAAttention`` additionally routes short prefills to a separate dense
        MLA prefill backend, and none of those accept ``attn_sink``, so prefill
        would silently drop the sink while decode applies it. Such a partially
        applied sink is not the trained architecture and corrupts the output, so
        opt out of the dense split instead.

        Prefills up to ``index_topk`` keep every token inside the sparse top-k,
        making the sparse path numerically equivalent to the dense one apart from
        also applying the sink.
        """
        attention_config = get_current_vllm_config().attention_config
        # Stock vLLM 0.24 already routes every SparseMLAAttentionImpl token
        # through forward_mqa.  The explicit switch exists only on the newer
        # HY4 feature branch this adapter was supplied against.
        if not hasattr(attention_config, "sparse_mla_force_mqa"):
            logger.info_once(
                "HYV4 learnable sink enabled: vLLM 0.24 sparse MLA already "
                "uses MQA for prefill and decode."
            )
            return
        if attention_config.sparse_mla_force_mqa:
            return
        attention_config.sparse_mla_force_mqa = True
        logger.info_once(
            "HYV4 learnable sink enabled: forcing sparse MQA for prefill too, "
            "as the dense MLA prefill backends cannot apply sinks."
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        if self.q_lora_rank is not None:
            assert self.q_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            q_c = self.q_a_proj(hidden_states)[0]
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.q_proj is not None
            q = self.q_proj(hidden_states)[0]

        assert self.kv_a_proj_with_mqa is not None
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add a head dim of 1 to k_pe.
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        output_shape = (
            hidden_states.shape[0],
            self.num_local_heads * self.v_head_dim,
        )
        # Single coarse eager break covering the indexer and MLA attention, as
        # the breakable cudagraph contract requires: everything that reads
        # per-batch metadata runs in one eager segment, so no tensor has to stay
        # alive across a capture-segment boundary.
        attn_out = torch.empty(
            output_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        self._indexer_and_attn(
            hidden_states, q_c, positions, q, kv_c_normed, k_pe, attn_out
        )

        if self.gated_mla and self.linear_gate is not None:
            gate_score = self.linear_gate(hidden_states)[0]
            if self.config.gating_type == "headwise":
                gate_score = gate_score.unsqueeze(-1)
                attn_out = attn_out.reshape(*attn_out.shape[:-1], -1, self.v_head_dim)
                attn_out = attn_out * torch.sigmoid(gate_score)
                attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)
            else:
                attn_out = attn_out * torch.sigmoid(gate_score)

        out, _ = self.o_proj(attn_out)
        return out

    @eager_break_during_capture
    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | None,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, heads * v_head_dim], written in place
    ) -> None:
        """Run the lightning indexer and MLA attention in one eager segment.

        Both read per-batch attention metadata, so under the breakable cudagraph
        they must not be captured. Keeping them in a single break (instead of one
        break each) also means the attention inputs never have to survive a
        capture-segment boundary. The nested ``sparse_attn_indexer`` and
        ``unified_mla_attention_with_output`` breaks short-circuit here, since
        the capture is no longer active inside an eager segment.
        """
        if self.indexer is not None and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        out.copy_(
            self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=out.shape,
            )
        )
