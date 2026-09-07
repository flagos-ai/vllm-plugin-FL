# SPDX-License-Identifier: Apache-2.0
"""GLM5-Next text runtime for pristine vLLM 0.24.

This combines vLLM 0.24's KDA recurrent layer, DeepSeek-V3.2 sparse MLA,
DeepSeek MoE, and mHC operators. The model-specific bounded KDA gate and the
kpool-compressed index/tail caches are kept plugin-owned.
"""

from collections.abc import Iterable
from types import MethodType

import torch
from einops import rearrange
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.compilation.decorators import support_torch_compile
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mhc import (
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLAAttention,
    DeepseekV2Model,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_fl.kernels.glm5_next.provider import use_nvidia_reference

if use_nvidia_reference():
    try:
        from vllm_fl.kernels.glm5_next.mhc_broadcast_tilelang import (
            mhc_pre_broadcast_tilelang,
        )
    except (ImportError, OSError):
        # H100 deployments may intentionally omit TileLang.  The model keeps
        # the exact materialized mHC fallback in that case; importing the
        # model must not make the optional vendor kernel mandatory.
        mhc_pre_broadcast_tilelang = None
else:
    mhc_pre_broadcast_tilelang = None

if current_platform.is_cuda():
    from vllm.model_executor.layers.fla.ops.kda import fused_recurrent_kda
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

    from vllm_fl.kernels.glm5_next.safe_kda import (
        chunk_kda_with_safe_gate,
        fused_safe_kda_gate,
    )
else:
    from vllm_fl.kernels.glm5_next.portable import (
        causal_conv1d_fn,
        causal_conv1d_update,
        chunk_kda_with_safe_gate,
        fused_recurrent_kda,
        safe_kda_gate as fused_safe_kda_gate,
    )

if use_nvidia_reference():
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
else:

    class SiluAndMulWithClamp(nn.Module):
        """Bounded SwiGLU without constructing vLLM's CUDA custom op."""

        def __init__(
            self,
            swiglu_limit: float,
            alpha: float = 1.0,
            beta: float = 0.0,
            **kwargs,
        ) -> None:
            super().__init__()
            del kwargs
            self.swiglu_limit = float(swiglu_limit)
            self.alpha = float(alpha)
            self.beta = float(beta)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dim = x.shape[-1] // 2
            if self.alpha == 1.0 and self.beta == 0.0:
                try:
                    from flag_gems.fused.silu_and_mul_with_clamp import (
                        silu_and_mul_with_clamp,
                    )

                    return silu_and_mul_with_clamp(
                        x[..., :dim], x[..., dim:], self.swiglu_limit
                    )
                except (ImportError, OSError, NotImplementedError, RuntimeError):
                    pass
            gate = x[..., :dim].clamp(max=self.swiglu_limit)
            up = x[..., dim:].clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            return gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)


from vllm_fl.kernels.glm5_next.indexer_backend import INDEXER_BACKEND
from vllm_fl.kernels.glm5_next.sparse_attn_indexer_kpool import (
    SparseAttnIndexerKpool,
)
from vllm_fl.models.glm5_next_kpool import (
    Glm5NextIndexerCache,
    Glm5NextTailCache,
)

logger = init_logger(__name__)


def _hc_expand(x: torch.Tensor, streams: int) -> torch.Tensor:
    return x.unsqueeze(1).expand(-1, streams, -1).contiguous()


def _hc_contract(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1)


class Glm5NextMLP(nn.Module):
    """Reference GLM5-Next SwiGLU, including the trained clamp."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        if swiglu_limit is None:
            raise ValueError("GLM5-Next requires a finite swiglu_limit")
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextMoE(nn.Module):
    """Reference GLM5-Next router and clamped shared/routed experts."""

    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.routed_scaling_factor = float(config.routed_scaling_factor)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_shared_experts = int(config.n_shared_experts)
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported."
            )

        router_dtype_name = getattr(config, "moe_router_dtype", "float32")
        router_dtype = getattr(torch, str(router_dtype_name))
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=router_dtype,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        swiglu_limit = config.swiglu_limit
        if swiglu_limit is None:
            raise ValueError("GLM5-Next requires a finite swiglu_limit")
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * config.n_shared_experts
                ),
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            router_logits_dtype=self.gate.out_dtype,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )[:num_tokens]
        return final_hidden_states.view(num_tokens, hidden_dim)


class Glm5NextLinearAttention(KimiGatedDeltaNetAttention):
    """v0.24 KDA projections with the reference bounded gate on both paths."""

    def __init__(self, config, vllm_config, prefix: str = "") -> None:
        super().__init__(config, vllm_config, prefix)
        kda_config = config.linear_attn_config or {}
        self.kda_lower_bound = float(kda_config.get("gate_lower_bound", -5.0))
        self._conv_state_dim_first = is_conv_state_dim_first()

        # GLM5-Next checkpoints store A_log as [num_heads], whereas the v0.24
        # runtime parameter is [1, 1, local_heads, 1].
        old_loader = self.A_log.weight_loader

        def load_a_log(param, loaded_weight):
            if loaded_weight.ndim == 1:
                loaded_weight = loaded_weight.view(1, 1, -1, 1)
            return old_loader(param, loaded_weight)

        self.A_log.weight_loader = load_a_log

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        del positions
        num_tokens = hidden_states.size(0)
        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]

        beta = self.b_proj(hidden_states)[0].float().sigmoid()
        g1 = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Keep the metadata-dependent KDA body visible to the breakable-CG
        # decorator. Routing through torch.ops.vllm.kda_attention makes the
        # custom op opaque and can freeze capture-time state indices/branches.
        self._forward(
            q_proj_states=q,
            k_proj_states=k,
            v_proj_states=v,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward_reference_safe_gate(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            #     # V1 profile run
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        constant_caches = self.kv_cache

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)

        conv_state_q, conv_state_k, conv_state_v = conv_state.chunk(3, dim=-2)

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )
        if attn_metadata_narrowed.num_prefills > 0:
            q_proj_states = q_proj_states.transpose(0, 1)
            k_proj_states = k_proj_states.transpose(0, 1)
            v_proj_states = v_proj_states.transpose(0, 1)
            q = causal_conv1d_fn(
                q_proj_states,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
        else:
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : attn_metadata_narrowed.num_actual_tokens
            ]
            q = causal_conv1d_update(
                q_proj_states,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_proj_states,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_proj_states,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim), (q, k, v)
        )

        if attn_metadata_narrowed.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            zero_idx = non_spec_state_indices_tensor[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_safe_gate(
                q=q,
                k=k,
                v=v,
                raw_g=g1,
                beta=beta,
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                lower_bound=self.kda_lower_bound,
            )
            # Init cache
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
        else:
            assert non_spec_query_start_loc is not None
            g1 = fused_safe_kda_gate(
                rearrange(g1, "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
                lower_bound=self.kda_lower_bound,
            ).unsqueeze(0)
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata_narrowed.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
            0, :num_actual_tokens
        ]

    _forward = _forward_reference_safe_gate


def _indexer_forward_nope(
    self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
) -> torch.Tensor:
    """DSA indexer projection when the checkpoint has zero RoPE channels."""
    del rotary_emb
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)
    kw, _ = self.wk_weights_proj(hidden_states)
    k = self.k_norm(kw[:, : self.head_dim])
    if getattr(self, "_wp_fp32", None) is None:
        self._wp_fp32 = (
            self.wk_weights_proj.weight.data[self.head_dim :, :]
            .t()
            .contiguous()
            .float()
        )
    weights = torch.mm(hidden_states.float(), self._wp_fp32)

    assert self.head_dim == 128 and self.quant_block_size == 128
    assert self.scale_fmt == "ue8m0"
    q = q.view(-1, self.head_dim)
    q_fp8, q_scale = INDEXER_BACKEND.fwht128_quant_fp8(q)
    q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
    q_scale = q_scale.view(-1, self.n_head, 1)
    weights = (
        weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
    ).squeeze(-1)
    gate_score = torch.nn.functional.linear(
        hidden_states, self.index_kpool_compress_gate
    )
    if self.n_head < 32:
        pad = 32 - self.n_head
        q_fp8 = torch.cat(
            [q_fp8, q_fp8.new_zeros(q_fp8.shape[0], pad, self.head_dim)],
            dim=1,
        )
        weights = torch.cat([weights, weights.new_zeros(weights.shape[0], pad)], dim=1)
    return self.indexer_op(
        hidden_states,
        q_fp8,
        k,
        weights,
        gate_score=gate_score,
        compress_ape=self.index_kpool_compress_ape,
        index_kpool=self.index_kpool,
        positions=positions,
    )


class Glm5NextMLAAttention(DeepseekV2MLAAttention):
    """DeepSeek sparse MLA with an explicit zero-RoPE fast path."""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        config = kwargs["config"]
        cache_config = kwargs["cache_config"]
        # vLLM's base MLA constructor eagerly creates its stock
        # ``SparseAttnIndexer`` before this class can replace it with the
        # plugin-owned kpool implementation below.  On a CUDA empty-build
        # wheel that stock constructor rejects the missing DeepGEMM ABI even
        # though all runtime indexer operations will be routed through the
        # portable backend.  Suppress only that constructor-time capability
        # guard, and only for GLM5's non-NVIDIA provider; the temporary value
        # is restored before model construction continues.  Explicit
        # ``VLLM_FL_GLM5_PROVIDER=nvidia`` still fails fast as intended.
        sparse_indexer_module = None
        original_has_deep_gemm = None
        mla_attention_module = None
        original_get_prefill_backend = None
        if current_platform.is_cuda() and not use_nvidia_reference():
            from vllm.model_executor.layers import sparse_attn_indexer
            from vllm.model_executor.layers.attention import (
                mla_attention as mla_attention_module,
            )

            from vllm_fl.dispatch.backends.flaggems.impl.mla_prefill import (
                FlagGemsMLAPrefillBackend,
            )

            sparse_indexer_module = sparse_attn_indexer
            original_has_deep_gemm = sparse_indexer_module.has_deep_gemm
            sparse_indexer_module.has_deep_gemm = lambda: True
            original_get_prefill_backend = (
                mla_attention_module.get_mla_prefill_backend
            )
            mla_attention_module.get_mla_prefill_backend = (
                lambda _vllm_config: FlagGemsMLAPrefillBackend
            )
        try:
            super().__init__(*args, **kwargs)
        finally:
            if sparse_indexer_module is not None:
                sparse_indexer_module.has_deep_gemm = original_has_deep_gemm
            if mla_attention_module is not None:
                mla_attention_module.get_mla_prefill_backend = (
                    original_get_prefill_backend
                )
        if self.indexer is not None and config.index_kpool_compress:
            indexer = self.indexer
            kpool = int(config.index_kpool)

            # Replace the stock per-token cache object registered by the base
            # constructor with the kpool-compressed cache at the same prefix.
            static_ctx = vllm_config.compilation_config.static_forward_context
            old_cache = indexer.k_cache
            assert static_ctx.get(old_cache.prefix) is old_cache
            del static_ctx[old_cache.prefix]

            indexer.index_kpool = kpool
            indexer.index_kpool_compress_ape = nn.Parameter(
                torch.zeros(kpool, indexer.head_dim, dtype=torch.float32)
            )
            indexer.index_kpool_compress_gate = nn.Parameter(
                torch.empty(
                    indexer.head_dim,
                    config.hidden_size,
                    dtype=torch.bfloat16,
                )
            )
            indexer.k_cache = Glm5NextIndexerCache(
                head_dim=(
                    indexer.head_dim + indexer.head_dim // indexer.quant_block_size * 4
                ),
                dtype=torch.uint8,
                prefix=old_cache.prefix,
                cache_config=cache_config,
                index_kpool=kpool,
            )
            indexer.tail_cache = Glm5NextTailCache(
                head_dim=indexer.head_dim,
                dtype=torch.bfloat16,
                prefix=f"{indexer.prefix}.tail_cache",
                cache_config=cache_config,
                index_kpool=kpool,
            )
            indexer.indexer_op = SparseAttnIndexerKpool(
                indexer.k_cache,
                indexer.quant_block_size,
                indexer.scale_fmt,
                indexer.topk_tokens,
                indexer.head_dim,
                indexer.max_model_len,
                indexer.max_total_seq_len,
                indexer.topk_indices_buffer,
                tail_cache=indexer.tail_cache,
            )

        if self.qk_rope_head_dim == 0:
            self.mla_attn.rotary_emb = None
            self.mla_attn.indexer_rope_emb = None
            if self.indexer is not None:
                self.indexer.forward = MethodType(_indexer_forward_nope, self.indexer)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.layer_idx = layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.mhc = bool(config.mhc)

        if config.is_kda_layer(layer_idx):
            self.self_attn = Glm5NextLinearAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                # The reference checkpoint keeps MLA projections in BF16.
                quant_config=None,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc:
            self.n = config.mhc_num_residual_streams
            d_model = self.n * config.hidden_size
            mix_hc = (2 + self.n) * self.n
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            # Populated after checkpoint loading.  Repeated first-layer
            # residual streams make the projection equivalent to a projection
            # with this stream-summed weight, without materializing the input
            # broadcast first.
            self.hc_attn_fn_broadcast: torch.Tensor | None = None
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MHCFusedPostPreOp()

    def _attention(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.self_attn, DeepseekV2MLAAttention):
            return self.self_attn(positions, hidden_states, None)
        return self.self_attn(hidden_states, positions)

    def _hc_pre(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm: RMSNorm,
    ):
        return self.mhc_pre_op(
            residual=residual,
            fn=fn,
            hc_scale=scale,
            hc_base=base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            norm_weight=norm.weight.data,
            norm_eps=norm.variance_epsilon,
        )

    def _hc_pre_broadcast(
        self,
        residual: torch.Tensor,
        norm: RMSNorm,
    ):
        assert mhc_pre_broadcast_tilelang is not None
        assert self.hc_attn_fn_broadcast is not None
        return mhc_pre_broadcast_tilelang(
            residual,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.mhc_post_mult_value,
            self.mhc_sinkhorn_iterations,
            norm_weight=norm.weight.data,
            norm_eps=norm.variance_epsilon,
            fn_broadcast=self.hc_attn_fn_broadcast,
        )

    def _hc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm: RMSNorm,
    ):
        return self.mhc_fused_post_pre_op(
            x=x,
            residual=residual,
            post_layer_mix=post,
            comb_res_mix=comb,
            fn=fn,
            hc_scale=scale,
            hc_base=base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            n_splits=1,
            tile_n=1,
            norm_weight=norm.weight.data,
            norm_eps=norm.variance_epsilon,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not self.mhc:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self._attention(positions, hidden_states)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual=residual
            )
            hidden_states = residual + self.mlp(hidden_states)
            return hidden_states, residual, None, None

        x = hidden_states
        if post is None:
            if (
                self.layer_idx == 0
                and x.dim() == 2
                and self.hc_attn_fn_broadcast is not None
                and mhc_pre_broadcast_tilelang is not None
            ):
                residual, post, comb, x = self._hc_pre_broadcast(
                    x, self.input_layernorm
                )
            else:
                if self.layer_idx == 0:
                    x = _hc_expand(x, self.n)
                residual = x
                post, comb, x = self._hc_pre(
                    x,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.input_layernorm,
                )
        else:
            assert residual is not None and comb is not None
            residual, post, comb, x = self._hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm,
            )

        x = self._attention(positions, x)
        assert residual is not None and post is not None and comb is not None
        residual, post, comb, x = self._hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.post_attention_layernorm,
        )
        x = self.mlp(x)

        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.mhc_post_op(x, residual, post, comb)
            return _hc_contract(x), None, None, None
        return x, residual, post, comb


@support_torch_compile
class Glm5NextModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size

        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            # kpool expands 512 pool ids to 2048 tokens, appends <=3 tail
            # tokens, and sparse MLA requires a 128-column aligned width.
            ((config.index_topk + config.index_kpool - 1 + 127) // 128 * 128),
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config,
                layer_idx,
                prefix,
                topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.use_mha = False
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = post = comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            post = comb = None

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The v0.24 DeepSeek loader handles fused MLA/indexer projections,
        # dense SwiGLU stacking, expert tensors, and direct KDA/mHC parameters.
        return DeepseekV2Model.load_weights(self, weights)

    def finalize_mhc_broadcast_weights(self) -> None:
        """Build the first-layer projection used by the NVIDIA fast path."""
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        if not use_nvidia_reference():
            return

        from vllm.utils.deep_gemm import is_deep_gemm_supported

        if not is_deep_gemm_supported():
            logger.warning_once(
                "GLM5-Next mHC broadcast specialization is disabled because "
                "DeepGEMM is unavailable; using the generic mHC path"
            )
            return

        layer = self.layers[self.start_layer]
        if isinstance(layer, Glm5NextDecoderLayer) and layer.layer_idx == 0:
            layer.hc_attn_fn_broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.n, self.config.hidden_size)
                .sum(dim=1)
                .contiguous()
            )
            logger.info_once(
                "Enabled GLM5-Next first-layer mhc_pre_broadcast_tilelang"
            )


class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_text_config
        self.model = Glm5NextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        config = vllm_config.model_config.hf_text_config
        speculative_config = vllm_config.speculative_config
        num_spec = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_heads,
            config.linear_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
            ignore_unexpected_prefixes=["model.visual."],
        )
        loaded = loader.load_weights(weights)

        # AutoWeightsLoader accepts a checkpoint as soon as every *present*
        # key has a destination; it does not verify the inverse condition that
        # every runtime parameter received a checkpoint tensor.  That is too
        # weak for a newly adapted architecture: one missed packed projection
        # otherwise remains torch.empty() and only appears later as meaningless
        # logits.  Audit the text model at the innermost CausalLM boundary so
        # vision-only parameters and outer HF prefix mapping cannot obscure the
        # result.
        expected = {name for name, _ in self.named_parameters()}
        missing = sorted(expected - loaded)
        unexpected = sorted(loaded - expected)
        logger.info(
            "GLM5-Next strict text weight audit: loaded=%d expected=%d "
            "missing=%d unexpected=%d",
            len(loaded),
            len(expected),
            len(missing),
            len(unexpected),
        )
        if unexpected:
            logger.warning(
                "GLM5-Next weight audit returned unexpected names: %s",
                unexpected[:32],
            )
        if missing:
            raise RuntimeError(
                "GLM5-Next checkpoint did not initialize all text parameters; "
                f"first missing names: {missing[:64]}"
            )
        self.model.finalize_mhc_broadcast_weights()
        return loaded


__all__ = ["Glm5NextForCausalLM"]
