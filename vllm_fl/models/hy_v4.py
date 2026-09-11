# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only HY4 model for vLLM serving."""

from __future__ import annotations

import os
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear

try:
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )
except ImportError:  # vLLM variants without the legacy mapping helper
    fused_moe_make_expert_params_mapping = None
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    get_pp_missing_layer_names,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors

from vllm_fl.configs.hy_v4 import HYV4Config
from vllm_fl.models.hy_v4_attention import (
    HYV4MLAAttention,
    compute_skip_topk_layers,
    is_skip_topk_indexer_weight,
)
from vllm_fl.ops.hy4_hc_projection import hc_n8_projection
from vllm_fl.ops.hy_v4_hc import (
    read_post_linear as _hyv4_hc_read_post_linear,
    writeback as _hyv4_hc_writeback,
)

logger = init_logger(__name__)


_HYV4_SHARED_EXPERTS_RUNNER_ENV = "VLLM_HY4_SHARED_EXPERTS_RUNNER"


def _hyv4_env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean HY4 opt-in without treating malformed values as enabled."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _hyv4_fp32_combine_and_reduce(
    routed: torch.Tensor,
    shared: torch.Tensor,
    *,
    tp_size: int,
    ep_size: int,
    is_sequence_parallel: bool,
    routed_output_is_global: bool,
    shared_output_is_global: bool,
    all_reduce: Callable[[torch.Tensor], torch.Tensor] = tensor_model_parallel_all_reduce,
) -> torch.Tensor:
    """Combine routed/shared outputs without a silent BF16 accumulation.

    The runner/SP paths keep the shared MLP's row projection unreduced.  The
    default non-SP fallback retains the original RowParallelLinear reduction;
    this helper therefore only performs a collective when its explicit branch
    ownership flags say one is still needed:

    * standalone vLLM ``FusedMoE`` always returns a TP/EP-global routed output,
      so the fallback reduces only shared output here when it is unreduced;
    * a caller that explicitly supplies a local routed output can request one
      late collective over the FP32 sum with ``routed_output_is_global=False``.

    Sequence parallelism is handled by the caller: its local chunks are summed
    in FP32 and then all-gathered, so no TP all-reduce occurs in this helper.
    ``all_reduce`` is injectable for deterministic unit tests.
    """
    combined = routed.float() + shared.float()

    if is_sequence_parallel:
        return combined

    needs_collective = tp_size > 1 or ep_size > 1
    if not needs_collective:
        return combined

    if routed_output_is_global:
        if shared_output_is_global:
            return combined
        # The routed result is already global.  Reduce only the unreduced
        # shared branch, then add in FP32.  The collective is deliberately
        # issued on the caller's stream after the aux-stream compute is joined.
        shared = all_reduce(shared.float()).float()
        return routed.float() + shared

    if shared_output_is_global:
        return all_reduce(routed.float()).float() + shared.float()

    # Neither branch is global yet.  One collective after the FP32 local add
    # preserves expert ownership and avoids the old two-collective path.
    return all_reduce(combined).float()


class _HYV4FP32RoutedOutput(nn.Module):
    """Cast only the routed output before vLLM's shared+routed add.

    vLLM 0.24 applies ``routed_output_transform`` outside the opaque MoE custom
    op.  Therefore the fake op may keep its BF16 output contract while the
    runner's final ``shared_output + fused_output`` is promoted to FP32.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.float()


def _try_load_mxfp8_indexer_wk(
    name: str,
    tensor: torch.Tensor,
    pending: dict[str, dict[str, torch.Tensor]],
    params_dict: dict[str, nn.Parameter],
    loaded_params: set[str],
) -> bool:
    """Fuse this checkpoint's ModelOpt MXFP8 indexer wk with weights_proj."""
    if ".indexer.wk." not in name:
        return False
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = name.endswith(".weight_scale") and tensor.dtype == torch.uint8
    if not is_weight and not is_scale:
        return False

    layer_prefix = name.rsplit(".wk.", 1)[0]
    entry = pending.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight, scale = entry["weight"], entry["scale"]
    del pending[layer_prefix]
    if weight.shape[:-1] != scale.shape[:-1]:
        raise ValueError(
            f"HY4 indexer MXFP8 shape mismatch: {weight.shape} vs {scale.shape}"
        )
    group_size = weight.shape[-1] // scale.shape[-1]
    if group_size != 32:
        raise ValueError(
            f"HY4 indexer MXFP8 expected group size 32, got {group_size}"
        )
    scales = torch.exp2(scale.to(torch.int16).float() - 127.0)
    scales = scales.repeat_interleave(group_size, dim=-1)
    weight_bf16 = (weight.float() * scales).to(torch.bfloat16)

    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


def _make_hyv4_expert_params_mapping(
    model: nn.Module,
) -> list[tuple[str, str, int, str]]:
    """Build the vLLM routed-expert mapping, with a small legacy fallback.

    vLLM 0.24 exports ``fused_moe_make_expert_params_mapping``.  A few
    plugin images carry the same RoutedExperts ABI without re-exporting that
    helper, so keep the normal no-EPLB mapping local rather than making HY4
    import fail before model registration.  EPLB physical-to-logical expert
    remapping still requires the canonical helper and fails explicitly when
    it is unavailable.
    """
    num_experts = int(model.config.n_routed_experts)
    num_redundant_experts = int(getattr(model, "n_redundant_experts", 0))
    helper = fused_moe_make_expert_params_mapping
    if callable(helper):
        try:
            return list(
                helper(
                    model,
                    ckpt_gate_proj_name="gate_proj",
                    ckpt_down_proj_name="down_proj",
                    ckpt_up_proj_name="up_proj",
                    num_experts=num_experts,
                    num_redundant_experts=num_redundant_experts,
                )
            )
        except (AttributeError, TypeError) as exc:
            if num_redundant_experts:
                raise RuntimeError(
                    "HY4 EPLB loading requires vLLM's expert mapping helper"
                ) from exc

    if num_redundant_experts:
        raise RuntimeError(
            "HY4 EPLB loading requires vLLM's expert mapping helper"
        )

    mapping = []
    for expert_id in range(num_experts):
        mapping.extend(
            [
                (
                    "experts.routed_experts.w13_",
                    f"experts.{expert_id}.gate_proj.",
                    expert_id,
                    "w1",
                ),
                (
                    "experts.routed_experts.w2_",
                    f"experts.{expert_id}.down_proj.",
                    expert_id,
                    "w2",
                ),
                (
                    "experts.routed_experts.w13_",
                    f"experts.{expert_id}.up_proj.",
                    expert_id,
                    "w3",
                ),
            ]
        )
    return mapping


class HYV4HyperConnection(nn.Module):
    """Four identity residual streams with token-dependent read/write gates."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.num_streams = config.hc_mult
        width = config.hc_mult * config.hidden_size
        self.hc_fn = nn.Parameter(
            torch.empty(2 * config.hc_mult, width, dtype=torch.float32)
        )
        self.hc_base = nn.Parameter(
            torch.empty(2 * config.hc_mult, dtype=torch.float32)
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.magnitude = float(config.hc_magnitude)
        self.hc_eps = float(config.hc_eps)
        self.normalize_eps = float(config.rms_norm_eps)

    def read(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse streams and return the write gates for the branch output."""
        flat = streams.flatten(1).float()
        gates = hc_n8_projection(flat, self.hc_fn, self.normalize_eps)
        return _hyv4_hc_read_post_linear(
            streams,
            gates,
            self.hc_scale,
            self.hc_base,
            self.magnitude,
            self.hc_eps,
            self.num_streams,
        )

    @staticmethod
    def write(
        streams: torch.Tensor,
        delta: torch.Tensor,
        write: torch.Tensor,
    ) -> torch.Tensor:
        """Write one branch result into all identity residual streams."""
        return _hyv4_hc_writeback(streams, delta, write, streams.shape[1])


class HYV4HyperLayer(nn.Module):
    """Checkpoint-compatible wrapper around a pre-branch hyper-connection."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.hc_pre = HYV4HyperConnection(config)


class HYV4HyperHead(nn.Module):
    """Collapse the four final residual streams before the final RMSNorm."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.num_streams = config.hc_mult
        width = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(config.hc_mult, width, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32)
        )
        self.hc_head_scale = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.hc_eps = float(config.hc_eps)
        self.normalize_eps = float(config.rms_norm_eps)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = streams.flatten(1).float()
        read = torch.sigmoid(
            hc_n8_projection(flat, self.hc_head_fn, self.normalize_eps)
            * self.hc_head_scale
            + self.hc_head_base
        ) + self.hc_eps
        hidden = (read.unsqueeze(-1) * streams.float()).sum(dim=1)
        return hidden.to(streams.dtype)


class HYV4DenseMLP(nn.Module):
    """Tensor-parallel unclamped SwiGLU used by dense and shared branches."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        disable_tp: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class HYV4MoE(nn.Module):
    """HY4 no-aux sigmoid routed experts plus one shared expert."""

    def __init__(
        self,
        config: HYV4Config,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        self.is_sequence_parallel = bool(parallel_config.use_sequence_parallel_moe)
        # The upstream runner owns aux-stream ordering and late all-reduce.
        # Keep it opt-in so the existing HY4 fallback remains easy to A/B.
        self._use_shared_experts_runner = _hyv4_env_flag(
            _HYV4_SHARED_EXPERTS_RUNNER_ENV
        )
        if config.hidden_act != "silu":
            raise ValueError("HY4 currently supports only the silu activation")

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # The SM90 DSV3 router kernel selected by vLLM for H=6144/E=256
        # requires BF16 weights. HY4 stores and evaluates the router in FP32,
        # so keep this model on GateLinear's FP32 F.linear fallback instead.
        self.gate.allow_dsv3_router_gemm = False
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        self.shared_experts = HYV4DenseMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            # Runner/SP outputs stay local until their FP32 combine.  The
            # default non-SP fallback intentionally retains HY4's original
            # BF16 shared reduction.  In SP, both projections are replicated,
            # matching DeepSeek's ``disable_tp=is_sequence_parallel`` contract.
            reduce_results=not (
                self._use_shared_experts_runner or self.is_sequence_parallel
            ),
            disable_tp=self.is_sequence_parallel,
            prefix=f"{prefix}.shared_experts",
        )
        eplb_config = parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = config.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts

        # The runner path combines a local shared result after casting routed
        # output to FP32.  The default non-SP fallback keeps the original
        # RowParallelLinear reduction and only uses the helper for the FP32
        # final add.  SP uses replicated weights and no row reduction.
        self.experts = FusedMoE(
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
            routed_scaling_factor=config.routed_scaling_factor,
            swiglu_limit=config.swiglu_limit,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            apply_routed_scale_to_output=False,
            shared_experts=(
                self.shared_experts if self._use_shared_experts_runner else None
            ),
            routed_output_transform=(
                _HYV4FP32RoutedOutput()
                if self._use_shared_experts_runner
                else None
            ),
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
            is_sequence_parallel=self.is_sequence_parallel,
        )
        self.n_local_physical_experts = self.experts.routed_experts.local_num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        num_tokens = hidden_states.shape[0]
        output_dtype = hidden_states.dtype
        if self.is_sequence_parallel:
            # Keep token ownership aligned between routed and shared branches;
            # the gather below restores the original token order.
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self._use_shared_experts_runner:
            # vLLM 0.24's SharedExperts owns aux-stream synchronization.  Its
            # runner applies the routed output transform outside the opaque
            # custom op, so the shared+routed add is FP32 while the fake op
            # retains its BF16 shape/dtype contract.
            combined = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
            if self.is_sequence_parallel:
                combined = tensor_model_parallel_all_gather(combined.float(), 0)
                combined = combined[:num_tokens]
            return combined.to(output_dtype).view(shape)

        if self.experts.is_internal_router:
            routed = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            routed = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        shared = self.shared_experts(hidden_states)
        moe_config = getattr(self.experts, "moe_config", None)
        tp_size_value = getattr(moe_config, "tp_size", None)
        tp_size = int(
            tp_size_value
            if tp_size_value is not None
            else get_tensor_model_parallel_world_size()
        )
        ep_size = int(getattr(moe_config, "ep_size", 1))
        combined = _hyv4_fp32_combine_and_reduce(
            routed,
            shared,
            tp_size=tp_size,
            ep_size=ep_size,
            is_sequence_parallel=self.is_sequence_parallel,
            # The standalone FusedMoE call above already performs either its
            # early or late reduction, so its returned routed output is global.
            routed_output_is_global=True,
            # The env=false non-SP fallback keeps RowParallelLinear's original
            # reduce_results=True behavior.  SP returns before this flag is
            # consulted because its shared weights are replicated.
            shared_output_is_global=not self.is_sequence_parallel,
        )
        if self.is_sequence_parallel:
            combined = tensor_model_parallel_all_gather(combined, 0)
            combined = combined[:num_tokens]
        combined = combined.to(output_dtype)
        return combined.view(shape)


class HYV4Attention(HYV4MLAAttention):
    """HY4 compressed MLA plus DSA indexer, gate, and learnable sink."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: HYV4Config,
        layer_idx: int,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__(
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
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            layer_idx=layer_idx,
        )


class HYV4DecoderLayer(nn.Module):
    """One HY4 attention/MLP block operating on four residual streams."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        config = typing.cast(HYV4Config, vllm_config.model_config.hf_config)
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        self.hc_attn_layer = HYV4HyperLayer(config)
        self.self_attn = HYV4Attention(
            vllm_config,
            config,
            layer_idx,
            f"{prefix}.self_attn",
            topk_indices_buffer,
        )
        self.hc_mlp_layer = HYV4HyperLayer(config)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = HYV4MoE(config, vllm_config, f"{prefix}.mlp")
        else:
            self.mlp = HYV4DenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        streams: torch.Tensor,
    ) -> torch.Tensor:
        hidden, write = self.hc_attn_layer.hc_pre.read(streams)
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(positions, hidden)
        streams = self.hc_attn_layer.hc_pre.write(streams, hidden, write)

        hidden, write = self.hc_mlp_layer.hc_pre.read(streams)
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        streams = self.hc_mlp_layer.hc_pre.write(streams, hidden, write)
        return streams


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class HYV4Model(nn.Module):
    """HY4 decoder backbone."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = typing.cast(HYV4Config, vllm_config.model_config.hf_config)
        self.config = config
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=vllm_config.device_config.device,
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
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HYV4DecoderLayer(
                vllm_config,
                prefix,
                self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.hc_head = HYV4HyperHead(config)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.hc_head = PPMissingLayer()
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["streams"],
            config.hc_mult * config.hidden_size,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                inputs_embeds = self.embed_input_ids(input_ids)
            streams = inputs_embeds.unsqueeze(1).expand(
                -1,
                self.config.hc_mult,
                -1,
            )
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline rank requires intermediate tensors")
            streams = intermediate_tensors["streams"].view(
                -1,
                self.config.hc_mult,
                self.config.hidden_size,
            )

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            streams = layer(positions, streams)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"streams": streams.flatten(1)})
        hidden_states = self.hc_head(streams)
        return self.norm(hidden_states)


class HYV4LogitsProcessor(LogitsProcessor):
    """Compute the excluded HY4 language head in FP32 when configured."""

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        hidden_states = hidden_states.to(lm_head.weight.dtype)
        logits = lm_head.quant_method.apply(
            lm_head,
            hidden_states,
            bias=embedding_bias,
        )
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits


class HYV4MixtureOfExperts(MixtureOfExperts):
    """Expose HY4 expert metadata to vLLM EPLB infrastructure."""

    moe_mlp_layers: list[HYV4MoE]

    def extract_moe_parameters(self, example_moe: HYV4MoE | None) -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError("HY4 local expert count changed unexpectedly")
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.routed_experts.update_expert_map()


class HYV4ForCausalLM(
    nn.Module,
    SupportsPP,
    HYV4MixtureOfExperts,
):
    """Native vLLM causal language model for the HY4 preview checkpoint."""

    # HY4 preview is a text-only checkpoint family: no released variant ships a
    # vision tower, image processor, or image-token id.  Pin the capability
    # flag so vLLM 0.24's registry classifies HY4 as text-only and rejects
    # multimodal requests instead of silently dropping image inputs.
    supports_multimodal = False

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = typing.cast(HYV4Config, vllm_config.model_config.hf_config)
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = HYV4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            lm_head_dtype = (
                torch.float32
                if config.enable_lm_head_fp32
                else vllm_config.model_config.dtype
            )
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                params_dtype=lm_head_dtype,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = HYV4LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.expert_weights = []
        self.num_expert_groups = 1
        self.moe_mlp_layers = []
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, HYV4MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts.routed_experts)
        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @staticmethod
    def _packed_expert_target(name: str) -> tuple[str, bool] | None:
        gate_up_name = ".mlp.experts.gate_up_proj"
        down_name = ".mlp.experts.down_proj"
        if gate_up_name in name:
            return (
                name.replace(
                    gate_up_name,
                    ".mlp.experts.routed_experts.w13_weight",
                ),
                True,
            )
        if down_name in name:
            return (
                name.replace(
                    down_name,
                    ".mlp.experts.routed_experts.w2_weight",
                ),
                False,
            )
        return None

    def _load_all_experts(
        self,
        name: str,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_any = False
        expert_ids = getattr(self, "_hy4_local_expert_ids", range(num_experts))
        if loaded_weight.shape[0] != len(expert_ids):
            raise ValueError(
                f"HY4 packed expert leading dimension {loaded_weight.shape[0]} "
                f"does not match {len(expert_ids)} selected experts for {name}"
            )
        for loaded_expert_id, expert_id in enumerate(expert_ids):
            loaded_any |= loader(
                param,
                loaded_weight[loaded_expert_id],
                name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
        return loaded_any

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load HY4 split or packed expert weights and their quant scales."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        stacked_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        pending_wk_mxfp8: dict[str, dict[str, torch.Tensor]] = {}
        pending_wk_fp8: dict[str, dict[str, torch.Tensor]] = {}
        # W8A8 HY4 checkpoints store each routed expert as three independent
        # tensors (``experts.<id>.{gate,up,down}_proj``), with a matching
        # ``.weight_scale`` tensor.  RoutedExperts exposes those tensors as
        # packed ``w13_{weight,weight_scale}``/``w2_{weight,weight_scale}``
        # parameters, so use vLLM's canonical mapping for both data kinds.
        # Keep all candidates per source name for EPLB configurations that
        # map more than one physical expert to the same logical checkpoint
        # expert.
        split_expert_mapping: dict[
            str, list[tuple[str, int, str]]
        ] = {}
        for param_name, weight_name, expert_id, shard_id in (
            _make_hyv4_expert_params_mapping(self)
        ):
            split_expert_mapping.setdefault(weight_name, []).append(
                (param_name, expert_id, shard_id)
            )
        pp_missing_layer_names = get_pp_missing_layer_names(self)
        skip_topk_layers = compute_skip_topk_layers(self.config)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        local_heads = self.config.num_attention_heads // tp_size

        for name, loaded_weight in weights:
            if name.startswith("model.mtp_layers."):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if is_skip_topk_indexer_weight(name, skip_topk_layers):
                continue
            if _try_load_mxfp8_indexer_wk(
                name,
                loaded_weight,
                pending_wk_mxfp8,
                params_dict,
                loaded_params,
            ):
                continue
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                pending_wk_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue

            # The W8A8 export uses split per-expert names.  Match the
            # ``experts.<id>.<projection>.`` part without treating shared
            # experts (``shared_experts``) as routed experts.  Scales must go
            # through the same weight loader as weights so channel-wise TP
            # sharding is preserved.
            split_mapping = None
            if ".experts." in name:
                parts = name.split(".")
                try:
                    experts_index = parts.index("experts")
                    expert_id = int(parts[experts_index + 1])
                    projection = parts[experts_index + 2]
                except (ValueError, IndexError):
                    pass
                else:
                    source_name = f"experts.{expert_id}.{projection}."
                    split_mapping = split_expert_mapping.get(source_name)
            if split_mapping is not None:
                mapped = False
                for target_name, expert_id, shard_id in split_mapping:
                    mapped_name = name.replace(
                        source_name, target_name, 1
                    )
                    if is_pp_missing_parameter(mapped_name, self):
                        continue
                    param = params_dict.get(mapped_name)
                    if param is None:
                        continue
                    loader = typing.cast(Callable[..., bool], param.weight_loader)
                    if loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    ):
                        loaded_params.add(mapped_name)
                        mapped = True
                # An expert not owned by this EP rank is still a recognized
                # checkpoint tensor; do not let it fall through to the
                # generic unexpected-parameter error.
                if split_mapping or mapped:
                    continue

            if name.endswith("learnable_sink_param"):
                start = tp_rank * local_heads
                loaded_weight = loaded_weight.narrow(0, start, local_heads)

            packed_expert = self._packed_expert_target(name)
            if packed_expert is not None:
                mapped, is_gate_up = packed_expert
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped not in params_dict:
                    raise ValueError(
                        f"HY4 packed expert tensor has no destination: {name} "
                        f"-> {mapped}"
                    )
                param = params_dict[mapped]
                if is_gate_up:
                    gate, up = loaded_weight.chunk(2, dim=-2)
                    loaded_gate = self._load_all_experts(
                        mapped,
                        param,
                        gate,
                        "w1",
                        self.config.n_routed_experts,
                    )
                    loaded_up = self._load_all_experts(
                        mapped,
                        param,
                        up,
                        "w3",
                        self.config.n_routed_experts,
                    )
                    loaded = loaded_gate or loaded_up
                else:
                    loaded = self._load_all_experts(
                        mapped,
                        param,
                        loaded_weight,
                        "w2",
                        self.config.n_routed_experts,
                    )
                if not loaded:
                    raise ValueError(f"No local HY4 expert accepted {name}")
                loaded_params.add(mapped)
                continue

            mapped_stacked = False
            for param_name, weight_name, shard_id in stacked_mapping:
                if weight_name not in name or "mlp.experts" in name:
                    continue
                if weight_name in ("wk", "weights_proj") and ".indexer." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped, self):
                    mapped_stacked = True
                    break
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                mapped_stacked = True
                break
            if mapped_stacked:
                continue

            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                raise ValueError(f"Unexpected HY4 checkpoint parameter: {name}")
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
            loaded_params.add(name)

        if pending_wk_mxfp8 or pending_wk_fp8:
            raise ValueError(
                "HY4 checkpoint coverage failed: incomplete FP8/MXFP8 indexer "
                "wk weight/scale pairs for "
                f"{sorted(set(pending_wk_mxfp8) | set(pending_wk_fp8))}"
            )

        runtime_attention_scales = (
            ".self_attn.attn.q_scale",
            ".self_attn.attn.k_scale",
            ".self_attn.attn.v_scale",
            ".self_attn.attn.prob_scale",
            ".self_attn.mla_attn.q_scale",
            ".self_attn.mla_attn.k_scale",
            ".self_attn.mla_attn.v_scale",
            ".self_attn.mla_attn.prob_scale",
        )
        missing = sorted(
            name
            for name in set(params_dict).difference(loaded_params)
            if not name.endswith(runtime_attention_scales)
        )
        if missing:
            preview = ", ".join(missing[:20])
            suffix = " ..." if len(missing) > 20 else ""
            raise ValueError(
                f"HY4 checkpoint coverage failed: {len(missing)} model "
                f"parameters were not loaded: {preview}{suffix}"
            )
        return loaded_params


__all__ = ["HYV4ForCausalLM"]
