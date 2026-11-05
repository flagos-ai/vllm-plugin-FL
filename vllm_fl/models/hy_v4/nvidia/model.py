# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hy4 text model, iHC boundaries, MoE and multi-token prediction.

Model structure and checkpoint loading share one module. Numerical providers
remain in the operator libraries and are selected through the fusion adapters.
"""

import copy
import typing
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from itertools import islice

import regex as re
import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    get_pp_missing_layer_names,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from ..config import _extend_layer_types, _normalize_hyv4_config
from ..fusions import maybe_ihc_pair_pre, triton_ihc_post, triton_ihc_pre
from ..indexer_policy import compute_skip_topk_layers, is_skip_topk_indexer_weight
from ..model_weights import (
    _create_mtp_quant_config,
    _get_spec_layer_idx_from_weight_name,
    _prepare_mtp_fp8_expert_scale,
    _remap_mtp_quant_exclusions,
    _resolve_fused_expert_param,
    _should_skip_missing_mtp_scale_param,
    hyv4_packed_modules_mapping,
)
from .attention import HYV4MLAAttention
from .fusions import maybe_ihc_post_pre, triton_ihc_supported
from vllm_fl.models.hy_v4.hpc_ihc import HpcIHCHead, HpcIHCPost, HpcIHCPre
from vllm_fl.utils import use_flaggems_op

logger = init_logger(__name__)


# Independent hyper-connections


class HYV4HCPreLayer(nn.Module):
    """iHC pre-processing layer (2D-activation adaptation).

    Steps:
        1. RMS-normalize the flattened ``[num_tokens, hc * d]`` input.
        2. Project to the pre/post gating logits.
        3. Turn the logits into sigmoid gates.
        4. Reduce over the channel dim with the pre gates.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        hidden_dim: int,
        hc_mult: int = 4,
        magnitude: float = 2.0,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        hc_eps: float = 1e-6,
        layernorm_epsilon: float = 1e-5,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.hc_mult = hc_mult
        self.magnitude = magnitude
        self.hc_eps = hc_eps
        self.layernorm_epsilon = layernorm_epsilon
        hc_in_dim = hc_mult * hidden_dim
        mix_hc = 2 * hc_mult  # pre + post only, no comb

        self.hc_fn = ReplicatedLinear(
            input_size=hc_in_dim,
            output_size=mix_hc,
            params_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.hc_fn",
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))

        self.reset_parameters(init_std, base_noise_std)

        # Optional single-kernel HPC replacement for the whole forward below.
        # ``norm_owner`` is left unset: the RMSNorm that follows the pre block
        # lives in the decoder layer and is not folded in yet.
        self.hpc_op: HpcIHCPre | None = None
        if HpcIHCPre.support(hc_mult, hidden_dim):
            self.hpc_op = HpcIHCPre(
                hc_mult=hc_mult,
                hidden_size=hidden_dim,
                magnitude=magnitude,
                hc_eps=hc_eps,
                norm_eps=layernorm_epsilon,
                fallback_op=self,
            )

    def reset_parameters(self, init_std: float, base_noise_std: float = 0.0) -> None:
        """Initialize the gate scale and per-channel gate bias."""
        del init_std  # hc_fn is initialized by ReplicatedLinear
        nn.init.constant_(self.hc_scale, 0.01)
        with torch.no_grad():
            self.hc_base[: self.hc_mult].fill_(
                -torch.log(torch.tensor(self.hc_mult - 1.0, dtype=self.hc_base.dtype))
            )
            self.hc_base[self.hc_mult : 2 * self.hc_mult].fill_(0.0)
            if base_noise_std > 0.0:
                self.hc_base.add_(torch.randn_like(self.hc_base) * base_noise_std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce the iHC channels and emit the post gates.

        Args:
            x: Input of shape ``[num_tokens, hc, d]``.

        Returns:
            A tuple of the pre-gated reduction ``[num_tokens, d]`` and the post
            gates ``[num_tokens, hc]`` consumed by `HYV4HCPostLayer`.

        """
        if self.hpc_op is not None:
            return self.hpc_op(x)
        if triton_ihc_supported(x):
            pair_result = maybe_ihc_pair_pre(self, x)
            if pair_result is not None:
                return pair_result
            if use_flaggems_op("ihc_pre"):
                return triton_ihc_pre(
                    x,
                    self.hc_fn.weight,
                    self.hc_scale,
                    self.hc_base,
                    self.magnitude,
                    self.hc_eps,
                    self.layernorm_epsilon,
                )

        shape = x.size()  # [num_tokens, hc, d]
        hc = self.hc_mult
        hc_eps = self.hc_eps

        x_flat = x.flatten(1).float()  # [num_tokens, hc*d]

        rsqrt = torch.rsqrt(
            x_flat.square().mean(-1, keepdim=True) + self.layernorm_epsilon
        )
        mixes = self.hc_fn(x_flat)[0] * rsqrt  # [num_tokens, 2*hc]

        pre_raw = mixes[..., :hc]
        post_raw = mixes[..., hc : 2 * hc]

        pre = (
            torch.sigmoid(
                pre_raw * self.hc_scale[0].float() + self.hc_base[:hc].float()
            )
            + hc_eps
        )
        post = (
            self.magnitude
            * torch.sigmoid(
                post_raw * self.hc_scale[1].float() + self.hc_base[hc : 2 * hc].float()
            )
            + hc_eps
        )

        y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)  # [num_tokens, d]
        return y.to(x.dtype), post


class HYV4HCPostLayer(nn.Module):
    """iHC post-processing layer (2D-activation adaptation).

    Applies post-gating to the sub-block output and adds the multi-channel
    residual (no comb mixing)::

        y[n, i, d] = post[n, i] * x[n, d] + residual[n, i, d]
    """

    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.config = config

        # Optional single-kernel HPC replacement for the whole forward below.
        # HYV4HCLayer constructs this boundary only when iHC is enabled.
        self.hpc_op: HpcIHCPost | None = None
        hc_mult = config.hc_mult
        if HpcIHCPost.support(hc_mult, config.hidden_size):
            self.hpc_op = HpcIHCPost(hc_mult=hc_mult, hidden_size=config.hidden_size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
    ) -> torch.Tensor:
        """Scatter the sub-block output back onto the iHC channels.

        Args:
            x: Attention/MLP output of shape ``[num_tokens, d]``.
            residual: Multi-channel residual ``[num_tokens, hc, d]``.
            post: Post gates ``[num_tokens, hc]`` from `HYV4HCPreLayer`.

        Returns:
            The updated residual channels ``[num_tokens, hc, d]``.

        """
        if self.hpc_op is not None:
            return self.hpc_op(x, residual, post)
        if triton_ihc_supported(x) and use_flaggems_op("ihc_post"):
            return triton_ihc_post(x, residual, post)

        dtype = x.dtype
        x = x.float()
        residual = residual.float()
        post = post.float()

        post_gated = post.unsqueeze(-1) * x.unsqueeze(-2)  # [num_tokens, hc, d]
        y = post_gated + residual
        return y.to(dtype)


class HYV4HCHeadLayer(nn.Module):
    """iHC head layer (2D-activation adaptation).

    Merges the iHC channels back into a single hidden state before the final
    layer norm, using an RMS-normed projection plus sigmoid-gated reduction.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        hidden_size: int,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_head_fn = ReplicatedLinear(
            input_size=hc_mult * hidden_size,
            output_size=hc_mult,
            params_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.hc_head_fn",
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.reset_parameters(init_std, base_noise_std)

        # Optional single-kernel HPC replacement for the whole forward below.
        self.hpc_op: HpcIHCHead | None = None
        if HpcIHCHead.support(hc_mult, hidden_size):
            self.hpc_op = HpcIHCHead(
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                hc_eps=hc_eps,
                norm_eps=config.rms_norm_eps,
                fallback_op=self,
            )

    def reset_parameters(
        self, init_std: float = 6e-3, base_noise_std: float = 0.0
    ) -> None:
        """Initialize the head gate scale and per-channel gate bias."""
        del init_std  # hc_head_fn is initialized by ReplicatedLinear
        nn.init.constant_(self.hc_head_scale, 0.01)
        with torch.no_grad():
            self.hc_head_base.fill_(
                -torch.log(
                    torch.tensor(self.hc_mult - 1.0, dtype=self.hc_head_base.dtype)
                )
            )
            if base_noise_std > 0.0:
                self.hc_head_base.add_(
                    torch.randn_like(self.hc_head_base) * base_noise_std
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge the iHC channels into a single hidden state.

        Args:
            x: Input of shape ``[num_tokens, hc, d]``.

        Returns:
            The merged hidden state ``[num_tokens, d]``.

        """
        if self.hpc_op is not None:
            return self.hpc_op(x)

        shape, x_dtype = x.size(), x.dtype

        x = x.flatten(1).float()  # [num_tokens, hc*d]
        rsqrt = torch.rsqrt(
            x.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        mixes = self.hc_head_fn(x)[0] * rsqrt  # [num_tokens, hc]
        pre = (
            torch.sigmoid(
                mixes * self.hc_head_scale.float() + self.hc_head_base.float()
            )
            + self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)  # [num_tokens, d]
        return y.to(x_dtype)


class HYV4HCLayer(nn.Module):
    """Wrapper owning one iHC boundary (pre + post) of a decoder sub-block."""

    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_ihc = config.enable_ihc
        if self.enable_ihc:
            self.hc_pre = HYV4HCPreLayer(
                config,
                config.hidden_size,
                config.hc_mult,
                config.hc_magnitude,
                init_std,
                base_noise_std,
                config.hc_eps,
                config.rms_norm_eps,
                prefix=f"{prefix}.hc_pre",
            )
            self.hc_post = HYV4HCPostLayer(config)

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the sub-block input to 3D when iHC is enabled."""
        if not self.enable_ihc:
            return hidden_states
        return self._prepare_input_to_3d(hidden_states)

    def _prepare_input_to_3d(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reshape the iHC input to ``[num_tokens, hc, h]``.

        Accepted inputs are ``[num_tokens, hc, h]`` (no-op), ``[num_tokens, h]``
        (broadcast over the channels) and ``[num_tokens, hc * h]`` (reshape).
        """
        if hidden_states.dim() == 3:
            return hidden_states
        if hidden_states.dim() != 2:
            raise RuntimeError(
                f"HC expects a 2D/3D tensor, got shape={tuple(hidden_states.shape)}"
            )

        d0, d1 = hidden_states.shape
        h = self.config.hidden_size
        hc = self.config.hc_mult

        if d1 == h:
            return hidden_states.unsqueeze(1).repeat(1, hc, 1)

        expected = hc * h
        if d1 == expected:
            return hidden_states.reshape(d0, hc, h)

        raise RuntimeError(
            f"HC expects last dim to be hidden_size ({h})"
            f"or hc_mult*hidden_size ({expected}), got {d1}. "
            f"hc_mult={hc}, hidden_size={h}."
        )

    def pre(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Reduce the iHC channels and produce the post gates.

        Returns:
            A tuple of the reduced hidden states ``[num_tokens, d]``, the post
            gates ``[num_tokens, hc]`` (``None`` when iHC is disabled) and the
            residual (the untouched input).

        """
        if not self.enable_ihc:
            return hidden_states, None, hidden_states
        reduced, post_gates = self.hc_pre(hidden_states)
        return reduced, post_gates, hidden_states

    def post(
        self,
        output_with_bias: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply post-gating and add the residual."""
        if not self.enable_ihc:
            return output_with_bias + residual
        assert post_gates is not None
        return self.hc_post(output_with_bias, residual, post_gates)


# Dense and routed experts


class HYV4FeedForward(nn.Module):
    """Dense SwiGLU feed-forward block.

    Used both for the dense decoder layers and for the MoE shared experts.
    The routed-expert SwiGLU clamp does not apply here, matching the reference
    implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        return out


class HYV4MoEFused(nn.Module):
    """HY V4 MoE layer with optional SwiGLU clamp support.

    When ``config.swiglu_limit > 0`` the routed experts use a clamped SwiGLU::

        gate = clamp(gate, max=limit)
        up = clamp(up, -limit, limit)
        output = silu(gate) * up

    Dense layers and shared experts are NOT clamped. The clamp is forwarded to
    ``FusedMoEFactory`` as ``swiglu_limit`` so it travels through the quant
    config to every expert backend instead of mutating global state.

    NOTE: The reference implementation additionally supports HPC gate kernels,
    TPCP/MLPSP sharded shared experts and a side-stream overlap for the shared
    experts. Those depend on infrastructure that is absent here, so this port
    keeps only the standard path. TODO: restore once available.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        top_k = config.num_experts_per_tok
        intermediate_size = config.expert_hidden_dim
        router_scaling_factor = config.router_scaling_factor
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts: HYV4FeedForward | None
        if config.num_shared_experts > 0:
            self.shared_experts = HYV4FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_hidden_dim * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
            )
        else:
            self.shared_experts = None

        self.expert_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )

        raw_swiglu_limit = config.swiglu_limit
        use_swiglu_clamp = bool(raw_swiglu_limit) and float(raw_swiglu_limit) > 0
        moe_swiglu_limit = float(raw_swiglu_limit) if use_swiglu_clamp else None

        self.experts = FusedMoEFactory(
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            renormalize=config.route_norm,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            scoring_func="sigmoid",
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=router_scaling_factor,
            e_score_correction_bias=self.expert_bias,
            shared_experts=self.shared_experts,
            swiglu_limit=moe_swiglu_limit,
        )
        self.prefix = prefix
        if use_swiglu_clamp:
            logger.debug_once(
                "HYV4MoEFused: swiglu_limit=%.1f enabled for routed experts",
                moe_swiglu_limit,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts); the gate stores fp32 weights
        # and emits fp32 logits, so no explicit upcast is needed here.
        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(orig_shape)


# Target transformer and checkpoint loading


class HYV4DecoderLayer(nn.Module):
    """One HY V4 decoder layer: MLA attention plus a dense or MoE MLP.

    When``config.enable_ihc`` is set the layer runs on ``hc_mult`` residual
    channels and each sub-block is wrapped by an `HYV4HCLayer` boundary;
    otherwise it uses the standard single-stream residual.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        vllm_config: VllmConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.enable_ihc = config.enable_ihc
        layer_idx = int(prefix.split(".")[-1])
        self.layer_idx = layer_idx

        max_position_embeddings = config.max_position_embeddings
        self.self_attn = HYV4MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            layer_idx=layer_idx,
            topk_indices_buffer=topk_indices_buffer,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.mlp_layer_types[layer_idx] == "dense":
            self.mlp = HYV4FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.block_type = "feedforward"
        else:
            self.mlp = HYV4MoEFused(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
            self.block_type = "moe"
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_attn_layer = HYV4HCLayer(
            config, layer_idx, prefix=f"{prefix}.hc_attn_layer"
        )
        self.hc_mlp_layer = HYV4HCLayer(
            config, layer_idx, prefix=f"{prefix}.hc_mlp_layer"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.enable_ihc:
            return self._forward_ihc(positions, hidden_states)
        return self._forward_normal(positions, hidden_states, residual)

    def _forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard single-residual-stream forward (iHC disabled)."""
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def _forward_ihc(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """IHC forward: each sub-block reduces and re-scatters the channels."""
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        post_pre = maybe_ihc_post_pre(self, hidden_states, residual, post_gates)
        if post_pre is None:
            hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)

            hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
            hidden_states, post_gates, residual = self.hc_mlp_layer.pre(hidden_states)
        else:
            hidden_states, post_gates, residual = post_pre
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post_gates)

        # Under iHC the residual is carried inside hidden_states.
        return hidden_states, None


class HYV4Model(nn.Module):
    """HY V4 backbone."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = _normalize_hyv4_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )
        self.config = config
        self.quant_config = quant_config
        self.enable_ihc = config.enable_ihc

        if get_pp_group().is_first_rank or (
            self.config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HYV4DecoderLayer(
                config=config,
                vllm_config=vllm_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        if self.enable_ihc:
            # iHC head layer: merge the residual channels into one stream.
            if get_pp_group().is_last_rank:
                self.hc_head = HYV4HCHeadLayer(
                    config,
                    hidden_size=config.hidden_size,
                    hc_mult=config.hc_mult,
                    hc_eps=config.hc_eps,
                    prefix=f"{prefix}.hc_head",
                )
            else:
                self.hc_head = PPMissingLayer()
            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states"], config.hc_mult * config.hidden_size
                )
            )
        else:
            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states", "residual"], config.hidden_size
                )
            )

        # MoE hyperparameters (consumed by EPLB).
        self._cached_expert_params_mapping: list[tuple[str, str, int, str]] | None = (
            None
        )
        self.expert_weights: MutableSequence[Sequence[torch.Tensor]] = []
        self.num_expert_groups = 1
        self.moe_layers: list[nn.Module] = []
        example_layer: HYV4MoEFused | None = None
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, HYV4DecoderLayer)
            if layer.block_type == "moe":
                assert isinstance(layer.mlp, HYV4MoEFused)
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            self.num_moe_layers = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_moe_layers = len(self.moe_layers)
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, HYV4MoEFused):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id) for weights, fp8
        # weight scales and fp8 activation scales.
        if self._cached_expert_params_mapping is None:
            self._cached_expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
                num_redundant_experts=self.num_redundant_experts,
            )
        return self._cached_expert_params_mapping

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            # In iHC mode the flattened [num_tokens, hc*h] tensor from the
            # previous PP stage is reshaped back to 3D by the first layer's
            # prepare_input, and residual is unused.
            residual = None if self.enable_ihc else intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            if self.enable_ihc:
                # hidden_states is [num_tokens, hc, h]; flatten the channel dim
                # for PP transfer (matches the 2D receive buffer).
                return IntermediateTensors({"hidden_states": hidden_states.flatten(1)})
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if self.enable_ihc:
            hidden_states = self.hc_head(hidden_states)
        else:
            hidden_states = hidden_states + residual

        return self.norm(hidden_states)

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # FP8 indexer wk dequant buffer (weight and scale arrive separately).
        pending_wk_fp8: dict[str, dict[str, torch.Tensor]] = {}
        pp_missing_layer_names = get_pp_missing_layer_names(self)
        skip_topk_layers = compute_skip_topk_layers(self.config)

        # Must not be cached on `self`: `process_weights_after_loading` swaps in
        # kernel-specific expert weights via `replace_parameter`, and a cache
        # outliving this call would pin the pre-shuffle storage (OOM on large
        # MoE) and make a later reload target orphaned tensors.
        params_dict = dict(self.named_parameters())
        # Split per-expert mapping (V3 style): experts.0.gate_proj.weight
        split_expert_params_mapping = self.get_expert_mapping()
        loaded_params: set[str] = set()

        # Sink weights are sharded like the q/k/v linears.
        sink_tp_size = get_tensor_model_parallel_world_size()
        sink_tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // sink_tp_size
        head_rank_start = n_local_head * sink_tp_rank
        head_rank_end = n_local_head * (sink_tp_rank + 1)
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        # Fused expert mapping: experts.gate_up_proj (all experts in one tensor).
        # The packed weights are owned by the RoutedExperts submodule, so the
        # targets are experts.routed_experts.[base_layer.]w{13,2}_weight.
        fused_expert_prefix = f".experts.routed_experts.{base_layer}"
        fused_expert_params_mapping = [
            (f"{fused_expert_prefix}w13_weight", ".experts.gate_up_proj", 0, "w1"),
            (f"{fused_expert_prefix}w2_weight", ".experts.down_proj", 0, "w2"),
        ]
        num_experts = self.config.num_experts

        def _should_skip_missing_param(param_name: str) -> bool:
            # Sparse checkpoints may contain indexer weights for layers that
            # fall back to dense MLA on unsupported GPUs.
            return ".indexer." in param_name

        def _is_split_expert_weight(weight_name: str) -> bool:
            """Whether this weight is in split (per-expert) format."""
            # Split format: mlp.experts.<id>.gate_proj.weight
            return bool(re.search(r"\.experts\.\d+\.", weight_name))

        def _is_fused_expert_weight(weight_name: str) -> bool:
            """Whether this weight is in fused (all-experts-packed) format."""
            return ".experts.gate_up_proj" in weight_name or (
                ".experts.down_proj" in weight_name
                and not _is_split_expert_weight(weight_name)
            )

        for name, loaded_weight in weights:
            if is_skip_topk_indexer_weight(name, skip_topk_layers):
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
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Names and stacked shard ids are normalized upstream by
            # AutoWeightsLoader, through quant_config.get_cache_scale_mapper()
            # and hf_to_vllm_mapper.
            stacked_shard_id = getattr(loaded_weight, "shard_id", None)
            if ".indexer.wk." in name:
                name = name.replace(".indexer.wk.", ".indexer.wk_weights_proj.")
                stacked_shard_id = 0
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            # Determine per-weight whether this is fused or split format.
            is_fused_expert = _is_fused_expert_weight(name)
            expert_params_mapping = (
                fused_expert_params_mapping
                if is_fused_expert
                else split_expert_params_mapping
            )

            is_expert_weight = False
            loaded_expert_param_names: set[str] = set()
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue

                # This is an expert weight and must not be attempted as any
                # other kind of weight later on.
                is_expert_weight = True

                # Do not modify `name`: the loop may continue past this point.
                name_mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if is_fused_expert:
                    if "experts.gate_up_proj" in name:
                        chunks = loaded_weight.chunk(2, dim=-2)
                        success_w1 = self.load_fused_expert_weights(
                            name_mapped, params_dict, chunks[0], "w1", num_experts
                        )
                        success_w3 = self.load_fused_expert_weights(
                            name_mapped, params_dict, chunks[1], "w3", num_experts
                        )
                        success = success_w1 and success_w3
                    else:
                        success = self.load_fused_expert_weights(
                            name_mapped,
                            params_dict,
                            loaded_weight,
                            shard_id,
                            num_experts,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    # Split per-expert format (V3 style).
                    if name_mapped not in params_dict:
                        if _should_skip_missing_param(name_mapped):
                            continue
                        logger.warning_once(
                            "Skipping unknown checkpoint weight: %s",
                            name_mapped,
                        )
                        continue
                    param = params_dict[name_mapped]
                    # Ask the weight loader whether it succeeded, otherwise we
                    # may skip experts that have other available replicas.
                    weight_loader: Callable[..., typing.Any] = param.weight_loader
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                if success:
                    if not is_fused_expert:
                        loaded_expert_param_names.add(name_mapped)
                        continue
                    name = name_mapped
                    break
            else:
                if loaded_expert_param_names:
                    loaded_params.update(loaded_expert_param_names)
                    continue
                if "learnable_sink_param" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    with torch.no_grad():
                        params_dict[name][:n].copy_(narrow_weight)
                else:
                    if is_expert_weight:
                        # An expert weight that is not mapped to this rank.
                        continue
                    if "gate.e_score_correction_bias" in name:
                        name = name.replace(
                            "gate.e_score_correction_bias", "expert_bias"
                        )
                    if "router.gate." in name:
                        name = name.replace("router.", "")
                    if "hc_fn" in name:
                        name = name.replace("hc_fn", "hc_fn.weight")
                    if "hc_head_fn" in name:
                        name = name.replace("hc_head_fn", "hc_head_fn.weight")

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        if _should_skip_missing_param(name):
                            continue
                        logger.warning_once(
                            "Skipping unknown checkpoint weight: %s",
                            name,
                        )
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    if stacked_shard_id is None:
                        weight_loader(param, loaded_weight)
                    else:
                        weight_loader(param, loaded_weight, stacked_shard_id)
            loaded_params.add(name)

        return loaded_params


class HYV4ForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
            ".indexer.weights_proj.": (".indexer.wk_weights_proj.", 1),
        }
    )
    packed_modules_mapping = hyv4_packed_modules_mapping()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if quant_config is not None:
            quant_config.packed_modules_mapping = self.packed_modules_mapping

        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.model = HYV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            # The head stays in the model dtype; ``enable_lm_head_fp32`` is
            # surfaced as ``head_dtype`` on the config (see HYV4Config), which
            # makes LogitsProcessor accumulate the projection straight into
            # fp32 instead of materializing an fp32 copy of the weight.
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            # With tie_word_embeddings, embed_tokens is kept on the last rank
            # (see HYV4Model.__init__) so the weight can be shared here.
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)

        if getattr(self.config, "soft_logits_capping", False):
            soft_cap = self.config.soft_logits_capping_logits
            logits = soft_cap * torch.nn.functional.tanh(logits / soft_cap)

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # MTP layers are loaded by the MTP head, not by the target model. They
        # appear as model.mtp_layers.<i>.* and as the layer ids past the backbone.
        mtp_start = self.config.num_hidden_layers
        drop_prefixes: dict[str, str | None] = {"model.mtp_layers.": None}
        for i in range(self.config.num_nextn_predict_layers):
            drop_prefixes[f"model.layers.{mtp_start + i}."] = None
        if self.config.tie_word_embeddings:
            drop_prefixes["lm_head."] = None
        drop_mapper = WeightsMapper(orig_to_new_prefix=drop_prefixes)
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper | drop_mapper)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


# Multi-token prediction and checkpoint mapping


class HYV4SharedHead(nn.Module):
    """Holds the draft LM head shared with the target model."""

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        # Use the same ``lm_head`` prefix as the backbone so that quant-config
        # exclusion lists (e.g. ModelOpt's ``exclude_modules: ["lm_head", ...]``)
        # also apply here: an unquantized checkpoint lm_head must not gain a
        # spurious ``weight_scale`` parameter on the draft side, which would
        # otherwise stay at its FP-min sentinel until the proposer swaps the
        # whole ParallelLMHead for the target model's ``lm_head``.
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix="lm_head",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class HYV4MultiTokenPredictorLayer(nn.Module):
    """A single MTP draft block."""

    def __init__(
        self,
        config: PreTrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        del model_config  # kept for signature parity with the other MTP heads
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.shared_head = HYV4SharedHead(config=config, quant_config=quant_config)

        # HYV4DecoderLayer indexes layer_types/mlp_layer_types by the numeric
        # layer id parsed from the prefix. MTP layers live after the backbone
        # layers, so extend the per-layer config lists to cover those ids.
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        mtp_config = copy.deepcopy(config)
        # MTP checkpoints do not carry iHC pre/post weights for mtp_layers, so
        # the draft blocks are built without iHC to match their structure.
        mtp_config.enable_ihc = False
        mtp_config.layer_types = _extend_layer_types(
            mtp_config.layer_types, layer_idx, "full_attention"
        )
        mtp_config.mlp_layer_types = _extend_layer_types(
            mtp_config.mlp_layer_types, layer_idx, "sparse"
        )

        self.mtp_block = HYV4DecoderLayer(
            config=mtp_config,
            vllm_config=vllm_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids  # embeddings are resolved by the caller
        assert inputs_embeds is not None
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states, _ = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class HYV4MultiTokenPredictor(nn.Module):
    """Owns the MTP draft blocks and their shared embedding / logits path."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        target_config = vllm_config.model_config.hf_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None, (
            "HYV4MTP can only be built as a speculative draft model"
        )
        draft_model_config = speculative_config.draft_model_config
        assert draft_model_config is not None
        config = _normalize_hyv4_config(draft_model_config.hf_config)
        self.mtp_start_layer_idx = target_config.num_hidden_layers
        self.num_mtp_layers = max(target_config.num_nextn_predict_layers, 1)
        self.quant_config = _remap_mtp_quant_exclusions(
            _create_mtp_quant_config(config, vllm_config.quant_config),
            self.mtp_start_layer_idx,
            self.num_mtp_layers,
        )
        if self.quant_config is not None:
            self.quant_config.packed_modules_mapping = HYV4MTP.packed_modules_mapping

        # Sparse DSA draft blocks must share the target model's top-k buffer.
        # The proposer injects it through HYV4MTP.set_topk_indices_buffer()
        # before any draft forward.
        self.topk_indices_buffer: torch.Tensor | None = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        self.layers = nn.ModuleDict(
            {
                str(idx): HYV4MultiTokenPredictorLayer(
                    config,
                    f"{prefix}.layers.{idx}",
                    vllm_config=vllm_config,
                    model_config=draft_model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=self.quant_config,
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.requires_topk_indices_buffer = any(
            layer.mtp_block.self_attn.is_sparse for layer in self.layers.values()
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.spec_step_idx: int = 0

    def set_skip_topk(self, skip: bool) -> None:
        """Toggle the draft indexer for ``index_share_for_mtp_iteration``.

        The proposer clears the flag for draft step 0 so the MTP layer builds
        its own top-k indices, then sets it for steps 1+ so they reuse what
        step 0 wrote into the shared buffer instead of re-running the indexer.
        """
        for layer in self.layers.values():
            layer.mtp_block.self_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        """Move the top-k rows at ``slot_ids`` to the front of the buffer.

        Step 0 writes one row per query token of the multi-token batch, while
        steps 1+ decode a single token per request and index the buffer from 0.
        Without this gather they would read another token's rows.
        """
        num_slots = slot_ids.numel()
        if self.topk_indices_buffer is None or num_slots == 0:
            return
        self.topk_indices_buffer[:num_slots] = self.topk_indices_buffer[slot_ids]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        token_positions = positions[0] if positions.ndim == 2 else positions
        inputs_embeds = torch.where(
            (token_positions == 0).unsqueeze(-1), 0, inputs_embeds
        )

        current_step_idx = self.spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_step_idx = self.spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        lm_head = mtp_layer.shared_head.head
        proj_input = mtp_layer.shared_head(hidden_states)
        return self.logits_processor(lm_head, proj_input)


class HYV4MTP(nn.Module):
    """HY V4 MTP draft head.

    Not a pipeline-parallel stage: the draft head always runs on a single rank,
    matching `DeepseekV32MTP` / `KimiK3MTP` / `HYV3MTP`.
    """

    packed_modules_mapping = hyv4_packed_modules_mapping()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = HYV4MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.quant_config = self.model.quant_config
        self.sampler = Sampler()

    def set_topk_indices_buffer(self, topk_indices_buffer: torch.Tensor) -> None:
        """Share the target sparse-index buffer with every draft consumer.

        Proposers that walk ``named_modules()`` instead of calling this reach
        the same consumers via ``HYV4MLAAttention.topk_indices_buffer``.
        """
        self.model.topk_indices_buffer = topk_indices_buffer
        for layer in self.model.layers.values():
            self_attn = layer.mtp_block.self_attn
            if not self_attn.is_sparse:
                continue

            indexer = self_attn.indexer
            assert indexer is not None, "Sparse HYV4 MTP attention requires an indexer"
            indexer.topk_indices_buffer = topk_indices_buffer
            indexer.indexer_op.topk_indices_buffer = topk_indices_buffer

            attn_impl = self_attn.mla_attn.impl
            attn_impl.topk_indices_buffer = topk_indices_buffer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        del intermediate_tensors  # the MTP head is single-stage
        if (
            self.model.requires_topk_indices_buffer
            and self.model.topk_indices_buffer is None
        ):
            raise RuntimeError(
                "HYV4 sparse MTP requires the target model's top-k indices buffer. "
                "The proposer must call HYV4MTP.set_topk_indices_buffer() before "
                "the first draft forward."
            )
        self.model.spec_step_idx = spec_step_idx
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        self.model.spec_step_idx = spec_step_idx
        return self.model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        if f"model.layers.{spec_layer}.embed_tokens" in name:
            return "__skip__"
        if f"model.layers.{spec_layer}.shared_head" in name:
            return "__skip__"

        spec_layer_weight_names = ["enorm", "hnorm", "eh_proj", "final_layernorm"]
        spec_layer_weight = any(
            weight_name in name for weight_name in spec_layer_weight_names
        )
        if not spec_layer_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.",
                f"model.layers.{spec_layer}.mtp_block.",
            )
        return name

    def _load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        if name not in params_dict:
            return False
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True
        return loaded_local_expert

    def _load_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        loaded_params: set[str],
        split_expert_params_mapping: list[tuple[str, str, int, str]],
        fused_expert_param_names: dict[tuple[str, str], str],
        num_experts: int,
    ) -> bool:
        """Load one routed-expert weight in either checkpoint layout.

        Args:
            name: Weight name already rewritten to draft-module naming.
            loaded_weight: The checkpoint tensor.
            params_dict: The draft model's named parameters.
            loaded_params: Set updated with the parameters that received a value.
            split_expert_params_mapping: Mapping for the per-expert layout.
            fused_expert_param_names: ``(mlp_prefix, tag) -> param name`` for the
                all-experts-packed layout.
            num_experts: Total number of routed experts.

        Returns:
            True when the weight was consumed (even if this rank holds none of
            the addressed experts).

        """
        base = name.split(".experts.")[0]
        for ckpt_proj, tag in (
            (".experts.gate_up_proj", "w13_weight"),
            (".experts.down_proj", "w2_weight"),
        ):
            if ckpt_proj not in name:
                continue
            param_base = fused_expert_param_names.get((base, tag))
            if param_base is None:
                return False
            # Keep the checkpoint suffix (e.g. `_scale_inv`) so block-scale
            # tensors land in the scale parameter, not in the weight.
            target = _resolve_fused_expert_param(
                param_base, name.split(ckpt_proj, 1)[1], params_dict
            )
            if target is None:
                return False
            if tag == "w13_weight":
                chunks = loaded_weight.chunk(2, dim=-2)
                loaded_w1 = self._load_fused_expert_weights(
                    target, params_dict, chunks[0], "w1", num_experts
                )
                loaded_w3 = self._load_fused_expert_weights(
                    target, params_dict, chunks[1], "w3", num_experts
                )
                loaded = loaded_w1 and loaded_w3
            else:
                loaded = self._load_fused_expert_weights(
                    target, params_dict, loaded_weight, "w2", num_experts
                )
            if loaded:
                loaded_params.add(target)
            # The weight belongs to the experts either way; never fall through.
            return True

        consumed = False
        for param_name, weight_name, expert_id, shard_id in split_expert_params_mapping:
            if weight_name not in name:
                continue
            consumed = True
            name_mapped = name.replace(weight_name, param_name)
            if name_mapped not in params_dict:
                continue
            param = params_dict[name_mapped]
            weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
            if weight_loader(
                param,
                loaded_weight,
                name_mapped,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            ):
                loaded_params.add(name_mapped)
        return consumed

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        pp_missing_layer_names = get_pp_missing_layer_names(self)
        loaded_params: set[str] = set()

        mtp_start = self.config.num_hidden_layers
        shared_weights = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": f"model.layers.{mtp_start}.shared_head.head.weight",
        }

        num_experts = self.config.n_routed_experts
        sink_tp_size = get_tensor_model_parallel_world_size()
        sink_tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // sink_tp_size
        head_rank_start = n_local_head * sink_tp_rank
        head_rank_end = n_local_head * (sink_tp_rank + 1)

        # Routed-expert weights come in two checkpoint layouts:
        #   split: mlp.experts.<id>.gate_proj.weight
        #   fused: mlp.experts.gate_up_proj (all experts in one tensor)
        # The split layout is resolved through the shared EPLB helper so the
        # target param names stay in sync with the RoutedExperts module layout;
        # the fused layout is resolved from the live parameter names.
        split_expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )
        fused_expert_param_names: dict[tuple[str, str], str] = {}
        for param_name in params_dict:
            for tag in ("w13_weight", "w2_weight"):
                if param_name.endswith(tag) and ".experts." in param_name:
                    base = param_name.split(".experts.")[0]
                    fused_expert_param_names[base, tag] = param_name
        stacked_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA runs both latent down-projections as one GEMM.
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
        ]
        # Sparse DSA draft blocks build an Indexer whose wk / weights_proj are
        # fused into a single MergedColumnParallelLinear (wk_weights_proj).
        indexer_stacked_mapping = [
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
        ]
        # FP8 indexer wk dequant buffer (weight and scale arrive separately).
        pending_wk_fp8: dict[str, dict[str, torch.Tensor]] = {}

        for name, loaded_weight in weights:
            if name in shared_weights:
                target_name = shared_weights[name]
                if target_name in params_dict:
                    param = params_dict[target_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(target_name)
                continue

            spec_layer = None
            if name.startswith("model.mtp_layers."):
                parts = name.split(".")
                if len(parts) > 3 and parts[2].isdigit():
                    spec_layer = mtp_start + int(parts[2])
                    name = name.replace(
                        f"model.mtp_layers.{parts[2]}.",
                        f"model.layers.{spec_layer}.",
                    )
            else:
                spec_layer = _get_spec_layer_idx_from_weight_name(self.config, name)

            if spec_layer is None:
                continue

            name = self._rewrite_spec_layer_name(spec_layer, name)
            if name == "__skip__":
                continue
            name, loaded_weight = _prepare_mtp_fp8_expert_scale(
                self.quant_config, name, loaded_weight
            )

            if "mlp.gate.e_score_correction_bias" in name:
                name = name.replace("gate.e_score_correction_bias", "expert_bias")

            is_loaded = False
            for param_name, weight_name, shard_id in stacked_mapping:
                if weight_name not in name or ".experts." in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    if is_pp_missing_parameter(name_mapped, self):
                        is_loaded = True
                    break
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                is_loaded = True
                break
            if is_loaded:
                continue

            # FP8 indexer wk: dequantize to BF16 and load into the fused
            # wk_weights_proj. PP-aware (skips layers not held by this rank).
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                pending_wk_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue

            # BF16 indexer wk / weights_proj: merge into the fused param.
            is_loaded = False
            for param_name, weight_name, shard_id in indexer_stacked_mapping:
                if weight_name not in name or "wk_weights" in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    if is_pp_missing_parameter(name_mapped, self):
                        is_loaded = True
                    break
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                is_loaded = True
                break
            if is_loaded:
                continue

            is_loaded = False
            if ".experts." in name:
                is_loaded = self._load_expert_weight(
                    name,
                    loaded_weight,
                    params_dict,
                    loaded_params,
                    split_expert_params_mapping,
                    fused_expert_param_names,
                    num_experts,
                )
            if is_loaded:
                continue

            if "learnable_sink_param" in name:
                if name in params_dict:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    with torch.no_grad():
                        params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                continue

            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            if remapped_name is None:
                continue
            name = remapped_name
            if name not in params_dict:
                if is_pp_missing_parameter(name, self):
                    continue
                if _should_skip_missing_mtp_scale_param(self.quant_config, name):
                    continue
                logger.warning_once("Skipping unknown MTP weight: %s", name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if pending_wk_fp8:
            raise ValueError(
                "Incomplete MTP FP8 indexer weight/scale pairs: "
                + ", ".join(sorted(pending_wk_fp8))
            )

        logger.info_once("HYV4 MTP draft model loaded: %d params", len(loaded_params))
        unassigned_all = sorted(set(params_dict) - loaded_params)
        # KVCacheScaleParameter (``k_scale`` / ``v_scale`` / ``q_scale`` /
        # ``prob_scale``) is created by ``BaseKVCacheMethod.create_weights`` for
        # every Attention layer that runs on an fp8-family quant method, and
        # its -1.0 sentinel is replaced by the runtime default (1.0) in
        # ``process_weights_after_loading`` -- which runs *after* this method.
        # HY V4 checkpoints intentionally omit these (``kv_cache_quant_algo:
        # null``); silence them only when the parameter is actually the
        # sentinel type so a genuine mismatch on a same-named tensor still
        # reaches the warning.
        from vllm.model_executor.layers.quantization.kv_cache import (
            KVCacheScaleParameter,
        )

        unassigned = [
            name
            for name in unassigned_all
            if not isinstance(params_dict.get(name), KVCacheScaleParameter)
        ]
        missing_scales = [name for name in unassigned if "weight_scale" in name]
        if missing_scales:
            raise ValueError(
                "Missing MTP checkpoint weight scales: " + ", ".join(missing_scales)
            )
        if unassigned:
            # A draft parameter with no checkpoint source keeps its sentinel
            # init value (FP8 scales start at finfo(float32).min), which
            # silently destroys draft quality instead of failing the load.
            logger.warning(
                "HYV4 MTP draft model: %d parameters received no checkpoint value: %s",
                len(unassigned),
                ", ".join(unassigned),
            )
        return loaded_params
