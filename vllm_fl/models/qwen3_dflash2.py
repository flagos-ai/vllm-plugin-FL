# SPDX-License-Identifier: Apache-2.0
"""DFlash 2 draft model support for the ARM CPU vLLM-FL runtime.

This is a vLLM 0.24-compatible adaptation of the upstream DFlash 2 model.
The target model and its LM head remain owned by vLLM; the checkpoint only
contains the five-layer drafter, dynamic convolutions, and path selector.
"""

import os
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)
from vllm.model_executor.models.qwen2 import Qwen2MLP as Qwen3MLP
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

_UINT32_MASK = (1 << 32) - 1
_PHILOX_KEY_A = 0x9E3779B9
_PHILOX_KEY_B = 0xBB67AE85
_PHILOX_ROUND_A = 0xD2511F53
_PHILOX_ROUND_B = 0xCD9E8D57
_UINT32_TO_UNIFORM = 4.6566127342e-10


def _philox4x32(seed: int, offset: int) -> tuple[int, int, int, int]:
    """Return Triton's 10-round Philox4x32 output for one scalar offset."""
    counter_0 = offset & _UINT32_MASK
    counter_1 = (offset >> 32) & _UINT32_MASK
    counter_2 = 0
    counter_3 = 0
    key_0 = seed & _UINT32_MASK
    key_1 = (seed >> 32) & _UINT32_MASK
    for _ in range(10):
        old_counter_0 = counter_0
        old_counter_2 = counter_2
        product_0 = _PHILOX_ROUND_B * old_counter_2
        product_1 = _PHILOX_ROUND_A * old_counter_0
        counter_0 = ((product_0 >> 32) & _UINT32_MASK) ^ counter_1 ^ key_0
        counter_1 = product_0 & _UINT32_MASK
        counter_2 = ((product_1 >> 32) & _UINT32_MASK) ^ counter_3 ^ key_1
        counter_3 = product_1 & _UINT32_MASK
        key_0 = (key_0 + _PHILOX_KEY_A) & _UINT32_MASK
        key_1 = (key_1 + _PHILOX_KEY_B) & _UINT32_MASK
    return counter_0, counter_1, counter_2, counter_3


def dflash2_keyed_uniform(
    seed: int,
    position: int,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Reproduce upstream Triton's token/position-keyed uniform samples.

    DFlash2 first hashes the request seed with the predecessor's absolute
    position, then uses each candidate token id as a Philox counter.  Keeping
    this CPU implementation bit-compatible makes sampling independent of the
    order in which top-k candidates happen to be returned.
    """
    position_seed = _philox4x32(seed, position)[0]
    words = [
        _philox4x32(position_seed, int(token_id))[0]
        for token_id in token_ids.detach().reshape(-1).tolist()
    ]
    # This is Triton's uint_to_uniform_float conversion.  It deliberately
    # folds the sign bit instead of dividing the unsigned word by 2**32.
    magnitudes = [word if word < (1 << 31) else _UINT32_MASK - word for word in words]
    return (
        torch.tensor(magnitudes, dtype=torch.float32, device=token_ids.device)
        .mul_(_UINT32_TO_UNIFORM)
        .reshape(token_ids.shape)
    )


def _keyed_gumbel_argmax(
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    seed: int,
    position: int,
    temperature: float,
) -> torch.Tensor:
    uniforms = dflash2_keyed_uniform(seed, position, candidate_ids).clamp_min_(
        _UINT32_TO_UNIFORM
    )
    noise = -torch.log(-torch.log1p(-uniforms))
    return (scores / temperature + noise).argmax()


def validate_dflash2_block_size(
    draft_config: dict,
    num_speculative_tokens: int,
) -> int:
    """Return a checkpoint-compatible runtime block size.

    DFlash counts the verified anchor in ``block_size`` whereas vLLM's
    ``num_speculative_tokens`` does not.  The released implementation allows
    shorter runtime blocks (and recommends them for quantized MLX kernels),
    but running beyond the checkpoint's trained block is not a safe tuning.
    """
    runtime_block_size = 1 + int(num_speculative_tokens)
    checkpoint_block_size = int(draft_config["block_size"])
    if runtime_block_size < 2 or runtime_block_size > checkpoint_block_size:
        raise ValueError(
            "DFlash2 runtime block size must be in [2, checkpoint block size] "
            f"but got {runtime_block_size} with checkpoint block size "
            f"{checkpoint_block_size}"
        )
    return runtime_block_size


def grouped_dynamic_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Apply the DFlash 2 block-local dynamic depthwise convolution."""
    if (
        os.environ.get("VLLM_FL_DFLASH2_NATIVE_CONV", "0") == "1"
        and hidden_states.device.type == "cpu"
        and hidden_states.dtype == torch.bfloat16
        and delta.dtype == torch.bfloat16
        and base.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and delta.stride(-1) == 1
        and base.is_contiguous()
        and hasattr(torch.ops.triton_jit_cpu, "dflash2_grouped_conv")
    ):
        return torch.ops.triton_jit_cpu.dflash2_grouped_conv(
            hidden_states,
            delta,
            base,
            block_size,
            group_size,
        )
    if os.environ.get("VLLM_FL_DFLASH2_SLICED_CONV", "0") == "1":
        return _grouped_dynamic_conv_sliced(
            hidden_states, delta, base, block_size, group_size
        )
    taps = base.shape[0]
    num_groups = hidden_states.shape[-1] // group_size
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + (
            coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
        )
    return output.flatten(-2)


def _grouped_dynamic_conv_sliced(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Apply the same convolution without padded shifts or position masks."""
    taps = base.shape[0]
    num_groups = hidden_states.shape[-1] // group_size
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    rows = hidden_states.shape[0]
    for tap in range(1, taps):
        for block_start in range(0, rows, block_size):
            block_end = min(block_start + block_size, rows)
            output_begin = block_start + tap
            if output_begin >= block_end:
                continue
            output[output_begin:block_end].add_(
                coefficients[output_begin:block_end, tap]
                * blocks[block_start : block_end - tap]
            )
    return output.flatten(-2)


def select_greedy_path(
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
) -> torch.Tensor:
    """Walk the greedy DFlash 2 path from already-projected hidden states."""
    predecessor = anchor_token_ids.to(torch.long)
    path = []
    batch = torch.arange(candidate_ids.shape[0], device=candidate_ids.device)
    use_batched_matvec = (
        os.environ.get("VLLM_FL_DFLASH2_BATCHED_GREEDY_SELECTOR", "0") == "1"
    )
    successors = successor_codebook[candidate_ids] if use_batched_matvec else None
    for position in range(candidate_ids.shape[1]):
        candidates = candidate_ids[:, position]
        predecessor_code = predecessor_codebook[predecessor]
        state = predecessor_code * hidden[:, position]
        if successors is not None:
            interaction = torch.bmm(
                successors[:, position], state.unsqueeze(-1)
            ).squeeze(-1)
        else:
            interaction = torch.einsum(
                "br,bkr->bk", state, successor_codebook[candidates]
            )
        scores = unary_logits[:, position] + interaction
        selected = scores.argmax(dim=-1)
        predecessor = candidates[batch, selected]
        path.append(predecessor)
    return torch.stack(path, dim=1)


def select_stochastic_path(
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: list[int],
    anchor_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the upstream keyed-Gumbel path and retain its exact sparse q."""
    predecessor = anchor_token_ids.to(torch.long)
    path = []
    probability_rows = []
    batch = torch.arange(candidate_ids.shape[0], device=candidate_ids.device)
    temperatures = temperatures[: candidate_ids.shape[0]].to(
        device=unary_logits.device,
        dtype=torch.float32,
    )
    anchor_positions = anchor_positions[: candidate_ids.shape[0]].to(
        device=unary_logits.device,
        dtype=torch.int64,
    )
    if len(seeds) < candidate_ids.shape[0]:
        raise ValueError("DFlash2 requires one sampling seed per request")
    for position in range(candidate_ids.shape[1]):
        candidates = candidate_ids[:, position]
        successor = successor_codebook[candidates]
        predecessor_code = predecessor_codebook[predecessor]
        scores = (
            unary_logits[:, position].float()
            + torch.einsum(
                "br,bkr->bk",
                predecessor_code * hidden[:, position],
                successor,
            ).float()
        )
        rows = []
        selected_rows = []
        for row_index in range(scores.shape[0]):
            temperature = float(temperatures[row_index])
            if temperature < 1.0e-5:
                selected = scores[row_index].argmax()
                probabilities = torch.zeros_like(scores[row_index])
                probabilities[selected] = 1.0
            else:
                probabilities = torch.softmax(
                    scores[row_index] / temperature,
                    dim=-1,
                    dtype=torch.float32,
                )
                selected = _keyed_gumbel_argmax(
                    scores[row_index],
                    candidates[row_index],
                    seeds[row_index],
                    int(anchor_positions[row_index]) + position,
                    temperature,
                )
            rows.append(probabilities)
            selected_rows.append(selected)
        position_probabilities = torch.stack(rows)
        selected = torch.stack(selected_rows)
        predecessor = candidates[batch, selected]
        path.append(predecessor)
        probability_rows.append(position_probabilities)
    return torch.stack(path, dim=1), torch.stack(probability_rows, dim=1)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}"
            )
        self.block_size = block_size
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return grouped_dynamic_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.group_size,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        taps = self.base_kernel.shape[1]
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3Attention(nn.Module):
    """Non-causal sliding-window attention used by the released Qwen drafter."""

    def __init__(
        self,
        config,
        prefix: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=getattr(config, "is_neox_style", True),
            rope_parameters=config.rope_parameters,
        )
        self.sliding_window = int(config.sliding_window)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )
        # vLLM's CPU decoder backend constructs a one-sided sliding
        # window even when CommonAttentionMetadata.causal is False. DFlash 2
        # is non-causal within the parallel draft block, so queries must see
        # both earlier and later query positions inside the configured window.
        backend_window = getattr(self.attn.impl, "sliding_window", None)
        if isinstance(backend_window, tuple) and len(backend_window) == 2:
            symmetric_window = self.sliding_window - 1
            self.attn.impl.sliding_window = (symmetric_window, symmetric_window)
        self.causal = False
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)
        q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        self.self_attn = DFlash2Qwen3Attention(
            config,
            prefix=maybe_prefix(prefix, "self_attn"),
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        runtime_block_size = validate_dflash2_block_size(
            draft_config,
            speculative_config.num_speculative_tokens,
        )
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            block_size=runtime_block_size,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        # Keep direct Parameters rather than Embedding modules because the
        # released checkpoint names omit the trailing ``.weight``.
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def select_greedy(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Walk the exact greedy DFlash 2 path without materializing KxK edges."""
        hidden = self.hidden_projection(hidden_states)
        return select_greedy_path(
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.predecessor_codebook,
            self.successor_codebook,
        )

    def select_stochastic(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        temperatures: torch.Tensor,
        seeds: list[int],
        anchor_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden_projection(hidden_states)
        return select_stochastic_path(
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.predecessor_codebook,
            self.successor_codebook,
            temperatures,
            seeds,
            anchor_positions,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    """DFlash model body with DFlash 2 convolutions and selector."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        current_vllm_config = get_current_vllm_config()
        static_context = current_vllm_config.compilation_config.static_forward_context
        for layer in self.layers:
            static_context.pop(layer.self_attn.attn.layer_name, None)
        self.layers = nn.ModuleList(
            [
                DFlash2Qwen3DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                    config=self.config,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )

        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        self.candidate_selector = CandidateSelector(
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            rank=int(draft_config["selector_rank"]),
            top_k=int(draft_config["selector_top_k"]),
            params_dtype=vllm_config.model_config.dtype,
            prefix=maybe_prefix(prefix, "candidate_selector"),
        )

        # FlagGems must treat these BF16 drafter weights separately from the
        # target model body; its online quantizer checks this marker.
        for module in self.modules():
            if hasattr(module, "weight"):
                module._flag_gems_spec_draft = True

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlash2Qwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # DFlashQwen3ForCausalLM.compute_logits expects this processor.
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None
        draft_config = self.config.dflash_config
        self.output_multiplier = float(draft_config.get("output_multiplier", 1.0))
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = softcap if softcap > 0 else None

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.compute_logits(hidden_states)
        selector = self.model.candidate_selector
        values, ids = torch.topk(logits, selector.top_k, dim=-1, sorted=False)
        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap
        return ids.to(torch.int64), values


EntryClass: Callable[..., nn.Module] = DFlash2Qwen3ForCausalLM
