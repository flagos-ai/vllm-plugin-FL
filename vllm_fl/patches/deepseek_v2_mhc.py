# SPDX-License-Identifier: Apache-2.0
"""TeleChat 29B (DeepSeekV3-based) mHC patch for vLLM 0.20.2.

Monkey-patches the original vllm deepseek_v2 module to add multi-head
residual Collar (mHC) support, which is required by TeleChat 29B models
that have ``num_residual_streams > 1`` in their config.
"""

import logging
import math
import typing

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# mHC helper functions
# ---------------------------------------------------------------------------


def _get_num_residual_streams(config) -> int:
    num_streams = getattr(config, "num_residual_streams", None)
    if num_streams is None and hasattr(config, "to_dict"):
        num_streams = config.to_dict().get("num_residual_streams", 1)
    return int(num_streams or 1)


def _use_mhc(config) -> bool:
    return _get_num_residual_streams(config) > 1


def _expand_to_mhc_streams(
    hidden_states: torch.Tensor,
    num_streams: int,
    hidden_size: int,
) -> torch.Tensor:
    """Convert token embeddings [T, C] to mHC layout [T, 1, n*C]."""
    if hidden_states.dim() == 3:
        return hidden_states
    num_tokens = hidden_states.shape[0]
    stream_dim = hidden_size * num_streams
    if hidden_states.shape[-1] == stream_dim:
        return hidden_states.view(num_tokens, 1, stream_dim)
    return (
        hidden_states.unsqueeze(1)
        .unsqueeze(2)
        .repeat(1, 1, num_streams, 1)
        .reshape(num_tokens, 1, stream_dim)
        .contiguous()
    )


def _to_mhc_stream_tensor(
    hidden_states: torch.Tensor,
    num_streams: int,
    hidden_size: int,
) -> torch.Tensor:
    """Normalize hidden states to mHC layout [T, 1, n*C]."""
    num_tokens = hidden_states.shape[0]
    stream_dim = num_streams * hidden_size
    if hidden_states.dim() == 3:
        if hidden_states.shape[-1] == hidden_size:
            return hidden_states.reshape(num_tokens, 1, stream_dim)
        if hidden_states.shape[1] == 1 and hidden_states.shape[-1] == stream_dim:
            return hidden_states
    if hidden_states.dim() == 2 and hidden_states.shape[-1] == stream_dim:
        return hidden_states.view(num_tokens, 1, stream_dim)
    raise ValueError(
        "Unexpected mHC hidden state shape "
        f"{tuple(hidden_states.shape)} for num_streams={num_streams}, "
        f"hidden_size={hidden_size}"
    )


def _contract_mhc_streams(
    hidden_states: torch.Tensor,
    num_streams: int,
    hidden_size: int,
) -> torch.Tensor:
    """Fallback collapse: mean over streams."""
    stream_states = _to_mhc_stream_tensor(hidden_states, num_streams, hidden_size)
    num_tokens = stream_states.shape[0]
    return stream_states.view(num_tokens, num_streams, hidden_size).mean(dim=1)


def _contract_mhc_output(
    hidden_states: torch.Tensor,
    num_streams: int,
    hidden_size: int,
    contract_module: "mHCModule | None",
) -> torch.Tensor:
    """Collapse n-stream states to [T, C] before final RMSNorm/lm_head."""
    if contract_module is None:
        return _contract_mhc_streams(hidden_states, num_streams, hidden_size)

    stream_states = _to_mhc_stream_tensor(hidden_states, num_streams, hidden_size)
    h_pre, _, _ = contract_module.compute_mappings(stream_states)
    return contract_module.aggregate(stream_states, h_pre).reshape(
        stream_states.shape[0], hidden_size
    )


def _is_mhc_weight(name: str) -> bool:
    return ".attn_hc." in name or ".ffn_hc." in name


def _normalize_weight_name(name: str) -> str:
    if name.startswith("model."):
        return name[len("model."):]
    return name


def _resolve_mhc_param_name(
    name: str, params_dict: dict[str, nn.Parameter]
) -> str | None:
    if name in params_dict:
        return name
    if ".mapping_weight" in name:
        legacy = name.replace(".mapping_weight", ".mapping_proj.weight")
        if legacy in params_dict:
            return legacy
    return None


def _load_mhc_split_bias(
    name: str,
    loaded_weight: torch.Tensor,
    param: nn.Parameter,
    num_streams: int,
) -> None:
    bias = param.data
    weight = loaded_weight.reshape(-1)
    n = num_streams
    if name.endswith(".bias_pre"):
        bias[:n].copy_(weight)
    elif name.endswith(".bias_post"):
        bias[n: 2 * n].copy_(weight)
    elif name.endswith(".bias_res"):
        bias[2 * n:].copy_(weight)


# ---------------------------------------------------------------------------
# mHC module classes
# ---------------------------------------------------------------------------


class SinkhornKnopp(nn.Module):
    eps = 1e-6

    def _sinkhorn_normalize(
        self, matrix: torch.Tensor, num_iterations: int
    ) -> torch.Tensor:
        for _ in range(num_iterations):
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + SinkhornKnopp.eps)
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + SinkhornKnopp.eps)
        return matrix

    def forward(
        self, h_res_logits: torch.Tensor, num_iterations: int
    ) -> torch.Tensor:
        matrix_init = torch.exp(
            h_res_logits - h_res_logits.max(dim=-1, keepdim=True).values
        )
        return self._sinkhorn_normalize(matrix_init, num_iterations)


class mHCModule(nn.Module):
    def __init__(self, config, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

        self.n = _get_num_residual_streams(config)
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = getattr(config, "mhc_sinkhorn_iterations", 20)
        self.h_res_clamp_min = getattr(config, "mhc_h_res_clamp_min", -30)
        self.h_res_clamp_max = getattr(config, "mhc_h_res_clamp_max", 30)

        self.mapping_weight = nn.Parameter(
            torch.empty(
                self.n * self.n + 2 * self.n,
                self.n * self.hidden_size,
            )
        )

        init_alpha = getattr(config, "mhc_init_gating_factor", 0.01)
        self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))
        self.bias = nn.Parameter(torch.zeros(self.n * self.n + 2 * self.n))
        self.norm_eps = 1e-6

    def _projection_and_get_norm(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_channels = x.shape[-1]
        scale = x.norm(dim=-1, keepdim=True) / math.sqrt(n_channels)
        scale = 1.0 / (scale + self.norm_eps)
        return torch.nn.functional.linear(x, self.mapping_weight), scale

    def _compute_h(
        self, proj: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = torch.cat(
            [
                self.alpha_pre.expand(self.n),
                self.alpha_post.expand(self.n),
                self.alpha_res.expand(self.n * self.n),
            ],
            dim=-1,
        )
        h = scale * proj * alpha + self.bias
        h_pre = h[..., : self.n].sigmoid()
        h_post = h[..., self.n: 2 * self.n].sigmoid() * 2
        h_res = h[..., 2 * self.n:]
        return h_pre, h_post, h_res

    def compute_mappings(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len, batch_size, _ = x.shape
        proj, scale = self._projection_and_get_norm(x)
        h_pre, h_post, h_res = self._compute_h(proj, scale)
        h_res = h_res.clamp(min=self.h_res_clamp_min, max=self.h_res_clamp_max)
        h_res = SinkhornKnopp()(
            h_res.view(seq_len, batch_size, self.n, self.n),
            self.sinkhorn_iterations,
        )
        return h_pre, h_post, h_res

    def aggregate(self, x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        x_streams = x.view(seq_len, batch_size, self.n, self.hidden_size)
        return (x_streams * h_pre.unsqueeze(-1)).sum(dim=2)

    def _apply_h_post(self, x: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = h_post.shape
        if x.dim() == 1:
            channels = x.shape[0]
            x_expanded = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                seq_len, batch_size, 1, channels
            )
        else:
            channels = x.shape[-1]
            x_expanded = x.unsqueeze(2)
        result = h_post.unsqueeze(-1) * x_expanded
        return result.view(seq_len, batch_size, self.n * channels)

    def apply_h_res(self, h_res: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = residual.shape
        h_res_batched = h_res.view(seq_len * batch_size, self.n, self.n)
        residual_batched = residual.view(
            seq_len, batch_size, self.n, self.hidden_size
        ).view(seq_len * batch_size, self.n, self.hidden_size)
        mixed = torch.bmm(h_res_batched, residual_batched)
        return mixed.view(seq_len, batch_size, self.n * self.hidden_size)

    def fused_h_res_h_post_bda_inference(
        self,
        h_res: torch.Tensor,
        original_residual: torch.Tensor,
        h_post: torch.Tensor,
        layer_output_with_bias: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        x, bias = layer_output_with_bias
        mixed = self.apply_h_res(h_res, original_residual)
        x_expanded = self._apply_h_post(x, h_post)
        bias_expanded = (
            self._apply_h_post(bias, h_post) if bias is not None else None
        )
        return self.get_bias_dropout_add((x_expanded, bias_expanded), mixed, 0.0)

    def get_bias_dropout_add(
        self,
        x_with_bias: tuple[torch.Tensor, torch.Tensor | None],
        residual: torch.Tensor,
        prob: float,
    ) -> torch.Tensor:
        x, bias = x_with_bias
        if x.dtype != residual.dtype:
            x = x.to(residual.dtype)
            if bias is not None:
                bias = bias.to(residual.dtype)
        out = torch.nn.functional.dropout(x, p=prob, inplace=False)
        if bias is not None:
            out = out + bias
        return out + residual

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)
        aggregated = self.aggregate(hidden_states, h_pre)
        return aggregated, h_res, h_post


# ---------------------------------------------------------------------------
# Monkey-patches
# ---------------------------------------------------------------------------


def _patch_decoder_layer():
    """Patch DeepseekV2DecoderLayer with mHC support."""
    import vllm.model_executor.models.deepseek_v2 as dsv2
    from vllm.config import VllmConfig
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2DecoderLayer,
        DeepseekV2MLP, DeepseekAttention,
    )

    _orig_init = DeepseekV2DecoderLayer.__init__
    _orig_forward = DeepseekV2DecoderLayer.forward

    def _patched_init(self, vllm_config: VllmConfig, prefix: str, **kwargs):
        _orig_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        config = vllm_config.model_config.hf_config
        if not _use_mhc(config):
            self.enable_mhc = False
            return

        self.enable_mhc = True
        self.n_streams = _get_num_residual_streams(config)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.attn_hc = mHCModule(config, layer_idx)
        self.ffn_hc = mHCModule(config, layer_idx)

    def _patched_forward(self, positions, hidden_states, residual, llama_4_scaling=None):
        if not self.enable_mhc:
            return _orig_forward(self, positions, hidden_states, residual, llama_4_scaling)

        # ---- mHC forward path ----
        seq_len = hidden_states.shape[0]
        batch_size = hidden_states.shape[1]
        channels = self.hidden_size

        if hidden_states.dim() == 2:
            hidden_states = _expand_to_mhc_streams(
                hidden_states, self.n_streams, self.hidden_size
            )

        origin_hidden_states = hidden_states
        aggregated_hidden_states, attention_res_weights, attention_post_weights = (
            self.attn_hc(origin_hidden_states)
        )

        hidden_states = aggregated_hidden_states.reshape(-1, channels)
        hidden_states = self.input_layernorm(hidden_states)
        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)
        hidden_states = hidden_states.reshape(seq_len, batch_size, channels)

        if (
            not isinstance(self.self_attn, DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor

        hidden_states = self.attn_hc.fused_h_res_h_post_bda_inference(
            h_res=attention_res_weights,
            original_residual=origin_hidden_states,
            h_post=attention_post_weights,
            layer_output_with_bias=(hidden_states, None),
        )

        origin_hidden_states = hidden_states
        aggregated_hidden_states, mlp_res_weights, mlp_post_weights = self.ffn_hc(
            origin_hidden_states
        )

        hidden_states = aggregated_hidden_states.reshape(-1, channels)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states.reshape(seq_len, batch_size, channels)

        if (
            isinstance(self.mlp, DeepseekV2MLP)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor

        hidden_states = self.ffn_hc.fused_h_res_h_post_bda_inference(
            h_res=mlp_res_weights,
            original_residual=origin_hidden_states,
            h_post=mlp_post_weights,
            layer_output_with_bias=(hidden_states, None),
        )
        return hidden_states, residual

    DeepseekV2DecoderLayer.__init__ = _patched_init
    DeepseekV2DecoderLayer.forward = _patched_forward
    logger.info("Patched DeepseekV2DecoderLayer: mHC enabled")


def _patch_model():
    """Patch DeepseekV2Model with mHC support."""
    import vllm.model_executor.models.deepseek_v2 as dsv2
    from vllm.config import VllmConfig
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2Model, PPMissingLayer, RMSNorm,
        make_empty_intermediate_tensors_factory,
    )

    _orig_model_init = DeepseekV2Model.__init__
    _orig_model_forward = DeepseekV2Model.forward

    def _patched_model_init(self, *, vllm_config: VllmConfig, prefix: str = ""):
        _orig_model_init(self, vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        if not _use_mhc(config):
            self.enable_mhc = False
            return

        self.enable_mhc = True
        self.n_streams = _get_num_residual_streams(config)
        logger.info(
            "DeepSeek mHC enabled with num_residual_streams=%d",
            self.n_streams,
        )
        mhc_hidden_size = config.hidden_size * self.n_streams

        from vllm.distributed import get_pp_group
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], mhc_hidden_size
        )

    def _patched_model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
        if not self.enable_mhc:
            return _orig_model_forward(
                self, input_ids, positions, intermediate_tensors, inputs_embeds
            )

        from vllm.distributed import get_pp_group
        from vllm.sequence import IntermediateTensors

        if intermediate_tensors is None:
            if input_ids is not None:
                hidden_states = self.embed_input_ids(input_ids)
            else:
                hidden_states = self.embed_input_ids(inputs_embeds)
            residual = None
            if self.enable_mhc:
                hidden_states = _expand_to_mhc_streams(
                    hidden_states, self.n_streams, self.config.hidden_size
                )
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            if self.enable_mhc and hidden_states.dim() == 2:
                hidden_states = hidden_states.view(
                    hidden_states.shape[0], 1,
                    self.n_streams * self.config.hidden_size
                )

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, PPMissingLayer):
                continue
            hidden_states = layer(
                positions, hidden_states, residual, None
            )
            if isinstance(hidden_states, tuple):
                hidden_states, residual = hidden_states

        if not get_pp_group().is_last_rank:
            if self.enable_mhc and hidden_states.dim() == 3:
                hidden_states = hidden_states.reshape(
                    hidden_states.shape[0], self.n_streams * self.config.hidden_size
                )
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.enable_mhc:
            contract_module = None
            if self.end_layer > self.start_layer:
                last_layer = self.layers[self.end_layer - 1]
                if isinstance(last_layer, torch.nn.Module) and getattr(last_layer, "enable_mhc", False):
                    contract_module = last_layer.ffn_hc
            hidden_states = _contract_mhc_output(
                hidden_states,
                self.n_streams,
                self.config.hidden_size,
                contract_module,
            )
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    DeepseekV2Model.__init__ = _patched_model_init
    DeepseekV2Model.forward = _patched_model_forward
    logger.info("Patched DeepseekV2Model: mHC enabled")


def _patch_load_weights():
    """Patch DeepseekV2ForCausalLM.load_weights to handle mHC weight names."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.utils import is_pp_missing_parameter

    import vllm.model_executor.models.deepseek_v2 as dsv2
    DeepseekV2ForCausalLM = dsv2.DeepseekV2ForCausalLM

    _orig_load_weights = DeepseekV2ForCausalLM.load_weights

    def _patched_load_weights(self, weights):
        config = self.config
        if not _use_mhc(config):
            return _orig_load_weights(self, weights)

        import vllm._aiter_ops as _aiter_ops_mod
        rocm_aiter_ops = _aiter_ops_mod.rocm_aiter_ops
        from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
        import vllm.model_executor.models.deepseek_v2 as dsv2

        rocm_aiter_moe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        mla_params_mapping = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        mha_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        _pending_wk_fp8: dict = {}
        indexer_fused_mapping = [
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        stacked_params_mapping.extend(indexer_fused_mapping)

        if self.use_mha:
            stacked_params_mapping.extend(mha_params_mapping)
        else:
            stacked_params_mapping.extend(mla_params_mapping)

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled
                else 0
            ),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        num_streams = _get_num_residual_streams(config)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # ---- mHC weight handling ----
            if _is_mhc_weight(name):
                if name.endswith((".bias_pre", ".bias_post", ".bias_res")):
                    fused_name = name.rsplit(".", 1)[0] + ".bias"
                    if fused_name in params_dict:
                        if not is_pp_missing_parameter(fused_name, self):
                            _load_mhc_split_bias(
                                name, loaded_weight, params_dict[fused_name], num_streams
                            )
                            loaded_params.add(fused_name)
                    continue

                resolved = _resolve_mhc_param_name(name, params_dict)
                if resolved is not None:
                    if not is_pp_missing_parameter(resolved, self):
                        param = params_dict[resolved]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(resolved)
                    continue

            # ---- Fallback: original logic for non-mHC weights ----
            spec_layer = dsv2.get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue

            is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled
                and ("mlp.shared_experts" in name)
            )

            if dsv2._try_load_fp8_indexer_wk(
                name, loaded_weight, _pending_wk_fp8, params_dict, loaded_params
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Expert weights + general fallback
                is_expert_weight = False
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    split_dim = (
                        1
                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                        else 0
                    )
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                        if loaded_weight.ndim == 1:
                            weight_to_load = loaded_weight[chunk_slice]
                        elif split_dim == 0:
                            weight_to_load = loaded_weight[chunk_slice, :]
                        else:
                            weight_to_load = loaded_weight[:, chunk_slice]
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue
                        is_expert_weight = True
                        name_mapped = chunk_name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            typing.Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            continue
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        name = dsv2.maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

            if name is not None and not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        return loaded_params

    DeepseekV2ForCausalLM.load_weights = _patched_load_weights
    logger.info("Patched DeepseekV2ForCausalLM.load_weights: mHC aware")


def apply_model_patches():
    """Apply all TeleChat/deepseek mHC patches to vllm's deepseek_v2 module."""
    _patch_decoder_layer()
    _patch_model()
    _patch_load_weights()
    logger.info("All TeleChat mHC patches applied successfully")
