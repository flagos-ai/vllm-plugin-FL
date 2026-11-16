# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM v0.32.0 for FL operator dispatch.
"""Qwen4Exp constructors for FL layers and quantization.

The upstream models own forward, weight loading and model interfaces.
Initialize the FL layers directly because upstream constructors bind their
own HC, PLE and attention classes.
"""

import torch
from torch import nn
from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
)
from vllm.model_executor.models.utils import (
    StageMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.models.qwen4_exp.nvidia import model as upstream
from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (
    enable_qwen4_exp_low_latency_gemm,
)
from vllm.models.qwen4_exp.nvidia.model import (
    _QSA_LAYER_TYPES,
    Qwen4ExpMixtureOfExperts,
    Qwen4ExpSparseMoeBlock,
    is_hc_sequence_parallel_enabled,
)
from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import (
    request_hc_down_silu_warmup,
)
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
from vllm.platforms import current_platform
from vllm.tokenizers.registry import cached_tokenizer_from_config

from .hyperconnection import GatedResidual, HyperConnectionConfig
from .ple_layer import Qwen4ExpPLELayer
from .quantization import configure_bf16_weights


class Qwen4ExpDecoderLayer(upstream.Qwen4ExpDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        # MoE SP also enables HC SP, while HC SP can be enabled independently.
        self.use_hc_sequence_parallel = is_hc_sequence_parallel_enabled(vllm_config)
        self.layer_idx = extract_layer_index(prefix)
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.ple: Qwen4ExpPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
                use_sequence_parallel=self.use_hc_sequence_parallel,
            )

        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_hc_sequence_parallel,
            )
        elif layer_type in _QSA_LAYER_TYPES:
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                    reduce_results=not self.use_hc_sequence_parallel,
                )
            else:
                qsa_cls = Qwen4ExpQSAAttention
                if current_platform.vendor_name == "thead":
                    from .thead_qsa import Qwen4ExpQSAAttention as TheadQSAAttention

                    qsa_cls = TheadQSAAttention
                self.self_attn = qsa_cls(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                    reduce_results=not self.use_hc_sequence_parallel,
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = Qwen4ExpSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
            reduce_results=not self.use_hc_sequence_parallel,
        )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )


class Qwen4ExpModel(upstream.Qwen4ExpModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        # MoE SP also enables HC SP, while HC SP can be enabled independently.
        self.use_hc_sequence_parallel = is_hc_sequence_parallel_enabled(vllm_config)
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.vocab_size = config.vocab_size
        self._qsa_layer_ids = frozenset(
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_type in _QSA_LAYER_TYPES
            and getattr(config, "indexer_n_heads", None) is not None
        )
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen4ExpSparseMoeBlock,
            "mlp",
        )
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size
        )

        self.hyper_connection_mixer: GatedResidual | None
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )
        else:
            self.hyper_connection_mixer = None

        spec_config = vllm_config.speculative_config
        # MTP HC multi-stream outputs: when speculative method=="mtp" and the
        # model uses HC with hc_count>1, retain the pre-final-mixer multi-stream
        # hidden state [T, hc_count*H] so the MTP drafter can feed a real
        # multi-stream backbone hidden on its first step (scheme A). Derived
        # purely from config (NOT node identity) so P/D nodes stay consistent.
        needs_mtp_hidden = (
            spec_config is not None
            and getattr(spec_config, "method", None) == "mtp"
            and get_pp_group().is_last_rank
        )
        if needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hc_count * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None


class Qwen4ExpForCausalLM(upstream.Qwen4ExpForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        if vllm_config.lora_config is not None:
            # LoRA does not support the merged QKV/indexer projection, so its
            # packed mapping must keep the indexer separate.
            self.packed_modules_mapping = self.packed_modules_mapping | {
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            }
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        configure_bf16_weights(vllm_config)
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = Qwen4ExpModel(
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
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)
        if self.model_config.dtype == torch.bfloat16:
            # Precompile the fused HC down+SiLU kernels for every CUDA-graph
            # capture size in the fused dispatch range, so no CuTe-DSL JIT
            # happens during graph capture.
            request_hc_down_silu_warmup(
                vllm_config.compilation_config.cudagraph_capture_sizes or (),
                self.config.hc_lowrank,
                self.config.hc_count,
                self.config.hidden_size * self.config.hc_count,
            )


class Qwen4ExpForConditionalGeneration(upstream.Qwen4ExpForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config
        if vllm_config.lora_config is not None:
            # LoRA does not support the merged QKV/indexer projection, so its
            # packed mapping must keep the indexer separate.
            self.packed_modules_mapping = self.packed_modules_mapping | {
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            }
        configure_bf16_weights(vllm_config)
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "Qwen4ExpForConditionalGeneration requires multimodal_config"
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
                    input_norm=build_mm_input_norm(self.model_config),
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
            self.language_model = Qwen4ExpForCausalLM(
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
        self.set_moe_parameters(self.language_model.model.layers)


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMixtureOfExperts",
    "Qwen4ExpModel",
    "Qwen4ExpSparseMoeBlock",
]
