# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-vendor inference-only Qwen3.8-Flash-Next model."""

import inspect
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)

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
from vllm.model_executor.layers.quantization import QuantizationConfig
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
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
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

try:
    from vllm.model_executor.models.utils import maybe_fuse_shared_experts
except ImportError:  # vLLM 0.24 predates optional AITER shared-expert fusion
    def maybe_fuse_shared_experts(weights, **_kwargs):
        return weights
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from ..common.hyperconnection import (
    GatedResidualSimple,
    HyperConnectionConfig,
    pack_gated_hc_projection_weights,
)
from ..config import Qwen3_8FlashNextConfig, Qwen3_8FlashNextTextConfig
from .ple_layer import Qwen3_8FlashNextPLELayer
from .qsa import Qwen3_8FlashNextQSAAttention

_HAS_NATIVE_STACKED_WEIGHTS_MAPPER = hasattr(
    WeightsMapper(), "orig_to_new_stacked"
)


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
) -> tuple[
    list[tuple[str, str, int, str]],
    list[tuple[str, str, int, str]],
] | None:
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
            if "reduce_results" in inspect.signature(
                QwenGatedDeltaNetAttention.__init__
            ).parameters:
                gdn_kwargs["reduce_results"] = False
            self.linear_attn = QwenGatedDeltaNetAttention(config, **gdn_kwargs)
            # vLLM 0.24 predates the constructor option. HC owns the single TP
            # reduction after each branch, so disable the GDN output linear's
            # internal reduction explicitly to avoid a double all-reduce.
            if "reduce_results" not in inspect.signature(
                QwenGatedDeltaNetAttention.__init__
            ).parameters:
                self.linear_attn.out_proj.reduce_results = False
            self._gdn_requires_output_buffer = "output" in inspect.signature(
                QwenGatedDeltaNetAttention.forward
            ).parameters
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
        prev_block_output: torch.Tensor | None = None,
        prev_injection: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Keep a clear failure mode for old callers that still try to pass a
        # separate residual tensor.  HC's pending state is a block output plus
        # injection logits, not the ordinary vLLM residual ABI.
        legacy_residual = kwargs.pop("residual", None)
        if legacy_residual is not None:
            raise ValueError("HC layers use prev_block_output/prev_injection")
        del kwargs
        if self.ple is not None:
            # PLE adds to the materialized multi-stream state.  It therefore
            # terminates a pending delayed combine before its external add.
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = self.attn_hyper_connection.combine_pending(
                    hidden_states,
                    prev_block_output,
                    prev_injection,
                )
                prev_block_output = prev_injection = None
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

        hidden_states, mixed, injection = self.attn_hyper_connection.combine_and_mix(
            hidden_states,
            prev_block_output,
            prev_injection,
        )
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
        if get_tensor_model_parallel_world_size() > 1:
            self_attention_output = tensor_model_parallel_all_reduce(
                self_attention_output
            )

        # Consume the attention injection at the next HC boundary and leave
        # the MLP output itself pending for the following layer/final mixer.
        hidden_states, mixed, next_injection = (
            self.mlp_hyper_connection.combine_and_mix(
                hidden_states,
                self_attention_output,
                injection,
            )
        )
        block_output = self.mlp(mixed)
        if get_tensor_model_parallel_world_size() > 1 and getattr(
            self.mlp, "requires_tp_all_reduce", True
        ):
            block_output = tensor_model_parallel_all_reduce(block_output)
        return hidden_states, block_output, next_injection


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

        spec_config = vllm_config.speculative_config
        # MTP HC multi-stream outputs: when speculative method=="mtp" and the
        # model uses HC with hc_count>1, retain the pre-final-mixer multi-stream
        # residual [T, hc_count*H] so the MTP drafter can feed a real
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
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        block_output: torch.Tensor | None = None
        injection: torch.Tensor | None = None
        last_layer = None
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
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
                if block_output is not None and injection is not None:
                    # Deepstack is an external add and must see the fully
                    # materialized state, so it explicitly ends the delayed
                    # combine chain for this layer.
                    hidden_states = layer.mlp_hyper_connection.combine_pending(
                        hidden_states,
                        block_output,
                        injection,
                    )
                    block_output = injection = None
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            # PP transports one tensor.  Materialize the pending MLP block at
            # the stage boundary instead of silently dropping its injection.
            if last_layer is not None and block_output is not None:
                assert injection is not None
                hidden_states = last_layer.mlp_hyper_connection.combine_pending(
                    hidden_states,
                    block_output,
                    injection,
                )
            return IntermediateTensors({"hidden_states": hidden_states})

        assert self.hyper_connection_mixer is not None
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states,
                block_output,
                injection,
            )
        )
        if self._mtp_hidden_buffer is not None:
            # Capture the pre-final-mixer multi-stream state
            # [T, hc_count*H] for the MTP drafter (zero extra compute:
            # this tensor is needed by the final mixer regardless).
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states

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
        pack_gated_hc_projection_weights(self)
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

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen3_8FlashNextModelState

        return Qwen3_8FlashNextModelState

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

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

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
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        pack_gated_hc_projection_weights(self)
        return loaded


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

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen3_8FlashNextModelState

        return Qwen3_8FlashNextModelState

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
            # vLLM 0.24's Qwen3.5 multimodal base does not provide the newer
            # _init_video_pruning() helper and does not support EVS pruning.
            # Preserve the existing day0 compatibility behavior.
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
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

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

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
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        pack_gated_hc_projection_weights(self)
        return loaded

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


__all__ = [
    "Qwen3_8FlashNextDecoderLayer",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen3_8FlashNextModel",
    "Qwen3_8FlashNextSparseMoeBlock",
]
