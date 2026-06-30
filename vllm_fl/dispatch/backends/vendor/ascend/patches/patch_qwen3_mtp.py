# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from the vllm-ascend project.
# mypy: ignore-errors

"""Ascend-specific patches for Qwen3.5/Qwen3.6 Multi-Token Prediction (MTP).

These patches are needed because:

1. The upstream ``vllm.v1.worker.utils.bind_kv_cache`` raises
   ``NotImplementedError`` when a single layer index maps to more than one
   attention module and the platform is not CUDA/XPU/CPU. With an MTP drafter
   model, both the target model and the drafter model contain attention layers
   with the same layer index, which triggers this error on Ascend NPUs.

2. The upstream ``Qwen3NextMultiTokenPredictor.forward`` consumes PP
   intermediate tensors on non-first PP ranks. On Ascend the MTP drafter is
   placed on the last PP rank and should instead always combine local token
   embeddings with the target hidden states passed in from the base model.
"""

import copy
import logging
from collections import defaultdict

import numpy as np
import torch
import vllm.v1.worker.utils as worker_utils
from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch 1: bind_kv_cache
# ---------------------------------------------------------------------------

def _ascend_bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict,
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """Bind the allocated KV cache to both ModelRunner and forward context.

    This is the Ascend-safe variant of ``vllm.v1.worker.utils.bind_kv_cache``.
    When multiple attention layers share the same layer index (e.g. target model
    + MTP drafter model), the upstream implementation raises
    ``NotImplementedError`` on non-CUDA platforms. Here we follow the CUDA path
    and keep only the first layer's KV cache in ``runner_kv_caches`` while
    still binding every layer's cache to ``forward_context``.
    """
    assert len(runner_kv_caches) == 0

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[worker_utils.extract_layer_index(layer_name, num_attn_module)].append(
            layer_name
        )

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        # Typical encoder-decoder / MTP drafter case: multiple attention modules
        # share the same layer index. Keep the first one for the runner list and
        # bind all of them to the forward context below.
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = [kv_cache]


def patch_bind_kv_cache() -> None:
    """Replace ``vllm.v1.worker.utils.bind_kv_cache`` with the Ascend version."""
    worker_utils.bind_kv_cache = _ascend_bind_kv_cache
    logger.info("Patched vllm.v1.worker.utils.bind_kv_cache for Ascend MTP")


# ---------------------------------------------------------------------------
# Patch 2: Qwen3NextMultiTokenPredictor.forward
# ---------------------------------------------------------------------------

def _ascend_qwen3_next_mtp_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    """MTP drafter forward that always uses local embeddings on the last PP rank.

    The upstream implementation only builds local embeddings on the first PP
    rank and consumes ``intermediate_tensors`` elsewhere. For Ascend the MTP
    drafter runs on the last PP rank together with the base model, so the last
    rank should always combine token embeddings with the target hidden states
    instead of receiving them through PP intermediate tensors.
    """
    if inputs_embeds is None:
        inputs_embeds = self.embed_input_ids(input_ids)
    assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
    inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
    hidden_states = self.pre_fc_norm_hidden(hidden_states)
    hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
    hidden_states = self.fc(hidden_states)
    residual = None

    current_step_idx = spec_step_idx % self.num_mtp_layers
    hidden_states, residual = self.layers[current_step_idx](
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {
                "hidden_states": hidden_states,
                "residual": residual,
            }
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def _disable_torch_compile_for_drafter(
    cls,
) -> None:
    """Do not compile the MTP drafter model on Ascend.

    The drafter is invoked with slices of persistent CPU-sized buffers
    (``input_ids``, ``hidden_states``). Under ``torch.compile`` these buffers
    cause shape specialization / constraint violations because their storage
    size is known at trace time. Running the drafter eagerly avoids the issue
    while the target model still uses the Ascend graph backend.
    """
    ignore_key = "_ignore_compile_vllm"
    setattr(cls, ignore_key, True)
    logger.info("Disabled torch.compile for %s on Ascend", cls.__name__)


def patch_qwen3_next_mtp() -> None:
    """Patch ``Qwen3NextMultiTokenPredictor.forward`` and weight loading."""
    try:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        from vllm.model_executor.models.qwen3_next import QwenNextMixtureOfExperts
        from vllm.model_executor.models.qwen3_next_mtp import (
            Qwen3NextMTP,
            Qwen3NextMultiTokenPredictor,
        )
    except ImportError:
        logger.debug("Qwen3NextMTP not available, skip MTP patch")
        return

    _disable_torch_compile_for_drafter(Qwen3NextMultiTokenPredictor)
    _disable_torch_compile_for_drafter(Qwen3NextMTP)

    _orig_load_model = BaseModelLoader.load_model

    def _ascend_base_load_model(
        self, vllm_config, model_config
    ):
        """Use eager safetensors loading for qwen3_next_mtp draft models.

        The memory-mapped lazy tensors produced by the default safetensors
        iterator can become ``UntypedStorage`` objects when they are passed
        through the MTP weight-loading pipeline on Ascend NPUs. Loading the
        files eagerly into CPU memory before filtering avoids this problem.

        ``initialize_model`` passes the top-level ``vllm_config.model_config``
        to the model's ``__init__``, but when loading the draft model the real
        config is the ``model_config`` argument. We temporarily replace
        ``vllm_config.model_config`` so that draft-only rewrites do not touch
        the target config.
        """
        if getattr(model_config.hf_config, "model_type", None) == "qwen3_next_mtp":
            load_config = vllm_config.load_config
            original_strategy = load_config.safetensors_load_strategy
            load_config.safetensors_load_strategy = "eager"
            original_model_config = vllm_config.model_config
            vllm_config.model_config = model_config
            try:
                return _orig_load_model(self, vllm_config, model_config)
            finally:
                vllm_config.model_config = original_model_config
                load_config.safetensors_load_strategy = original_strategy
        return _orig_load_model(self, vllm_config, model_config)

    _orig_set_moe_parameters = QwenNextMixtureOfExperts.set_moe_parameters

    def _ascend_set_moe_parameters(self):
        try:
            _orig_set_moe_parameters(self)
        except RuntimeError as e:
            # Dense qwen3.5/qwen3.5 models do not have any MoE layers in the MTP
            # module. The upstream mixin raises in that case, so we swallow the
            # specific error and leave the MoE-related attributes unset.
            if "No Qwen3Next layer found" in str(e):
                logger.debug(
                    "Skipping set_moe_parameters for dense Ascend MTP model"
                )
            else:
                raise

    _orig_load_weights = Qwen3NextMTP.load_weights

    def _ascend_qwen3_next_mtp_load_weights(
        self, weights: "Iterable[tuple[str, torch.Tensor]]"
    ) -> "set[str]":
        """Load Qwen3NextMTP weights from qwen3.5/qwen3.5_moe checkpoints.

        Upstream ``Qwen3NextMTP.load_weights`` expects the qwen3_next layout
        where shared embedding/head weights live at ``model.embed_tokens``
        and ``lm_head``. Qwen3.5/Qwen3.6 checkpoints wrap them under
        ``model.language_model.*``; strip that prefix before delegating to the
        original loader. The ``mtp.*`` weights are left untouched and are
        remapped to ``model.*`` by the upstream loader.
        """

        def _remap_language_model_prefix(weights):
            for name, tensor in weights:
                if name.startswith("model.language_model.embed_tokens"):
                    # The shared token embedding lives under the multimodal
                    # ``language_model`` wrapper in qwen3.5/qwen3.5_moe
                    # checkpoints, but Qwen3NextMTP keeps it inside ``model``.
                    name = name.replace(
                        "model.language_model.embed_tokens",
                        "model.embed_tokens",
                        1,
                    )
                elif name.startswith("model.language_model."):
                    name = name[len("model.language_model."):]
                yield name, tensor

        return _orig_load_weights(self, _remap_language_model_prefix(weights))

    _orig_init = Qwen3NextMTP.__init__

    def _ascend_qwen3_next_mtp_init(
        self, *, vllm_config, prefix: str = ""
    ):
        """Rewrite qwen3.5/qwen3.5_moe draft configs before building the model.

        ``ModelConfig`` may be reconstructed in worker processes without going
        through the SpeculativeConfig hf_config_override path. Rewriting here
        guarantees that ``Qwen3NextMTP`` always sees a flat ``qwen3_next_mtp``
        config with the vocabulary size and layer fields it expects.

        We deep-copy the config before rewriting so that the target model's
        ``hf_config`` is not corrupted when the draft and target configs share
        the same object (e.g. through ``AutoConfig`` caching).
        """
        cfg = vllm_config.model_config.hf_config
        if cfg.model_type in ("qwen3_5", "qwen3_5_moe"):
            cfg = copy.deepcopy(cfg)
            _rewrite_qwen3_5_config_for_mtp(cfg)
            vllm_config.model_config.hf_config = cfg
        return _orig_init(self, vllm_config=vllm_config, prefix=prefix)

    BaseModelLoader.load_model = _ascend_base_load_model
    QwenNextMixtureOfExperts.set_moe_parameters = _ascend_set_moe_parameters
    Qwen3NextMultiTokenPredictor.forward = _ascend_qwen3_next_mtp_forward
    Qwen3NextMTP.__init__ = _ascend_qwen3_next_mtp_init
    Qwen3NextMTP.load_weights = _ascend_qwen3_next_mtp_load_weights
    logger.info("Patched Qwen3NextMTP loader/init/forward/load_weights for Ascend")


# ---------------------------------------------------------------------------
# Patch 3: Qwen3_5MultiTokenPredictor.forward (if present in this vLLM version)
# ---------------------------------------------------------------------------

def patch_qwen3_5_mtp() -> None:
    """Patch ``Qwen3_5MultiTokenPredictor.forward`` for Ascend if it exists."""
    try:
        from vllm.model_executor.models.qwen3_5_mtp import (
            Qwen3_5MultiTokenPredictor,
        )
    except ImportError:
        logger.debug("Qwen3_5MultiTokenPredictor not available, skip MTP patch")
        return

    Qwen3_5MultiTokenPredictor.forward = _ascend_qwen3_next_mtp_forward
    logger.info("Patched Qwen3_5MultiTokenPredictor.forward for Ascend")


# ---------------------------------------------------------------------------
# Patch 4: SpeculativeConfig hf_config_override for qwen3.5/qwen3.6 MTP
# ---------------------------------------------------------------------------

_SPECULATIVE_OVERRIDE_PATCHED = False
_ORIG_HF_CONFIG_OVERRIDE = None


def _rewrite_qwen3_5_config_for_mtp(hf_config):
    """Flatten a qwen3.5/qwen3.5_moe config so it can drive ``Qwen3NextMTP``."""
    original_model_type = hf_config.model_type

    # Qwen3.5/Qwen3.6 checkpoints store language-model-specific fields in
    # ``text_config`` (e.g. vocab_size, hidden_size, num_hidden_layers).
    # ``Qwen3NextMTP`` expects these fields directly on the config object,
    # so flatten them before rewriting the model type.
    text_cfg = getattr(hf_config, "text_config", None)
    if text_cfg is not None:
        text_items = (
            text_cfg.items()
            if isinstance(text_cfg, dict)
            else text_cfg.__dict__.items()
        )
        for key, value in text_items:
            if key in ("model_type", "architectures", "text_config"):
                continue
            if not hasattr(hf_config, key) or getattr(
                hf_config, key, None
            ) is None:
                setattr(hf_config, key, value)
        # Dense models do not define ``num_experts``; set a safe default so
        # the generic Qwen3Next MoE code path skips expert mapping.
        if not hasattr(hf_config, "num_experts"):
            hf_config.num_experts = 0

    hf_config.model_type = "qwen3_next_mtp"
    n_predict = getattr(
        hf_config, "num_nextn_predict_layers", None
    ) or getattr(hf_config, "mtp_num_hidden_layers", None)
    hf_config.update(
        {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
    )
    logger.info(
        "Rewrote %s draft config to qwen3_next_mtp for Ascend MTP",
        original_model_type,
    )


def _ascend_hf_config_override(hf_config):
    """Map qwen3.5/qwen3.6 configs to the upstream qwen3_next_mtp type.

    Defined at module level so that the callable can be pickled and sent to
    worker processes (a closure capturing a local variable would not survive
    the ``ModelConfig`` serialization).
    """
    global _ORIG_HF_CONFIG_OVERRIDE
    if _ORIG_HF_CONFIG_OVERRIDE is None:
        from vllm.config.speculative import SpeculativeConfig

        _ORIG_HF_CONFIG_OVERRIDE = SpeculativeConfig.hf_config_override

    # The draft config may share the same hf_config object with the target
    # config (e.g. through AutoConfig caching). Deep-copy before rewriting so
    # the target model keeps its original qwen3.5/qwen3.5_moe type.
    hf_config = copy.deepcopy(hf_config)
    hf_config = _ORIG_HF_CONFIG_OVERRIDE(hf_config)
    if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
        _rewrite_qwen3_5_config_for_mtp(hf_config)
    return hf_config


def patch_speculative_config_override() -> None:
    """Install the module-level ``_ascend_hf_config_override``.

    Upstream vLLM only auto-detects ``qwen3_next`` models as MTP. Qwen3.5 and
    Qwen3.6 checkpoints use ``qwen3_5`` / ``qwen3_5_moe`` model_type but share
    the same MTP structure, so we rewrite the draft model config to
    ``qwen3_next_mtp`` so that ``Qwen3NextMTP`` / ``Qwen3NextMultiTokenPredictor``
    are used. This lets the Ascend MTP forward patch above take effect.

    This patch must be applied at platform-plugin import time, before the main
    process creates ``SpeculativeConfig`` (and therefore ``draft_model_config``).
    """
    global _SPECULATIVE_OVERRIDE_PATCHED, _ORIG_HF_CONFIG_OVERRIDE
    if _SPECULATIVE_OVERRIDE_PATCHED:
        return

    try:
        from vllm.config.speculative import SpeculativeConfig
    except ImportError:
        logger.debug("SpeculativeConfig not available, skip override patch")
        return

    _ORIG_HF_CONFIG_OVERRIDE = SpeculativeConfig.hf_config_override
    SpeculativeConfig.hf_config_override = _ascend_hf_config_override
    _SPECULATIVE_OVERRIDE_PATCHED = True
    logger.info("Patched SpeculativeConfig.hf_config_override for Ascend MTP")


def _normalize_qwen3_5_config(hf_config):
    """Add aliases that upstream vLLM expects for qwen3.5/qwen3.5_moe configs.

    The multimodal target model config uses ``image_token_id`` but the
    speculative-decoding drafter code looks for ``image_token_index``.
    """
    if not hasattr(hf_config, "image_token_index") and hasattr(
        hf_config, "image_token_id"
    ):
        hf_config.image_token_index = hf_config.image_token_id


def patch_model_config_for_qwen3_mtp() -> None:
    """Patch ``ModelConfig.__post_init__`` for qwen3.5/qwen3.5_moe MTP.

    The ``hf_overrides`` callable attached to the draft ``ModelConfig`` is not
    always preserved when the config is sent to worker processes. This wrapper
    rewrites the config after the upstream post-init:

    - For target configs, it just adds the ``image_token_index`` alias.
    - For draft configs, it flattens ``text_config`` and rewrites the model type
      to ``qwen3_next_mtp`` so the correct model class is resolved.
    """
    try:
        from vllm.config import ModelConfig
        from vllm.transformers_utils.config import get_hf_text_config
    except ImportError:
        logger.debug("ModelConfig not available, skip model config patch")
        return

    if getattr(patch_model_config_for_qwen3_mtp, "_patched", False):
        return

    _orig_post_init = ModelConfig.__post_init__

    def _ascend_model_config_post_init(self, *args, **kwargs):
        _orig_post_init(self, *args, **kwargs)
        if self.hf_config.model_type not in ("qwen3_5", "qwen3_5_moe"):
            return

        # Deep-copy so that target and draft configs do not share the same
        # hf_config object after one of them is rewritten.
        self.hf_config = copy.deepcopy(self.hf_config)

        _normalize_qwen3_5_config(self.hf_config)

        logger.info(
            "[Ascend MTP] ModelConfig post-init: runner=%s runner_type=%s "
            "model_type=%s architectures=%s",
            self.runner,
            self.runner_type,
            self.hf_config.model_type,
            getattr(self.hf_config, "architectures", None),
        )

        if self.runner == "draft":
            _rewrite_qwen3_5_config_for_mtp(self.hf_config)
            self.hf_text_config = get_hf_text_config(self.hf_config)
            # Re-resolve the model class now that the architecture has changed.
            registry = self.registry
            model_info, arch = registry.inspect_model_cls(self.architectures, self)
            self._model_info = model_info
            self._architecture = arch
            logger.info("Resolved architecture: %s", arch)

    ModelConfig.__post_init__ = _ascend_model_config_post_init
    patch_model_config_for_qwen3_mtp._patched = True
    logger.info("Patched ModelConfig.__post_init__ for Ascend MTP")


def patch_qwen3_next_mtp_multimodal_flag() -> None:
    """Force ``Qwen3NextMTP`` configs to be treated as text-only.

    ``ModelConfig`` may be reconstructed in worker processes before the
    platform-level post-init patch is active, leaving the stale ``_model_info``
    from the original qwen3.5/qwen3.5_moe config. That can cause the multimodal
    registry to believe the draft model supports multimodal inputs. Since
    ``Qwen3NextMTP`` is text-only, override the property.
    """
    try:
        from vllm.config import ModelConfig
    except ImportError:
        logger.debug("ModelConfig not available, skip multimodal flag patch")
        return

    if getattr(ModelConfig, "_ascend_qwen3_next_mtp_mm_patched", False):
        return

    _orig = ModelConfig.is_multimodal_model.fget

    @property
    def _ascend_is_multimodal_model(self):
        if getattr(self.hf_config, "model_type", None) == "qwen3_next_mtp":
            return False
        return _orig(self)

    ModelConfig.is_multimodal_model = _ascend_is_multimodal_model
    ModelConfig._ascend_qwen3_next_mtp_mm_patched = True
    logger.info("Patched ModelConfig.is_multimodal_model for Ascend MTP")


def _lazy_init_triton_device_props() -> None:
    """Initialize Ascend triton device properties if they are not ready yet."""
    try:
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
        init_device_properties_triton()
    except Exception:
        pass


def _wrap_with_triton_init(fn):
    """Call device-property init before delegating to ``fn``."""

    def wrapper(*args, **kwargs):
        _lazy_init_triton_device_props()
        return fn(*args, **kwargs)

    return wrapper


def patch_ascend_rejection_sampler() -> None:
    """Replace upstream rejection-sampler helpers with Ascend implementations.

    The upstream Triton kernels used by ``vllm.v1.sample.rejection_sampler``
    are not compatible with Ascend NPUs. vllm-ascend provides Ascend-tuned
    versions of the same module-level helpers; swap them in so that speculative
    decoding / MTP works on Ascend through vllm-plugin-FL.
    """
    try:
        import vllm.v1.sample.rejection_sampler as rs
        from vllm_ascend.sample.rejection_sampler import (
            apply_sampling_constraints,
            expand_batch_to_tokens,
            rejection_sample,
        )
    except Exception as e:
        logger.debug("Ascend rejection sampler helpers not available: %s", e)
        return

    if getattr(patch_ascend_rejection_sampler, "_patched", False):
        return

    rs.apply_sampling_constraints = _wrap_with_triton_init(apply_sampling_constraints)
    rs.rejection_sample = _wrap_with_triton_init(rejection_sample)
    rs.expand_batch_to_tokens = _wrap_with_triton_init(expand_batch_to_tokens)
    patch_ascend_rejection_sampler._patched = True
    logger.info("Patched rejection sampler helpers for Ascend")


def patch_ascend_eagle_proposer() -> None:
    """Replace upstream Eagle padded-batch Triton kernels with PyTorch.

    The upstream ``eagle_prepare_next_token_padded_kernel`` and
    ``eagle_prepare_inputs_padded_kernel`` use Triton patterns that fail to
    compile on Ascend NPUs.  Re-implement the same logic with standard PyTorch
    ops so that EAGLE / MTP padded-batch speculative decoding works through
    vllm-plugin-FL.
    """
    try:
        import vllm.v1.spec_decode.eagle as eagle_mod
    except Exception as e:
        logger.debug("Ascend EagleProposer patch not applicable: %s", e)
        return

    if getattr(patch_ascend_eagle_proposer, "_patched", False):
        return

    def _prepare_next_token_ids_padded(
        self,
        common_attn_metadata,
        sampled_token_ids: torch.Tensor,
        requests,
        gpu_input_batch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Precompute backup tokens for when there is no valid next token.
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        vocab_size = gpu_input_batch.vocab_size
        discard_mask = discard_request_mask[:batch_size]

        # Valid sampled tokens are in [0, vocab_size); -1 means rejected.
        valid_mask = (sampled_token_ids != -1) & (sampled_token_ids < vocab_size)
        valid_count = valid_mask.sum(dim=1).to(torch.int32)
        valid_count = torch.where(
            discard_mask, torch.zeros_like(valid_count), valid_count
        )

        # Find the last valid token index in each row.
        positions = torch.arange(
            num_tokens, device=device, dtype=torch.int64
        ).unsqueeze(0).expand(batch_size, -1)
        last_valid_pos = torch.where(
            valid_mask,
            positions,
            torch.full_like(positions, -1),
        ).max(dim=1).values
        last_valid_pos_safe = torch.clamp(last_valid_pos, min=0)

        selected = sampled_token_ids.gather(
            1, last_valid_pos_safe.unsqueeze(1)
        ).squeeze(1).to(torch.int32)
        has_valid = (valid_count > 0) & (~discard_mask)

        next_token_ids = torch.where(
            has_valid,
            selected,
            backup_tokens_gpu[:batch_size],
        )
        return next_token_ids, valid_count

    def _prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        cu_num_draft_tokens = spec_decode_metadata.cu_num_draft_tokens.to(
            torch.int32
        )
        num_draft_tokens = cu_num_draft_tokens.clone()
        if num_reqs > 1:
            num_draft_tokens[1:] -= cu_num_draft_tokens[:-1]

        num_rejected_tokens = torch.where(
            num_draft_tokens > 0,
            num_draft_tokens + 1 - valid_sampled_tokens_count,
            torch.zeros_like(num_draft_tokens),
        )

        q_last_tok_idx = common_attn_metadata.query_start_loc[1:] - 1
        token_indices_to_sample = q_last_tok_idx - num_rejected_tokens
        token_indices_to_sample = token_indices_to_sample.to(torch.int32)

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )
        return spec_common_attn_metadata, token_indices_to_sample

    eagle_mod.EagleProposer.prepare_next_token_ids_padded = (
        _prepare_next_token_ids_padded
    )
    eagle_mod.EagleProposer.prepare_inputs_padded = _prepare_inputs_padded
    patch_ascend_eagle_proposer._patched = True
    logger.info("Patched EagleProposer padded helpers for Ascend")


def patch_mrope_for_graph_mode_mtp() -> None:
    """Make ``MRotaryEmbedding.forward_native`` graph-safe for text positions.

    The upstream implementation uses ``query.view(num_tokens, -1, self.head_size)``
    where ``num_tokens = positions.shape[-1]``. Under ``torch.compile`` with
    dynamic batch sizes this explicit symbolic dimension causes a data-dependent
    guard inside the view meta. The MTP drafter model is compiled together with
    the target model on Ascend, so we replace the view with a shape that only
    uses concrete static dimensions (``num_heads`` and ``head_size``) and an
    inferred leading dimension.
    """
    try:
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            MRotaryEmbedding,
        )
    except ImportError:
        logger.debug("MRotaryEmbedding not available, skip graph-safe patch")
        return

    if getattr(patch_mrope_for_graph_mode_mtp, "_patched", False):
        return

    _orig_forward_native = MRotaryEmbedding.forward_native

    def _graph_safe_forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        # The multimodal (positions.ndim == 2) path is not exercised by the MTP
        # drafter and is left unchanged to avoid altering target-model behavior.
        if positions.ndim == 2:
            return _orig_forward_native(self, positions, query, key, offsets)

        self._match_cos_sin_cache_dtype(query)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        num_heads = query_shape[-1] // self.head_size
        query = query.view(-1, num_heads, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self.apply_rotary_emb.forward_native(query_rot, cos, sin)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        num_kv_heads = key_shape[-1] // self.head_size
        key = key.view(-1, num_kv_heads, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self.apply_rotary_emb.forward_native(key_rot, cos, sin)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

        return query, key

    MRotaryEmbedding.forward_native = _graph_safe_forward_native
    patch_mrope_for_graph_mode_mtp._patched = True
    logger.info("Patched MRotaryEmbedding.forward_native for graph-safe MTP")


def patch_qwen3_mtp_platform() -> None:
    """Apply only the platform-level MTP patch.

    ``SpeculativeConfig`` is constructed in the main (API) process before worker
    processes are spawned, so the hf_config_override patch must be installed
    during platform plugin registration. ``ModelConfig`` is reconstructed in
    worker processes, so its post-init is also patched here.
    """
    patch_speculative_config_override()
    patch_model_config_for_qwen3_mtp()
    patch_qwen3_next_mtp_multimodal_flag()


def patch_mamba_speculative_support() -> None:
    """Allow qwen3.5/qwen3.5_moe models with Mamba layers to use MTP.

    Upstream vLLM only permits ``qwen3_next`` models to combine Mamba layers
    with speculative decoding. The Ascend MTP patches above make qwen3.5/3.6
    work the same way, so expand the allow-list.
    """
    try:
        from vllm.model_executor.layers.mamba.abstract import MambaBase
        from vllm.v1.kv_cache_interface import MambaSpec
    except ImportError:
        logger.debug("MambaBase not available, skip speculative support patch")
        return

    _orig_get_kv_cache_spec = MambaBase.get_kv_cache_spec

    def _ascend_mamba_get_kv_cache_spec(self, vllm_config):
        model_type = vllm_config.model_config.hf_config.model_type
        architectures = getattr(vllm_config.model_config.hf_config, "architectures", None)
        logger.info(
            "[Ascend MTP] MambaBase.get_kv_cache_spec called for model_type=%s "
            "architectures=%s speculative_config=%s",
            model_type, architectures, vllm_config.speculative_config is not None,
        )
        if (
            vllm_config.speculative_config is not None
            and model_type
            not in ("qwen3_next", "qwen3_next_mtp", "qwen3_5", "qwen3_5_moe")
        ):
            logger.info(
                "[Ascend MTP] model_type %s not in allow-list, delegating to upstream",
                model_type,
            )
            return _orig_get_kv_cache_spec(self, vllm_config)

        mamba_block_size = vllm_config.cache_config.mamba_block_size
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
        )

    MambaBase.get_kv_cache_spec = _ascend_mamba_get_kv_cache_spec
    logger.info("Patched MambaBase.get_kv_cache_spec for Ascend MTP")


def patch_qwen3_5_mtp_post_init() -> None:
    """Clear stale target-model ``_model_info`` after draft config is cloned.

    When the speculative config clones the base config into a draft config,
    the cloned ``ModelConfig`` object still carries the target model's cached
    ``_model_info`` and ``_architecture``. For ``Qwen3NextMTP`` this causes
    vLLM to resolve the draft architecture as the target architecture. Clear
    the cached values so that the draft model is re-resolved correctly.
    """
    try:
        from vllm.config import ModelConfig
    except ImportError:
        logger.debug("ModelConfig not available, skip qwen3.5 MTP post-init patch")
        return

    _orig_post_init = ModelConfig.__post_init__

    def _ascend_qwen3_5_mtp_post_init(self):
        _orig_post_init(self)
        if getattr(self.hf_config, "model_type", None) == "qwen3_next_mtp":
            return
        if (
            self.speculative_config is not None
            and getattr(self.speculative_config, "_draft_hf_config", None) is not None
            and self.speculative_config._draft_hf_config.model_type == "qwen3_next_mtp"
        ):
            self._model_info = None
            self._architecture = None

    ModelConfig.__post_init__ = _ascend_qwen3_5_mtp_post_init
    logger.info("Patched ModelConfig.__post_init__ for qwen3.5 MTP draft")


def patch_qwen3_mtp_config_overrides() -> None:
    """Apply ``SpeculativeConfig.hf_config_override`` without mutating target config.

    Upstream vLLM applies the override dict to the shared ``hf_config`` object,
    which means the target model's config is also rewritten (e.g.
    ``model_type='qwen3_next_mtp'``). Apply the override only to the draft
    config copy that ``SpeculativeConfig`` keeps internally.
    """
    try:
        from vllm.config import ModelConfig
    except ImportError:
        logger.debug("ModelConfig not available, skip MTP config override patch")
        return

    _orig = ModelConfig.__post_init__

    def _ascend_mtp_config_overrides(self):
        if (
            self.speculative_config is not None
            and getattr(self.speculative_config, "hf_config_override", None)
            and getattr(self.speculative_config, "_draft_hf_config", None) is not None
        ):
            override = self.speculative_config.hf_config_override
            for key, value in override.items():
                if hasattr(self.speculative_config._draft_hf_config, key):
                    setattr(self.speculative_config._draft_hf_config, key, value)
        _orig(self)

    ModelConfig.__post_init__ = _ascend_mtp_config_overrides
    logger.info("Patched ModelConfig.__post_init__ for MTP config overrides")


def patch_qwen3_mtp() -> None:
    """Apply all Ascend MTP-related runtime patches."""
    patch_bind_kv_cache()
    patch_qwen3_next_mtp()
    patch_qwen3_5_mtp()
    patch_mamba_speculative_support()
    patch_qwen3_next_mtp_multimodal_flag()
    patch_qwen3_5_mtp_post_init()
    patch_qwen3_mtp_config_overrides()
    patch_ascend_rejection_sampler()
    patch_ascend_eagle_proposer()
    patch_mrope_for_graph_mode_mtp()
