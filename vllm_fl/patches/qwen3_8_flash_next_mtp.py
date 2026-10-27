# SPDX-License-Identifier: Apache-2.0
"""Narrow Qwen4Exp adapters for the vLLM 0.24 native MTP proposer.

The sampler, token alignment, KV ownership and acceptance bookkeeping remain
upstream. Only Qwen's draft config, 4H feedback allocation and tuple return
contract differ from the 0.24 defaults.
"""
from __future__ import annotations

import copy
import functools

_TARGET_TYPES = frozenset({
    "qwen4_exp", "qwen4_exp_text", "qwen3_8_flash_next", "qwen3_8_flash_next_text",
})
_MTP_ARCHITECTURES = ("Qwen3_8FlashNextMTP", "Qwen4ExpMTP")


def is_qwen38_mtp(config) -> bool:
    return bool(getattr(config, "qwen38_mtp", False)) or bool(
        set(getattr(config, "architectures", None) or ()) & set(_MTP_ARCHITECTURES)
    )


def make_qwen38_draft_config(config):
    """Produce a separate text-only draft config without mutating the target."""
    text_config = copy.deepcopy(getattr(config, "text_config", config))
    count = int(getattr(text_config, "mtp_num_hidden_layers", 1))
    if count != 1:
        raise ValueError("Qwen3.8 checkpoint must contain exactly one MTP layer")
    if int(getattr(text_config, "hc_count", 0)) != 4:
        raise ValueError("Qwen3.8 MTP requires four residual streams")
    # 'mtp' is a native 0.24 MTPModelTypes tag. Architecture keeps model identity.
    text_config.model_type = "mtp"
    text_config.architectures = ["Qwen3_8FlashNextMTP"]
    text_config.qwen38_mtp = True
    text_config.hc_mult = text_config.hc_count
    text_config.n_predict = count
    text_config.ple_layer_ids = []
    text_config.supports_qwen38_ngram_context = False
    text_config.index_share_for_mtp_iteration = False
    for name in ("rope_parameters", "rope_scaling"):
        rope = getattr(text_config, name, None)
        if isinstance(rope, dict):
            rope.pop("mrope_section", None)
            rope.pop("mrope_interleaved", None)
    return text_config


def _uses_qwen38_draft(proposer) -> bool:
    draft = getattr(proposer, "draft_model_config", None)
    return draft is not None and is_qwen38_mtp(draft.hf_config)


def install_qwen38_mtp_compat() -> None:
    """Register draft config early without importing either runtime proposer."""
    from vllm import __version__ as vllm_version
    from vllm.config.speculative import SpeculativeConfig
    from vllm.model_executor.models import ModelRegistry

    for architecture in _MTP_ARCHITECTURES:
        if architecture not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                architecture,
                "vllm_fl.models.qwen3_8_flash_next.gpu.mtp:Qwen3_8FlashNextMTP",
            )

    original_override = SpeculativeConfig.hf_config_override
    if not getattr(original_override, "_qwen38_mtp_compat", False):
        @functools.wraps(original_override)
        def override(config):
            model_type = getattr(config, "model_type", None)
            if model_type in _TARGET_TYPES or is_qwen38_mtp(config):
                if not str(vllm_version).startswith("0.24."):
                    raise RuntimeError(
                        "Qwen3.8 MTP compatibility is pinned to vLLM 0.24"
                    )
                return make_qwen38_draft_config(config)
            return original_override(config)
        override._qwen38_mtp_compat = True
        SpeculativeConfig.hf_config_override = staticmethod(override)


def install_qwen38_mtp_runtime_compat() -> None:
    """Install legacy runtime hooks only in workers constructing Qwen MTP."""
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    original_init = SpecDecodeBaseProposer.__init__
    if not getattr(original_init, "_qwen38_mtp_compat", False):
        @functools.wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not _uses_qwen38_draft(self):
                return
            if self.parallel_drafting:
                raise NotImplementedError("Qwen3.8 MTP uses sequential drafting")
            config = self.draft_model_config.hf_config
            hidden_size = self.draft_model_config.get_hidden_size() * config.hc_mult
            if self.hidden_size != hidden_size:
                # Initialization only, before capture or any request can use it.
                self.hidden_size = hidden_size
                self.hidden_states = self.hidden_states.new_zeros(
                    (self.max_num_tokens, hidden_size)
                )
        init._qwen38_mtp_compat = True
        SpecDecodeBaseProposer.__init__ = init

    original_tuple = SpecDecodeBaseProposer.model_returns_tuple
    if not getattr(original_tuple, "_qwen38_mtp_compat", False):
        @functools.wraps(original_tuple)
        def returns_tuple(self):
            if _uses_qwen38_draft(self):
                return True
            return original_tuple(self)
        returns_tuple._qwen38_mtp_compat = True
        SpecDecodeBaseProposer.model_returns_tuple = returns_tuple
