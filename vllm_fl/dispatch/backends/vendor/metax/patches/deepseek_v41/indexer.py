# SPDX-License-Identifier: Apache-2.0
"""Enforce the optional indexer workspace bound before scheduler admission."""

from functools import wraps
from inspect import signature


def install_indexer_request_guard():
    """vLLM 0.32 exposes the final token count here for both local and MP engines."""
    from vllm.v1.engine.core import EngineCore

    from vllm_fl.models.deepseek_v41.config import indexer_max_context

    original = EngineCore.preprocess_add_request
    if getattr(original, "_fl_v41_indexer_limit", False):
        return
    if tuple(signature(original).parameters) != ("self", "request"):
        raise RuntimeError(
            "Unsupported vLLM EngineCore request preprocessing signature"
        )

    @wraps(original)
    def preprocess(self, request):
        req, wave = original(self, request)
        limit = indexer_max_context(self.vllm_config)
        # The native preprocessor includes multimodal and embedding lengths.
        # Resumable inputs can append more tokens after this admission check.
        if (
            limit is not None
            and req.sampling_params is not None
            and (req.resumable or req.num_prompt_tokens + req.max_tokens > limit)
        ):
            raise ValueError(
                f"The V4.1 indexer requires a complete input and a declared "
                f"prompt plus output length of at most {limit} tokens"
            )
        return req, wave

    preprocess._fl_v41_indexer_limit = True
    EngineCore.preprocess_add_request = preprocess
