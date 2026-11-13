# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-external PLE preparation and speculative state hooks for vLLM 0.24."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

import numpy as np

from .qwen3_8_flash_next_ple import _mrv1_prepare, _validate_base
from .qwen3_8_flash_next_sampling import install_sampling_contract

_ple_spec_context: ContextVar[tuple | None] = ContextVar(
    "qwen38_ple_spec", default=None
)


def is_metax():
    # Keep the worker hook importable before model runtime modules are loaded.
    from vllm.platforms import current_platform

    return getattr(current_platform, "vendor_name", None) == "metax"


def get_ple_spec_metadata():
    return _ple_spec_context.get()


def spec_decode_counts(query_lengths, is_prefilling, num_spec):
    """Include zero-draft decodes so the previous accepted offset is applied."""
    lengths = np.asarray(query_lengths, dtype=np.int32)
    prefill = np.asarray(is_prefilling, dtype=np.bool_)
    if lengths.shape != prefill.shape or np.any(lengths < 0):
        raise ValueError("inconsistent request lengths/prefill flags")
    counts = np.full(lengths.shape, -1, dtype=np.int32)
    decode = ~prefill & (lengths > 0) & (lengths <= num_spec + 1)
    counts[decode] = lengths[decode] - 1
    return counts


def _uses_ple(runner):
    if not is_metax():
        return False
    config = runner.model_config.hf_text_config
    return bool(
        getattr(config, "supports_qwen38_ngram_context", False)
        and getattr(config, "ple_layer_ids", ())
    )


def install_qwen38_runner_hooks(runner_cls):
    """One idempotent, model-guarded preparation hook for ModelRunnerFL/MRV1."""
    if runner_cls.__dict__.get("_qwen38_ple_hooks_installed", False):
        return
    original_init = runner_cls.__init__
    original_kwargs = runner_cls._init_model_kwargs
    original_dummy = runner_cls._dummy_run
    original_attn = runner_cls._build_attention_metadata
    original_capture = getattr(runner_cls, "_warmup_and_capture", None)
    original_dispatch = getattr(
        runner_cls, "_determine_batch_execution_and_padding", None
    )

    @wraps(original_init)
    def runner_init(self, *args, **kwargs):
        config = kwargs.get("vllm_config", args[0] if args else None)
        if config is not None:
            text_config = config.model_config.hf_text_config
            spec_config = config.speculative_config
            if (
                is_metax()
                and getattr(text_config, "supports_qwen38_ngram_context", False)
                and spec_config is not None
                and spec_config.method == "mtp"
            ):
                from vllm_fl.patches.qwen3_8_flash_next_mtp import (
                    install_qwen38_mtp_runtime_compat,
                )

                install_qwen38_mtp_runtime_compat()
        result = original_init(self, *args, **kwargs)
        if _uses_ple(self):
            install_sampling_contract(self.input_batch)
        return result

    @wraps(original_kwargs)
    def model_kwargs(self, *args, **kwargs):
        result = original_kwargs(self, *args, **kwargs)
        if _uses_ple(self):
            install_sampling_contract(self.input_batch)
            result.update(_mrv1_prepare(self))
        return result

    @wraps(original_dummy)
    def dummy_run(self, num_tokens, *args, **kwargs):
        if not _uses_ple(self):
            return original_dummy(self, num_tokens, *args, **kwargs)
        previous = getattr(self, "_qwen38_dummy_tokens", None)
        self._qwen38_dummy_tokens = num_tokens
        try:
            return original_dummy(self, num_tokens, *args, **kwargs)
        finally:
            self._qwen38_dummy_tokens = previous

    @wraps(original_attn)
    def attention_metadata(self, *args, **kwargs):
        if not _uses_ple(self):
            return original_attn(self, *args, **kwargs)
        if self.speculative_config is None:
            if self.vllm_config.additional_config.get(
                "qwen38_fused_qsa_metadata", False
            ):
                from vllm_fl.models.qwen3_8_flash_next.common.qsa_packing import (
                    qsa_metadata_build_scope,
                )

                with qsa_metadata_build_scope():
                    return original_attn(self, *args, **kwargs)
            return original_attn(self, *args, **kwargs)
        _validate_base(self.vllm_config, require_sync=True)
        if not kwargs.get("for_cudagraph_capture", False):
            nreq = self.input_batch.num_reqs
            lengths = np.diff(self.query_start_loc.np[: nreq + 1])
            prefill = (
                self.input_batch.num_computed_tokens_cpu[:nreq]
                < self.input_batch.num_prompt_tokens[:nreq]
            )
            counts = spec_decode_counts(
                lengths, prefill, self.speculative_config.num_speculative_tokens
            )
            self.num_decode_draft_tokens.np.fill(-1)
            self.num_decode_draft_tokens.np[:nreq] = counts
            self.num_decode_draft_tokens.copy_to_gpu()
        kwargs["use_spec_decode"] = True
        token = _ple_spec_context.set(
            (self.num_accepted_tokens.gpu, self.num_decode_draft_tokens.cpu)
        )
        try:
            return original_attn(self, *args, **kwargs)
        finally:
            _ple_spec_context.reset(token)

    if original_dispatch is not None:

        @wraps(original_dispatch)
        def dispatch(self, *args, **kwargs):
            # Native KV setup can replace InputBatch after runner construction.
            uses_ple = _uses_ple(self)
            if uses_ple:
                install_sampling_contract(self.input_batch)
            if uses_ple and getattr(self, "_qwen38_dummy_tokens", None) is None:
                nreq = kwargs.get("num_reqs", args[1] if len(args) > 1 else 0)
                batch = self.input_batch
                if np.any(
                    batch.num_computed_tokens_cpu[:nreq]
                    < batch.num_prompt_tokens[:nreq]
                ):
                    # A two-token prompt or final prefill chunk can match K1's
                    # uniform decode shape. Its recurrent initialization must
                    # still execute eagerly, independent of tensor shape.
                    kwargs["force_eager"] = True
            return original_dispatch(self, *args, **kwargs)

        runner_cls._determine_batch_execution_and_padding = dispatch

    @wraps(original_capture)
    def warmup_and_capture(self, *args, **kwargs):
        enabled = _uses_ple(self) and self.vllm_config.additional_config.get(
            "qwen38_qsa_length_buckets", False
        )
        if not enabled:
            return original_capture(self, *args, **kwargs)
        from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import (
            qsa_capture_columns,
        )

        # Largest first allows the shared graph pool to reuse its allocations.
        # The original warmup/capture method also participates in graph-memory
        # profiling, so its estimate includes all four score-width variants.
        for columns in (0, 8192, 2048, 512):
            with qsa_capture_columns(columns):
                original_capture(self, *args, **kwargs)

    if original_capture is not None:
        runner_cls._warmup_and_capture = warmup_and_capture
    runner_cls.__init__ = runner_init
    runner_cls._init_model_kwargs = model_kwargs
    runner_cls._dummy_run = dummy_run
    runner_cls._build_attention_metadata = attention_metadata
    runner_cls._qwen38_ple_hooks_installed = True
