# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-external PLE preparation and speculative state hooks for vLLM 0.24."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

import numpy as np
import torch


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


def committed_context_cpu(token_ids, computed_tokens, context_len, eos_token_id):
    """Read the committed prefix only, excluding tentative draft suffixes."""
    if token_ids.device.type != "cpu" or token_ids.ndim != 2:
        raise ValueError("committed token history must be a CPU matrix")
    ends = torch.as_tensor(computed_tokens, dtype=torch.int64, device="cpu")
    if ends.ndim != 1 or ends.numel() > token_ids.shape[0]:
        raise ValueError("invalid committed token counts")
    if context_len < 1 or bool(((ends < 0) | (ends > token_ids.shape[1])).any()):
        raise ValueError("committed count is outside its history")
    columns = ends[:, None] + torch.arange(-context_len, 0, dtype=torch.int64)
    values = token_ids[: ends.numel()].gather(1, columns.clamp_min(0))
    return torch.where(columns >= 0, values, values.new_full((), eos_token_id)).long()


def _uses_ple(runner):
    config = runner.model_config.hf_text_config
    return bool(
        is_metax() and getattr(config, "supports_qwen38_ngram_context", False)
        and getattr(config, "ple_layer_ids", ())
    )


def _validate_base(vllm_config, require_sync=False):
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError("Qwen PLE base integration requires PP1")
    if (
        vllm_config.cache_config.mamba_cache_mode != "none"
        or vllm_config.cache_config.enable_prefix_caching
    ):
        raise NotImplementedError(
            "Qwen PLE requires Mamba none and prefix caching disabled"
        )
    if (
        require_sync
        and getattr(vllm_config.scheduler_config, "async_scheduling", None) is not False
    ):
        raise NotImplementedError(
            "Qwen MRV1 host lookup requires explicit async_scheduling=False"
        )


class _HostInputBuffers:
    def __init__(self, model, config, max_num_reqs, max_num_tokens, device):
        self.config = config
        self.context_len = int(config.ngram_size) - 1
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.context = torch.full(
            (max_num_reqs, self.context_len),
            int(config.eos_token_id),
            dtype=torch.int64,
            device=device,
        )
        from vllm_fl.models.qwen3_8_flash_next.gpu.ple_layer import Qwen3_8FlashNextPLELayer

        self.layers = [
            m for m in model.modules() if isinstance(m, Qwen3_8FlashNextPLELayer)
        ]
        if not self.layers:
            raise RuntimeError("configured PLE layer not found in the loaded target")

    def prepare(self, input_ids, query_start_loc, context, dummy=False):
        nreq = query_start_loc.numel() - 1
        if nreq > self.context.shape[0]:
            raise ValueError("PLE requests exceed fixed buffers")
        self.query_start_loc[: nreq + 1].copy_(query_start_loc)
        self.query_start_loc[nreq + 1 :].fill_(query_start_loc[-1])
        self.context.fill_(int(self.config.eos_token_id))
        self.context[:nreq].copy_(context)
        for layer in self.layers:
            layer.prepare_host_lookup(
                input_ids, self.query_start_loc, self.context, dummy=dummy
            )
        return dict(query_start_loc=self.query_start_loc, ngram_context=self.context)


def _mrv1_prepare(runner):
    _validate_base(runner.vllm_config, require_sync=True)
    config = runner.model_config.hf_text_config
    buffers = getattr(runner, "_qwen38_host_inputs", None)
    if buffers is None:
        buffers = _HostInputBuffers(
            runner.get_model(),
            config,
            runner.max_num_reqs,
            runner.max_num_tokens,
            runner.device,
        )
        runner._qwen38_host_inputs = buffers
    dummy_tokens = getattr(runner, "_qwen38_dummy_tokens", None)
    if dummy_tokens is not None:
        starts = torch.tensor(
            [0, min(dummy_tokens, runner.max_num_tokens)], dtype=torch.int32
        )
        context = torch.full(
            (1, buffers.context_len), int(config.eos_token_id), dtype=torch.int64
        )
        return buffers.prepare(runner.input_ids.gpu, starts, context, dummy=True)
    batch = runner.input_batch
    nreq = batch.num_reqs
    context = committed_context_cpu(
        batch.token_ids_cpu_tensor[:nreq],
        batch.num_computed_tokens_cpu[:nreq],
        buffers.context_len,
        int(config.eos_token_id),
    )
    return buffers.prepare(
        runner.input_ids.gpu, runner.query_start_loc.cpu[: nreq + 1], context
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
    original_dispatch = getattr(runner_cls, "_determine_batch_execution_and_padding", None)

    @wraps(original_init)
    def runner_init(self, *args, **kwargs):
        config = kwargs.get("vllm_config", args[0] if args else None)
        if config is not None:
            text_config = config.model_config.hf_text_config
            spec_config = config.speculative_config
            if (
                is_metax() and getattr(text_config, "supports_qwen38_ngram_context", False)
                and spec_config is not None
                and spec_config.method == "mtp"
            ):
                from vllm_fl.patches.qwen3_8_flash_next_mtp import (
                    install_qwen38_mtp_runtime_compat,
                )

                install_qwen38_mtp_runtime_compat()
        return original_init(self, *args, **kwargs)

    @wraps(original_kwargs)
    def model_kwargs(self, *args, **kwargs):
        result = original_kwargs(self, *args, **kwargs)
        if _uses_ple(self):
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
        if not _uses_ple(self) or self.speculative_config is None:
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
            if _uses_ple(self) and getattr(self, "_qwen38_dummy_tokens", None) is None:
                nreq = kwargs.get("num_reqs", args[1] if len(args) > 1 else 0)
                batch = self.input_batch
                if np.any(batch.num_computed_tokens_cpu[:nreq] < batch.num_prompt_tokens[:nreq]):
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
        from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import qsa_capture_columns

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
