# SPDX-License-Identifier: Apache-2.0
"""MRV1 PLE input state, synchronous history and asynchronous GPU suffixes."""

from __future__ import annotations

import numpy as np
import torch


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


def async_context_plan(
    req_ids,
    request_states,
    computed_tokens,
    query_start_loc,
    token_ids,
    previous,
    context_len,
    eos_token_id,
):
    """Map active rows to GPU history; seed new/resumed rows from native state.

    MRV1 fills continuous async output histories with -1 placeholders. They
    must never seed PLE. For requests absent from the previous batch, native
    _update_states restores all_token_ids before preparing model inputs.
    """
    nreq = len(req_ids)
    starts = query_start_loc.tolist()
    if len(set(req_ids)) != nreq or len(starts) != nreq + 1:
        raise ValueError("invalid async PLE request boundaries")
    if not starts or starts[0] != 0 or any(b < a for a, b in zip(starts, starts[1:])):
        raise ValueError("invalid async PLE packed offsets")
    metadata = torch.full((nreq, context_len + 1), eos_token_id, dtype=torch.int32)
    metadata[:, 0] = -1
    next_previous = {}
    for row, req_id in enumerate(req_ids):
        state = request_states[req_id]
        begin = int(computed_tokens[row])
        if begin < 0 or begin > token_ids.shape[1]:
            raise ValueError("async PLE computed position outside token history")
        prior = previous.get(req_id)
        if begin > 0 and prior is not None and prior[0] is state and prior[1] == begin:
            metadata[row, 0] = prior[2]
        elif begin:
            first = max(0, begin - context_len)
            seed = token_ids[row, first:begin]
            if bool((seed < 0).any()):
                raise ValueError(
                    "async PLE cannot seed from unresolved token placeholders"
                )
            metadata[row, 1 + context_len - seed.numel() :].copy_(seed)
        next_previous[req_id] = (state, begin + starts[row + 1] - starts[row], row)
    return metadata, next_previous


class _AsyncNgramContext:
    """GPU-resident consumed-token suffix, independent of mutable batch rows.

    Mirrors upstream Qwen4ExpModelState's GPU-authoritative context. MRV1
    lacks MRV2's all_token_ids GPU table; a double-buffered n-gram suffix is
    sufficient when speculation is off. Recompute/resume seeds come from the
    native restored history. No sampled token is read back to the CPU here.
    """

    def __init__(self, max_num_reqs, context_len, eos_token_id, device):
        self.context_len = context_len
        self.eos_token_id = eos_token_id
        self.history = torch.full(
            (max_num_reqs, context_len),
            eos_token_id,
            dtype=torch.int32,
            device=device,
        )
        self.next_history = torch.empty_like(self.history)
        self.previous = {}

    def prepare(self, runner, output):
        from vllm_fl.models.qwen3_8_flash_next.gpu.ops.ple_history import (
            update_ple_history,
        )

        batch = runner.input_batch
        nreq = batch.num_reqs
        metadata, next_previous = async_context_plan(
            batch.req_ids,
            runner.requests,
            batch.num_computed_tokens_cpu[:nreq],
            runner.query_start_loc.cpu[: nreq + 1],
            batch.token_ids_cpu_tensor[:nreq],
            self.previous,
            self.context_len,
            self.eos_token_id,
        )
        # Per-call pinned allocations are retained by PyTorch's copy allocator
        # until H2D completes; a later CPU step cannot overwrite this metadata.
        metadata_gpu = metadata.pin_memory().to(output.device, non_blocking=True)
        update_ple_history(
            runner.input_ids.gpu,
            runner.query_start_loc.gpu,
            metadata_gpu,
            self.history,
            self.next_history,
            output,
            nreq,
            self.eos_token_id,
        )
        self.history, self.next_history = self.next_history, self.history
        self.previous = next_previous
        return output


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
        from vllm_fl.models.qwen3_8_flash_next.gpu.ple_layer import (
            Qwen3_8FlashNextPLELayer,
        )

        self.layers = [
            m for m in model.modules() if isinstance(m, Qwen3_8FlashNextPLELayer)
        ]
        if not self.layers:
            raise RuntimeError("configured PLE layer not found in the loaded target")

    def prepare(
        self,
        input_ids,
        query_start_loc,
        context,
        dummy=False,
        use_cpu_metadata=False,
        gpu_query_start_loc=None,
    ):
        nreq = query_start_loc.numel() - 1
        if nreq > self.context.shape[0]:
            raise ValueError("PLE requests exceed fixed buffers")
        source = (
            query_start_loc
            if gpu_query_start_loc is None
            else gpu_query_start_loc[: nreq + 1]
        )
        self.query_start_loc[: nreq + 1].copy_(source)
        self.query_start_loc[nreq + 1 :].fill_(query_start_loc[-1])
        if context is not self.context:
            self.context.fill_(int(self.config.eos_token_id))
            self.context[:nreq].copy_(context)
        for layer in self.layers:
            uva = getattr(getattr(layer, "ple_embedding", None), "_uva_lookup", None)
            if uva is not None:
                layer.prepare_host_lookup(
                    input_ids,
                    self.query_start_loc,
                    self.context,
                    dummy=dummy,
                    num_reqs=nreq,
                    num_tokens=int(query_start_loc[-1]),
                )
            else:
                layer.prepare_host_lookup(
                    input_ids,
                    query_start_loc if use_cpu_metadata else self.query_start_loc,
                    context if use_cpu_metadata else self.context,
                    dummy=dummy,
                )
        return dict(query_start_loc=self.query_start_loc, ngram_context=self.context)


def _mrv1_prepare(runner):
    _validate_base(
        runner.vllm_config,
        require_sync=runner.speculative_config is not None,
    )
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
    if runner.vllm_config.scheduler_config.async_scheduling:
        if getattr(runner, "enable_prompt_embeds", False):
            raise NotImplementedError(
                "Async PLE requires token IDs, not prompt embeddings"
            )
        history = getattr(buffers, "async_history", None)
        if history is None:
            history = _AsyncNgramContext(
                runner.max_num_reqs,
                buffers.context_len,
                int(config.eos_token_id),
                runner.device,
            )
            buffers.async_history = history
        context = history.prepare(runner, buffers.context)
    else:
        context = committed_context_cpu(
            batch.token_ids_cpu_tensor[:nreq],
            batch.num_computed_tokens_cpu[:nreq],
            buffers.context_len,
            int(config.eos_token_id),
        )
    use_cpu_metadata = bool(
        runner.vllm_config.additional_config.get("qwen38_ple_cpu_metadata", False)
    )
    lookup_ids = runner.input_ids.gpu
    if (
        use_cpu_metadata
        and runner.vllm_config.additional_config.get("qwen38_ngram_mode", "sharded")
        not in {"uva_broadcast", "uva_sharded", "uva_auto"}
        and runner.speculative_config is None
        and nreq > 0
        and not getattr(runner, "enable_prompt_embeds", False)
        and np.all(
            batch.num_computed_tokens_cpu[:nreq] < batch.num_prompt_tokens[:nreq]
        )
    ):
        # Pure prefill IDs are gathered into this pinned CPU mirror by MRV1.
        # Decode/draft IDs can be updated only on GPU, so retain that source.
        lookup_ids = runner.input_ids.cpu
    prepare_options = {"use_cpu_metadata": use_cpu_metadata}
    if runner.vllm_config.additional_config.get("qwen38_ngram_mode", "sharded") in {
        "uva_broadcast",
        "uva_sharded",
        "uva_auto",
    }:
        # MRV1 already queued this GPU mirror before updating the GPU history.
        # A D2D copy preserves our fixed forward buffer without a blocking H2D.
        prepare_options["gpu_query_start_loc"] = runner.query_start_loc.gpu
    return buffers.prepare(
        lookup_ids,
        runner.query_start_loc.cpu[: nreq + 1],
        context,
        **prepare_options,
    )
