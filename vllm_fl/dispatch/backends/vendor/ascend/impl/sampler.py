# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ascend top-k/top-p and opt-in sampling overlap integration."""

import torch
from vllm import envs
from vllm.config.model import LogprobsMode
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p_pytorch,
    random_sample,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.utils import record_function_or_nullcontext

_ASCEND_TOP_K_TOP_P_OP = "_C_ascend::npu_apply_top_k_top_p"
_GLOBAL_RANDOM_SAMPLE_STREAM = None


def _fill_exponential_(
    q: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> None:
    """Fill every row while preserving per-request generator semantics."""
    batch_size = q.shape[0]
    invalid_indices = [index for index in generators if not 0 <= index < batch_size]
    if invalid_indices:
        raise IndexError(
            "Generator indices must be within the sampling batch: "
            f"batch_size={batch_size}, invalid={invalid_indices}"
        )

    # None/partial generators need a default draw for every row. When every
    # row has its own generator, avoid the redundant unseeded fill.
    if len(generators) != batch_size:
        q.exponential_()
    for index, generator in generators.items():
        q[index].exponential_(generator=generator)


def _global_random_sample_stream():
    """Return the process-local random sampling stream, creating it lazily."""
    global _GLOBAL_RANDOM_SAMPLE_STREAM
    if _GLOBAL_RANDOM_SAMPLE_STREAM is None:
        _GLOBAL_RANDOM_SAMPLE_STREAM = torch.npu.Stream()
    return _GLOBAL_RANDOM_SAMPLE_STREAM


def global_stream_random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Match vLLM-Ascend's default random sampling stream ordering."""
    stream = _global_random_sample_stream()
    with record_function_or_nullcontext("sampler: exponential_submit"):
        with torch.npu.stream(stream):
            q = torch.empty_like(probs)
            _fill_exponential_(q, generators)
    with record_function_or_nullcontext("sampler: wait_random_stream"):
        torch.npu.current_stream().wait_stream(stream)
    with record_function_or_nullcontext("sampler: div_argmax"):
        return probs.div_(q).argmax(dim=-1).view(-1)


def _has_ascend_top_k_top_p_kernel() -> bool:
    """Check both the operator schema and its NPU dispatch kernel."""
    has_kernel = getattr(
        torch._C,
        "_dispatch_has_kernel_for_dispatch_key",
        None,
    )
    if has_kernel is None:
        return False
    try:
        return has_kernel(_ASCEND_TOP_K_TOP_P_OP, "PrivateUse1")
    except RuntimeError:
        # PyTorch raises when the qualified operator schema is not registered.
        return False


def _get_ascend_top_k_top_p_op():
    """Return the AscendC sampler op only when its NPU kernel is registered."""
    if not _has_ascend_top_k_top_p_kernel():
        return None
    ascend_ops = getattr(torch.ops, "_C_ascend", None)
    if ascend_ops is None:
        return None
    return getattr(ascend_ops, "npu_apply_top_k_top_p", None)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Apply Ascend top-k/top-p, with a fixed PyTorch host fallback."""
    if k is None and p is None:
        return logits

    # vLLM-Ascend intentionally avoids its device-specific op in batch
    # invariant mode. Use the fixed PyTorch implementation directly instead
    # of vLLM's dynamic wrapper, which may dispatch to Triton on large batches.
    if envs.VLLM_BATCH_INVARIANT:
        return apply_top_k_top_p_pytorch(logits, k, p)

    ascend_op = _get_ascend_top_k_top_p_op()
    if ascend_op is not None:
        return ascend_op(logits, k=k, p=p)

    # The extension schema can be absent in a Python-only installation. Never
    # fall back through vLLM's apply_top_k_top_p because B >= 8 can select its
    # CUDA-oriented Triton implementation on Ascend.
    return apply_top_k_top_p_pytorch(logits, k, p)


class AscendTopKTopPSampler(TopKTopPSampler):
    """Top-k/top-p sampler using the AscendC filter when it is available."""

    def __init__(
        self,
        *,
        enable_global_stream_random_sample: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.enable_global_stream_random_sample = (
            enable_global_stream_random_sample
        )
        # NPU resources and the random tensor are intentionally lazy so
        # importing/constructing this sampler remains CPU-testable.
        self._async_exponential_stream = None
        self._async_exponential_event = None
        self._async_exponential_device: torch.device | None = None
        self._pending_q: torch.Tensor | None = None
        self._pending_q_event = None

    @property
    def has_pending_async_exponential(self) -> bool:
        return self._pending_q is not None

    def _ensure_async_resources(self, device: torch.device) -> tuple[object, object]:
        if self._async_exponential_stream is None:
            self._async_exponential_stream = torch.npu.Stream(device=device)
            self._async_exponential_event = torch.npu.Event()
            self._async_exponential_device = device
        elif self._async_exponential_device != device:
            raise RuntimeError(
                "Async exponential resources cannot change device: "
                f"created={self._async_exponential_device}, requested={device}"
            )
        assert self._async_exponential_event is not None
        return self._async_exponential_stream, self._async_exponential_event

    def prepare_async_exponential(
        self,
        batch_size: int,
        vocab_size: int,
        generators: dict[int, torch.Generator],
        device: torch.device,
    ) -> None:
        """Launch FP32 exponential random generation on a dedicated stream."""
        if envs.VLLM_BATCH_INVARIANT:
            raise RuntimeError(
                "Async exponential is disabled by VLLM_BATCH_INVARIANT=1"
            )
        if self.has_pending_async_exponential:
            raise RuntimeError(
                "Async exponential prepare called before consuming pending q"
            )
        if batch_size <= 0 or vocab_size <= 0:
            raise ValueError(
                "Async exponential shape must be positive: "
                f"batch_size={batch_size}, vocab_size={vocab_size}"
            )

        device = torch.device(device)
        stream, event = self._ensure_async_resources(device)
        current_stream = torch.npu.current_stream(device)
        with record_function_or_nullcontext("sampler: exponential_submit"):
            stream.wait_stream(current_stream)
            with torch.npu.stream(stream):
                q = torch.empty(
                    (batch_size, vocab_size),
                    device=device,
                    dtype=torch.float32,
                )
                _fill_exponential_(q, generators)
                event.record()

        self._pending_q = q
        self._pending_q_event = event

    @staticmethod
    def _record_q_stream(q: torch.Tensor) -> None:
        q.record_stream(torch.npu.current_stream(q.device))

    def discard_async_exponential(self) -> bool:
        """Discard a prepared q after an upstream request failure."""
        q = self._pending_q
        event = self._pending_q_event
        self._pending_q = None
        self._pending_q_event = None
        if q is None:
            return False

        # The producer may still be using q. Finish it before releasing the
        # last reference; this API only runs on exceptional request paths.
        if event is not None:
            event.synchronize()
        return True

    def _consume_async_exponential(
        self,
        probs: torch.Tensor,
    ) -> torch.Tensor | None:
        q = self._pending_q
        event = self._pending_q_event
        if q is None:
            return None
        if event is None:
            self._pending_q = None
            raise RuntimeError("Async exponential state has q without an event")

        # One prepare has exactly one consume, including error paths. Use an
        # event wait here rather than switching sampling to a global stream.
        self._pending_q = None
        self._pending_q_event = None
        with record_function_or_nullcontext("sampler: wait_random_stream"):
            event.synchronize()

        mismatches: list[str] = []
        if q.shape != probs.shape:
            mismatches.append(f"shape q={tuple(q.shape)} probs={tuple(probs.shape)}")
        if q.device != probs.device:
            mismatches.append(f"device q={q.device} probs={probs.device}")
        if q.dtype != torch.float32:
            mismatches.append(f"dtype q={q.dtype} expected={torch.float32}")
        if probs.dtype != torch.float32:
            mismatches.append(f"probs_dtype={probs.dtype} expected={torch.float32}")
        if mismatches:
            raise RuntimeError(
                "Prepared async exponential q does not match sampling probs: "
                + "; ".join(mismatches)
            )

        self._record_q_stream(q)
        return q

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = apply_top_k_top_p(logits, k, p)

        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        with record_function_or_nullcontext("sampler: softmax"):
            probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = self._consume_async_exponential(probs)
        if q is not None:
            with record_function_or_nullcontext("sampler: div_argmax"):
                sampled = probs.div_(q).argmax(dim=-1).view(-1)
        elif (
            self.enable_global_stream_random_sample
            and not envs.VLLM_BATCH_INVARIANT
        ):
            sampled = global_stream_random_sample(probs, generators)
        else:
            sampled = random_sample(probs, generators)
        return sampled, logits_to_return


class AscendSampler(Sampler):
    """vLLM sampler with Ascend filtering and optional random overlap."""

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        *,
        enable_global_stream_random_sample: bool = True,
    ) -> None:
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler(
            logprobs_mode=logprobs_mode,
            enable_global_stream_random_sample=(
                enable_global_stream_random_sample
            ),
        )

    def prepare_async_exponential(
        self,
        batch_size: int,
        vocab_size: int,
        generators: dict[int, torch.Generator],
        device: torch.device,
    ) -> None:
        self.topk_topp_sampler.prepare_async_exponential(
            batch_size,
            vocab_size,
            generators,
            device,
        )

    def discard_async_exponential(self) -> bool:
        return self.topk_topp_sampler.discard_async_exponential()
