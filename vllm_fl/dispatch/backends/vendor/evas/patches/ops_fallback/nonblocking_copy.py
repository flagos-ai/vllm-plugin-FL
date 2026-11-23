# Copyright (c) 2026 BAAI. All rights reserved.
"""Force blocking CPU->LPU ``Tensor.copy_`` (drop ``non_blocking=True``).

H2D copies with ``non_blocking=True`` from a temporary pinned CPU source
corrupt the device buffer or wedge the copy engine. The split is source
lifetime, which a wrapper cannot observe, so every CPU->LPU ``copy_`` is
downgraded to blocking — a recycled source can never race the DMA. D2D copies
are left untouched, and upstream files are not modified.

"""

from __future__ import annotations


def apply_lpu_nonblocking_copy_patch() -> None:
    """Wrap ``torch.Tensor.copy_``: CPU->LPU copies are forced blocking.

    Idempotent — safe to call multiple times.
    """
    import torch

    if getattr(torch, "_lpu_nonblocking_copy_patched", False):
        return

    _orig_copy_ = torch.Tensor.copy_

    def _lpu_copy_(self, *args, **kwargs):
        # Accept both positional and keyword non_blocking (torch.Tensor.copy_
        # signature: copy_(src, non_blocking=False)).
        non_blocking = kwargs.get("non_blocking", False)
        if len(args) >= 2:
            non_blocking = args[1]
        if non_blocking:
            src = args[0] if args else kwargs.get("src", kwargs.get("other"))
            src_device = getattr(src, "device", None) if src is not None else None
            if (
                self.device.type == "lpu"
                and src_device is not None
                and src_device.type == "cpu"
            ):
                # CPU -> LPU: drop non_blocking (see module docstring).
                if len(args) >= 2:
                    args = (args[0], False) + args[2:]
                else:
                    kwargs["non_blocking"] = False
        return _orig_copy_(self, *args, **kwargs)

    torch.Tensor.copy_ = _lpu_copy_
    torch._lpu_nonblocking_copy_patched = True  # type: ignore[attr-defined]


def apply_lpu_verified_copy_to_gpu_patch() -> None:
    """Wrap ``CpuGpuBuffer.copy_to_gpu`` with read-back verification + retry.

    run15 step-probe evidence: TP2's blocking input-prep copies (qsl,
    input_ids) silently did NOT land — the device buffers still held stale
    garbage (other tensors' float bits) when the forward consumed them,
    wedging the device. The retry is a fresh blocking copy; verification is
    a D2H read-back compared against the CPU source (the buffers are small —
    qsl/seq_lens/input_ids — so the read-back cost is negligible).
    """
    import torch
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if getattr(torch, "_lpu_verified_copy_to_gpu_patched", False):
        return

    from vllm.v1.utils import CpuGpuBuffer

    _orig_copy_to_gpu = CpuGpuBuffer.copy_to_gpu

    def _verified_copy_to_gpu(self, n: int | None = None):
        res = _orig_copy_to_gpu(self, n)
        if self.gpu.device.type != "lpu":
            return res
        for _attempt in range(8):
            cpu_view = self.cpu if n is None else self.cpu[:n]
            gpu_view = self.gpu if n is None else self.gpu[:n]
            got = gpu_view.cpu()
            if torch.equal(got, cpu_view):
                if _attempt:
                    logger.warning(
                        "[lpu-copy] CpuGpuBuffer %s H2D landed after %d "
                        "retries", tuple(self.gpu.shape), _attempt,
                    )
                return res
            # Re-copy (fresh blocking transfer) and verify again.
            res = gpu_view.copy_(cpu_view, non_blocking=False)
        logger.error(
            "[lpu-copy] CpuGpuBuffer %s H2D never verified after 8 "
            "attempts; proceeding", tuple(self.gpu.shape),
        )
        return res

    CpuGpuBuffer.copy_to_gpu = _verified_copy_to_gpu  # type: ignore[method-assign]
    torch._lpu_verified_copy_to_gpu_patched = True  # type: ignore[attr-defined]
