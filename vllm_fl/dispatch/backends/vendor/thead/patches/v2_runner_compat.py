# Copyright (c) 2026 BAAI. All rights reserved.

"""T-Head PPU adaptations for the upstream V2 model runner.

The upstream V2 runner is written for a CUDA build. It stages host-side
updates through ``UvaBuffer``, which needs pinned memory that also maps into
the device address space, and it captures graphs through the context manager
bound at import time by ``cudagraph_utils``, which asserts on a CUDA/Xpu
communicator.

Neither holds on PPU with a ``VLLM_TARGET_DEVICE=empty`` build: there is no
``_C::get_cuda_view_from_cpu_tensor``, and FlagCX is the collective backend.
Both adaptations below run before the V2 runner is constructed and leave the
upstream behaviour untouched wherever the probe succeeds.

Removal condition: delete this module once the vLLM build exposes
``_C::get_cuda_view_from_cpu_tensor`` on PPU, or once FlagCX satisfies the
communicator assertion in ``cudagraph_utils.graph_capture``.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def mapped_host_buffers_available() -> bool:
    """Whether this build can expose pinned host memory as a device view.

    ``is_uva_available()`` only reports the pinned-memory capability, while the
    V2 buffers also need the memory to be *mapped*. Probe the op itself: a
    backend can pin memory without being able to map it into its device address
    space.
    """
    from vllm.utils.platform_utils import is_uva_available
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    if not is_uva_available():
        return False
    try:
        get_accelerator_view_from_cpu_tensor(
            torch.zeros(1, dtype=torch.int32, pin_memory=True)
        )
    except RuntimeError:
        logger.warning_once(
            "Pinned host memory cannot be mapped into the device address space "
            "by this vLLM build; the V2 runner will copy its UVA buffers "
            "explicitly instead."
        )
        return False
    return True


def _uva_unavailable() -> bool:
    """Stand-in for ``is_uva_available`` on builds without mapped host memory."""
    return False


def install_v2_runner_compat() -> None:
    """Adapt the upstream V2 runner internals that assume a CUDA build/runtime.

    Call before the V2 runner is constructed so that both buffers and graph
    capture are created with the adapted helpers.
    """
    from vllm.v1.worker.gpu import buffer_utils

    # Only override the answer when this build cannot map host memory, so CUDA
    # builds keep the upstream path untouched.
    if not mapped_host_buffers_available():
        buffer_utils.is_uva_available = _uva_unavailable

    # ``cudagraph_utils`` binds the process-group context manager at import
    # time; reuse the device-agnostic context that the V1 ``ModelRunnerFL``
    # resolves for this platform instead.
    from vllm.v1.worker.gpu import cudagraph_utils

    from vllm_fl.worker.model_runner import graph_capture as fl_graph_capture

    cudagraph_utils.graph_capture = fl_graph_capture
