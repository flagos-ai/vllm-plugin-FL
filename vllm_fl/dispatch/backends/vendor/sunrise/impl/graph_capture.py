# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Sunrise stream adapters for vLLM 0.28 distributed graph capture."""

import sys
from contextlib import contextmanager, nullcontext
from functools import wraps
from types import ModuleType


def _replace_cached_aliases(name, original, replacement):
    # Consumers can import these helpers before worker initialization. Update
    # only globals that still reference the exact upstream function.
    for module in tuple(sys.modules.values()):
        if isinstance(module, ModuleType):
            namespace = vars(module)
            if namespace.get(name) is original:
                namespace[name] = replacement


def patch_graph_capture_runtime():
    """Keep TP/PP and communicator capture contexts on the live PTPU stream."""
    from vllm.distributed import parallel_state
    from vllm.platforms import current_platform
    from vllm.utils import torch_utils

    if current_platform.device_type != "ptpu":
        return
    group_cls = parallel_state.GroupCoordinator
    if getattr(group_cls.graph_capture, "_sunrise_graph_capture_patched", False):
        return

    original_group_capture = group_cls.graph_capture
    original_graph_capture = parallel_state.graph_capture
    original_current_stream = torch_utils.current_stream

    @wraps(original_current_stream)
    def current_stream():
        if current_platform.device_type == "ptpu":
            # vLLM's CUDA TLS cache is not updated by torch.ptpu.stream().
            # This also keeps FlagCX collectives on the stream being captured.
            return current_platform.torch_device_fn.current_stream()
        return original_current_stream()

    @contextmanager
    @wraps(original_group_capture)
    def group_graph_capture(self, graph_capture_context=None):
        if current_platform.device_type != "ptpu":
            with original_group_capture(self, graph_capture_context) as context:
                yield context
            return

        device_fn = current_platform.torch_device_fn
        context = graph_capture_context
        if context is None:
            context = parallel_state.GraphCaptureContext(device_fn.Stream())
        stream = context.stream

        # CommunicatorFL uses FlagCX directly and has no CUDA custom allreduce
        # object. Retain a communicator's capture hooks when it provides them.
        maybe_ca_context = nullcontext()
        maybe_aiter_context = nullcontext()
        communicator = self.device_communicator
        if communicator is not None:
            ca_comm = getattr(communicator, "ca_comm", None)
            if ca_comm is not None:
                maybe_ca_context = ca_comm.capture()

            from vllm._aiter_ops import rocm_aiter_ops

            if rocm_aiter_ops.is_enabled():
                aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
                if aiter_ar is not None:
                    maybe_aiter_context = aiter_ar.capture()

        previous_stream = device_fn.current_stream()
        if previous_stream != stream:
            stream.wait_stream(previous_stream)
        with device_fn.stream(stream), maybe_ca_context, maybe_aiter_context:
            yield context

    @contextmanager
    @wraps(original_graph_capture)
    def graph_capture(device, graph_capture_context=None):
        if current_platform.device_type != "ptpu":
            with original_graph_capture(device, graph_capture_context) as context:
                yield context
            return

        context = graph_capture_context
        if context is None:
            context = parallel_state.GraphCaptureContext(
                current_platform.torch_device_fn.Stream(device=device)
            )
        with (
            parallel_state.get_tp_group().graph_capture(context),
            parallel_state.get_pp_group().graph_capture(context),
        ):
            yield context

    group_graph_capture._sunrise_graph_capture_patched = True
    group_cls.graph_capture = group_graph_capture
    _replace_cached_aliases("graph_capture", original_graph_capture, graph_capture)
    _replace_cached_aliases("current_stream", original_current_stream, current_stream)
