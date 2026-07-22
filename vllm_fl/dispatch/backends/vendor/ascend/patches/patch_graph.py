# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/compilation/acl_graph.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Ascend-specific ACL graph extensions for vllm-plugin-FL.

This module is intentionally separated from the generic graph wrapper so that
Ascend behavior (stream sync, graph-param workspaces, capture-error diagnosis,
etc.) is injected at runtime rather than hard-coded into the multi-hardware
framework.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import weakref
from dataclasses import dataclass
from typing import Any, ClassVar, Optional
from unittest.mock import patch

import torch

from vllm.compilation.counter import compilation_counter
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.platforms import current_platform

from vllm_fl.compilation.graph import register_graph_wrapper_backend

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Stream-resource capture error diagnostics (CANN error 207008)
# --------------------------------------------------------------------------- #
_STREAM_RESOURCE_ERROR_CODE = "207008"
_STREAM_RESOURCE_ERROR_MARKERS = (
    "insufficient_stream_resources",
    "stream resources are insufficient",
)
_STREAM_RESOURCE_GUIDANCE = (
    "ACL graph capture failed with a known stream-resource exhaustion "
    "signature. Consider upgrading to a newer HDK/CANN stack, reducing "
    "cudagraph_capture_sizes, lowering max_cudagraph_capture_size, preferring "
    "FULL or FULL_DECODE_ONLY for mostly uniform decode workloads, or "
    "temporarily disabling graph mode to confirm the failure is capture-related."
)


def _is_stream_resource_capture_error(exc: RuntimeError) -> bool:
    message = str(exc)
    lowered_message = message.lower()
    has_error_code = _STREAM_RESOURCE_ERROR_CODE in message
    has_stream_resource_marker = any(
        marker in lowered_message for marker in _STREAM_RESOURCE_ERROR_MARKERS)
    return has_stream_resource_marker or (has_error_code
                                          and "stream resource" in lowered_message)


def _raise_stream_resource_capture_error(exc: RuntimeError) -> None:
    raise RuntimeError(
        f"{_STREAM_RESOURCE_GUIDANCE}\nOriginal error:\n{exc}") from exc


# --------------------------------------------------------------------------- #
# Graph parameter bookkeeping for attention/MLA workspace reuse across captures
# --------------------------------------------------------------------------- #
@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[Any]]
    attn_params: dict[int, list[tuple]]
    conv1d_params: dict[int, list[tuple]]  # for causal conv1d params
    conv1d_handles: dict[int, list[Any]]  # for causal conv1d params handles
    conv1d_events: dict[int, list[torch.npu.ExternalEvent]]  # for causal conv1d params events


_graph_params: Optional[GraphParams] = None
_draft_graph_params: Optional[GraphParams] = None
_draft_graph_prefill_params: Optional[GraphParams] = None

def reset_graph_params() -> None:
    global _graph_params, _draft_graph_params, _draft_graph_prefill_params
    _graph_params = None
    _draft_graph_params = None
    _draft_graph_prefill_params = None


def _make_empty_graph_params(capture_sizes: list[int]) -> GraphParams:
    return GraphParams(
        {size: [] for size in capture_sizes},
        {size: None for size in capture_sizes},
        {size: [] for size in capture_sizes},
        {size: [] for size in capture_sizes},
        {size: [] for size in capture_sizes},
        {size: [] for size in capture_sizes},
        {size: [] for size in capture_sizes},
    )


def set_graph_params(aclgraph_capture_sizes: list[int]) -> None:
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = _make_empty_graph_params(aclgraph_capture_sizes)


def ensure_graph_params(aclgraph_capture_sizes: list[int]) -> GraphParams:
    """Initialize graph bookkeeping and add any newly resolved capture sizes.

    The final ACL graph sizes are not known until attention backends have
    resolved the cudagraph mode.  This helper is deliberately idempotent so a
    worker can run that resolution more than once without losing graph handles
    that have already been registered.
    """
    global _graph_params
    if _graph_params is None:
        _graph_params = _make_empty_graph_params(aclgraph_capture_sizes)
        return _graph_params

    for size in aclgraph_capture_sizes:
        _graph_params.events.setdefault(size, [])
        _graph_params.workspaces.setdefault(size, None)
        _graph_params.handles.setdefault(size, [])
        _graph_params.attn_params.setdefault(size, [])
        _graph_params.conv1d_params.setdefault(size, [])
        _graph_params.conv1d_handles.setdefault(size, [])
        _graph_params.conv1d_events.setdefault(size, [])
    return _graph_params


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor) -> None:
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = workspace


def get_graph_params() -> Optional[GraphParams]:
    return _graph_params


def set_draft_graph_params(aclgraph_capture_sizes: list[int]) -> None:
    global _draft_graph_params
    if _draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    _draft_graph_params = _make_empty_graph_params(aclgraph_capture_sizes)


def update_draft_graph_params_workspaces(num_tokens: int, workspace: Any) -> None:
    global _draft_graph_params
    if _draft_graph_params is not None:
        _draft_graph_params.workspaces[num_tokens] = workspace


def get_draft_graph_params() -> Optional[GraphParams]:
    return _draft_graph_params


def set_draft_graph_prefill_params(aclgraph_capture_sizes: list[int]) -> None:
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        raise ValueError("DraftGraph prefill parameters have already been set!")
    _draft_graph_prefill_params = _make_empty_graph_params(aclgraph_capture_sizes)


def update_draft_graph_prefill_params_workspaces(num_tokens: int,
                                                  workspace: Any) -> None:
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        _draft_graph_prefill_params.workspaces[num_tokens] = workspace


def get_draft_graph_prefill_params() -> Optional[GraphParams]:
    return _draft_graph_prefill_params


def weak_ref_tensors(tensor: Any) -> Any:
    """Convert tensors to weak references to save memory during graph replay."""
    from vllm_fl.compilation.graph import weak_ref_tensors as _generic_weak_ref
    return _generic_weak_ref(tensor)


def weak_ref_workspaces(params: Optional[GraphParams]) -> None:
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            continue
        params.workspaces[num_tokens] = weak_ref_tensors(
            params.workspaces[num_tokens])


def _has_runtime_graph_updates(
    params: Optional[GraphParams], num_tokens: Optional[int]
) -> bool:
    """Whether replay can race with graph-task parameter updates.

    The host-blocking replay barrier inherited from vLLM-Ascend is only
    required when captured attention/conv1d graph tasks are updated on a
    separate stream. Empty maps let graph paths without such tasks skip it.
    """
    if params is None:
        return False

    runtime_update_maps = (
        params.events,
        params.handles,
        params.attn_params,
        params.conv1d_events,
        params.conv1d_handles,
        params.conv1d_params,
    )
    if num_tokens is None:
        return any(
            any(bucket for bucket in values.values())
            for values in runtime_update_maps
        )
    return any(values.get(num_tokens) for values in runtime_update_maps)


def update_full_graph_params(
    attn_backends,
    update_stream,
    forward_context,
    num_tokens: int,
    vllm_config: VllmConfig,
    speculative_config=None,
    num_dcp_pcp_tokens: Optional[int] = None,
    draft_attn_metadatas=None,
) -> None:
    """Dispatch graph-param updates to the attention backend and GDN conv1d."""
    if not isinstance(attn_backends, (list, tuple, set)):
        attn_backends = (attn_backends,)

    updated_impls = set()
    for attn_backend in attn_backends:
        try:
            impl_cls = attn_backend.get_impl_cls()
        except NotImplementedError:
            # Hybrid state-space/GDN backends may implement attention through
            # custom ops without exposing the standard AttentionImpl hook.
            # Their graph-task updates are handled by the conv1d path below.
            continue
        if impl_cls in updated_impls or not hasattr(impl_cls, "update_graph_params"):
            continue
        updated_impls.add(impl_cls)
        impl_cls.update_graph_params(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config,
            num_dcp_pcp_tokens,
            draft_attn_metadatas,
        )

    has_attention_tasks = any(
        params is not None and params.handles.get(num_tokens)
        for params in (
            get_graph_params(),
            get_draft_graph_params(),
            get_draft_graph_prefill_params(),
        )
    )
    if has_attention_tasks and not updated_impls:
        raise RuntimeError(
            "FULL ACL graph has captured attention task handles, but no "
            "attention graph-parameter updater was resolved. Replaying would "
            "deadlock while waiting for an unrecorded ExternalEvent."
        )

    # Optional GDN conv1d update (only available when vllm-ascend gdn is present).
    try:
        from vllm_ascend.ops.gdn import update_conv1d_graph_params
        update_conv1d_graph_params(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            getattr(forward_context, "is_draft_model", False),
            draft_attn_metadatas,
        )
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Ascend backend mixin for GraphWrapper
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class _ACLGraphEntry:
    """Internal entry used by the mixin; mirrors generic GraphEntry fields."""
    batch_descriptor: BatchDescriptor
    aclgraph: Any | None = None
    output: Any | None = None
    input_addresses: Optional[list[int]] = None


class ACLGraphBackendMixin:
    """
    Backend-specific mixin that supplies Ascend ACL graph behavior to the
    generic GraphWrapper.

    The mixin is instantiated once per GraphWrapper and receives hook calls
    during capture and replay.  It mirrors the workflow of
    `vllm_ascend.compilation.acl_graph.ACLGraphWrapper` but stays out of the
    generic code path.
    """

    _all_instances: ClassVar[weakref.WeakSet["ACLGraphBackendMixin"]] = weakref.WeakSet()

    @classmethod
    def clear_all_graphs(cls) -> None:
        for instance in list(cls._all_instances):
            instance.wrapper.concrete_graph_entries.clear()

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.vllm_config = wrapper.vllm_config
        self.runtime_mode = wrapper.runtime_mode
        self.aclgraph_options = wrapper.graph_options
        self.use_eagle = getattr(wrapper, "use_eagle", False)
        self.enable_enpu = getattr(wrapper, "enable_enpu", False)
        # Emergency rollback/A-B switch for stacks that update graph tasks
        # outside the GraphParams bookkeeping used below.
        self.force_replay_sync = os.environ.get(
            "VLLM_FL_FORCE_ACLGRAPH_REPLAY_SYNC", "0"
        ).lower() in ("1", "true", "yes", "on")
        self.is_debugging_mode = wrapper.is_debugging_mode
        self._runnable_str = str(
            wrapper.runnable) if self.is_debugging_mode else None
        ACLGraphBackendMixin._all_instances.add(self)

    def _is_stream_resource_capture_error(self, exc: RuntimeError) -> bool:
        return _is_stream_resource_capture_error(exc)

    def _sync_offloader_before_capture(self) -> None:
        try:
            from vllm.model_executor.offloader.base import get_offloader
            get_offloader().sync_prev_onload()
        except Exception:
            pass

    def _join_offloader_after_forward(self) -> None:
        try:
            from vllm.model_executor.offloader.base import get_offloader
            get_offloader().join_after_forward()
        except Exception:
            pass

    def before_capture(self, entry, args, kwargs) -> None:
        self._sync_offloader_before_capture()
        forward_context = get_forward_context()
        self._previous_capturing = getattr(forward_context, "capturing", False)
        forward_context.capturing = True

    def wrap_capture_context(self, entry, stack) -> None:
        # For NPU, torch.npu.empty_cache is the function that needs to be
        # disabled when gc_disable is enabled.  The generic wrapper already
        # patches PlatformFL.empty_cache; patch torch.npu.empty_cache as well.
        if self.aclgraph_options.gc_disable:
            stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

    def after_capture(self, entry, output, args, kwargs) -> Any:
        self._join_offloader_after_forward()

        forward_context = get_forward_context()
        forward_context.capturing = getattr(
            self, "_previous_capturing", False
        )

        # Convert attention workspace tensors to weak refs to save memory.
        weak_ref_workspaces(get_graph_params())
        weak_ref_workspaces(get_draft_graph_params())
        weak_ref_workspaces(get_draft_graph_prefill_params())

        # The generic wrapper will weak-ref the output again; return the
        # original output so PyTorch can manage memory correctly during capture.
        return output

    def capture_error_handler(self, exc: BaseException) -> None:
        try:
            forward_context = get_forward_context()
            forward_context.capturing = getattr(
                self, "_previous_capturing", False
            )
        except Exception:
            pass
        if isinstance(exc, RuntimeError) and self._is_stream_resource_capture_error(exc):
            _raise_stream_resource_capture_error(exc)

    def before_replay(self, entry, args, kwargs) -> None:
        # In async scheduling or multi-threaded scenarios, ensure host-side
        # attention-param updates stay ordered with graph execution.
        # When enable_enpu is on, model_runner orders update vs replay; skip.
        # When FULL + EAGLE draft (merge path), replay does not need barrier.
        is_draft_eagle = False
        try:
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX
            is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        except Exception:
            pass

        need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
        num_tokens = getattr(entry.batch_descriptor, "num_tokens", None)
        has_runtime_updates = any(
            _has_runtime_graph_updates(params, num_tokens)
            for params in (
                get_graph_params(),
                get_draft_graph_params(),
                get_draft_graph_prefill_params(),
            )
        )
        if (
            not self.enable_enpu
            and need_sync
            and (self.force_replay_sync or has_runtime_updates)
        ):
            torch.npu.current_stream().synchronize()

    def weak_ref_tensors(self, tensor: Any) -> Any:
        # Ascend does not yet have a dedicated weak-ref csrc op; fall back to
        # the generic implementation which currently returns the tensor as-is.
        return tensor


def patch_graph() -> None:
    """Register the Ascend ACL graph backend mixin."""
    if current_platform.device_type != "npu":
        logger.info(
            "Skipping ACL graph patch: current platform is not NPU (%s)",
            current_platform.device_type)
        return
    register_graph_wrapper_backend("npu", ACLGraphBackendMixin)
    logger.info("Registered Ascend ACL graph backend mixin for GraphWrapper")
