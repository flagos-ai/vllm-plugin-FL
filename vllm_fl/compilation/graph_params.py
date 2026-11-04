# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-descriptor accelerator task parameters for full graph replay.

The graph runtime owns the lifetime of this registry. Attention backends own
the operator-specific parameter records stored in it. Keeping the registry in
``compilation`` lets every full-graph runner use the same capture lifecycle
without pulling vendor operator implementations into the runner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.forward_context import BatchDescriptor


@dataclass
class GraphParams:
    """Captured device-task state indexed like GraphWrapper's graph entries."""

    events: dict[BatchDescriptor, list[Any]] = field(default_factory=dict)
    workspaces: dict[BatchDescriptor, Any | None] = field(default_factory=dict)
    handles: dict[BatchDescriptor, list[Any]] = field(default_factory=dict)
    attn_params: dict[BatchDescriptor, list[Any]] = field(default_factory=dict)

    def prepare_capture(self, descriptor: BatchDescriptor) -> None:
        num_tokens = descriptor.num_tokens
        if (
            not isinstance(num_tokens, int)
            or isinstance(num_tokens, bool)
            or num_tokens <= 0
        ):
            raise ValueError(f"Graph capture size must be positive, got {num_tokens}")
        if descriptor in self.attn_params and (
            self.attn_params[descriptor]
            or self.handles[descriptor]
            or self.events[descriptor]
        ):
            raise RuntimeError(
                f"Graph task parameters for {descriptor} already contain captured state"
            )
        self.events[descriptor] = []
        self.workspaces[descriptor] = None
        self.handles[descriptor] = []
        self.attn_params[descriptor] = []

    def require_capture(self, descriptor: BatchDescriptor) -> None:
        if descriptor not in self.attn_params:
            raise RuntimeError(
                f"Graph task parameters were not prepared for {descriptor}"
            )

    def clear(self) -> None:
        self.events.clear()
        self.workspaces.clear()
        self.handles.clear()
        self.attn_params.clear()


_GRAPH_PARAMS = GraphParams()


def get_graph_params() -> GraphParams:
    return _GRAPH_PARAMS


def prepare_graph_params(descriptor: BatchDescriptor) -> GraphParams:
    _GRAPH_PARAMS.prepare_capture(descriptor)
    return _GRAPH_PARAMS


def clear_graph_params() -> None:
    _GRAPH_PARAMS.clear()


def weak_ref_workspace(
    descriptor: BatchDescriptor, weak_ref: Callable[[Any], Any]
) -> None:
    """Drop ownership while preserving the address retained by the NPU graph."""
    params = get_graph_params()
    params.require_capture(descriptor)
    workspace = params.workspaces[descriptor]
    if workspace is None:
        return
    params.workspaces[descriptor] = weak_ref(workspace)
