# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-shape accelerator task parameters for full graph replay.

The graph runtime owns the lifetime of this registry. Attention backends own
the operator-specific parameter records stored in it. Keeping the registry in
``compilation`` lets every full-graph runner use the same capture lifecycle
without pulling vendor operator implementations into the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphParams:
    """Captured device-task state indexed by the full-graph token shape."""

    events: dict[int, list[Any]] = field(default_factory=dict)
    workspaces: dict[int, Any | None] = field(default_factory=dict)
    handles: dict[int, list[Any]] = field(default_factory=dict)
    attn_params: dict[int, list[Any]] = field(default_factory=dict)

    def prepare_shape(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            raise ValueError(f"Graph capture size must be positive, got {num_tokens}")
        if num_tokens in self.attn_params and (
            self.attn_params[num_tokens]
            or self.handles[num_tokens]
            or self.events[num_tokens]
        ):
            raise RuntimeError(
                f"Graph task parameters for capture size {num_tokens} "
                "already contain captured state"
            )
        self.events[num_tokens] = []
        self.workspaces[num_tokens] = None
        self.handles[num_tokens] = []
        self.attn_params[num_tokens] = []

    def require_shape(self, num_tokens: int) -> None:
        if num_tokens not in self.attn_params:
            raise RuntimeError(
                f"Graph task parameters were not prepared for capture size {num_tokens}"
            )

    def clear(self) -> None:
        self.events.clear()
        self.workspaces.clear()
        self.handles.clear()
        self.attn_params.clear()


_GRAPH_PARAMS = GraphParams()


def get_graph_params() -> GraphParams:
    return _GRAPH_PARAMS


def prepare_graph_params(num_tokens: int) -> GraphParams:
    _GRAPH_PARAMS.prepare_shape(num_tokens)
    return _GRAPH_PARAMS


def clear_graph_params() -> None:
    _GRAPH_PARAMS.clear()


def weak_ref_workspace(num_tokens: int) -> None:
    """Drop ownership while preserving the address retained by the NPU graph."""
    params = get_graph_params()
    params.require_shape(num_tokens)
    workspace = params.workspaces[num_tokens]
    if workspace is None:
        return
    try:
        from vllm.utils.torch_utils import weak_ref_tensors

        weak_workspace = weak_ref_tensors(workspace)
    except Exception:
        weak_workspace = workspace
    params.workspaces[num_tokens] = weak_workspace
