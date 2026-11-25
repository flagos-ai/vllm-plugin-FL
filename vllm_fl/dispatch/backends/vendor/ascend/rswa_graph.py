# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner-owned state for Ascend paged R-SWA full decode graphs."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from vllm.platforms import current_platform

from vllm_fl.compilation.graph_params import GraphParams
from vllm_fl.compilation.graph_runtime import GraphRuntimeController


@dataclass(frozen=True)
class RswaGraphBuffers:
    """One runner's fixed-address inputs and workspace for decode buckets."""

    capture_sizes: frozenset[int]
    attn_mask: Any
    block_table: Any
    workspace: Any

    @property
    def nbytes(self) -> int:
        return self.attn_mask.nbytes + self.block_table.nbytes + self.workspace.nbytes


@dataclass(kw_only=True)
class RswaGraphParams(GraphParams):
    rswa_buffers: RswaGraphBuffers
    skip_pages_logged: bool = False

    def prepare_shape(self, num_tokens: int) -> None:
        if num_tokens not in self.rswa_buffers.capture_sizes:
            raise RuntimeError(
                f"R-SWA graph buffers were not sized for {num_tokens} tokens"
            )
        super().prepare_shape(num_tokens)
        self.workspaces[num_tokens] = self.rswa_buffers.workspace


class RswaGraphRuntime(GraphRuntimeController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._params: RswaGraphParams | None = None
        self._replay_stream = None

    def reserve_graph_memory(self, device: Any) -> int:
        from .impl.attention import reserve_rswa_graph_buffers

        if self._params is None:
            buffers = reserve_rswa_graph_buffers(self.vllm_config, device)
            if buffers is None:
                return 0
            self.update_stream = current_platform.torch_device_fn.Stream(device=device)
            self._params = RswaGraphParams(rswa_buffers=buffers)
        return self._params.rswa_buffers.nbytes

    @contextmanager
    def _stage_inputs(self, context, num_tokens):
        if self.attention_impls is None or len(self.attention_impls) != 1:
            raise RuntimeError(
                "R-SWA graphs require one bound Ascend attention backend"
            )
        context.rswa_graph_params = self._params
        try:
            self.attention_impls[0].stage_rswa_graph_inputs(context, num_tokens)
            yield
        finally:
            del context.rswa_graph_params
            if hasattr(context, "rswa_read_lengths"):
                del context.rswa_read_lengths
            if hasattr(context, "rswa_graph_inputs"):
                del context.rswa_graph_inputs

    @contextmanager
    def capture_scope(self, context):
        assert not self._active
        if self._params is None:
            raise RuntimeError("R-SWA graph buffers must be reserved before capture")
        num_tokens = self._num_tokens(context)
        self._params.prepare_shape(num_tokens)
        was_capturing = getattr(context, "capturing", False)
        self._active = True
        context.capturing = True
        try:
            with self._stage_inputs(context, num_tokens):
                yield
        except BaseException:
            self._params.discard_shape(num_tokens)
            raise
        finally:
            context.capturing = was_capturing
            self._active = False

    @contextmanager
    def replay_scope(self, context):
        assert not self._active
        stream = current_platform.torch_device_fn.current_stream()
        if self._replay_stream is not None and self._replay_stream != stream:
            stream.wait_stream(self._replay_stream)
        self._replay_stream = stream
        with (
            self._stage_inputs(context, self._num_tokens(context)),
            super().replay_scope(context),
        ):
            yield

    def clear_decoder_graphs(self) -> None:
        if self._replay_stream is not None:
            self._replay_stream.synchronize()
        if self.wrapper_type is not None:
            self.wrapper_type.clear_all_graphs(self)
        if self._params is not None:
            self._params.clear()
        self._params = None
        self._replay_stream = None
        self.update_stream = None
