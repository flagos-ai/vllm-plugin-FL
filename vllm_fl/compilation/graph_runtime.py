# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NPU task lifecycle hooks for FL's existing graph wrapper.

Task ordering follows vllm-ascend v0.24.0rc1 acl_graph.py and
model_runner_v1.py. Graph allocation, wrapper dispatch and input buffers remain
owned by FL's existing GraphWrapper and vLLM runner.
"""

from contextlib import contextmanager
from typing import Any

from vllm.distributed.parallel_state import GraphCaptureContext
from vllm.platforms import current_platform


class GraphRuntimeController:
    """Hold ordinary-attention task updaters for one NPU runner."""

    def __init__(self, *, vllm_config: Any, full_graph_wrapper_type: Any = None):
        self.vllm_config = vllm_config
        self.wrapper_type = full_graph_wrapper_type
        self.attention_impls: tuple[type, ...] | None = None
        self.update_stream = None
        self._active = False

    @staticmethod
    def _num_tokens(context: Any) -> int:
        num_tokens = getattr(getattr(context, "batch_descriptor", None), "num_tokens", None)
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            raise RuntimeError("Ascend FULL graphs require integer BatchDescriptor.num_tokens")
        return num_tokens

    @contextmanager
    def capture_scope(self, context: Any):
        from vllm_fl.compilation.graph_params import (
            prepare_graph_params,
            weak_ref_workspace,
        )

        assert not self._active
        num_tokens = self._num_tokens(context)
        prepare_graph_params(num_tokens)
        self._active = True
        context.capturing = True
        try:
            yield
        finally:
            try:
                weak_ref_workspace(num_tokens)
            finally:
                context.capturing = False
                self._active = False

    @contextmanager
    def replay_scope(self, context: Any):
        assert not self._active
        self._active = True
        try:
            # rc1 waits for the previous replay before releasing this step's
            # external events with task updates on a separate stream.
            current_platform.torch_device_fn.current_stream().synchronize()
            yield
            self._update_attention(context)
        finally:
            self._active = False

    def _update_attention(self, context: Any) -> None:
        if self.attention_impls is None:
            raise RuntimeError("Ascend attention backends must be bound before FULL replay")
        if self.attention_impls:
            if self.update_stream is None:
                self.update_stream = current_platform.torch_device_fn.Stream()
            for impl in self.attention_impls:
                impl.update_graph_params(
                    self.update_stream, context, self._num_tokens(context), self.vllm_config
                )

    def bind_attention_groups(self, attn_groups: Any) -> None:
        """Resolve backend factories while the runner's config is in scope."""
        from vllm.v1.attention.backend import AttentionBackend

        implementations = []
        seen_backends = set()
        for groups in attn_groups:
            for group in groups:
                backend = group.backend
                if backend in seen_backends:
                    continue
                seen_backends.add(backend)
                factory = getattr(backend, "get_impl_cls", None)
                if factory is None or factory is AttentionBackend.get_impl_cls:
                    continue
                try:
                    impl = factory()
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to resolve Ascend graph backend {backend.__name__}"
                    ) from exc
                if not callable(getattr(impl, "update_graph_params", None)):
                    raise RuntimeError(
                        f"Ascend graph backend {backend.__name__} lacks update_graph_params"
                    )
                if impl not in implementations:
                    implementations.append(impl)
        self.attention_impls = tuple(implementations)

    def clear_decoder_graphs(self) -> None:
        from vllm_fl.compilation.graph_params import clear_graph_params

        if self.wrapper_type is not None:
            self.wrapper_type.clear_all_graphs()
        clear_graph_params()


def get_graph_capture(default_capture):
    """Use rc1's separate NPU capture stream; retain the caller elsewhere."""
    @contextmanager
    def capture(device):
        if current_platform.device_type != "npu":
            with default_capture(device=device) as context:
                yield context
            return
        device_fn = current_platform.torch_device_fn
        context = GraphCaptureContext(device_fn.Stream(device=device))
        current_stream = device_fn.current_stream()
        if context.stream != current_stream:
            context.stream.wait_stream(current_stream)
        with device_fn.stream(context.stream):
            yield context

    return capture
