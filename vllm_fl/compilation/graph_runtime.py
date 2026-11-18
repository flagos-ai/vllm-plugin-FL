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
        self._rswa_params = None
        self._rswa_replay_stream = None

    def prepare_rswa_graph(self, device: Any) -> int:
        """Reserve the R-SWA graph buffers before the worker sizes its KV cache."""
        from vllm_fl.models.unlimited_ocr import get_ocr_rswa_window

        config = self.vllm_config
        if (
            get_ocr_rswa_window(config.model_config) is None
            or not config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            return 0

        if self._rswa_params is None:
            from vllm_fl.compilation.graph_params import GraphParams
            from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
                reserve_rswa_graph_buffers,
            )

            buffers = reserve_rswa_graph_buffers(config, device)
            if buffers is None:
                return 0
            self.update_stream = current_platform.torch_device_fn.Stream(device=device)
            self._rswa_params = GraphParams(rswa_buffers=buffers)

        return self._rswa_params.rswa_buffers.nbytes

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
        rswa_params = self._rswa_params
        if rswa_params is None:
            params = prepare_graph_params(num_tokens)
        else:
            params = rswa_params
            params.prepare_shape(num_tokens)

        was_capturing = getattr(context, "capturing", False)
        self._active = True
        context.capturing = True
        try:
            if rswa_params is not None:
                context.rswa_graph_params = rswa_params
                self._stage_rswa_graph_inputs(context, num_tokens)
            yield
            if rswa_params is None:
                weak_ref_workspace(num_tokens)
        except BaseException:
            params.discard_shape(num_tokens)
            raise
        finally:
            if rswa_params is not None:
                del context.rswa_graph_params
                if hasattr(context, "rswa_graph_inputs"):
                    del context.rswa_graph_inputs
            context.capturing = was_capturing
            self._active = False

    @contextmanager
    def replay_scope(self, context: Any):
        assert not self._active
        rswa_params = self._rswa_params
        if rswa_params is not None:
            context.rswa_graph_params = rswa_params
        self._active = True
        try:
            stream = current_platform.torch_device_fn.current_stream()
            if rswa_params is not None:
                # Only one graph per runner may use the shared buffers at a time.
                previous_stream = self._rswa_replay_stream
                if previous_stream is not None and previous_stream != stream:
                    stream.wait_stream(previous_stream)
                self._stage_rswa_graph_inputs(context, self._num_tokens(context))
            # rc1 waits for the previous replay before releasing this step's
            # external events with task updates on a separate stream.
            # On R-SWA this also completes the metadata copies on the replay stream.
            stream.synchronize()
            yield
            if rswa_params is not None:
                self._rswa_replay_stream = stream
            self._update_attention(context)
        finally:
            if rswa_params is not None:
                del context.rswa_graph_params
            self._active = False

    def _stage_rswa_graph_inputs(self, context: Any, num_tokens: int) -> None:
        if not self.attention_impls or len(self.attention_impls) != 1:
            raise RuntimeError(
                "R-SWA graphs require one bound Ascend attention backend"
            )
        self.attention_impls[0].stage_rswa_graph_inputs(context, num_tokens)

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

        if self._rswa_params is not None:
            if self._rswa_replay_stream is not None:
                self._rswa_replay_stream.synchronize()
            if self.wrapper_type is not None:
                self.wrapper_type.clear_all_graphs(self)
            self._rswa_params.clear()
            self._rswa_params = None
            self._rswa_replay_stream = None
            self.update_stream = None
            return

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
