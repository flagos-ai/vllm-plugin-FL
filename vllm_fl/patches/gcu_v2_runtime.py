# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU V2 graph and indirect Mamba state-address integration."""

from contextlib import contextmanager
from functools import wraps

import torch


def install_gcu_v2_runtime():
    from vllm.platforms import current_platform

    if current_platform.device_type != "gcu":
        return

    from vllm.distributed.parallel_state import GraphCaptureContext
    from vllm.v1.worker.gpu import cudagraph_utils
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

    if not getattr(cudagraph_utils.graph_capture, "_fl_gcu_graph", False):

        @contextmanager
        def graph_capture(device):
            device_fn = current_platform.torch_device_fn
            stream = device_fn.Stream(device=device)
            stream.wait_stream(device_fn.current_stream())
            with device_fn.stream(stream):
                yield GraphCaptureContext(stream)

        graph_capture._fl_gcu_graph = True
        cudagraph_utils.graph_capture = graph_capture

    original = MambaSpecDecodeGPUContext._populate_metadata
    if getattr(original, "_fl_gcu_addresses", False):
        return

    @wraps(original)
    def populate_metadata(self, *args, **kwargs):
        original(self, *args, **kwargs)
        from vllm_fl.dispatch.backends.vendor.gcu.impl.kv_zeroer import (
            _device_segment_addresses,
        )

        # These tables contain torch allocation handles, not Triton device VAs.
        # Translate once, before warmup/capture, retaining stable tensor storage.
        self.is_initialized = False
        tables = (
            self.state_base_addrs,
            self.block_table_ptrs,
            self.aligned_index_block_table_ptrs,
        )
        translated = []
        for name, table in zip(
            ("state_base_addrs", "block_table_ptrs", "aligned_index_block_table_ptrs"),
            tables,
        ):
            try:
                addresses = _device_segment_addresses(
                    table.view(torch.uint64).flatten()
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Mamba {name} device-address translation failed"
                ) from exc
            translated.append(addresses)
        for table, addresses in zip(tables, translated):
            table.copy_(addresses.view(table.dtype).reshape(table.shape))
        self.is_initialized = True

    populate_metadata._fl_gcu_addresses = True
    MambaSpecDecodeGPUContext._populate_metadata = populate_metadata
