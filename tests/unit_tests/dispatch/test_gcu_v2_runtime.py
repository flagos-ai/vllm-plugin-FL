# Copyright (c) 2026 BAAI. All rights reserved.
"""Real-device stream-cache contract for the V2 graph capture boundary."""

import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("raise_inside", [False, True])
def test_gcu_v2_capture_stream_cache_and_restore(raise_inside):
    import torch

    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import current_stream

    if current_platform.device_type != "gcu":
        pytest.skip("Requires GCU")
    from vllm.v1.worker.gpu import cudagraph_utils

    from vllm_fl.dispatch.backends.vendor.gcu.v2_runtime import install_gcu_v2_runtime

    install_gcu_v2_runtime()
    device_fn = current_platform.torch_device_fn
    previous = device_fn.current_stream()
    previous_cached = current_stream()
    try:
        with cudagraph_utils.graph_capture(device=torch.device("gcu:0")) as context:
            assert device_fn.current_stream() == context.stream
            assert current_stream() == context.stream
            if raise_inside:
                raise ValueError("capture body failed")
    except ValueError as exc:
        assert raise_inside
        assert str(exc) == "capture body failed"
    assert device_fn.current_stream() == previous
    assert current_stream() == previous_cached
