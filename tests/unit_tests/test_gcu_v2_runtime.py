# Copyright (c) 2026 BAAI. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("device", ["gcu", "cuda"])
def test_mamba_pointer_tables(monkeypatch, device):
    import vllm.platforms
    from vllm.v1.worker.gpu import cudagraph_utils
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext as Context

    from vllm_fl.dispatch.backends.vendor.gcu.impl import kv_zeroer
    from vllm_fl.patches.gcu_v2_runtime import install_gcu_v2_runtime

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(device_type=device)
    )
    monkeypatch.setattr(cudagraph_utils, "graph_capture", object())

    def populate(self):
        self.is_initialized = True

    monkeypatch.setattr(Context, "_populate_metadata", populate)
    seen = []

    def translate(table):
        seen.append(table)
        assert table.dtype == torch.uint64
        assert table.ndim == 1
        return (table.view(torch.int64) + 100).view(torch.uint64)

    monkeypatch.setattr(kv_zeroer, "_device_segment_addresses", translate)
    install_gcu_v2_runtime()
    installed = Context._populate_metadata
    install_gcu_v2_runtime()
    assert Context._populate_metadata is installed
    tables = [torch.tensor([[i + 1 - (1 << 63)]], dtype=torch.int64) for i in range(3)]
    state = SimpleNamespace(
        state_base_addrs=tables[0],
        block_table_ptrs=tables[1],
        aligned_index_block_table_ptrs=tables[2],
    )
    installed(state)
    assert state.is_initialized
    assert [t.item() for t in tables] == (
        [i - (1 << 63) for i in ([101, 102, 103] if device == "gcu" else [1, 2, 3])]
    )
    assert len(seen) == (3 if device == "gcu" else 0)


def test_graph_uses_platform_stream(monkeypatch):
    import vllm.platforms
    from vllm.v1.worker.gpu import cudagraph_utils
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext as Context

    from vllm_fl.patches.gcu_v2_runtime import install_gcu_v2_runtime

    events = []
    stream = SimpleNamespace(wait_stream=lambda s: events.append(("wait", s)))

    @contextmanager
    def scope(s):
        assert s is stream
        events.append("enter")
        yield
        events.append("exit")

    fn = SimpleNamespace(
        Stream=lambda device: stream, current_stream=lambda: "current", stream=scope
    )
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(device_type="gcu", torch_device_fn=fn),
    )
    monkeypatch.setattr(cudagraph_utils, "graph_capture", object())
    monkeypatch.setattr(Context, "_populate_metadata", lambda self: None)
    install_gcu_v2_runtime()
    with cudagraph_utils.graph_capture(device="gcu:0") as ctx:
        assert ctx.stream is stream
    assert events == [("wait", "current"), "enter", "exit"]


def test_mamba_translation_failure_keeps_original_tables(monkeypatch):
    import vllm.platforms
    from vllm.v1.worker.gpu import cudagraph_utils
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext as Context

    from vllm_fl.dispatch.backends.vendor.gcu.impl import kv_zeroer
    from vllm_fl.patches.gcu_v2_runtime import install_gcu_v2_runtime

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(device_type="gcu")
    )
    monkeypatch.setattr(cudagraph_utils, "graph_capture", object())
    monkeypatch.setattr(Context, "_populate_metadata", lambda self: None)
    calls = []

    def translate(table):
        calls.append(table)
        if len(calls) == 2:
            raise RuntimeError("SDK query failed")
        return torch.tensor([100], dtype=torch.uint64)

    monkeypatch.setattr(kv_zeroer, "_device_segment_addresses", translate)
    install_gcu_v2_runtime()
    tables = [torch.tensor([[i + 1]], dtype=torch.int64) for i in range(3)]
    state = SimpleNamespace(
        state_base_addrs=tables[0],
        block_table_ptrs=tables[1],
        aligned_index_block_table_ptrs=tables[2],
        is_initialized=True,
    )
    with pytest.raises(RuntimeError, match="Mamba block_table_ptrs"):
        Context._populate_metadata(state)
    assert not state.is_initialized
    assert [t.item() for t in tables] == [1, 2, 3]


def test_enflame_metadata_scalar_fill_uses_native_backend():
    from pathlib import Path

    import yaml

    import vllm_fl

    config = yaml.safe_load(
        (Path(vllm_fl.__file__).parent / "dispatch/config/enflame.yaml").read_text()
    )
    assert "fill_scalar_" in config["flagos_blacklist"]
    # The workaround must not disable all FlagGems tensor copies or fills.
    assert "copy_" not in config["flagos_blacklist"]
    assert "fill_tensor_" not in config["flagos_blacklist"]
