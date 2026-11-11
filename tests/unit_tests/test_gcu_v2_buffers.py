# Copyright (c) 2025 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("device", ["gcu", "cuda", "cpu"])
@pytest.mark.parametrize("available", [False, True])
def test_gcu_v2_uses_explicit_copies_only_on_gcu(monkeypatch, device, available):
    import vllm.platforms
    from vllm.v1.worker.gpu import buffer_utils
    from vllm.v1.worker.gpu.block_table import BlockTables

    monkeypatch.setattr(BlockTables, "_make_ptr_tensor", BlockTables._make_ptr_tensor)

    from vllm_fl.patches.gcu_v2_buffers import install_gcu_v2_buffers

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(device_type=device)
    )
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: available)
    install_gcu_v2_buffers()
    guarded = buffer_utils.is_uva_available
    pointer_builder = BlockTables._make_ptr_tensor
    assert guarded() is (available and device != "gcu")
    install_gcu_v2_buffers()
    assert buffer_utils.is_uva_available is guarded
    assert BlockTables._make_ptr_tensor is pointer_builder


@pytest.mark.parametrize("device", ["gcu", "cuda"])
def test_block_table_pointer_translation_is_gcu_only(monkeypatch, device):
    import vllm.platforms
    from vllm.v1.worker.gpu import buffer_utils
    from vllm.v1.worker.gpu.block_table import BlockTables

    from vllm_fl.dispatch.backends.vendor.gcu.impl import kv_zeroer
    from vllm_fl.patches.gcu_v2_buffers import install_gcu_v2_buffers

    raw, translated = object(), object()
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(device_type=device)
    )
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    monkeypatch.setattr(BlockTables, "_make_ptr_tensor", lambda self, tensors: raw)
    seen = []

    def translate(value):
        seen.append(value)
        return translated

    monkeypatch.setattr(kv_zeroer, "_device_segment_addresses", translate)
    install_gcu_v2_buffers()
    result = BlockTables._make_ptr_tensor(
        SimpleNamespace(device=SimpleNamespace(type=device)), []
    )
    assert result is (translated if device == "gcu" else raw)
    assert seen == ([raw] if device == "gcu" else [])
