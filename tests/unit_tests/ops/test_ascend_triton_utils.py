def test_legacy_triton_utils_share_canonical_device_properties(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.ascend.impl import triton_utils as legacy
    from vllm_fl.dispatch.backends.vendor.ascend.impl.triton import (
        triton_utils as canonical,
    )

    monkeypatch.setattr(canonical, "_NUM_AICORE", 20)
    monkeypatch.setattr(canonical, "_NUM_VECTORCORE", 40)

    assert legacy.get_aicore_num() == 20
    assert legacy.get_vectorcore_num() == 40
