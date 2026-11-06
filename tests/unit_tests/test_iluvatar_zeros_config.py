"""Regression for large KV backing allocations on BI-V150."""

from vllm_fl.utils import get_flag_gems_whitelist_blacklist


def test_iluvatar_native_zero_dependency_is_excluded(monkeypatch):
    # Exercise the actual worker config loader, not a duplicate YAML parser.
    from types import SimpleNamespace

    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="iluvatar")
    )
    for key in (
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    ):
        monkeypatch.delenv(key, raising=False)
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert not whitelist
    assert {"zeros", "zero_", "scatter_"} <= set(blacklist)
    # Native SDPA invokes the inner flash op, so excluding only the outer
    # registration does not avoid the BF16 D72/N256 NaN regression.
    assert {"_scaled_dot_product_flash_attention", "_flash_attention_forward"} <= set(
        blacklist
    )
    assert "_scaled_dot_product_attention_math" not in blacklist
    assert "mm" not in blacklist
    assert "rms_norm" not in blacklist
