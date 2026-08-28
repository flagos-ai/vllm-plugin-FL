# Copyright (c) 2025 BAAI. All rights reserved.

"""Runner-selection contracts for vLLM 0.28.0."""


def test_register_preserves_upstream_runner_selection(monkeypatch):
    from types import SimpleNamespace

    import vllm_fl
    import vllm_fl.utils

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(vllm_fl, "_patch_custom_ops", lambda: None)
    monkeypatch.setattr(vllm_fl, "_patch_flash_attn_import", lambda: None)
    monkeypatch.setattr(vllm_fl, "_get_op_config", lambda: None)
    monkeypatch.setattr(
        vllm_fl.utils,
        "DeviceInfo",
        lambda: SimpleNamespace(vendor_name="metax"),
    )

    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    assert "VLLM_USE_V2_MODEL_RUNNER" not in __import__("os").environ


def test_register_preserves_explicit_v1_request(monkeypatch):
    from types import SimpleNamespace

    import vllm_fl
    import vllm_fl.utils

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(vllm_fl, "_patch_custom_ops", lambda: None)
    monkeypatch.setattr(vllm_fl, "_patch_flash_attn_import", lambda: None)
    monkeypatch.setattr(vllm_fl, "_get_op_config", lambda: None)
    monkeypatch.setattr(
        vllm_fl.utils,
        "DeviceInfo",
        lambda: SimpleNamespace(vendor_name="metax"),
    )

    vllm_fl.register()

    assert __import__("os").environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_register_selects_native_cuda_semantics_for_nvidia(monkeypatch):
    from types import SimpleNamespace

    import vllm_fl
    import vllm_fl.utils

    monkeypatch.setattr(vllm_fl, "_patch_custom_ops", lambda: None)
    monkeypatch.setattr(vllm_fl, "_patch_flash_attn_import", lambda: None)
    monkeypatch.setattr(vllm_fl, "_get_op_config", lambda: None)
    monkeypatch.setattr(
        vllm_fl.utils,
        "DeviceInfo",
        lambda: SimpleNamespace(vendor_name="nvidia"),
    )

    assert vllm_fl.register() == "vllm_fl.nvidia_platform.NvidiaPlatformFL"
