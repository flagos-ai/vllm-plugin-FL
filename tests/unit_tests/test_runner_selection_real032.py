"""Exercise upstream runner policy using real 0.32 configuration objects.

These are configuration tests, not device/engine acceptance tests.
"""

import pytest


@pytest.mark.parametrize("vendor", ["nvidia", "iluvatar"])
@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_real_config_upstream_runner_policy(monkeypatch, vendor, setting):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.platforms import current_platform

    from vllm_fl.platform import PlatformFL

    if setting is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", setting)
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    monkeypatch.setattr(
        current_platform, "check_and_update_config", PlatformFL.check_and_update_config
    )
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    if setting is not None:
        assert config.use_v2_model_runner is (setting == "1")
    else:
        import vllm.config.vllm as config_module

        assert config.use_v2_model_runner is bool(config_module.HAS_TRITON)
        # Real unsupported-feature detection, not a mocked runner boolean.
        config.parallel_config.distributed_executor_backend = "external_launcher"
        config.parallel_config.pipeline_parallel_size = 2
        assert "pipeline parallelism with external_launcher" in (
            config._get_v2_model_runner_unsupported_features()
        )
        assert config.use_v2_model_runner is False


@pytest.mark.parametrize(
    "vendor,device_type",
    [
        ("ascend", "npu"),
        ("mthreads", "musa"),
        ("enflame", "gcu"),
        ("gcu", "gcu"),
        ("tsingmicro", "txda"),
    ],
)
@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_retained_v1_registration_to_real_config(
    monkeypatch, vendor, device_type, setting
):
    from types import SimpleNamespace

    from vllm.config import DeviceConfig, VllmConfig
    from vllm.platforms import current_platform

    import vllm_fl
    import vllm_fl.utils
    from vllm_fl.platform import PlatformFL

    if setting is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", setting)
    # Only hardware discovery/import side effects are isolated. Registration,
    # upstream config construction and the vendor rejection policy remain real.
    for name in (
        "_init_vendor_device",
        "_arm_cpu_platform",
        "_patch_custom_ops",
        "_patch_flash_attn_import",
        "_get_op_config",
    ):
        monkeypatch.setattr(vllm_fl, name, lambda: None)
    monkeypatch.setattr(
        vllm_fl.utils, "DeviceInfo", lambda: SimpleNamespace(vendor_name=vendor)
    )
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(PlatformFL, "device_type", device_type)
    monkeypatch.setattr(
        current_platform, "check_and_update_config", PlatformFL.check_and_update_config
    )
    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    if setting == "1":
        with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
            VllmConfig(device_config=DeviceConfig(device="cpu"))
    else:
        from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

        config = VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE
            ),
        )
        assert config.use_v2_model_runner is False
        if device_type == "txda":
            for mode, graph_mode, message in (
                (
                    CompilationMode.NONE,
                    CUDAGraphMode.FULL,
                    "TXDA graph mode is not supported",
                ),
                (
                    CompilationMode.VLLM_COMPILE,
                    CUDAGraphMode.NONE,
                    "eager execution only",
                ),
            ):
                config.compilation_config.mode = mode
                config.compilation_config.cudagraph_mode = graph_mode
                with pytest.raises(ValueError, match=message):
                    PlatformFL.check_and_update_config(config)
