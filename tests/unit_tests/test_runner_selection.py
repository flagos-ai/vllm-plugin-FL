# Copyright (c) 2025 BAAI. All rights reserved.

"""Runner-selection contracts for vLLM 0.28.0."""

import pytest


@pytest.mark.parametrize("requested, expected", [(None, "0"), ("0", "0"), ("1", "1")])
def test_ascend_defaults_to_fl_runner(monkeypatch, requested, expected):
    import os
    from types import SimpleNamespace

    import vllm_fl
    import vllm_fl.utils

    if requested is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", requested)
    monkeypatch.setattr(vllm_fl, "_arm_cpu_platform", lambda: None)
    monkeypatch.setattr(vllm_fl, "_patch_custom_ops", lambda: None)
    monkeypatch.setattr(vllm_fl, "_patch_flash_attn_import", lambda: None)
    monkeypatch.setattr(vllm_fl, "_get_op_config", lambda: None)
    monkeypatch.setattr(
        vllm_fl.utils, "DeviceInfo", lambda: SimpleNamespace(vendor_name="ascend")
    )

    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == expected


def test_ascend_rejects_explicit_upstream_v2_runner(monkeypatch):
    from types import SimpleNamespace

    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        PlatformFL.check_and_update_config(SimpleNamespace(use_v2_model_runner=True))


def test_ascend_rejects_non_eager_execution(monkeypatch):
    from types import SimpleNamespace

    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = SimpleNamespace(
        use_v2_model_runner=False,
        model_config=SimpleNamespace(enforce_eager=False),
    )
    with pytest.raises(ValueError, match="--enforce-eager"):
        PlatformFL.check_and_update_config(config)


def test_ascend_revalidates_parallelism_after_glm_dense_fallback(monkeypatch):
    from types import SimpleNamespace

    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        index_topk=128,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        num_attention_heads=64,
    )
    arch_config = SimpleNamespace(
        is_deepseek_mla=True,
        head_size=576,
        total_num_kv_heads=1,
    )

    class ModelConfig:
        def __init__(self):
            self.enforce_eager = True
            self.hf_text_config = hf_config
            self.max_model_len = 128
            self.model_arch_config = arch_config

        @property
        def use_mla(self):
            return self.model_arch_config.is_deepseek_mla

        def verify_with_parallel_config(self, parallel_config):
            assert parallel_config.decode_context_parallel_size == 2
            assert self.use_mla is False
            assert self.model_arch_config.head_size == 256
            assert self.model_arch_config.total_num_kv_heads == 64
            raise RuntimeError("dense parallelism revalidated")

    config = SimpleNamespace(
        use_v2_model_runner=False,
        model_config=ModelConfig(),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )

    with pytest.raises(RuntimeError, match="dense parallelism revalidated"):
        PlatformFL.check_and_update_config(config)


def test_ascend_does_not_advertise_static_graph_support(monkeypatch):
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    monkeypatch.setattr(PlatformFL, "vendor_name", "ascend")

    assert not PlatformFL.support_static_graph_mode()


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
