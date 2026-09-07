"""Model-scoped FlagGems policy tests for Qwen3.8-Flash-Next."""

from types import SimpleNamespace

import pytest

from vllm_fl.patches.qwen3_8_flash_next import (
    apply_native_index_select_policy,
    needs_native_index_select,
    should_skip_generic_flaggems_aten,
)


def _config(model_type: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
    )


def _architecture_config(architecture: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=""),
            hf_config=SimpleNamespace(
                model_type="", architectures=[architecture]
            ),
            architectures=[architecture],
        )
    )


def test_qwen4_merges_native_index_select_with_platform_blacklist():
    config = _config("qwen4_exp_text")
    assert needs_native_index_select(config)
    whitelist, blacklist = apply_native_index_select_policy(
        config, None, ["copy_", "index"]
    )
    assert whitelist is None
    assert blacklist == [
        "copy_", "index", "index_select", "conv1d", "_conv_depthwise2d",
        "conv2d", "pad", "constant_pad_nd"
    ]


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen3_8_flash_next"])
def test_policy_accepts_outer_multimodal_model_type(model_type):
    config = _config(model_type)
    _, blacklist = apply_native_index_select_policy(config, None, None)
    assert blacklist == [
        "index_select", "conv1d", "_conv_depthwise2d", "conv2d", "pad",
        "constant_pad_nd"
    ]


def test_policy_accepts_checkpoint_architecture_fallback():
    config = _architecture_config("Qwen4ExpForConditionalGeneration")
    _, blacklist = apply_native_index_select_policy(config, None, None)
    assert blacklist == [
        "index_select", "conv1d", "_conv_depthwise2d", "conv2d", "pad",
        "constant_pad_nd"
    ]


def test_policy_adds_measured_nvidia_decode_fast_paths():
    config = _config("qwen4_exp_text")
    _, blacklist = apply_native_index_select_policy(
        config, None, ["copy_"], vendor_name="nvidia"
    )
    assert blacklist == [
        "copy_", "index_select", "conv1d", "_conv_depthwise2d", "conv2d",
        "pad", "constant_pad_nd", "repeat_interleave_tensor",
        "repeat_interleave_self_tensor", "linear", "mm", "mm_out", "addmm",
        "addmm_out", "addmm_", "addmm_dtype", "addmm_dtype_out"
    ]


@pytest.mark.parametrize("vendor_name", ["amd", "metax", "iluvatar", "mthreads"])
def test_policy_keeps_vendor_flaggems_linear_and_repeat_interleave(vendor_name):
    config = _config("qwen4_exp_text")
    _, blacklist = apply_native_index_select_policy(
        config, None, None, vendor_name=vendor_name
    )
    assert "linear" not in blacklist
    assert "mm" not in blacklist
    assert "addmm" not in blacklist
    assert "repeat_interleave_tensor" not in blacklist
    assert "repeat_interleave_self_tensor" not in blacklist


@pytest.mark.parametrize("vendor_name", ["amd", "metax", "iluvatar", "mthreads"])
def test_other_accelerators_keep_generic_flaggems_aten(vendor_name):
    config = _config("qwen4_exp_text")
    assert not should_skip_generic_flaggems_aten(
        config, vendor_name=vendor_name, whitelist=None
    )


def test_nvidia_qwen4_skips_generic_flaggems_aten_by_default():
    config = _config("qwen4_exp_text")
    assert should_skip_generic_flaggems_aten(
        config, vendor_name="nvidia", whitelist=None
    )


def test_explicit_whitelist_overrides_nvidia_generic_aten_policy():
    config = _config("qwen4_exp_text")
    assert not should_skip_generic_flaggems_aten(
        config, vendor_name="nvidia", whitelist=["silu"]
    )


def test_policy_is_idempotent_and_preserves_explicit_whitelist():
    config = _config("qwen3_8_flash_next_text")
    _, blacklist = apply_native_index_select_policy(config, None, ["index_select"])
    assert blacklist == [
        "index_select", "conv1d", "_conv_depthwise2d", "conv2d", "pad",
        "constant_pad_nd"
    ]
    whitelist, blacklist = apply_native_index_select_policy(
        config, ["add"], ["copy_"]
    )
    assert whitelist == ["add"]
    assert blacklist == ["copy_"]


def test_policy_rejects_unsafe_explicit_whitelist():
    config = _config("qwen4_exp_text")
    with pytest.raises(ValueError, match="requires native PLE runtime primitives"):
        apply_native_index_select_policy(config, ["add", "index_select"], ["copy_"])

    with pytest.raises(ValueError, match="constant_pad_nd"):
        apply_native_index_select_policy(
            config, ["add", "constant_pad_nd"], ["copy_"]
        )

    with pytest.raises(ValueError, match="pad"):
        apply_native_index_select_policy(config, ["add", "pad"], ["copy_"])

    with pytest.raises(ValueError, match="conv1d"):
        apply_native_index_select_policy(config, ["add", "conv1d"], ["copy_"])

    with pytest.raises(ValueError, match="linear"):
        apply_native_index_select_policy(
            config, ["add", "linear"], ["copy_"], vendor_name="nvidia"
        )

    # A non-NVIDIA backend may keep its own tuned FlagGems linear path.
    whitelist, blacklist = apply_native_index_select_policy(
        config, ["add", "linear"], ["copy_"], vendor_name="metax"
    )
    assert whitelist == ["add", "linear"]
    assert blacklist == ["copy_"]


def test_policy_does_not_change_other_models():
    config = _config("qwen3_text")
    assert not needs_native_index_select(config)
    whitelist, blacklist = apply_native_index_select_policy(config, None, ["copy_"])
    assert whitelist is None
    assert blacklist == ["copy_"]
