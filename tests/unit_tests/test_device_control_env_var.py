import pytest

from vllm_fl.utils import VENDOR_DEVICE_MAP, get_device_control_env_var

_OVERRIDE = "FLAGOS_DEVICE_CONTROL_ENV_VAR"


def test_every_vendor_declares_a_device_control_env_var():
    for vendor, device_info in VENDOR_DEVICE_MAP.items():
        value = device_info.get("device_control_env_var")
        assert isinstance(value, str) and value.strip(), (
            f"vendor {vendor!r} is missing device_control_env_var"
        )


@pytest.mark.parametrize(
    ("vendor", "expected"),
    [
        ("nvidia", "CUDA_VISIBLE_DEVICES"),
        ("iluvatar", "ILUVATAR_VISIBLE_DEVICES"),
        ("metax", "MACA_VISIBLE_DEVICES"),
        ("hygon", "HIP_VISIBLE_DEVICES"),
        ("thead", "CUDA_VISIBLE_DEVICES"),
        ("ascend", "ASCEND_RT_VISIBLE_DEVICES"),
        ("mthreads", "MUSA_VISIBLE_DEVICES"),
        ("sunrise", "TANG_VISIBLE_DEVICES"),
    ],
)
def test_known_vendors_resolve(monkeypatch, vendor, expected):
    monkeypatch.delenv(_OVERRIDE, raising=False)
    assert get_device_control_env_var(vendor) == expected


def test_cuda_device_type_does_not_imply_cuda_control_variable(monkeypatch):
    monkeypatch.delenv(_OVERRIDE, raising=False)
    assert VENDOR_DEVICE_MAP["iluvatar"]["device_type"] == "cuda"
    assert get_device_control_env_var("iluvatar") == "ILUVATAR_VISIBLE_DEVICES"
    assert VENDOR_DEVICE_MAP["metax"]["device_type"] == "cuda"
    assert get_device_control_env_var("metax") == "MACA_VISIBLE_DEVICES"
    assert VENDOR_DEVICE_MAP["hygon"]["device_type"] == "cuda"
    assert get_device_control_env_var("hygon") == "HIP_VISIBLE_DEVICES"


def test_unknown_vendor_falls_back_to_cuda(monkeypatch):
    monkeypatch.delenv(_OVERRIDE, raising=False)
    assert get_device_control_env_var("not_a_vendor") == "CUDA_VISIBLE_DEVICES"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv(_OVERRIDE, "CUSTOM_VISIBLE_DEVICES")
    assert get_device_control_env_var("ascend") == "CUSTOM_VISIBLE_DEVICES"
    assert get_device_control_env_var("not_a_vendor") == "CUSTOM_VISIBLE_DEVICES"


def test_blank_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv(_OVERRIDE, "   ")
    assert get_device_control_env_var("ascend") == "ASCEND_RT_VISIBLE_DEVICES"


def test_platform_picks_up_the_dispatched_value():
    from vllm.platforms import current_platform

    from vllm_fl.platform import PlatformFL

    expected = get_device_control_env_var(current_platform.vendor_name)
    assert PlatformFL.device_control_env_var == expected
