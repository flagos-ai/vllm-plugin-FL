import pytest

from vllm_fl.utils import get_device_control_env_var

_OVERRIDE = "FLAGOS_DEVICE_CONTROL_ENV_VAR"
_PLACEHOLDER = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"


def test_nvidia_uses_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv(_OVERRIDE, raising=False)
    assert get_device_control_env_var("nvidia") == "CUDA_VISIBLE_DEVICES"


@pytest.mark.parametrize("vendor", ["enflame", "gcu"])
def test_enflame_aliases_use_tops_visible_devices(monkeypatch, vendor):
    monkeypatch.delenv(_OVERRIDE, raising=False)
    assert get_device_control_env_var(vendor) == "TOPS_VISIBLE_DEVICES"


@pytest.mark.parametrize(
    "vendor",
    [
        "ascend",
        "iluvatar",
        "metax",
        "mthreads",
        "sunrise",
        "hygon",
        "thead",
        "not_a_vendor",
    ],
)
def test_unvalidated_vendors_preserve_vllm_base_noop(monkeypatch, vendor):
    monkeypatch.delenv(_OVERRIDE, raising=False)
    assert get_device_control_env_var(vendor) == _PLACEHOLDER


@pytest.mark.parametrize("vendor", ["ascend", "enflame", "gcu", "not_a_vendor"])
def test_env_override_wins(monkeypatch, vendor):
    monkeypatch.setenv(_OVERRIDE, "CUSTOM_VISIBLE_DEVICES")
    assert get_device_control_env_var(vendor) == "CUSTOM_VISIBLE_DEVICES"


@pytest.mark.parametrize(
    "vendor, expected",
    [
        ("ascend", _PLACEHOLDER),
        ("enflame", "TOPS_VISIBLE_DEVICES"),
        ("gcu", "TOPS_VISIBLE_DEVICES"),
    ],
)
def test_blank_env_override_is_ignored(monkeypatch, vendor, expected):
    monkeypatch.setenv(_OVERRIDE, "   ")
    assert get_device_control_env_var(vendor) == expected


def test_platform_picks_up_the_dispatched_value():
    from vllm_fl.platform import PlatformFL

    expected = get_device_control_env_var(PlatformFL.vendor_name)
    assert PlatformFL.device_control_env_var == expected
