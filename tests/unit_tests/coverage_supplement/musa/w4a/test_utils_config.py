# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for vllm_fl.utils configuration/env helpers.

Uses only the real utility module and local doubles; no accelerator, network
or remote access is involved.
"""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_fl.utils as utils

_ENV_KEYS = (
    "GEMS_VENDOR",
    "FLAGOS_DEVICE_CONTROL_ENV_VAR",
    "VLLM_FL_PREFER_ENABLED",
    "VLLM_FL_PREFER",
    "USE_FLAGGEMS",
    "VLLM_FL_FLAGOS_WHITELIST",
    "VLLM_FL_FLAGOS_BLACKLIST",
    "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    "VLLM_FL_OOT_WHITELIST",
    "VLLM_FL_OOT_BLACKLIST",
    "VLLM_FL_OOT_ENABLED",
    "VLLM_FL_OP_CONFIG",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _install_fake_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parts = name.split(".")
    for index in range(1, len(parts)):
        parent_name = ".".join(parts[:index])
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = ModuleType(parent_name)
            parent.__path__ = []
            monkeypatch.setitem(sys.modules, parent_name, parent)
        for key, value in attrs.items():
            pass
    parent_name = ".".join(parts[:-1])
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, parts[-1], module, raising=False)
    return module


def test_env_vendor_name_normalizes(monkeypatch):
    assert utils._env_vendor_name() == ""
    monkeypatch.setenv("GEMS_VENDOR", "  MTHREADS ")
    assert utils._env_vendor_name() == "mthreads"


@pytest.mark.parametrize(
    "vendor,device_type,device_name",
    [
        ("nvidia", "cuda", "nvidia"),
        ("ascend", "npu", "npu"),
        ("iluvatar", "cuda", "cuda"),
        ("metax", "cuda", "metax"),
        ("mthreads", "musa", "musa"),
        ("sunrise", "ptpu", "ptpu"),
        ("hygon", "cuda", "cuda"),
        ("amd", "cuda", "cuda"),
        ("thead", "cuda", "thead"),
        ("tsingmicro", "txda", "txda"),
        ("gcu", "gcu", "gcu"),
        ("enflame", "gcu", "gcu"),
        ("kunlunxin", "cuda", "kunlunxin"),
    ],
)
def test_vendor_device_map_lookups(vendor, device_type, device_name):
    assert utils.get_device_type(vendor) == device_type
    assert utils.get_device_name(vendor) == device_name


@pytest.mark.parametrize("vendor", ["", "   ", None, 3])
def test_vendor_field_rejects_bad_vendor(vendor):
    with pytest.raises(ValueError, match="non-empty string"):
        utils._get_vendor_device_field(vendor, "device_type")


def test_vendor_field_rejects_unknown_vendor():
    with pytest.raises(ValueError, match="not found in VENDOR_DEVICE_MAP"):
        utils.get_device_type("not-a-real-vendor")


@pytest.mark.parametrize("field_value", ["", "   ", None, 5])
def test_vendor_field_rejects_missing_or_non_string(monkeypatch, field_value):
    monkeypatch.setitem(
        utils.VENDOR_DEVICE_MAP,
        "probe",
        {"device_type": field_value, "device_name": "probe"},
    )
    with pytest.raises(ValueError, match="missing or empty"):
        utils.get_device_type("probe")


def test_device_control_env_var_defaults_and_override(monkeypatch):
    assert utils.get_device_control_env_var("nvidia") == "CUDA_VISIBLE_DEVICES"
    assert utils.get_device_control_env_var("tsingmicro") == "TXDA_VISIBLE_DEVICES"
    assert utils.get_device_control_env_var("ascend") == (
        utils._DEVICE_CONTROL_ENV_VAR_PLACEHOLDER
    )
    monkeypatch.setenv("FLAGOS_DEVICE_CONTROL_ENV_VAR", "  CUSTOM_VISIBLE  ")
    assert utils.get_device_control_env_var("ascend") == "CUSTOM_VISIBLE"
    monkeypatch.setenv("FLAGOS_DEVICE_CONTROL_ENV_VAR", "   ")
    assert utils.get_device_control_env_var("nvidia") == "CUDA_VISIBLE_DEVICES"


@pytest.mark.parametrize(
    "prefer_enabled,prefer,use_gems,default,expected",
    [
        ("false", "", None, True, False),
        ("0", "", None, True, False),
        ("True", "flagos", None, True, True),
        ("True", "other", None, True, False),
        ("True", "", "true", False, True),
        ("True", "", "1", False, True),
        ("True", "", "false", True, False),
        ("True", "", None, False, False),
        ("True", "", None, True, True),
    ],
)
def test_use_flaggems_precedence(
    monkeypatch, prefer_enabled, prefer, use_gems, default, expected
):
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", prefer_enabled)
    if prefer:
        monkeypatch.setenv("VLLM_FL_PREFER", prefer)
    if use_gems is not None:
        monkeypatch.setenv("USE_FLAGGEMS", use_gems)
    assert utils.use_flaggems(default=default) is expected


def test_flagos_whitelist_env_wins(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", " rms_norm , silu_and_mul , ")
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "cos")
    whitelist, blacklist = utils.get_flag_gems_whitelist_blacklist()
    assert whitelist == ["rms_norm", "silu_and_mul"]
    assert blacklist is None


def test_flagos_conflict_raises(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "a")
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "b")
    with pytest.raises(ValueError, match="Cannot set both"):
        utils.get_flag_gems_whitelist_blacklist()


def test_flagos_blacklist_env_and_append_dedup(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "b, a")
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "a,c,,c")
    _, blacklist = utils.get_flag_gems_whitelist_blacklist()
    assert blacklist == ["b", "a", "c"]


def test_flagos_append_on_empty_blacklist(monkeypatch):
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=lambda: [],
        get_flagos_blacklist=lambda: [],
    )
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "x")
    _, blacklist = utils.get_flag_gems_whitelist_blacklist()
    assert blacklist == ["x"]


def test_flagos_config_whitelist(monkeypatch):
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=lambda: ["rms_norm"],
        get_flagos_blacklist=lambda: [],
    )
    whitelist, blacklist = utils.get_flag_gems_whitelist_blacklist()
    assert whitelist == ["rms_norm"]
    assert blacklist is None


def test_flagos_config_blacklist(monkeypatch):
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=lambda: [],
        get_flagos_blacklist=lambda: ["gelu"],
    )
    whitelist, blacklist = utils.get_flag_gems_whitelist_blacklist()
    assert whitelist is None
    assert blacklist == ["gelu"]


def test_flagos_config_both_raises(monkeypatch):
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=lambda: ["a"],
        get_flagos_blacklist=lambda: ["b"],
    )
    with pytest.raises(ValueError, match="cannot define both"):
        utils.get_flag_gems_whitelist_blacklist()


def test_flagos_config_helper_failure_is_swallowed(monkeypatch):
    def boom():
        raise RuntimeError("config load failed")

    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=boom,
        get_flagos_blacklist=boom,
    )
    assert utils.get_flag_gems_whitelist_blacklist() == (None, None)


def test_use_flaggems_op_matrix(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "rms_norm")
    assert utils.use_flaggems_op("rms_norm") is True
    assert utils.use_flaggems_op("gelu") is False

    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST")
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "gelu")
    assert utils.use_flaggems_op("gelu") is False
    assert utils.use_flaggems_op("rms_norm") is True

    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST")
    monkeypatch.setenv("USE_FLAGGEMS", "false")
    assert utils.use_flaggems_op("rms_norm") is False


def test_use_flaggems_op_default_allow(monkeypatch):
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=lambda: [],
        get_flagos_blacklist=lambda: [],
    )
    assert utils.use_flaggems_op("anything") is True


def test_op_config_unset_and_empty(monkeypatch):
    monkeypatch.delenv("VLLM_FL_OP_CONFIG", raising=False)
    utils._load_op_config_from_env()
    assert utils.get_op_config() is None
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", "   ")
    utils._load_op_config_from_env()
    assert utils.get_op_config() is None


def test_op_config_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="file not found"):
        utils._load_op_config_from_env()


def test_op_config_invalid_json(monkeypatch, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(path))
    with pytest.raises(ValueError, match="Invalid VLLM_FL_OP_CONFIG JSON"):
        utils._load_op_config_from_env()


def test_op_config_not_an_object(monkeypatch, tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(path))
    with pytest.raises(ValueError, match="must be a JSON object"):
        utils._load_op_config_from_env()


def test_op_config_non_string_values(monkeypatch, tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(path))
    with pytest.raises(ValueError, match="map strings to strings"):
        utils._load_op_config_from_env()


def test_op_config_valid(monkeypatch, tmp_path):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps({"a": "b", "c": "d"}), encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(path))
    utils._load_op_config_from_env()
    assert utils.get_op_config() == {"a": "b", "c": "d"}
    monkeypatch.delenv("VLLM_FL_OP_CONFIG")
    utils._load_op_config_from_env()
    assert utils.get_op_config() is None


def test_device_info_properties_and_supported(monkeypatch):
    info = object.__new__(utils.DeviceInfo)
    detected = SimpleNamespace(
        vendor_name="mthreads", name="musa", dispatch_key="PrivateUse1"
    )
    backend = SimpleNamespace(
        gen_torch_device_object=Mock(return_value="torch_device"),
        get_torch_backend_device_fn=Mock(return_value="torch_backend"),
    )
    info._backend = backend
    info.device = detected
    info.supported_device = ["mthreads", "nvidia"]
    assert info.dispatch_key == "PrivateUse1"
    assert info.vendor_name == "mthreads"
    assert info.device_type == "musa"
    assert info.torch_device_fn == "torch_device"
    assert info.torch_backend_device == "torch_backend"
    assert info.get_supported_device() is True

    info.supported_device = ["nvidia"]
    with pytest.raises(NotImplementedError, match="not support now"):
        info.get_supported_device()


def test_get_flaggems_all_ops(monkeypatch):
    _install_fake_module(
        monkeypatch,
        "flag_gems",
        _FULL_CONFIG=(("rms_norm", object()), ("silu", object(), {})),
    )
    assert utils.get_flaggems_all_ops() == ["rms_norm", "silu"]

    _install_fake_module(monkeypatch, "flag_gems")
    assert utils.get_flaggems_all_ops() == []


def test_oot_whitelist(monkeypatch):
    assert utils.get_oot_whitelist() is None
    monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", " rms_norm ,, silu ")
    assert utils.get_oot_whitelist() == ["rms_norm", "silu"]


def test_oot_blacklist_whitelist_wins(monkeypatch):
    monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", "rms_norm")
    monkeypatch.setenv("VLLM_FL_OOT_BLACKLIST", "silu")
    assert utils.get_oot_blacklist() is None


def test_oot_blacklist_env(monkeypatch):
    monkeypatch.setenv("VLLM_FL_OOT_BLACKLIST", " silu , gelu ")
    assert utils.get_oot_blacklist() == ["silu", "gelu"]


def test_oot_blacklist_config(monkeypatch):
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_oot_blacklist=lambda: ["from_config"],
    )
    assert utils.get_oot_blacklist() == ["from_config"]


def test_oot_blacklist_config_failure(monkeypatch):
    def boom():
        raise RuntimeError("nope")

    _install_fake_module(
        monkeypatch, "vllm_fl.dispatch.config", get_oot_blacklist=boom
    )
    assert utils.get_oot_blacklist() is None


@pytest.mark.parametrize(
    "prefer,enabled,expected",
    [
        ("false", "1", False),
        ("True", "1", True),
        ("True", "true", True),
        ("True", "0", False),
        ("True", "false", False),
    ],
)
def test_is_oot_enabled(monkeypatch, prefer, enabled, expected):
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", prefer)
    monkeypatch.setenv("VLLM_FL_OOT_ENABLED", enabled)
    assert utils.is_oot_enabled() is expected


def test_oot_op_names_are_unique():
    assert utils.OOT_OP_NAMES
    assert len(set(utils.OOT_OP_NAMES)) == len(utils.OOT_OP_NAMES)
