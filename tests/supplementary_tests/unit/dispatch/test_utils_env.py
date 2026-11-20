# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for vllm_fl.utils env-driven configuration helpers.

Covers the functions not already exercised by existing tests:
``_load_op_config_from_env`` (direct call), VENDOR_DEVICE_MAP accessors,
OOT whitelist/blacklist resolution, and the silent-degradation path of
``get_flag_gems_whitelist_blacklist``.  ``get_device_control_env_var`` is
covered by tests/unit_tests/utils/test_device_control_env_var.py and
``get_op_config`` import-time behaviour by test_utils_import_boundaries.py,
so neither is duplicated here.
"""

import json
import re
import sys

import pytest

import vllm_fl.utils as utils
from vllm_fl.utils import (
    get_device_name,
    get_device_type,
    get_flag_gems_whitelist_blacklist,
    get_flaggems_all_ops,
    get_oot_blacklist,
    get_oot_whitelist,
    get_op_config,
    is_oot_enabled,
    use_flaggems,
    use_flaggems_op,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, clean_fl_env):
    """Keep every VLLM_FL_* / whitelist var unset around each test."""
    for var in (
        "VLLM_FL_OP_CONFIG",
        "VLLM_FL_OOT_WHITELIST",
        "VLLM_FL_OOT_BLACKLIST",
        "VLLM_FL_OOT_ENABLED",
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_FL_PREFER",
        "USE_FLAGGEMS",
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
        "FLAGOS_DEVICE_CONTROL_ENV_VAR",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


class TestLoadOpConfigFromEnv:
    """_load_op_config_from_env parsing and error messages (direct calls)."""

    @pytest.mark.supplementary
    def test_unset_and_blank_disable_config(self, monkeypatch):
        """A missing or whitespace-only VLLM_FL_OP_CONFIG disables the map."""
        monkeypatch.delenv("VLLM_FL_OP_CONFIG", raising=False)
        utils._load_op_config_from_env()
        assert get_op_config() is None
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", "   ")
        utils._load_op_config_from_env()
        assert get_op_config() is None

    @pytest.mark.supplementary
    def test_valid_config_roundtrip(self, tmp_path, monkeypatch):
        """A well-formed string-to-string JSON object loads verbatim."""
        config = tmp_path / "op.json"
        config.write_text(
            json.dumps({"op_a": "ref", "op_b": "flagos"}), encoding="utf-8"
        )
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(config))
        utils._load_op_config_from_env()
        assert get_op_config() == {"op_a": "ref", "op_b": "flagos"}

    @pytest.mark.supplementary
    def test_missing_file_message(self, tmp_path, monkeypatch):
        """A nonexistent path raises ValueError naming the file."""
        missing = tmp_path / "nope.json"
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(missing))
        with pytest.raises(
            ValueError,
            match=f"VLLM_FL_OP_CONFIG file not found: {re.escape(str(missing))}",
        ):
            utils._load_op_config_from_env()

    @pytest.mark.supplementary
    def test_invalid_json_message(self, tmp_path, monkeypatch):
        """Broken JSON raises the documented parse error."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(bad))
        with pytest.raises(ValueError, match="Invalid VLLM_FL_OP_CONFIG JSON file."):
            utils._load_op_config_from_env()

    @pytest.mark.supplementary
    @pytest.mark.parametrize("payload", ["[1, 2]", "null", '"a string"', "3"])
    def test_non_object_message(self, tmp_path, monkeypatch, payload):
        """Any non-object top level raises the JSON-object error."""
        config = tmp_path / "arr.json"
        config.write_text(payload, encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(config))
        with pytest.raises(
            ValueError, match="VLLM_FL_OP_CONFIG must be a JSON object."
        ):
            utils._load_op_config_from_env()

    @pytest.mark.supplementary
    def test_non_string_values_message(self, tmp_path, monkeypatch):
        """Non-string keys or values raise the strings-to-strings error."""
        config = tmp_path / "vals.json"
        config.write_text(json.dumps({"op": 3}), encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(config))
        with pytest.raises(
            ValueError, match="VLLM_FL_OP_CONFIG must map strings to strings."
        ):
            utils._load_op_config_from_env()


class TestVendorDeviceMap:
    """VENDOR_DEVICE_MAP accessors and their error paths."""

    @pytest.mark.supplementary
    @pytest.mark.parametrize(
        ("vendor", "device_type", "device_name"),
        [
            ("nvidia", "cuda", "nvidia"),
            ("ascend", "npu", "npu"),
            ("metax", "cuda", "metax"),
            ("mthreads", "musa", "musa"),
            ("amd", "cuda", "cuda"),
            ("kunlunxin", "cuda", "kunlunxin"),
        ],
    )
    def test_known_vendors_resolve(self, vendor, device_type, device_name):
        """Registered vendors report their configured device type/name."""
        assert get_device_type(vendor) == device_type
        assert get_device_name(vendor) == device_name

    @pytest.mark.supplementary
    @pytest.mark.parametrize("bad", ["", "   "])
    def test_blank_vendor_name_rejected(self, bad):
        """A blank vendor name raises the documented ValueError."""
        with pytest.raises(ValueError, match="vendor_name must be a non-empty string."):
            get_device_type(bad)

    @pytest.mark.supplementary
    def test_unknown_vendor_rejected(self):
        """An unregistered vendor raises with its name in the message."""
        with pytest.raises(
            ValueError, match="Vendor 'nope' not found in VENDOR_DEVICE_MAP."
        ):
            get_device_name("nope")

    @pytest.mark.supplementary
    def test_missing_field_rejected(self, monkeypatch):
        """A vendor entry lacking the requested field raises."""
        monkeypatch.setitem(utils.VENDOR_DEVICE_MAP, "broken", {"device_type": "x"})
        with pytest.raises(
            ValueError,
            match="Field 'device_name' for vendor 'broken' is missing or empty",
        ):
            get_device_name("broken")

    @pytest.mark.supplementary
    def test_map_covers_all_registered_backends(self):
        """Every documented backend vendor key exists in the map."""
        expected = {
            "nvidia",
            "ascend",
            "iluvatar",
            "metax",
            "mthreads",
            "sunrise",
            "hygon",
            "amd",
            "thead",
            "tsingmicro",
            "gcu",
            "enflame",
            "kunlunxin",
        }
        assert expected <= set(utils.VENDOR_DEVICE_MAP)


class TestFlagGemsSelection:
    """use_flaggems / use_flaggems_op env precedence."""

    @pytest.mark.supplementary
    def test_defaults_to_true(self):
        """With no env vars, FlagGems is preferred."""
        assert use_flaggems() is True
        assert use_flaggems_op("any_op") is True

    @pytest.mark.supplementary
    def test_use_flaggems_false_disables_everything(self, monkeypatch):
        """USE_FLAGGEMS=0 turns both helpers off regardless of lists."""
        monkeypatch.setenv("USE_FLAGGEMS", "0")
        assert use_flaggems() is False
        assert use_flaggems_op("any_op") is False

    @pytest.mark.supplementary
    def test_prefer_enabled_gate(self, monkeypatch):
        """VLLM_FL_PREFER_ENABLED=0 disables FlagGems even when USE_FLAGGEMS=1."""
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "0")
        monkeypatch.setenv("USE_FLAGGEMS", "1")
        assert use_flaggems() is False
        assert is_oot_enabled() is False

    @pytest.mark.supplementary
    def test_prefer_backend_other_disables(self, monkeypatch):
        """A VLLM_FL_PREFER other than 'flagos' disables FlagGems."""
        monkeypatch.setenv("VLLM_FL_PREFER", "native")
        assert use_flaggems() is False

    @pytest.mark.supplementary
    def test_prefer_backend_flagos_enables(self, monkeypatch):
        """VLLM_FL_PREFER=flagos keeps FlagGems on."""
        monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
        assert use_flaggems() is True

    @pytest.mark.supplementary
    def test_default_flag_allows_override(self, monkeypatch):
        """use_flaggems(default=False) is honoured when no env var is set."""
        assert use_flaggems(default=False) is False

    @pytest.mark.supplementary
    def test_whitelist_restricts_ops(self, monkeypatch):
        """A whitelist admits only listed ops."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "rms_norm, silu_and_mul")
        assert use_flaggems_op("rms_norm") is True
        assert use_flaggems_op("mm") is False

    @pytest.mark.supplementary
    def test_blacklist_denies_ops(self, monkeypatch):
        """A blacklist denies only listed ops."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "rms_norm")
        assert use_flaggems_op("rms_norm") is False
        assert use_flaggems_op("mm") is True

    @pytest.mark.supplementary
    def test_whitelist_overrides_blacklist_env(self, monkeypatch):
        """Setting both env vars is a hard error."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "rms_norm")
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "mm")
        with pytest.raises(
            ValueError, match="Cannot set both VLLM_FL_FLAGOS_WHITELIST"
        ):
            use_flaggems_op("rms_norm")

    @pytest.mark.supplementary
    def test_blacklist_append_extends_without_duplicates(self, monkeypatch):
        """BLACKLIST_APPEND appends new ops while preserving order."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "mm, add")
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "add, sub, mm")
        whitelist, blacklist = get_flag_gems_whitelist_blacklist()
        assert whitelist is None
        assert blacklist == ["mm", "add", "sub"]

    @pytest.mark.supplementary
    def test_whitelist_env_disables_append(self, monkeypatch):
        """A whitelist completely overrides any appended blacklist."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "mm")
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "add")
        whitelist, blacklist = get_flag_gems_whitelist_blacklist()
        assert whitelist == ["mm"]
        assert blacklist is None

    @pytest.mark.supplementary
    def test_silent_degradation_on_config_import_failure(self, monkeypatch):
        """A synthetic failure inside config loading degrades to (None, None).

        get_flag_gems_whitelist_blacklist swallows non-ValueError exceptions
        from the platform-config import so a broken config never blocks
        startup — this test deliberately pins that silent-degradation path.
        """
        import builtins

        real_import = builtins.__import__

        def broken_import(name, *args, **kwargs):
            if name == "vllm_fl.dispatch.config":
                raise RuntimeError("synthetic config failure")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        whitelist, blacklist = get_flag_gems_whitelist_blacklist()
        assert whitelist is None
        assert blacklist is None

    @pytest.mark.supplementary
    def test_config_conflict_valueerror_reraised(self, monkeypatch):
        """A ValueError from the config layer propagates (not swallowed)."""
        import vllm_fl.dispatch.config as config_mod

        def raise_conflict():
            raise ValueError(
                "Platform config cannot define both flagos_whitelist and flagos_blacklist."
            )

        monkeypatch.setattr(config_mod, "get_flagos_whitelist", lambda: ["a"])
        monkeypatch.setattr(config_mod, "get_flagos_blacklist", lambda: ["b"])
        with pytest.raises(
            ValueError, match="cannot define both flagos_whitelist and flagos_blacklist"
        ):
            get_flag_gems_whitelist_blacklist()

    @pytest.mark.supplementary
    def test_get_flaggems_all_ops_requires_flag_gems(self, monkeypatch):
        """Without flag_gems installed, the import error propagates."""
        saved = sys.modules.pop("flag_gems", None)
        try:
            with pytest.raises(ModuleNotFoundError):
                get_flaggems_all_ops()
        finally:
            if saved is not None:
                sys.modules["flag_gems"] = saved


class TestOotWhitelistBlacklist:
    """OOT whitelist/blacklist resolution used by custom_ops.register_oot_ops."""

    @pytest.mark.supplementary
    def test_unset_whitelist_returns_none(self):
        """No whitelist env → None (register all)."""
        assert get_oot_whitelist() is None

    @pytest.mark.supplementary
    def test_whitelist_parses_and_strips(self, monkeypatch):
        """The whitelist splits on commas, trims, and drops blanks."""
        monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", " silu_and_mul , rms_norm,, ")
        assert get_oot_whitelist() == ["silu_and_mul", "rms_norm"]

    @pytest.mark.supplementary
    def test_whitelist_ignores_blacklist(self, monkeypatch):
        """When a whitelist is set the blacklist is ignored (returns None)."""
        monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", "silu_and_mul")
        monkeypatch.setenv("VLLM_FL_OOT_BLACKLIST", "rms_norm")
        assert get_oot_blacklist() is None

    @pytest.mark.supplementary
    def test_blacklist_from_env(self, monkeypatch):
        """The blacklist env var parses like the whitelist."""
        monkeypatch.setenv("VLLM_FL_OOT_BLACKLIST", "rms_norm, rotary_embedding")
        assert get_oot_blacklist() == ["rms_norm", "rotary_embedding"]

    @pytest.mark.supplementary
    def test_blacklist_falls_back_to_config(self, monkeypatch):
        """Without env vars the platform-config blacklist is used."""
        import vllm_fl.dispatch.config as config_mod

        monkeypatch.setattr(config_mod, "get_oot_blacklist", lambda: ["cfg_op"])
        assert get_oot_blacklist() == ["cfg_op"]

    @pytest.mark.supplementary
    def test_blacklist_config_error_degrades_to_none(self, monkeypatch):
        """A broken platform config degrades to None rather than raising."""
        import vllm_fl.dispatch.config as config_mod

        def boom():
            raise RuntimeError("no config here")

        monkeypatch.setattr(config_mod, "get_oot_blacklist", boom)
        assert get_oot_blacklist() is None

    @pytest.mark.supplementary
    def test_is_oot_enabled_variants(self, monkeypatch):
        """OOT enablement honours the gate then VLLM_FL_OOT_ENABLED."""
        assert is_oot_enabled() is True  # default
        monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "0")
        assert is_oot_enabled() is False
        monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "true")
        assert is_oot_enabled() is True
        monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "1")
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "0")
        assert is_oot_enabled() is False
