# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vllm_fl.utils env gates: FlagGems selection, op config, OOT lists (E1).

Environment variables are the CONTRACT R2 sanctioned boundary, so every test
drives behavior through monkeypatch.setenv/delenv. Expected values come from
VENDOR_DEVICE_MAP, the nvidia.yaml platform blacklist and the qwen3_8 model
defaults (all verified against source).
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

# The shim import is load-bearing: it must precede vllm_fl/vllm
# imports on hosts without triton. Do not alphabetize.
# isort: split
from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)

# isort: split
import vllm_fl.utils as fl_utils

_FLAGGEMS_ENV_VARS = (
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
    "FLAGOS_DEVICE_CONTROL_ENV_VAR",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove every env var under test before each test."""
    for name in _FLAGGEMS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestUseFlaggems:
    """The VLLM_FL_PREFER / USE_FLAGGEMS environment ladder."""

    def test_defaults_to_true_and_honors_default_argument(self):
        """Unset env vars: default True, and default=False flips it."""
        assert fl_utils.use_flaggems() is True
        assert fl_utils.use_flaggems(default=False) is False

    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [("1", True), ("true", True), ("0", False), ("false", False), ("yes", False)],
    )
    def test_use_flaggems_env_values(self, monkeypatch, env_value, expected):
        """USE_FLAGGEMS accepts true/1 and rejects everything else."""
        monkeypatch.setenv("USE_FLAGGEMS", env_value)
        assert fl_utils.use_flaggems(default=False) is expected

    def test_use_flaggems_env_overrides_default(self, monkeypatch):
        """An explicit USE_FLAGGEMS=1 beats default=False."""
        monkeypatch.setenv("USE_FLAGGEMS", "1")
        assert fl_utils.use_flaggems(default=False) is True

    def test_prefer_backend_flagos_keeps_flaggems(self, monkeypatch):
        """VLLM_FL_PREFER=flagos matches the FlagOS backend."""
        monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
        assert fl_utils.use_flaggems() is True

    def test_prefer_backend_other_disables_flaggems(self, monkeypatch):
        """A non-flagos VLLM_FL_PREFER disables FlagGems entirely."""
        monkeypatch.setenv("VLLM_FL_PREFER", "vendor")
        assert fl_utils.use_flaggems() is False

    @pytest.mark.parametrize("value", ["0", "false", "no"])
    def test_prefer_enabled_gate_disables_everything(self, monkeypatch, value):
        """VLLM_FL_PREFER_ENABLED=false wins over all later env vars."""
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", value)
        monkeypatch.setenv("USE_FLAGGEMS", "1")
        assert fl_utils.use_flaggems() is False


class TestOpConfigFromEnv:
    """_load_op_config_from_env validation and reload behavior."""

    def test_unset_empty_and_whitespace_paths_yield_none(self, monkeypatch):
        """A missing, empty or whitespace-only VLLM_FL_OP_CONFIG clears state."""
        monkeypatch.delenv("VLLM_FL_OP_CONFIG", raising=False)
        fl_utils._load_op_config_from_env()
        assert fl_utils.get_op_config() is None

        monkeypatch.setenv("VLLM_FL_OP_CONFIG", "")
        fl_utils._load_op_config_from_env()
        assert fl_utils.get_op_config() is None

        monkeypatch.setenv("VLLM_FL_OP_CONFIG", "   ")
        fl_utils._load_op_config_from_env()
        assert fl_utils.get_op_config() is None

    def test_missing_file_raises_value_failure(self, monkeypatch, tmp_path):
        """A non-existent config path raises with the path in the message."""
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(tmp_path / "absent.json"))
        with pytest.raises(ValueError, match="file not found"):
            fl_utils._load_op_config_from_env()
        assert fl_utils.get_op_config() is None

    def test_invalid_json_raises_value_failure(self, monkeypatch, tmp_path):
        """Malformed JSON is re-raised as an actionable ValueError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(bad))
        with pytest.raises(ValueError, match="Invalid VLLM_FL_OP_CONFIG JSON"):
            fl_utils._load_op_config_from_env()
        assert fl_utils.get_op_config() is None

    def test_non_object_json_raises_value_failure(self, monkeypatch, tmp_path):
        """A JSON array is rejected: the config must be an object."""
        non_object = tmp_path / "non_object.json"
        non_object.write_text("[1, 2]")
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(non_object))
        with pytest.raises(ValueError, match="must be a JSON object"):
            fl_utils._load_op_config_from_env()
        assert fl_utils.get_op_config() is None

    def test_non_string_values_raise_value_failure(self, monkeypatch, tmp_path):
        """Values must map strings to strings; integers are rejected."""
        bad_values = tmp_path / "bad_values.json"
        bad_values.write_text('{"op": 1}')
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(bad_values))
        with pytest.raises(ValueError, match="strings to strings"):
            fl_utils._load_op_config_from_env()
        assert fl_utils.get_op_config() is None

    def test_valid_config_is_loaded_and_reloadable(self, monkeypatch, tmp_path):
        """A valid string-to-string JSON config loads verbatim."""
        config_file = tmp_path / "good.json"
        config_file.write_text(json.dumps({"backend": "flaggems"}))
        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(config_file))
        fl_utils._load_op_config_from_env()
        assert fl_utils.get_op_config() == {"backend": "flaggems"}

        monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(tmp_path / "second.json"))
        (tmp_path / "second.json").write_text(json.dumps({"backend": "vendor"}))
        fl_utils._load_op_config_from_env()
        assert fl_utils.get_op_config() == {"backend": "vendor"}


class TestWhitelistBlacklist:
    """get_flag_gems_whitelist_blacklist priority ladder and use_flaggems_op."""

    def test_whitelist_and_blacklist_env_conflict_raises(self, monkeypatch):
        """Setting both env vars at once is a configuration error."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "op_a")
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "op_b")
        with pytest.raises(ValueError, match="simultaneously"):
            fl_utils.get_flag_gems_whitelist_blacklist()
        # After clearing the conflicting pair, a plain blacklist parses again.
        monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST")
        _, blacklist = fl_utils.get_flag_gems_whitelist_blacklist()
        assert blacklist == ["op_b"]

    def test_whitelist_env_parses_and_overrides_blacklist(self, monkeypatch):
        """A whitelist strips whitespace and completely overrides any blacklist."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", " op_a , op_b ,")
        whitelist, blacklist = fl_utils.get_flag_gems_whitelist_blacklist()
        assert whitelist == ["op_a", "op_b"]
        assert blacklist is None

    def test_blacklist_env_parses(self, monkeypatch):
        """A blacklist env var parses with whitespace stripping."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", " op_a , op_b ,")
        whitelist, blacklist = fl_utils.get_flag_gems_whitelist_blacklist()
        assert whitelist is None
        assert blacklist == ["op_a", "op_b"]

    def test_platform_config_blacklist_loaded_for_nvidia(self, monkeypatch):
        """The nvidia.yaml platform blacklist becomes the default blacklist."""
        from vllm import platforms as vllm_platforms

        monkeypatch.setattr(
            vllm_platforms,
            "current_platform",
            NS(vendor_name="nvidia"),
            raising=False,
        )
        _, blacklist = fl_utils.get_flag_gems_whitelist_blacklist()
        assert blacklist == [
            "empty",
            "index_put_",
            "index_put",
            "_index_put_impl_",
            "nonzero",
            "copy_",
            "to_copy",
            "index",
            "cat",
            "unfold",
            "linear",
            "mm_out",
        ]

    def test_blacklist_append_dedupes_preserving_order(self, monkeypatch):
        """VLLM_FL_FLAGOS_BLACKLIST_APPEND appends unseen ops, skips dupes."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "cat")
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "cat, zzz , cat,")
        _, blacklist = fl_utils.get_flag_gems_whitelist_blacklist()
        assert blacklist == ["cat", "zzz"]

    def test_model_defaults_merge_for_qwen3_8_architecture(self, monkeypatch):
        """Qwen3_8 model defaults merge (deduped) with the current blacklist."""
        import vllm_fl.dispatch.config as dispatch_config

        monkeypatch.setattr(dispatch_config, "get_flagos_blacklist", lambda: None)
        self._ensure_qwen3_8_registered()
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "zzz")
        vllm_config = NS(
            model_config=NS(architectures=["Qwen3_8FlashNextForConditionalGeneration"])
        )
        _, blacklist = fl_utils.get_flag_gems_whitelist_blacklist(
            vllm_config=vllm_config, vendor_name="nvidia"
        )
        assert blacklist[0] == "index_select"
        assert "addmm_" in blacklist and "sort_stable" in blacklist
        assert blacklist[-1] == "zzz"
        assert len(blacklist) == len(set(blacklist))

    def test_model_defaults_empty_for_other_vendors(self, monkeypatch):
        """The qwen3_8 defaults table is nvidia-specific."""
        import vllm_fl.dispatch.config as dispatch_config

        monkeypatch.setattr(dispatch_config, "get_flagos_blacklist", lambda: None)
        self._ensure_qwen3_8_registered()
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "zzz")
        vllm_config = NS(
            model_config=NS(architectures=["Qwen3_8FlashNextForConditionalGeneration"])
        )
        _, blacklist = fl_utils.get_flag_gems_whitelist_blacklist(
            vllm_config=vllm_config, vendor_name="ascend"
        )
        assert blacklist == ["zzz"]

    @staticmethod
    def _ensure_qwen3_8_registered():
        """Register the qwen3_8 model config (idempotent) for the lookup."""
        import sys
        import types as types_module

        try:
            import vllm.transformers_utils.configs.qwen3_next  # noqa: F401
        except ImportError:
            # vLLM 0.33 moved this config module; the A100 image (0.24) ships
            # it natively. Stub only the class the patch imports.
            from transformers import PretrainedConfig

            stub = types_module.ModuleType("vllm.transformers_utils.configs.qwen3_next")

            class Qwen3NextConfig(PretrainedConfig):
                model_type = "qwen3_next"

            stub.Qwen3NextConfig = Qwen3NextConfig
            sys.modules["vllm.transformers_utils.configs.qwen3_next"] = stub

        from vllm_fl.patches.qwen3_8_flash_next import (
            apply_qwen3_8_flash_next_patches,
        )

        apply_qwen3_8_flash_next_patches()

    def test_use_flaggems_op_whitelist_membership(self, monkeypatch):
        """With a whitelist, only member ops use FlagGems."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "silu_and_mul, rms_norm")
        assert fl_utils.use_flaggems_op("silu_and_mul") is True
        assert fl_utils.use_flaggems_op("rms_norm_and_clamp") is False

    def test_use_flaggems_op_blacklist_denies_listed_ops(self, monkeypatch):
        """With a blacklist, listed ops fall back and others use FlagGems."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "op_a, op_b")
        assert fl_utils.use_flaggems_op("op_a") is False
        assert fl_utils.use_flaggems_op("op_c") is True

    def test_use_flaggems_op_respects_master_disable(self, monkeypatch):
        """VLLM_FL_PREFER_ENABLED=false denies every op."""
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "false")
        assert fl_utils.use_flaggems_op("anything") is False
