# Copyright (c) 2025 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl.envs`` module-level env dispatch.

``vllm_fl.envs`` resolves its three exported names lazily through the module
``__getattr__`` protocol so each read re-evaluates ``os.environ``.  These
tests pin the exact parsing semantics (truthy spellings, defaults, explicit
vs. unset) without any GPU or vllm dependency.
"""

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove every variable the envs module reads so defaults are visible."""
    for key in (
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_FL_PREFER",
        "USE_FLAGGEMS",
        "FLAGGEMS_ENABLE_OPLIST_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


class TestPreferEnabled:
    """``VLLM_FL_PREFER_ENABLED`` defaults to True and accepts 1/true."""

    def test_default_is_true_when_unset(self):
        """With the variable unset the lazy lookup must report True."""
        import vllm_fl.envs as envs

        assert envs.VLLM_FL_PREFER_ENABLED is True

    @pytest.mark.parametrize("value", ["True", "true", "TRUE", "1"])
    def test_truthy_spellings(self, monkeypatch, value):
        """Every accepted truthy spelling resolves to True."""
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", value)
        import vllm_fl.envs as envs

        assert envs.VLLM_FL_PREFER_ENABLED is True

    @pytest.mark.parametrize("value", ["False", "false", "0", "no", ""])
    def test_falsy_spellings(self, monkeypatch, value):
        """Anything outside {true, 1} resolves to False."""
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", value)
        import vllm_fl.envs as envs

        assert envs.VLLM_FL_PREFER_ENABLED is False


class TestFlaggemsOplistPath:
    """``FLAGGEMS_ENABLE_OPLIST_PATH`` falls back to the FlagGems default."""

    def test_default_path_when_unset(self):
        """The documented default path is returned when unset."""
        import vllm_fl.envs as envs

        assert envs.FLAGGEMS_ENABLE_OPLIST_PATH == "/tmp/flaggems_enable_oplist.txt"

    def test_env_override_wins(self, monkeypatch):
        """An explicitly set path is returned verbatim."""
        monkeypatch.setenv("FLAGGEMS_ENABLE_OPLIST_PATH", "custom/oplist.txt")
        import vllm_fl.envs as envs

        assert envs.FLAGGEMS_ENABLE_OPLIST_PATH == "custom/oplist.txt"


class TestUseFlaggems:
    """``USE_FLAGGEMS`` delegates to ``vllm_fl.utils.use_flaggems``."""

    def test_default_true_without_env_vars(self):
        """With no VLLM_FL_PREFER / USE_FLAGGEMS set, FlagGems is enabled."""
        import vllm_fl.envs as envs

        assert envs.USE_FLAGGEMS is True

    @pytest.mark.parametrize("value", ["1", "true", "True"])
    def test_explicit_enable(self, monkeypatch, value):
        """Explicit USE_FLAGGEMS=1/true keeps FlagGems on."""
        monkeypatch.setenv("USE_FLAGGEMS", value)
        import vllm_fl.envs as envs

        assert envs.USE_FLAGGEMS is True

    @pytest.mark.parametrize("value", ["0", "false", ""])
    def test_explicit_disable(self, monkeypatch, value):
        """Explicit USE_FLAGGEMS=0/false turns FlagGems off."""
        monkeypatch.setenv("USE_FLAGGEMS", value)
        import vllm_fl.envs as envs

        assert envs.USE_FLAGGEMS is False

    def test_prefer_enabled_gate_disables_flaggems(self, monkeypatch):
        """VLLM_FL_PREFER_ENABLED=false forces USE_FLAGGEMS False."""
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "false")
        import vllm_fl.envs as envs

        assert envs.USE_FLAGGEMS is False

    def test_non_flagos_backend_disables_flaggems(self, monkeypatch):
        """A different VLLM_FL_PREFER backend keeps FlagGems disabled."""
        monkeypatch.setenv("VLLM_FL_PREFER", "vendor")
        import vllm_fl.envs as envs

        assert envs.USE_FLAGGEMS is False


class TestLazyGetattr:
    """Module-level protocol behaviour of the lazy env dispatcher."""

    def test_unknown_attribute_raises(self):
        """An unknown name raises AttributeError naming the module."""
        import vllm_fl.envs as envs

        with pytest.raises(
            AttributeError, match="module 'vllm_fl.envs' has no attribute"
        ):
            envs.DEFINITELY_NOT_AN_ENV  # noqa: B018

    def test_dir_lists_all_env_names(self):
        """``__dir__`` exposes exactly the three known env names."""
        import vllm_fl.envs as envs

        listing = dir(envs)
        assert "VLLM_FL_PREFER_ENABLED" in listing
        assert "USE_FLAGGEMS" in listing
        assert "FLAGGEMS_ENABLE_OPLIST_PATH" in listing


class TestIsSet:
    """``is_set`` distinguishes explicit values from defaults."""

    def test_set_when_env_var_present(self, monkeypatch):
        """is_set reports True only when os.environ actually holds it."""
        import vllm_fl.envs as envs

        monkeypatch.delenv("VLLM_FL_PREFER_ENABLED", raising=False)
        assert envs.is_set("VLLM_FL_PREFER_ENABLED") is False
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "false")
        assert envs.is_set("VLLM_FL_PREFER_ENABLED") is True

    def test_empty_string_counts_as_set(self, monkeypatch):
        """An empty value is still an explicit set."""
        import vllm_fl.envs as envs

        monkeypatch.setenv("USE_FLAGGEMS", "")
        assert envs.is_set("USE_FLAGGEMS") is True

    def test_unknown_name_raises(self):
        """Unknown names raise AttributeError instead of returning False."""
        import vllm_fl.envs as envs

        with pytest.raises(AttributeError, match="has no attribute"):
            envs.is_set("NOT_A_KNOWN_ENV")
