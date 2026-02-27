# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unit tests for FlagGems operator whitelist/blacklist functionality.

Tests use_flaggems_op() and get_flag_gems_whitelist_blacklist() from vllm_fl.utils.
"""

import pytest

from vllm_fl.utils import get_flag_gems_whitelist_blacklist, use_flaggems_op


def _env_for_flaggems_enabled(monkeypatch):
    """Set env so use_flaggems() returns True (FlagGems enabled)."""
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "True")
    monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
    monkeypatch.setenv("USE_FLAGGEMS", "1")


# -----------------------------------------------------------------------------
# use_flaggems_op() - whitelist / blacklist
# -----------------------------------------------------------------------------


def test_use_flaggems_op_no_whitelist_no_blacklist_all_allowed(monkeypatch):
    """When neither whitelist nor blacklist is set, all ops are allowed."""
    _env_for_flaggems_enabled(monkeypatch)
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)

    assert use_flaggems_op("silu_and_mul") is True
    assert use_flaggems_op("rms_norm") is True
    assert use_flaggems_op("rotary_embedding") is True
    assert use_flaggems_op("any_other_op") is True


def test_use_flaggems_op_whitelist_only_in_list_allowed(monkeypatch):
    """When whitelist is set, only ops in the list are allowed."""
    _env_for_flaggems_enabled(monkeypatch)
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "silu_and_mul,rms_norm")
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)

    assert use_flaggems_op("silu_and_mul") is True
    assert use_flaggems_op("rms_norm") is True
    assert use_flaggems_op("rotary_embedding") is False
    assert use_flaggems_op("other_op") is False


def test_use_flaggems_op_whitelist_strips_spaces(monkeypatch):
    """Whitelist parsing strips spaces around operator names."""
    _env_for_flaggems_enabled(monkeypatch)
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", " silu_and_mul , rms_norm ")
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)

    assert use_flaggems_op("silu_and_mul") is True
    assert use_flaggems_op("rms_norm") is True
    assert use_flaggems_op("rotary_embedding") is False


def test_use_flaggems_op_blacklist_only_listed_disallowed(monkeypatch):
    """When blacklist is set, listed ops are disallowed, others allowed."""
    _env_for_flaggems_enabled(monkeypatch)
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "rms_norm,rotary_embedding")

    assert use_flaggems_op("silu_and_mul") is True
    assert use_flaggems_op("rms_norm") is False
    assert use_flaggems_op("rotary_embedding") is False
    assert use_flaggems_op("other_op") is True


def test_use_flaggems_op_whitelist_and_blacklist_both_set_raises(monkeypatch):
    """When both whitelist and blacklist are set, ValueError is raised."""
    _env_for_flaggems_enabled(monkeypatch)
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "silu_and_mul,rms_norm")
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "rms_norm,other")

    with pytest.raises(ValueError) as exc_info:
        use_flaggems_op("silu_and_mul")
    # Implementation disallows setting both whitelist and blacklist simultaneously
    assert "VLLM_FL_FLAGOS_WHITELIST" in str(exc_info.value)
    assert "VLLM_FL_FLAGOS_BLACKLIST" in str(exc_info.value)


def test_use_flaggems_op_flaggems_disabled_returns_false(monkeypatch):
    """When USE_FLAGGEMS is 0, use_flaggems_op always returns False."""
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "True")
    monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
    monkeypatch.setenv("USE_FLAGGEMS", "0")
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "silu_and_mul")
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)

    assert use_flaggems_op("silu_and_mul") is False
    assert use_flaggems_op("rms_norm") is False


def test_use_flaggems_op_default_when_flaggems_unset(monkeypatch):
    """When USE_FLAGGEMS is unset, default parameter is used for use_flaggems."""
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "True")
    monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
    monkeypatch.delenv("USE_FLAGGEMS", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)

    assert use_flaggems_op("silu_and_mul", default=True) is True
    assert use_flaggems_op("silu_and_mul", default=False) is False


# -----------------------------------------------------------------------------
# get_flag_gems_whitelist_blacklist()
# -----------------------------------------------------------------------------


def test_get_flag_gems_whitelist_blacklist_neither_set(monkeypatch):
    """When neither env is set, returns (None, None)."""
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert whitelist is None
    assert blacklist is None


def test_get_flag_gems_whitelist_blacklist_whitelist_only(monkeypatch):
    """When only whitelist is set, returns (list, None)."""
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "silu_and_mul,rms_norm")
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert whitelist == ["silu_and_mul", "rms_norm"]
    assert blacklist is None


def test_get_flag_gems_whitelist_blacklist_whitelist_strips_spaces(monkeypatch):
    """Whitelist parsing strips spaces."""
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", " a , b , c ")
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert whitelist == ["a", "b", "c"]
    assert blacklist is None


def test_get_flag_gems_whitelist_blacklist_blacklist_only(monkeypatch):
    """When only blacklist is set, returns (None, list)."""
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "index,index_put_")

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert whitelist is None
    assert blacklist == ["index", "index_put_"]


def test_get_flag_gems_whitelist_blacklist_both_set_raises(monkeypatch):
    """When both whitelist and blacklist are set, ValueError is raised."""
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "silu_and_mul")
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "rms_norm")

    with pytest.raises(ValueError) as exc_info:
        get_flag_gems_whitelist_blacklist()
    assert "VLLM_FL_FLAGOS_WHITELIST" in str(exc_info.value)
    assert "VLLM_FL_FLAGOS_BLACKLIST" in str(exc_info.value)
    assert (
        "simultaneously" in str(exc_info.value).lower()
        or "both" in str(exc_info.value).lower()
    )


def test_get_flag_gems_whitelist_blacklist_empty_strings(monkeypatch):
    """Empty or whitespace-only env values yield None / empty list handling."""
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "")
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "")

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert whitelist is None
    assert blacklist is None
