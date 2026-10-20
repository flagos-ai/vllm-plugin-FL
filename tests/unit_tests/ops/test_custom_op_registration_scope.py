# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("install_hooks", [None, False, True])
def test_explicit_moe_hook_scope(monkeypatch, enabled, install_hooks):
    import vllm.platforms

    import vllm_fl.dispatch.config.utils as config_utils
    import vllm_fl.ops.custom_ops as ops
    import vllm_fl.utils as utils

    oracle, factory = Mock(), Mock()
    monkeypatch.setattr(ops, "_patch_unquantized_moe_oracle", oracle)
    monkeypatch.setattr(ops, "_patch_fused_moe_factory", factory)
    monkeypatch.setattr(ops, "OOT_OPS", {})
    monkeypatch.setattr(utils, "is_oot_enabled", lambda: enabled)
    monkeypatch.setattr(utils, "get_oot_whitelist", lambda: None)
    monkeypatch.setattr(utils, "get_oot_blacklist", lambda: [])
    monkeypatch.setattr(config_utils, "get_platform_name", lambda: "metax")
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(device_type="cuda", vendor_name="metax"),
    )
    kwargs = {} if install_hooks is None else {"install_moe_hooks": install_hooks}
    ops.register_oot_ops(**kwargs)
    expected = install_hooks is not False
    assert oracle.call_count == int(expected)
    assert factory.call_count == int(expected and enabled)
