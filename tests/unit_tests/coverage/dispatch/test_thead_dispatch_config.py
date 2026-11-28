# Copyright 2026 BAAI. All rights reserved.

"""thead dispatch configuration contracts.

``vllm_fl/dispatch/config/thead.yaml`` had no loading test — neither in
``tests/unit_tests/dispatch/test_config_utils.py`` (which pins kunlunxin,
enflame, musa, nvidia) nor in the supplementary tree. These tests follow
that kunlunxin pattern and pin the thead rows:

* the attention route is ``[vendor:thead, flagos, reference]`` — the vendor
  FA3 bridge is tried first, with the FlagGems and PyTorch reference paths
  as fallbacks (``strict: false``);
* the shared ops (rms_norm / silu_and_mul / rotary_embedding) route
  flagos -> reference, with no vendor attempt;
* the prefer default is flagos and the FlagOS blacklist is empty;
* the runner-side platform file ``tests/platforms/thead.yaml`` exports the
  container contract via ``env_defaults`` (GEMS_VENDOR=thead,
  VLLM_TARGET_DEVICE=empty, FLAGCX_PATH=...) that the gate recipes rely on.

The dispatch config rows are platform-agnostic data loading (the same test
style ``test_config_utils.py`` uses for other vendors), so this file is not
gated on a thead session.
"""

from __future__ import annotations

import pytest

from vllm_fl.dispatch.config.utils import (
    get_config_path,
    load_platform_config,
)

pytestmark = [pytest.mark.supplementary]


class TestDispatchYaml:
    """``vllm_fl/dispatch/config/thead.yaml`` loads with the pinned policy."""

    def test_config_path_resolves(self):
        path = get_config_path("thead")
        assert path is not None and path.name == "thead.yaml"

    def test_prefer_is_flagos(self):
        config = load_platform_config("thead")
        assert config["prefer"] == "flagos"

    def test_strict_mode_is_off(self):
        """A failed backend attempt falls through to the next in order."""
        config = load_platform_config("thead")
        assert config["strict"] is False

    def test_attention_routes_vendor_first(self):
        """PPU attention must try the vendor FA3 bridge before flagos."""
        config = load_platform_config("thead")
        assert config["op_backends"]["attention_backend"] == [
            "vendor:thead",
            "flagos",
            "reference",
        ]

    @pytest.mark.parametrize(
        "op",
        ["rms_norm", "silu_and_mul", "rotary_embedding"],
    )
    def test_shared_ops_route_flagos_then_reference(self, op):
        """The shared ops have no thead vendor path: flagos, then torch."""
        config = load_platform_config("thead")
        assert config["op_backends"][op] == ["flagos", "reference"]

    def test_flagos_blacklist_is_empty(self):
        """No FlagOS operator is blacklisted for thead.

        The yaml key carries no entries (parses as None); both None and []
        mean "no blacklist" — assert the loader's effective result too, which
        is the value the dispatch policy actually consumes.
        """
        from vllm_fl.dispatch.config.utils import get_flagos_blacklist

        config = load_platform_config("thead")
        assert config.get("flagos_blacklist") in (None, [])
        assert get_flagos_blacklist(config) in (None, [])


class TestRunnerPlatformYaml:
    """``tests/platforms/thead.yaml`` — the container contract the gate uses."""

    @pytest.fixture()
    def platform_config(self):
        from tests.utils.platform_config import PlatformConfig

        return PlatformConfig.load("thead")

    def test_platform_and_vendor_are_thead(self, platform_config):
        assert platform_config.platform == "thead"
        assert platform_config.vendor == "thead"

    def test_env_defaults_export_the_container_contract(self, platform_config):
        env = platform_config.env_defaults
        assert env.get("GEMS_VENDOR") == "thead", (
            "the gate recipe resolves the thead platform through GEMS_VENDOR"
        )
        assert env.get("VLLM_TARGET_DEVICE") == "empty"
        assert env.get("VLLM_PLUGINS") == "fl"
        assert env.get("USE_FLAGGEMS") == "1"

    def test_env_defaults_carry_the_flagcx_path(self, platform_config):
        """FLAGCX_PATH is what routes dist_backend to flagcx on the PPU."""
        assert platform_config.env_defaults.get("FLAGCX_PATH")

    def test_zw810e_documents_compute_capability_8_0(self, platform_config):
        """The CC window asserted by the capability-contract suite and the
        live platform probe is documented here as 8.0."""
        zw810e = platform_config.device_types.get("zw810e")
        assert zw810e is not None, "zw810e must be a documented device type"
        assert str(zw810e.compute_capability) == "8.0"
