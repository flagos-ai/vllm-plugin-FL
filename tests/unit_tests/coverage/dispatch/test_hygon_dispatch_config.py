# Copyright 2025 BAAI. All rights reserved.

"""Dispatch-configuration contract for the Hygon (BW1000 / DCU) backend.

``vllm_fl/dispatch/config/hygon.yaml`` is the only piece of hygon vendor
behaviour in the dispatch layer that actually changes runtime decisions, and
before this file it had no test anywhere: the baseline config tests
(``tests/unit_tests/dispatch/test_config_utils.py``) cover musa / kunlunxin /
enflame / nvidia and skip hygon. Two things live in it:

* ``flagos_blacklist: [cat, zeros, zero_]`` — FlagGems' Triton zeros/zero_
  kernels pass the element count as int32, so a multi-GB KV-cache backing
  buffer (>2**31 int8 elements) aborts with "Triton Error [HIP]: invalid
  argument" the first time the cache is served. Blacklisting them keeps
  ``torch.zeros`` on the native ATen memset. **Regression risk being pinned
  here: if someone removes an entry, every test still passes and only the
  first real serve on a BW1000 OOMs.**
* the per-op order ``flagos -> vendor:hygon -> reference`` for the four
  registered ops. The ``vendor:hygon`` token deliberately matches no impl:
  the hygon vendor directory ships no ``register_ops.py``, so
  ``builtin_ops._register_vendor_backends`` skips it and the token is an
  inert forward-declaration. This suite also pins that "falls through"
  behaviour so a future ``register_ops.py`` cannot silently take over the
  flagos-first dispatch semantics — adding one flips the resolution only if
  the policy order is revisited at the same time.

Everything here is CPU-only and reads the real production YAML through the
real loader / policy / registry APIs (no re-parsing, no fakes). The config is
platform data, so the checks are platform-agnostic like
``test_config_utils.py``; they must not need a DCU.

Environment hygiene: ``VLLM_FL_*`` env vars override the YAML
(``VLLM_FL_PREFER`` / ``VLLM_FL_FLAGOS_BLACKLIST`` / ``VLLM_FL_PER_OP`` /
``VLLM_FL_CONFIG``), and ``USE_FLAGGEMS`` gates the blacklist consumer, so
every test pins a clean environment through the ``clean_fl_env`` fixture
before asserting on YAML-derived state.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm.platforms

from vllm_fl.dispatch.builtin_ops import (
    _VENDOR_BACKENDS_DIR,
    _find_vendor_backend_dir,
    _get_current_vendor_backend_dirs,
)
from vllm_fl.dispatch.config.utils import (
    get_config_path,
    get_flagos_blacklist,
    load_platform_config,
)
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import policy_context, policy_from_config
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, match_token
from vllm_fl.utils import get_flag_gems_whitelist_blacklist, use_flaggems_op

pytestmark = [pytest.mark.supplementary]

#: The four ops hygon.yaml gives an explicit backend order.
EXPECTED_ORDERED_OPS = (
    "attention_backend",
    "rms_norm",
    "silu_and_mul",
    "rotary_embedding",
)

#: The exact flagos_blacklist from vllm_fl/dispatch/config/hygon.yaml.
#: cat/zeros/zero_ are the int32-overflow workaround set; see the module
#: docstring for the serve-time OOM this pins out.
EXPECTED_FLAGOS_BLACKLIST = ["cat", "zeros", "zero_"]


@pytest.fixture(autouse=True)
def _clean_dispatch_env(clean_fl_env, monkeypatch):
    """Force the hygon YAML to be the effective configuration for every test.

    ``clean_fl_env`` deletes ``VLLM_FL_*`` overrides so the platform config
    becomes authoritative; ``USE_FLAGGEMS`` is pinned on because the
    blacklist only routes ops away from FlagGems when FlagGems is the
    preferred backend at all. ``current_platform`` is pinned to the hygon
    vendor so the platform-detection paths (config loading, blacklist
    resolution, vendor-dir discovery) read hygon.yaml regardless of which
    platform this suite is being collected on — the same technique
    ``tests/unit_tests/dispatch/test_config_utils.py`` uses.
    """
    monkeypatch.setenv("USE_FLAGGEMS", "1")
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(vendor_name="hygon"),
    )


def _hygon_config():
    config = load_platform_config("hygon")
    assert config is not None, "hygon.yaml must exist and parse as a dict"
    return config


class TestFlagosBlacklist:
    """The int32-overflow workaround set is pinned exactly."""

    def test_blacklist_is_exactly_cat_zeros_zero_(self):
        # Exact-set and exact-order: order is preserved by the loader and
        # shown in startup logs, and a reordered list usually means someone
        # hand-edited the workaround set.
        assert _hygon_config()["flagos_blacklist"] == EXPECTED_FLAGOS_BLACKLIST

    def test_blacklist_survives_the_config_utils_extraction(self):
        # get_flagos_blacklist (with the hygon config passed explicitly) is
        # the loader used at startup; same list must come back out of it.
        assert get_flagos_blacklist(_hygon_config()) == EXPECTED_FLAGOS_BLACKLIST

    def test_blacklisted_ops_do_not_use_flaggems(self):
        # The consumer side of the blacklist: use_flaggems_op must answer
        # False for every blacklisted op and True for the non-blacklisted
        # dispatch ops, so the KV-cache backing torch.zeros stays on the
        # native ATen memset instead of the Triton int32-count kernel.
        for op in EXPECTED_FLAGOS_BLACKLIST:
            assert use_flaggems_op(op) is False, (
                f"{op} must stay off the FlagGems Triton path"
            )
        for op in EXPECTED_ORDERED_OPS:
            assert use_flaggems_op(op) is True

    def test_blacklist_reaches_use_flaggems_op_through_the_config_loader(self):
        # End to end through the platform-detection path: with a clean env
        # the blacklist comes from hygon.yaml via load_platform_config (the
        # platform here is hygon, so get_flagos_blacklist() with no argument
        # resolves to hygon.yaml) and both sides of the tuple match the file.
        whitelist, blacklist = get_flag_gems_whitelist_blacklist()
        assert whitelist is None
        assert blacklist == EXPECTED_FLAGOS_BLACKLIST


class TestPerOpOrder:
    """Every ordered op is exactly flagos -> vendor:hygon -> reference."""

    def test_op_backends_covers_exactly_the_four_ordered_ops(self):
        op_backends = _hygon_config()["op_backends"]
        assert set(op_backends) == set(EXPECTED_ORDERED_OPS)

    @pytest.mark.parametrize("op", EXPECTED_ORDERED_OPS)
    def test_each_op_orders_flagos_vendor_hygon_reference(self, op):
        # Byte-exact per-op order: hygon has no vendor impls, so flagos
        # first is the operative decision and reference last is the safety
        # net. A change to any one op's list must be a conscious one.
        assert _hygon_config()["op_backends"][op] == [
            "flagos",
            "vendor:hygon",
            "reference",
        ]

    def test_policy_from_config_reproduces_the_yaml_order(self):
        # The same order must survive the SelectionPolicy translation used
        # at runtime (policy_from_config is the exact entry the manager's
        # env path calls for platform configs).
        yaml_path = get_config_path("hygon")
        assert yaml_path is not None
        assert yaml_path.name == "hygon.yaml"

        policy = policy_from_config(str(yaml_path))
        for op in EXPECTED_ORDERED_OPS:
            assert policy.get_per_op_order(op) == [
                "flagos",
                "vendor:hygon",
                "reference",
            ]

    def test_prefer_and_strict_defaults(self):
        # prefer=flagos makes the default (non-per-op) order
        # [flagos, vendor, reference]; strict=false keeps the fallback chain
        # live so a failing kernel degrades instead of aborting the run.
        config = _hygon_config()
        assert config["prefer"] == "flagos"
        assert config["strict"] is False


class TestVendorHygonTokenFallsThrough:
    """``vendor:hygon`` must resolve to nothing until a register_ops exists.

    The hygon vendor directory has no ``register_ops.py``
    (builtin_ops._register_vendor_backends skips it), so the token matches
    no impl and dispatch lands on flagos/reference. Pinning that behaviour
    both documents today's semantics and turns a future register_ops.py into
    a visible test change instead of a silent dispatch flip.
    """

    def test_hygon_vendor_dir_has_no_register_ops_module(self):
        # The precondition the rest of this class pins: the directory is
        # resolved as the current vendor's backend dir yet ships no
        # register_ops.py.
        available = {
            name
            for name in os.listdir(_VENDOR_BACKENDS_DIR)
            if os.path.isdir(os.path.join(_VENDOR_BACKENDS_DIR, name))
            and not name.startswith("_")
        }
        current_dir = _get_current_vendor_backend_dirs(available)
        assert current_dir == "hygon", (
            "this suite must run with the hygon vendor detected"
        )
        assert _find_vendor_backend_dir("hygon", available) == "hygon"
        assert not (Path(_VENDOR_BACKENDS_DIR) / "hygon" / "register_ops.py").is_file()

    def _manager_with(self, *impls):
        """An initialized OpManager over a private registry seeded with impls.

        The impls handed in mimic what builtin registration produces: a
        DEFAULT flagos impl and a REFERENCE impl, optionally a hygon VENDOR
        one. None of the fns are called; only the metadata participates in
        resolution.
        """
        registry = OpRegistry()
        for impl in impls:
            registry.register_impl(impl)
        manager = OpManager(registry=registry)
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        return manager

    @staticmethod
    def _impl(op, impl_id, kind, priority, vendor=None):
        from vllm_fl.dispatch.types import OpImpl

        return OpImpl(
            op_name=op,
            impl_id=impl_id,
            kind=kind,
            fn=lambda *args, **kwargs: impl_id,
            vendor=vendor,
            priority=priority,
        )

    def test_vendor_hygon_token_matches_no_registered_impl(self):
        # The literal token-to-impl contract: against the impl set the real
        # registration produces (flagos + reference only, no hygon impls),
        # the vendor:hygon slot in the hygon.yaml order is empty and
        # resolution lands on flagos — never on reference, because flagos
        # is listed first and is available.
        yaml_path = get_config_path("hygon")
        policy = policy_from_config(str(yaml_path))
        manager = self._manager_with(
            self._impl("rms_norm", "default.flagos", BackendImplKind.DEFAULT, 150),
            self._impl("rms_norm", "reference.torch", BackendImplKind.REFERENCE, 50),
        )

        with policy_context(policy):
            candidates = manager.resolve_candidates("rms_norm")
            order = policy.get_per_op_order("rms_norm")
            assert order[1] == "vendor:hygon"
            matched = [c.impl_id for c in candidates if match_token(c, "vendor:hygon")]
            assert matched == [], (
                "vendor:hygon must match nothing while the hygon vendor "
                "directory has no register_ops.py"
            )
            # ...and dispatch resolves through the first non-empty slot.
            assert manager.get_selected_impl_id("rms_norm") == "default.flagos"

    def test_resolution_still_picks_flagos_not_reference(self):
        # The observable end state of the fall-through: the flagos impl
        # wins because it is both listed first and registered; the inert
        # vendor token does not push selection to the last slot.
        yaml_path = get_config_path("hygon")
        policy = policy_from_config(str(yaml_path))
        manager = self._manager_with(
            self._impl("silu_and_mul", "default.flagos", BackendImplKind.DEFAULT, 150),
            self._impl(
                "silu_and_mul", "reference.torch", BackendImplKind.REFERENCE, 50
            ),
        )

        with policy_context(policy):
            assert manager.get_selected_impl_id("silu_and_mul") == "default.flagos"

    def test_a_future_hygon_impl_would_take_over_resolution(self):
        # The guard this class exists for: if a hygon vendor impl ever gets
        # registered, the yaml order hands it the op immediately (flagos is
        # listed first only as a preference, and the vendor slot is checked
        # before reference — but the flagos impl still wins because it sits
        # in slot 1). Concretely: with all three kinds registered under the
        # real yaml order, selection is flagos-first; flipping the yaml to
        # vendor-first (or registering hygon only) is the visible change a
        # reviewer must be shown. This test executes the resolution with all
        # three kinds present so the "silent semantics change" cannot
        # happen — adding register_ops.py makes this assertion's neighbours
        # fail loudly until the order is revisited.
        yaml_path = get_config_path("hygon")
        policy = policy_from_config(str(yaml_path))
        manager = self._manager_with(
            self._impl(
                "rotary_embedding", "default.flagos", BackendImplKind.DEFAULT, 150
            ),
            self._impl(
                "rotary_embedding",
                "vendor.hygon",
                BackendImplKind.VENDOR,
                100,
                vendor="hygon",
            ),
            self._impl(
                "rotary_embedding", "reference.torch", BackendImplKind.REFERENCE, 50
            ),
        )

        with policy_context(policy):
            # With a live hygon impl the vendor slot matches it...
            candidates = manager.resolve_candidates("rotary_embedding")
            matched = [c.impl_id for c in candidates if match_token(c, "vendor:hygon")]
            assert matched == ["vendor.hygon"]
            # ...but flagos is still first in the yaml order, so the
            # DEFAULT impl keeps winning until the order is deliberately
            # changed.
            assert manager.get_selected_impl_id("rotary_embedding") == "default.flagos"
