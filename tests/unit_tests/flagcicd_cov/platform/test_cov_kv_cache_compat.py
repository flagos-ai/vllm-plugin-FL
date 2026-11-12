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

"""vllm_fl.compat.kv_cache owner registration and binding flow (E1).

Order note: these tests invoke the real upstream bind helper
(vllm.v1.worker.utils through vllm_fl.worker.kv_cache_utils), whose import
chain transitively imports vllm's attention package. Sibling suites that
stub that package in sys.modules are order-sensitive to any earlier import;
run such suites before this one (pytest's default alphabetical order does).

The full bind flow runs against the real upstream helper
(vllm.v1.worker.utils.bind_kv_cache through vllm_fl.worker.kv_cache_utils)
on the CPU-shaped platform; on vLLM 0.24 the upstream helper calls layer
hooks itself, so the owner path exercises preflight plus upstream binding.
Owner classes are real classes registered through the public API; the
module-level registered-type set is restored after each test.
"""

from __future__ import annotations

import pytest
import torch

from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)
from vllm_fl.compat import kv_cache as compat_kv_cache


class _Owner:
    """KV-cache owner recording validation and binding."""

    def __init__(self):
        self.validated = None
        self.bound = None

    def validate_kv_cache(self, kv_cache):
        self.validated = kv_cache

    def bind_kv_cache(self, kv_cache):
        self.bound = kv_cache


class _BadValidateOwner:
    """Owner whose validation rejects the cache before any mutation."""

    def __init__(self):
        self.bound = None

    def validate_kv_cache(self, kv_cache):
        raise ValueError("unsupported kv cache layout")

    def bind_kv_cache(self, kv_cache):
        self.bound = kv_cache


@pytest.fixture(autouse=True)
def _restore_registered_owners():
    """Snapshot the module-level owner registry for every test."""
    saved = set(compat_kv_cache._REGISTERED_OWNER_TYPES)
    yield
    compat_kv_cache._REGISTERED_OWNER_TYPES.clear()
    compat_kv_cache._REGISTERED_OWNER_TYPES.update(saved)


class TestRegisterKvCacheOwner:
    """register_kv_cache_owner validation and decorator use."""

    def test_owner_without_hook_rejected(self):
        """A class without a callable bind_kv_cache raises TypeError."""
        rejected = type("NoHook", (), {})
        with pytest.raises(TypeError, match="no callable bind_kv_cache"):
            compat_kv_cache.register_kv_cache_owner(rejected)
        assert not compat_kv_cache.is_registered_owner(rejected())

    def test_non_callable_hook_rejected(self):
        """A non-callable bind_kv_cache attribute is equally rejected."""
        rejected = type("AttrHook", (), {"bind_kv_cache": "not-callable"})
        with pytest.raises(TypeError, match="no callable"):
            compat_kv_cache.register_kv_cache_owner(rejected)
        assert not compat_kv_cache.is_registered_owner(rejected())

    def test_decorator_returns_class_and_registers(self):
        """Used as a decorator the function returns the class unchanged."""

        @compat_kv_cache.register_kv_cache_owner
        class Decorated:
            def bind_kv_cache(self, kv_cache):
                return None

        assert Decorated.__name__ == "Decorated"
        assert compat_kv_cache.is_registered_owner(Decorated())

    def test_is_registered_owner_subclass_semantics(self):
        """Subclasses of a registered owner are recognized too."""
        compat_kv_cache.register_kv_cache_owner(_Owner)

        class Child(_Owner):
            pass

        assert compat_kv_cache.is_registered_owner(Child())
        assert not compat_kv_cache.is_registered_owner(object())


class TestUpstreamProbe:
    """The one-shot upstream hook behavior probe."""

    def test_probe_returns_boolean(self):
        """The probe result is a boolean (True on vLLM 0.24+)."""
        result = compat_kv_cache.upstream_binds_kv_cache_owners()
        assert isinstance(result, bool)

    def test_probe_result_is_cached(self):
        """The lru_cache holds a single cached result per process."""
        compat_kv_cache.upstream_binds_kv_cache_owners()
        info = compat_kv_cache.upstream_binds_kv_cache_owners.cache_info()
        assert info.currsize == 1


class TestCollectAndPreflight:
    """Owner collection and pre-validation over forward contexts."""

    def test_collect_owners_gathers_registered_layers_with_caches(self):
        """Only registered owners present in the context are collected."""
        compat_kv_cache.register_kv_cache_owner(_Owner)
        owner = _Owner()
        cache = torch.zeros(2)
        owners = compat_kv_cache._collect_owners(
            {"model.layers.0.attn": owner}, {"model.layers.0.attn": cache}
        )
        assert len(owners) == 1
        name, layer, kv_cache = owners[0]
        assert name == "model.layers.0.attn"
        assert layer is owner
        assert kv_cache is cache

    def test_collect_owners_skips_unregistered_and_missing_layers(self):
        """Non-owner layers and absent context entries are not collected."""
        compat_kv_cache.register_kv_cache_owner(_Owner)
        cache = torch.zeros(2)
        assert compat_kv_cache._collect_owners(
            {"model.layers.1.attn": object()}, {"model.layers.1.attn": cache}
        ) == []
        assert compat_kv_cache._collect_owners({}, {"model.layers.2.attn": cache}) == []

    def test_preflight_calls_validate_when_present(self):
        """_preflight_owners invokes validate_kv_cache with the cache."""
        cache = torch.zeros(2)
        owner = _Owner()
        compat_kv_cache._preflight_owners([("m.0", owner, cache)])
        assert owner.validated is cache

    def test_preflight_skips_owners_without_validate(self):
        """An owner without validate_kv_cache passes preflight untouched."""

        class NoValidate:
            def bind_kv_cache(self, kv_cache):
                self.bound = kv_cache

        owner = NoValidate()
        compat_kv_cache._preflight_owners([("m.0", owner, torch.zeros(1))])
        assert not hasattr(owner, "bound")


class TestBindKvCache:
    """The end-to-end bind flow with owner hooks."""

    def test_bind_validates_and_binds_owner(self):
        """A registered owner is validated and receives the bound cache."""
        compat_kv_cache.register_kv_cache_owner(_Owner)
        owner = _Owner()
        cache = torch.zeros(2, 3)
        runner_caches = []
        compat_kv_cache.bind_kv_cache(
            {"model.layers.0.attn": cache},
            {"model.layers.0.attn": owner},
            runner_caches,
            1,
        )
        assert owner.validated is cache
        assert owner.bound is cache
        assert len(runner_caches) >= 1

    def test_preflight_failure_leaves_state_untouched(self):
        """A validation error aborts the bind before any mutation."""
        compat_kv_cache.register_kv_cache_owner(_BadValidateOwner)
        bad_owner = _BadValidateOwner()
        runner_caches = []
        with pytest.raises(ValueError, match="unsupported kv cache layout"):
            compat_kv_cache.bind_kv_cache(
                {"model.layers.1.attn": torch.zeros(1)},
                {"model.layers.1.attn": bad_owner},
                runner_caches,
                1,
            )
        assert bad_owner.bound is None
        assert runner_caches == []

    def test_bind_without_owners_still_binds_upstream(self):
        """With no registered owners the plain upstream bind still runs.

        The upstream helper requires a forward-context entry per layer (it
        calls layer.bind_kv_cache itself on vLLM 0.24+), so a plain hook layer
        is supplied; the assertion checks the runner cache wiring happened.
        """

        class PlainLayer:
            def __init__(self):
                self.kv_cache = None

            def bind_kv_cache(self, kv_cache):
                self.kv_cache = kv_cache

        layer = PlainLayer()
        cache = torch.zeros(2)
        runner_caches = []
        compat_kv_cache.bind_kv_cache(
            {"model.layers.3.attn": cache}, {"model.layers.3.attn": layer}, runner_caches, 1
        )
        assert runner_caches == [cache]
        assert layer.kv_cache is cache
