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

"""HY4 runtime resolution and prefill-backend selection contracts.

``_flaggems_operator_allowed`` gates FlagGems dispatch on the operator
policy (whitelist/blacklist env vars are a sanctioned boundary), the
prefill selection honors explicit choices and unavailable implementations,
and ``validate_hy4_runtime`` rejects unusable KV-cache combinations before
weights are allocated. ``prepare_hy4_runtime``/``install_hy4_flaggems_
fallback`` intentionally stay untested: they mutate process-global vLLM
symbols, which the module docstring itself forbids outside HY4 worker
construction.
"""

import pytest

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.hy4.runtime import (
    HY4RuntimePlan,
    _flaggems_operator_allowed,
    _prefill_is_available,
    select_hy4_prefill_backend,
    validate_hy4_runtime,
)

pytestmark = [pytest.mark.gpu]

# Ops with a DEFAULT (FlagGems) registration in the dispatch registry; the
# A100 image ships real flag_gems so these resolve on the test host.
DEFAULT_KIND_OP = "silu_and_mul"
UNREGISTERED_OP = "definitely_not_a_registered_operator"


@pytest.fixture
def clean_dispatch_env(monkeypatch):
    """Isolate the policy env vars use_flaggems_op reads on every call.

    The platform config blacklist (priority 3) is host state; tests that
    care about the default path set an explicit harmless blacklist so the
    result never depends on the runner image. Other hy4 suites register
    implementations on the default manager, so rebuild the manager and
    policy around every test that relies on the built-in registry.
    """
    from vllm_fl.dispatch.manager import reset_default_manager
    from vllm_fl.dispatch.policy import reset_global_policy

    for var in (
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
        "VLLM_FL_PREFER",
        "VLLM_FL_PREFER_ENABLED",
        "USE_FLAGGEMS",
        "VLLM_FL_OP_CONFIG",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_default_manager()
    reset_global_policy()
    yield
    reset_default_manager()
    reset_global_policy()


class TestFlaggemsOperatorAllowed:
    """Policy gate over the dispatch registry."""

    def test_blacklisted_operator_is_denied(self, clean_dispatch_env, monkeypatch):
        """Blacklisted operator is denied."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", DEFAULT_KIND_OP)
        assert _flaggems_operator_allowed(DEFAULT_KIND_OP) is False

    def test_registered_default_implementation_is_allowed(
        self, clean_dispatch_env, monkeypatch
    ):
        # silu_and_mul has FlagGems (DEFAULT-kind) implementations registered
        # and is reachable on the CUDA image; whitelisting it makes the
        # policy decision independent of the host's platform config.
        """Registered default implementation is allowed."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", DEFAULT_KIND_OP)
        assert _flaggems_operator_allowed(DEFAULT_KIND_OP) is True

    def test_unregistered_operator_reports_false_not_an_error(self, clean_dispatch_env):
        # resolve_candidates raises RuntimeError for an unknown op; the gate
        # converts that into a plain "not allowed" answer.
        """Unregistered operator reports false not an error."""
        assert _flaggems_operator_allowed(UNREGISTERED_OP) is False

    def test_whitelist_restricted_to_another_op_denies_this_one(
        self, clean_dispatch_env, monkeypatch
    ):
        """Whitelist restricted to another operator denies this one."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "gelu_and_mul")
        assert _flaggems_operator_allowed(DEFAULT_KIND_OP) is False


class _FakeBackend:
    """Minimal backend class exposing exactly the selector's read surface."""

    name = "FAKE_TEST_BACKEND"

    @classmethod
    def is_available(cls):
        return _FakeBackend.available

    @classmethod
    def get_name(cls):
        return _FakeBackend.name


def _vllm_config(explicit_backend):
    """A config namespace with only the attributes the selector reads."""
    from types import SimpleNamespace

    attention_config = SimpleNamespace(mla_prefill_backend=explicit_backend)
    return SimpleNamespace(attention_config=attention_config)


class TestSelectHy4PrefillBackend:
    """Automatic and explicit MLA prefill backend selection."""

    def test_available_native_backend_is_returned(self, clean_dispatch_env):
        """Available native backend is returned."""
        _FakeBackend.available = True
        config = _vllm_config(explicit_backend=None)
        selected = select_hy4_prefill_backend(config, lambda cfg: _FakeBackend)
        assert selected is _FakeBackend

    def test_unavailable_backend_automatic_falls_back_to_none(self, clean_dispatch_env):
        # No explicit choice: automatic absence means the FlagGems portable
        # path takes over (selector returns None).
        """Unavailable backend automatic falls back to none."""
        _FakeBackend.available = False
        config = _vllm_config(explicit_backend=None)
        assert select_hy4_prefill_backend(config, lambda cfg: _FakeBackend) is None

    def test_unavailable_explicit_backend_raises(self, clean_dispatch_env):
        """Unavailable explicit backend raises."""
        _FakeBackend.available = False
        config = _vllm_config(explicit_backend=_FakeBackend)
        with pytest.raises(RuntimeError, match="no callable implementation"):
            select_hy4_prefill_backend(config, lambda cfg: _FakeBackend)

    def test_import_error_is_swallowed_only_when_automatic(self, clean_dispatch_env):
        """Import error is swallowed only when automatic."""

        def broken_selector(cfg):
            raise ImportError("backend module missing")

        # Automatic: the selector reports "unavailable" via None.
        assert select_hy4_prefill_backend(_vllm_config(None), broken_selector) is None
        # Explicit: the user asked for it, so the failure is loud.
        with pytest.raises(ImportError):
            select_hy4_prefill_backend(_vllm_config(_FakeBackend), broken_selector)

    def test_exhausted_automatic_candidates_value_error_becomes_none(
        self, clean_dispatch_env
    ):
        """Exhausted automatic candidates value error becomes none."""

        def exhausted(cfg):
            raise ValueError("No valid MLA prefill backend found for the given config")

        assert select_hy4_prefill_backend(_vllm_config(None), exhausted) is None

    def test_other_value_errors_stay_visible(self, clean_dispatch_env):
        """Other value errors stay visible."""

        def broken(cfg):
            raise ValueError("some unrelated configuration problem")

        with pytest.raises(ValueError, match="unrelated configuration"):
            select_hy4_prefill_backend(_vllm_config(None), broken)


class TestPrefillIsAvailable:
    """Availability probes used by the selection above."""

    def test_flash_attn_backend_without_varlen_callable_is_unavailable(
        self, monkeypatch
    ):
        # FLASH_ATTN is validated further than is_available(): the captured
        # ``prefill.flash_attn`` module must expose a callable
        # flash_attn_varlen_func. The source resolves it as an attribute of
        # the parent package, so patch that attribute rather than sys.modules.
        """Flash attn backend without variable-length callable is unavailable."""
        import types

        from vllm.v1.attention.backends.mla import prefill as prefill_pkg

        class FlashAttnLike:
            @classmethod
            def is_available(cls):
                return True

            @classmethod
            def get_name(cls):
                return "FLASH_ATTN"

        monkeypatch.setattr(
            prefill_pkg,
            "flash_attn",
            types.SimpleNamespace(flash_attn_varlen_func=None),
            raising=False,
        )
        assert _prefill_is_available(FlashAttnLike) is False

    def test_flash_attn_backend_with_callable_varlen_passes(self, monkeypatch):
        # The same FLASH_ATTN-named backend with a callable varlen entry
        # point is accepted (line 111's True return). Patch the attribute the
        # source imports so the test is order-independent.
        """Flash attn backend with callable variable-length passes."""
        import types

        from vllm.v1.attention.backends.mla import prefill as prefill_pkg

        class FlashAttnLike:
            @classmethod
            def is_available(cls):
                return True

            @classmethod
            def get_name(cls):
                return "FLASH_ATTN"

        monkeypatch.setattr(
            prefill_pkg,
            "flash_attn",
            types.SimpleNamespace(flash_attn_varlen_func=lambda *a, **k: None),
            raising=False,
        )
        assert _prefill_is_available(FlashAttnLike) is True

    def test_available_non_flash_attn_backend_passes(self):
        """Available non-Flash attn backend passes."""
        _FakeBackend.available = True
        assert _prefill_is_available(_FakeBackend) is True

    def test_unavailable_backend_reports_false(self):
        """Unavailable backend reports false."""
        _FakeBackend.available = False
        assert _prefill_is_available(_FakeBackend) is False


class TestValidateHy4Runtime:
    """Pre-allocation rejection of unusable runtime configurations."""

    def _minimal_vllm_config(self, cache_dtype="auto", model_dtype=None):
        from types import SimpleNamespace

        import torch

        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(
                index_topk=2048,
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
            ),
            dtype=model_dtype or torch.bfloat16,
            enforce_eager=True,
        )
        return SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype=cache_dtype),
            model_config=model_config,
            attention_config=SimpleNamespace(mla_prefill_backend=None),
        )

    def test_unsupported_kv_cache_dtype_is_rejected(self, device, clean_dispatch_env):
        """Unsupported KV cache dtype is rejected."""
        config = self._minimal_vllm_config(cache_dtype="fp8_e5m2")
        with pytest.raises(ValueError, match="Unsupported HY4 KV cache dtype"):
            validate_hy4_runtime(config)

    def test_portable_auto_kv_requires_bfloat16_model_dtype(
        self, device, clean_dispatch_env
    ):
        # With enforce_eager=True the graph-safe entry-point requirement is
        # bypassed; an fp32 model with a portable (FlagGems) indexer and
        # auto KV must be rejected before weight allocation.
        """Portable auto KV requires bfloat16 model dtype."""
        import torch

        config = self._minimal_vllm_config(
            cache_dtype="auto", model_dtype=torch.float32
        )
        with pytest.raises(ValueError, match="auto KV requires bfloat16 model dtype"):
            validate_hy4_runtime(config)


class TestRuntimePlanContract:
    """The immutable plan record handed to every HY4 layer."""

    def test_fields_round_trip_and_read_only(self):
        """Preserve runtime plan fields and reject assignments after construction."""
        import dataclasses
        from types import MappingProxyType

        plan = HY4RuntimePlan(
            provider="flaggems",
            query_quantizer=lambda x: x,
            operations=MappingProxyType({"silu_and_mul": lambda x: x}),
            prefill_backend=None,
            kv_cache_dtypes=("auto", "bfloat16"),
        )
        assert plan.provider == "flaggems"
        assert callable(plan.query_quantizer)
        assert set(plan.operations) == {"silu_and_mul"}
        assert plan.prefill_backend is None
        assert plan.kv_cache_dtypes == ("auto", "bfloat16")
        # The plan is shared across layer construction; validate_hy4_runtime
        # hands out a read-only operations view and the record itself is
        # frozen, so no layer can retarget another layer's operators.
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.provider = "native"
        with pytest.raises(TypeError):
            plan.operations["late"] = lambda x: x
