# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin dispatch backend class.

``KunlunxinBackend`` (``vllm_fl.dispatch.backends.vendor.kunlunxin.kunlunxin``)
gates every operator behind a lazily imported impl module, so all tests here
drive the backend with recorder stubs instead of vendor hardware:

* availability probing (``torch_xmlir`` plus the masqueraded CUDA device) and
  its memoization on the ``_available`` class attribute;
* the attention-backend class-path contract, including the MLA rejections;
* per-operator delegation: each public op method must forward its arguments
  verbatim to the matching function in ``...kunlunxin.impl.*`` and return
  whatever that function returns.

Order independence: every ``sys.modules`` entry a test touches (the
``torch_xmlir`` stand-in, the ``xtorch_ops`` stub, the re-imported impl tree)
is restored exactly on exit, and the memoized ``_available`` class attribute
is reset around each test.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.supplementary]

from vllm_fl.dispatch.backends.vendor.kunlunxin.kunlunxin import KunlunxinBackend

ATTENTION_BACKEND_PATH = (
    "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention"
    ".KunlunxinAttentionBackend"
)

_IMPL_PACKAGE = "vllm_fl.dispatch.backends.vendor.kunlunxin.impl"
_IMPL_TREE = [
    _IMPL_PACKAGE,
    f"{_IMPL_PACKAGE}.activation",
    f"{_IMPL_PACKAGE}.normalization",
    f"{_IMPL_PACKAGE}.rotary",
    f"{_IMPL_PACKAGE}.fused_gdn_gating",
    f"{_IMPL_PACKAGE}.causal_conv1d",
    f"{_IMPL_PACKAGE}.fla",
    f"{_IMPL_PACKAGE}.fla.chunk",
    f"{_IMPL_PACKAGE}.fla.fused_recurrent",
    f"{_IMPL_PACKAGE}.fused_moe",
    f"{_IMPL_PACKAGE}.fused_moe.fused_moe",
    f"{_IMPL_PACKAGE}.fused_moe.experts_selector",
]


@contextlib.contextmanager
def _torch_xmlir_absent():
    """Make ``import torch_xmlir`` fail without re-executing a real module.

    A ``None`` entry in ``sys.modules`` halts the import with ImportError, so
    this works both on hosts without the package (entry simply absent) and on
    Kunlunxin hosts where the real module is already imported.
    """
    saved = sys.modules.get("torch_xmlir")
    sys.modules["torch_xmlir"] = None
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("torch_xmlir", None)
        else:
            sys.modules["torch_xmlir"] = saved


@contextlib.contextmanager
def _torch_xmlir_present():
    """Expose a dummy ``torch_xmlir`` so the import inside is_available passes."""
    saved = sys.modules.get("torch_xmlir")
    sys.modules["torch_xmlir"] = ModuleType("torch_xmlir")
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("torch_xmlir", None)
        else:
            sys.modules["torch_xmlir"] = saved


@contextlib.contextmanager
def _xtorch_ops_stub():
    """Install an ``xtorch_ops`` stub and re-import the kunlunxin impl tree.

    Impl modules that import ``xtorch_ops`` at module scope can only load
    with the stub in place, so the whole ``...kunlunxin.impl`` tree is popped
    first and re-imported lazily by whoever needs it.  Every touched
    ``sys.modules`` entry is restored to its exact previous state on exit, so
    no stub ever leaks into another test.
    """
    stub = ModuleType("xtorch_ops")
    names = [*_IMPL_TREE, "xtorch_ops"]
    saved = {name: sys.modules.get(name) for name in names}
    for name in _IMPL_TREE:
        sys.modules.pop(name, None)
    sys.modules["xtorch_ops"] = stub
    try:
        yield stub
    finally:
        for name, entry in saved.items():
            if entry is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = entry


@pytest.fixture(autouse=True)
def _fresh_availability_cache():
    """Reset the memoized availability flag around every test in this module."""
    saved = KunlunxinBackend._available
    KunlunxinBackend._available = None
    yield
    KunlunxinBackend._available = saved


class TestAvailability:
    """is_available() probes torch_xmlir plus the masqueraded CUDA device."""

    @pytest.mark.platforms("kunlunxin")
    def test_false_when_torch_xmlir_missing(self):
        with _torch_xmlir_absent():
            assert KunlunxinBackend().is_available() is False
            assert KunlunxinBackend._available is False

    @pytest.mark.platforms("kunlunxin")
    def test_true_when_torch_xmlir_and_cuda_device_present(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        with _torch_xmlir_present():
            assert KunlunxinBackend().is_available() is True
            assert KunlunxinBackend._available is True

    @pytest.mark.platforms("kunlunxin")
    def test_false_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        with _torch_xmlir_present():
            assert KunlunxinBackend().is_available() is False

    @pytest.mark.platforms("kunlunxin")
    def test_false_when_device_count_is_zero(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        with _torch_xmlir_present():
            assert KunlunxinBackend().is_available() is False

    @pytest.mark.platforms("kunlunxin")
    def test_false_when_cuda_probe_raises_attribute_error(self, monkeypatch):
        # The reduced Kunlunxin torch build is the reason this backend also
        # catches AttributeError from the torch.cuda surface.
        monkeypatch.delattr(torch.cuda, "is_available")
        with _torch_xmlir_present():
            assert KunlunxinBackend().is_available() is False

    @pytest.mark.platforms("kunlunxin")
    def test_probe_runs_once_across_repeated_calls(self, monkeypatch):
        probes = []

        def counting_is_available():
            probes.append(1)
            return True

        monkeypatch.setattr(torch.cuda, "is_available", counting_is_available)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        with _torch_xmlir_present():
            backend = KunlunxinBackend()
            assert backend.is_available() is True
            assert backend.is_available() is True
            assert backend.is_available() is True
        assert len(probes) == 1

    @pytest.mark.platforms("kunlunxin")
    def test_negative_result_is_memoized(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        backend = KunlunxinBackend()
        with _torch_xmlir_absent():
            assert backend.is_available() is False
        # Hardware "appears" afterwards; the cached False must still win.
        with _torch_xmlir_present():
            assert backend.is_available() is False


class TestBackendIdentity:
    @pytest.mark.platforms("kunlunxin")
    def test_name_and_vendor_properties(self):
        backend = KunlunxinBackend()
        assert backend.name == "kunlunxin"
        assert backend.vendor == "kunlunxin"


class TestAttentionBackend:
    @pytest.mark.platforms("kunlunxin")
    def test_returns_impl_class_path_by_default(self):
        assert KunlunxinBackend().attention_backend() == ATTENTION_BACKEND_PATH

    @pytest.mark.platforms("kunlunxin")
    def test_use_sparse_alone_is_accepted(self):
        assert KunlunxinBackend().attention_backend(use_sparse=True) == (
            ATTENTION_BACKEND_PATH
        )

    @pytest.mark.platforms("kunlunxin")
    def test_rejects_mla(self):
        with pytest.raises(
            NotImplementedError,
            match="MLA attention is not implemented for Kunlunxin yet",
        ):
            KunlunxinBackend().attention_backend(use_mla=True)

    @pytest.mark.platforms("kunlunxin")
    def test_rejects_mla_with_sparse(self):
        with pytest.raises(
            NotImplementedError,
            match="MLA with sparse attention is not implemented for Kunlunxin yet",
        ):
            KunlunxinBackend().attention_backend(use_mla=True, use_sparse=True)


# Sentinels forwarded through the delegation cases below.  Tuple comparison
# in the assertions relies on object identity, so the same objects must be
# reused by call and expectation.
_OBJ = object()
_X = torch.randn(2, 6)
_R = torch.randn(2, 6)
_Q = torch.randn(3, 2, 8)
_K = torch.randn(3, 1, 8)
_COS = torch.randn(5, 4)
_SIN = torch.randn(5, 4)
_POS = torch.tensor([0, 1, 3])
_W = torch.randn(4, 2)
_TW = torch.randn(2, 3)
_TI = torch.zeros(2, 3, dtype=torch.int32)
_GO = torch.randn(2, 4)
_BIAS = torch.randn(4)

_DELEGATION_CASES = [
    pytest.param(
        "silu_and_mul",
        f"{_IMPL_PACKAGE}.activation",
        "silu_and_mul_kunlunxin",
        (_OBJ, _X),
        {},
        {},
        id="silu_and_mul",
    ),
    pytest.param(
        "rms_norm",
        f"{_IMPL_PACKAGE}.normalization",
        "rms_norm_kunlunxin",
        (_OBJ, _X, _R),
        {},
        {},
        id="rms_norm-with-residual",
    ),
    pytest.param(
        "rotary_embedding",
        f"{_IMPL_PACKAGE}.rotary",
        "rotary_embedding_kunlunxin",
        (_OBJ, _Q, _K, _COS, _SIN, _POS),
        {},
        {"rotary_interleaved": False, "inplace": True},
        id="rotary_embedding-defaults",
    ),
    pytest.param(
        "rotary_embedding",
        f"{_IMPL_PACKAGE}.rotary",
        "rotary_embedding_kunlunxin",
        (_OBJ, _Q, _K, _COS, _SIN, _POS),
        {"rotary_interleaved": True, "inplace": False},
        {"rotary_interleaved": True, "inplace": False},
        id="rotary_embedding-custom",
    ),
    pytest.param(
        "chunk_gated_delta_rule_fwd",
        f"{_IMPL_PACKAGE}.fla.chunk",
        "chunk_gated_delta_rule_fwd",
        (1, "two"),
        {"scale": 0.5},
        {"scale": 0.5},
        id="chunk_gated_delta_rule_fwd",
    ),
    pytest.param(
        "fused_recurrent_gated_delta_rule_fwd",
        f"{_IMPL_PACKAGE}.fla.fused_recurrent",
        "fused_recurrent_gated_delta_rule_fwd",
        (_X,),
        {"cu_seqlens": None},
        {"cu_seqlens": None},
        id="fused_recurrent_gated_delta_rule_fwd",
    ),
    pytest.param(
        "causal_conv1d_fn",
        f"{_IMPL_PACKAGE}.causal_conv1d",
        "causal_conv1d_fn_kunlunxin",
        (_X, _W, None, _X, _POS),
        {"activation": "silu"},
        {"activation": "silu"},
        id="causal_conv1d_fn",
    ),
    pytest.param(
        "causal_conv1d_update",
        f"{_IMPL_PACKAGE}.causal_conv1d",
        "causal_conv1d_update_kunlunxin",
        (_X, _W),
        {"activation": None},
        {"activation": None},
        id="causal_conv1d_update",
    ),
    pytest.param(
        "fused_gdn_gating",
        f"{_IMPL_PACKAGE}.fused_gdn_gating",
        "fused_gdn_gating_kunlunxin",
        (_X, _X, _X),
        {"beta": 2.0, "threshold": 15.0},
        {"beta": 2.0, "threshold": 15.0},
        id="fused_gdn_gating",
    ),
    pytest.param(
        "fused_experts",
        f"{_IMPL_PACKAGE}.fused_moe.fused_moe",
        "fused_experts_impl",
        (_X, _W, _W, _TW, _TI),
        {"activation": "relu", "inplace": True},
        {"activation": "relu", "inplace": True},
        id="fused_experts",
    ),
    pytest.param(
        "topk_softmax",
        f"{_IMPL_PACKAGE}.fused_moe.experts_selector",
        "vllm_topk_softmax",
        (_TW, _TI, _TI, _GO, True),
        {},
        {},
        id="topk_softmax",
    ),
    pytest.param(
        "grouped_topk",
        f"{_IMPL_PACKAGE}.fused_moe.experts_selector",
        "grouped_topk",
        (_GO, 2, 1, 3, True, 2.5, _BIAS, 1),
        {},
        {},
        id="grouped_topk",
    ),
]


class TestOpDelegation:
    """Each op method forwards verbatim to its impl module's function."""

    @pytest.mark.platforms("kunlunxin")
    @pytest.mark.parametrize(
        ("method", "impl_module", "impl_function", "args", "call_kwargs", "forwarded_kwargs"),
        _DELEGATION_CASES,
    )
    def test_op_method_delegates_to_impl_function(
        self,
        monkeypatch,
        method,
        impl_module,
        impl_function,
        args,
        call_kwargs,
        forwarded_kwargs,
    ):
        sentinel = object()
        calls = []

        def recorder(*call_args, **call_kwargs):
            calls.append((call_args, call_kwargs))
            return sentinel

        with _xtorch_ops_stub():
            impl = importlib.import_module(impl_module)
            monkeypatch.setattr(impl, impl_function, recorder)
            result = getattr(KunlunxinBackend(), method)(*args, **call_kwargs)

        assert result is sentinel
        assert calls == [(args, forwarded_kwargs)]
