# Copyright (c) 2026 BAAI. All rights reserved.
"""Per-operator MetaX availability without optional MCOPLIB installations."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_fl/dispatch/backends/vendor/metax"


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registration(monkeypatch):
    package = ModuleType("metax_native_registration_test")
    package.__path__ = [str(SOURCE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for suffix in ("patches", "metax", "impl", "impl.attention", "impl.attention.ops"):
        module = ModuleType(f"{package.__name__}.{suffix}")
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module.__name__, module)

    class Backend:
        available = True

        def is_available(self):
            return self.available

        def silu_and_mul(self, *args):
            return "native silu"

        rms_norm = rotary_embedding = topk_softmax = invoke_fused_moe_triton_kernel = (
            silu_and_mul
        )

        def moe_align_block_size(self, *args):
            return args

        def attention_backend(self, **kwargs):
            return "MacaFlashAttentionBackend"

    sys.modules[f"{package.__name__}.metax"].MacaBackend = Backend
    _load(monkeypatch, "vllm_fl.dispatch.types", ROOT / "vllm_fl/dispatch/types.py")
    _load(
        monkeypatch,
        f"{package.__name__}.impl.attention.ops._kernel_utils",
        SOURCE / "impl/attention/ops/_kernel_utils.py",
    )
    module = _load(
        monkeypatch, f"{package.__name__}.register_ops", SOURCE / "register_ops.py"
    )
    impls = []
    module.register_builtins(SimpleNamespace(register_many=impls.extend))
    return {impl.op_name: impl for impl in impls}, Backend


def _no_mcoplib(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcoplib", None)
    monkeypatch.setitem(sys.modules, "mcoplib.triton_fused_moe", None)


def test_schema_only_image_retains_attention_without_advertising_native_ops(
    registration, monkeypatch
):
    impls, _ = registration
    monkeypatch.setattr(
        torch._C, "_dispatch_has_kernel_for_dispatch_key", lambda name, key: False
    )
    _no_mcoplib(monkeypatch)
    assert impls["attention_backend"].is_available()
    assert impls["attention_backend"].fn() == "MacaFlashAttentionBackend"
    align = impls["moe_align_block_size"]
    assert align.impl_id == "vendor.metax"
    assert align.is_available()
    assert align.fn("ids", 4, 7, None, True, True) == ("ids", 4, 7, None, True, True)
    assert not any(
        impl.is_available()
        for name, impl in impls.items()
        if name not in {"attention_backend", "moe_align_block_size"}
    )


@pytest.mark.parametrize(
    "key",
    [
        "CUDA",
        "CompositeImplicitAutograd",
        "CompositeExplicitAutograd",
        "CompositeExplicitAutogradNonFunctional",
    ],
)
def test_each_operator_requires_its_own_executable_kernels(
    registration, monkeypatch, key
):
    impls, _ = registration
    kernels = {
        "_C::silu_and_mul",
        "_C::rms_norm",
        "_C::rotary_embedding",
        "_moe_C::topk_softmax",
    }
    monkeypatch.setattr(
        torch._C,
        "_dispatch_has_kernel_for_dispatch_key",
        lambda name, dispatch: dispatch == key and name in kernels,
    )
    _no_mcoplib(monkeypatch)
    assert impls["silu_and_mul"].is_available()
    assert impls["rotary_embedding"].is_available()
    assert impls["topk_softmax"].is_available()
    assert not impls["rms_norm"].is_available()  # Residual branch is still absent.
    kernels.add("_C::fused_add_rms_norm")
    assert impls["rms_norm"].is_available()
    kernels.remove("_C::silu_and_mul")
    assert not impls["silu_and_mul"].is_available()
    assert impls["attention_backend"].is_available()


def test_precompiled_moe_requires_callable_export_and_available_device(
    registration, monkeypatch
):
    impls, backend = registration
    package = ModuleType("mcoplib")
    package.__path__ = []
    module = ModuleType("mcoplib.triton_fused_moe")
    monkeypatch.setitem(sys.modules, "mcoplib", package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    moe = impls["invoke_fused_moe_triton_kernel"]
    assert not moe.is_available()
    module.fused_moe_triton_kernel = object()
    assert not moe.is_available()
    module.fused_moe_triton_kernel = lambda *args: None
    assert moe.is_available()
    backend.available = False
    assert not any(impl.is_available() for impl in impls.values())


def test_meta_kernels_do_not_advertise_native_operators(registration, monkeypatch):
    impls, _ = registration
    monkeypatch.setattr(
        torch._C,
        "_dispatch_has_kernel_for_dispatch_key",
        lambda name, key: key == "Meta",
    )
    _no_mcoplib(monkeypatch)
    assert impls["attention_backend"].is_available()
    assert not impls["silu_and_mul"].is_available()
    assert not impls["rms_norm"].is_available()
    assert not impls["rotary_embedding"].is_available()
    assert not impls["topk_softmax"].is_available()
