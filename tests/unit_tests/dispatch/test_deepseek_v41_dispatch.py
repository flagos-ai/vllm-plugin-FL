# Copyright (c) 2026 BAAI. All rights reserved.
"""Public API availability and transparent dispatch; no numerical oracle reuse."""

import inspect
import sys
from types import ModuleType

import pytest

from vllm_fl.dispatch.backends.flaggems.impl import deepseek_v41 as bridge


@pytest.fixture
def public(monkeypatch):
    root = ModuleType("flaggems_vllm")
    root.__path__ = []
    calls = []
    result = object()

    class Callable:
        def __init__(self, name, parameters):
            self.name = name
            self.__signature__ = inspect.Signature(
                [
                    inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for n in sorted(parameters)
                ]
            )

        def __call__(self, *args, **kwargs):
            calls.append((self.name, args, kwargs))
            return (result, None) if self.name == "flash_mla_with_kvcache" else result

    for name, parameters in bridge.V41_PUBLIC_OPERATORS.items():
        target = "flash_mla_with_kvcache" if name == "v41_sparse_attention" else name
        setattr(root, target, Callable(target, parameters))
    metadata = ModuleType("flaggems_vllm.ops.flash_mla_with_kvcache")
    metadata.get_mla_metadata = lambda: (object(), None)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", root)
    monkeypatch.setitem(
        sys.modules, "flaggems_vllm.ops.flash_mla_with_kvcache", metadata
    )
    return root, calls, result


@pytest.mark.parametrize("name", list(bridge.V41_PUBLIC_OPERATORS))
def test_exact_public_capability_before_execution(public, name):
    root, calls, result = public
    assert bridge.is_v41_public_available(name)
    target = "flash_mla_with_kvcache" if name == "v41_sparse_attention" else name
    setattr(root, target, lambda: None)
    assert not bridge.is_v41_public_available(name)
    setattr(root, target, None)
    assert not bridge.is_v41_public_available(name)
    assert calls == []


@pytest.mark.parametrize(
    "name", [x for x in bridge.V41_PUBLIC_OPERATORS if x != "v41_sparse_attention"]
)
def test_forwarding_preserves_buffers_and_execution_errors(public, name):
    root, calls, result = public
    a, b = object(), object()
    assert bridge.get_public_forwarder(name)(a, b, marker=a) is result
    assert calls == [(name, (a, b), {"marker": a})]
    error = RuntimeError("real execution failed")

    def fail(*args, **kwargs):
        raise error

    setattr(root, name, fail)
    with pytest.raises(RuntimeError) as raised:
        bridge.get_public_forwarder(name)(a, b)
    assert raised.value is error and len(calls) == 1


def test_sparse_attention_public_metadata_and_exact_contract(public):
    root, calls, result = public
    values = [object() for _ in range(6)]
    assert (
        bridge.get_public_forwarder("v41_sparse_attention")(
            *values[:4], extra_cache=values[4], extra_indices=values[5]
        )
        is result
    )
    name, args, kwargs = calls[0]
    assert name == "flash_mla_with_kvcache" and args[:2] == tuple(values[:2])
    assert args[2:5] == (None, None, 512)
    assert kwargs["indices"] is values[2] and kwargs["attn_sink"] is values[3]
    assert (
        kwargs["extra_k_cache"] is values[4]
        and kwargs["extra_indices_in_kvcache"] is values[5]
    )
    assert kwargs["is_fp8_kvcache"] and not kwargs["causal"]


def test_v41_requires_q_norm_control_before_execution(public):
    root, calls, _ = public
    name = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    implementation = getattr(root, name)
    signature = implementation.__signature__
    implementation.__signature__ = signature.replace(
        parameters=[
            p for p in signature.parameters.values() if p.name != "apply_q_norm"
        ]
    )
    assert not bridge.is_v41_public_available(name)
    assert calls == []


def test_v41_q_norm_control_is_forwarded_unchanged(public):
    _, calls, result = public
    name = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    q, kv, cache = object(), object(), object()
    assert bridge.get_public_forwarder(name)(q, kv, cache, apply_q_norm=False) is result
    assert calls == [(name, (q, kv, cache), {"apply_q_norm": False})]


def test_registered_compressor_preserves_all_ten_arguments_and_result(monkeypatch):
    import flaggems_vllm

    from vllm_fl.dispatch.backends.flaggems import register_ops

    registered = []

    class Registry:
        def register_many(self, values):
            registered.extend(values)

    monkeypatch.setattr(
        register_ops,
        "use_flaggems_op",
        lambda name: name == "deepseek_v41_compress_norm",
    )
    register_ops.register_builtins(Registry())
    assert len(registered) == 1
    impl = registered[0]
    assert (
        impl.op_name == "deepseek_v41_compress_norm"
        and impl.impl_id == "default.flagos"
    )
    args = tuple(object() for _ in range(10))
    expected = object()

    def public_compressor(*received):
        assert received == args
        return expected

    monkeypatch.setattr(flaggems_vllm, impl.op_name, public_compressor)
    assert impl.fn(*args) is expected
