"""Keep vendor codegen behavior while preserving integer tensor ABIs."""

from types import SimpleNamespace

import pytest

from vllm_fl.dispatch.backends.vendor.gcu.impl import inductor


def vendor_compile(graph_lowering_self):
    wrapper_code = graph_lowering_self
    wrapper_code.value = wrapper_code.value.replace("int64", "int32")
    wrapper_code.value = wrapper_code.value.replace("i64", "i32")
    wrapper_code.value = wrapper_code.value.replace("float64", "float32")
    wrapper_code.value = wrapper_code.value.replace("f64", "f32")
    wrapper_code.value = wrapper_code.value.replace("9223372036854775808", str(2**31))
    wrapper_code.value = wrapper_code.value.replace(
        "9223372036854775807", str(2**31 - 1)
    )
    wrapper_code.vendor_loaded = True
    return wrapper_code


def test_preserves_vendor_actions_and_integer_storage_abi(monkeypatch):
    monkeypatch.delenv("FL_GCU_PRESERVE_DTYPE_CODEGEN", raising=False)
    helper = type("Helper", (), {"compile_to_module": staticmethod(vendor_compile)})
    provider = SimpleNamespace(GCUCompileHelper=helper)
    inductor._install_gcu_dtype_preserving_codegen(provider)
    adapted = helper.compile_to_module
    inductor._install_gcu_dtype_preserving_codegen(provider)
    assert helper.compile_to_module is adapted
    code = "*i64 torch.int64 float64 f64 -9223372036854775808 9223372036854775807"
    wrapper = SimpleNamespace(value=code)
    assert adapted(wrapper) is wrapper
    assert wrapper.vendor_loaded
    assert wrapper.value == code.replace("float64", "float32").replace("f64", "f32")
    assert adapted.__wrapped__ is vendor_compile


@pytest.mark.parametrize("reason", ["source", "abi", "opt_out"])
def test_unknown_vendor_or_disabled_adapter_is_not_replaced(monkeypatch, reason):
    warnings = []
    monkeypatch.setattr(
        inductor.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    original = vendor_compile
    if reason == "abi":
        original = lambda self, extra: None
    if reason == "source":
        monkeypatch.setattr(
            inductor.inspect,
            "getsource",
            lambda fn: (
                "def changed(graph_lowering_self):\n    return graph_lowering_self\n"
            ),
        )
    monkeypatch.setenv(
        "FL_GCU_PRESERVE_DTYPE_CODEGEN", "0" if reason == "opt_out" else "1"
    )
    helper = type("Helper", (), {"compile_to_module": staticmethod(original)})
    inductor._install_gcu_dtype_preserving_codegen(
        SimpleNamespace(GCUCompileHelper=helper)
    )
    assert helper.compile_to_module is original
    if reason != "opt_out":
        assert any(
            "Skipping GCU dtype codegen adapter" in message for message in warnings
        )
    else:
        assert not warnings
