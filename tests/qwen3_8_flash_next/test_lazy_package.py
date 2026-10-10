"""Configuration imports and compatibility exports must remain lazy."""

import importlib
import subprocess
import sys
from types import SimpleNamespace


def test_package_import_does_not_load_gpu_model_or_register_ops():
    code = """
import sys
import vllm_fl.models.qwen3_8_flash_next as package
for suffix in ('.gpu.model', '.gpu.qsa', '.gpu.ops.hyperconnection'):
    assert package.__name__ + suffix not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_exports_resolve_to_current_classes(monkeypatch):
    package = importlib.import_module("vllm_fl.models.qwen3_8_flash_next")
    causal, conditional, mtp = object(), object(), object()
    monkeypatch.setitem(
        sys.modules,
        package.__name__ + ".gpu.model",
        SimpleNamespace(
            Qwen3_8FlashNextForCausalLM=causal,
            Qwen3_8FlashNextForConditionalGeneration=conditional,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        package.__name__ + ".gpu.mtp",
        SimpleNamespace(Qwen3_8FlashNextMTP=mtp),
    )
    assert package.Qwen4ExpForCausalLM is package.Qwen3_8FlashNextForCausalLM is causal
    assert (
        package.Qwen4ExpForConditionalGeneration
        is package.Qwen3_8FlashNextForConditionalGeneration
        is conditional
    )
    assert package.Qwen4ExpMTP is package.Qwen3_8FlashNextMTP is mtp


def test_qsa_quantization_policy_imports_without_model():
    policy = importlib.import_module(
        "vllm_fl.models.qwen3_8_flash_next.gpu.quantization"
    )
    assert (
        policy.without_modelopt_fp4(SimpleNamespace(get_name=lambda: "modelopt_fp4"))
        is None
    )
    other = SimpleNamespace(get_name=lambda: "compressed-tensors")
    assert policy.without_modelopt_fp4(other) is other
    assert policy.without_modelopt_fp4(None) is None
