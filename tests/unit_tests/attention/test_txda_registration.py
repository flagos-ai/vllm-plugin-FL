# Copyright (c) 2026 BAAI. All rights reserved.

"""Exercise real TXDA registration without loading accelerator libraries."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
TXDA = ROOT / "vllm_fl/dispatch/backends/vendor/txda"


@pytest.fixture
def txda_modules(monkeypatch):
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    # Package shells avoid activating the installed vLLM/FlagGems environment.
    package = ModuleType("vllm_fl.dispatch.backends.vendor.txda")
    package.__path__ = [str(TXDA)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    load("vllm_fl.dispatch.backends.base", ROOT / "vllm_fl/dispatch/backends/base.py")
    load("vllm_fl.dispatch.types", ROOT / "vllm_fl/dispatch/types.py")
    backend = load(package.__name__ + ".txda", TXDA / "txda.py")
    register = load(package.__name__ + ".register_ops", TXDA / "register_ops.py")
    return backend, register


def test_backend_import_and_unavailable_registration(txda_modules, monkeypatch):
    backend, register = txda_modules
    monkeypatch.setitem(sys.modules, "torch_txda", None)
    impls = []
    register.register_builtins(SimpleNamespace(register_many=impls.extend))
    assert len(impls) == 1
    assert impls[0].op_name == "attention_backend"
    assert impls[0].impl_id == "vendor.txda"
    assert not impls[0].is_available()


def test_available_registration_uses_v028_selector(txda_modules, monkeypatch):
    backend, register = txda_modules
    monkeypatch.setitem(sys.modules, "torch_txda", ModuleType("torch_txda"))
    monkeypatch.setattr(
        torch,
        "txda",
        SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        raising=False,
    )
    impls = []
    register.register_builtins(SimpleNamespace(register_many=impls.extend))
    assert impls[0].is_available()
    assert (
        impls[0]
        .fn(use_mla=False, use_sparse=False)
        .endswith("TxdaSDPAAttentionBackend")
    )
    for flags in [dict(use_mla=True), dict(use_sparse=True)]:
        with pytest.raises(NotImplementedError, match="MLA or sparse"):
            impls[0].fn(**flags)


def test_dispatch_config_cannot_fall_back_to_known_bad_kernels():
    config = yaml.safe_load(
        (ROOT / "vllm_fl/dispatch/config/tsingmicro.yaml").read_text()
    )
    assert config["op_backends"]["attention_backend"] == ["vendor"]
    assert config["op_backends"]["rms_norm"] == ["reference"]
    assert config["op_backends"]["rotary_embedding"] == ["reference"]
    assert config["op_backends"]["silu_and_mul"] == ["reference"]
    allowed = set(config["flagos_whitelist"])
    assert not {"copy", "copy_", "to_copy", "linear", "mm", "mm_out"} & allowed
    # A bare add/mul list lets Dense run but leaves MoE routing unregistered.
    assert {
        "topk_softmax", "moe_align_block_size",
        "invoke_fused_moe_triton_kernel", "moe_sum",
    } <= allowed
    assert not config.get("flagos_blacklist")


def test_seeded_sampling_keeps_native_exponential_fallback():
    # TXDA's native CPU fallback accepts CPU generators. FlagGems attempts to
    # unpack their MT19937 state as two Philox words and crashes during warmup.
    config = yaml.safe_load(
        (ROOT / "vllm_fl/dispatch/config/tsingmicro.yaml").read_text()
    )
    assert "exponential_" not in config["flagos_whitelist"]
