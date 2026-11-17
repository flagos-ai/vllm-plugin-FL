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

    # register_ops imports `.impl.patchs` at module load; the patchs package
    # imports vllm.model_executor, which is not available in this shell. Give
    # it an empty package so registration itself stays under test.
    patchs = ModuleType(package.__name__ + ".impl.patchs")
    monkeypatch.setitem(sys.modules, patchs.__name__, patchs)
    impl_pkg = ModuleType(package.__name__ + ".impl")
    impl_pkg.__path__ = [str(TXDA / "impl")]
    monkeypatch.setitem(sys.modules, impl_pkg.__name__, impl_pkg)

    backend = load(package.__name__ + ".txda", TXDA / "txda.py")
    register = load(package.__name__ + ".register_ops", TXDA / "register_ops.py")
    return backend, register


def test_backend_import_and_unavailable_registration(txda_modules, monkeypatch):
    backend, register = txda_modules
    monkeypatch.setitem(sys.modules, "torch_txda", None)
    impls = []
    register.register_builtins(SimpleNamespace(register_many=impls.extend))
    assert len(impls) == 3
    op_names = [impl.op_name for impl in impls]
    assert op_names == [
        "attention_backend",
        "invoke_fused_moe_triton_kernel",
        "rms_norm",
    ]
    assert all(impl.impl_id == "vendor.txda" for impl in impls)
    assert not any(impl.is_available() for impl in impls)


def test_available_registration_selects_flaggems_attention(txda_modules, monkeypatch):
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
    attn = impls[0]
    assert attn.is_available()
    # 0.32: the vendor backend routes attention through FlagGems
    # (flash_attn_varlen_func), validated on TX8110 with flaggems_vllm
    # 0.3.0.dev1 + triton 3.7.1. The 0.28 SDPA fallback is gone.
    assert attn.fn(use_mla=False, use_sparse=False).endswith("AttentionFLBackend")
    assert attn.fn(use_mla=True).endswith("MLAFLBackend")
    with pytest.raises(ValueError, match="sparse"):
        attn.fn(use_mla=False, use_sparse=True)


def test_dispatch_config_routes_txda_kernels():
    config = yaml.safe_load(
        (ROOT / "vllm_fl/dispatch/config/tsingmicro.yaml").read_text()
    )
    # 0.32: attention goes through the FlagGems TX81 flash kernel (vendor),
    # not the stock TritonAttentionBackend (flagos) — keep vendor first.
    assert config["op_backends"]["attention_backend"] == [
        "vendor",
        "flagos",
        "reference",
    ]
    # rms_norm falls back across all three; the vendor impl is pure-torch
    # (FlagGems Triton kernel exceeds the TXDA shared-memory limit).
    assert config["op_backends"]["rms_norm"] == ["flagos", "vendor", "reference"]
    assert config["op_backends"]["invoke_fused_moe_triton_kernel"] == [
        "flagos",
        "vendor",
        "reference",
    ]
    # The 0.28 flagos_whitelist (add/mul + MoE selectors) was dropped: the
    # dispatch backend registration now scopes the FlagGems ATen replacement
    # itself, so no whitelist is needed. Instead the txda config keeps the
    # memory-movement ops on TXDA-native kernels (FlagGems Triton copies are
    # far slower) via flagos_blacklist.
    assert "flagos_whitelist" not in config
    assert set(config.get("flagos_blacklist", [])) == {
        "copy",
        "copy_",
        "cat",
        "stack",
        "hstack",
        "index_select",
        "index_put",
        "index_put_",
        "index_add",
        "index_add_",
        "as_strided_copy",
        "select_scatter",
        "repeat",
        "tile",
        "flip",
        "embedding",
        "slice_backward",
        "diagonal_copy",
    }
