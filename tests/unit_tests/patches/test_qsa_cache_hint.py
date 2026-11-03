# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU contracts for the real upstream QSA JIT functions; no kernels launch."""

import ast
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import triton

from tests.utils.device_utils import get_backend

from vllm_fl.patches import qsa_cache_hint as patch_module

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA native QSA contract"
)


@pytest.fixture
def qsa_kernels(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    # Fresh JIT objects isolate both source and in-memory caches. Never mutate
    # the objects used by another test or initialize a device/compile a kernel.
    kernels = {}
    for name in patch_module._KERNEL_NAMES:
        original = getattr(qsa_indexer, name)
        kernel = triton.jit(
            original.fn,
            do_not_specialize=[
                param.name for param in original.params if param.do_not_specialize
            ],
            do_not_specialize_on_alignment=[
                param.name
                for param in original.params
                if param.do_not_specialize_on_alignment
            ],
            debug=original.debug,
            noinline=original.noinline,
        )
        assert not kernel.device_caches
        monkeypatch.setattr(qsa_indexer, name, kernel)
        kernels[name] = kernel
    monkeypatch.setattr(triton, "__version__", "3.6.0")
    monkeypatch.setattr(patch_module, "version", lambda _name: "0.7.0+git317a0249")
    monkeypatch.setattr(torch.version, "hip", None)
    capability = Mock(return_value=(9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    return qsa_indexer, kernels, capability


def _config(architecture="Qwen4ExpForConditionalGeneration"):
    return SimpleNamespace(architecture=architecture)


def _ast(source):
    return ast.dump(ast.parse(source), include_attributes=False)


@pytest.mark.parametrize(
    "architecture", ["Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"]
)
def test_only_optional_load_hints_change_on_real_upstream_jit(
    qsa_kernels, architecture
):
    module, kernels, capability = qsa_kernels
    before = {name: kernel.src for name, kernel in kernels.items()}
    identities = {
        name: (kernel.fn, kernel.signature, tuple(kernel.params))
        for name, kernel in kernels.items()
    }
    originals = {
        name: getattr(module, name)
        for name in (
            "_topk",
            "_prefill_logits",
            "qsa_select_paged_decode",
            "qsa_select_paged_prefill",
            "_expand_qsa_indices_kernel",
        )
    }
    old_hashes = {name: kernel.cache_key for name, kernel in kernels.items()}
    device = torch.device("cuda:3")
    assert patch_module.patch_qsa_cache_hint(_config(architecture), device)
    capability.assert_called_once_with(device)

    for name, kernel in kernels.items():
        assert getattr(module, name) is kernel
        assert (kernel.fn, kernel.signature, tuple(kernel.params)) == identities[name]
        assert kernel._fl_qsa_original_src == before[name]
        expected = ast.parse(before[name])
        hint_calls = [
            node
            for node in ast.walk(expected)
            if isinstance(node, ast.Call)
            and any(kw.arg == "eviction_policy" for kw in node.keywords)
        ]
        assert len(hint_calls) == 1
        hint_call = hint_calls[0]
        assert isinstance(hint_call.func, ast.Attribute)
        assert hint_call.func.attr == "load"
        assert "k_cache_ptr" in ast.unparse(hint_call.args[0])
        hint_call.keywords = [
            kw for kw in hint_call.keywords if kw.arg != "eviction_policy"
        ]
        assert _ast(kernel.src) == ast.dump(expected, include_attributes=False)
        assert kernel.cache_key != old_hashes[name]
        assert not kernel.device_caches
    prefill = kernels["_qsa_mqa_paged_prefill_kernel"]
    assert {p.name for p in prefill.params if p.do_not_specialize} == {
        "num_rows",
        "query_offset",
    }
    assert all(
        getattr(module, name) is original for name, original in originals.items()
    )

    after = {name: (kernel.src, kernel.cache_key) for name, kernel in kernels.items()}
    assert patch_module.patch_qsa_cache_hint(_config(architecture), device) is False
    assert after == {
        name: (kernel.src, kernel.cache_key) for name, kernel in kernels.items()
    }


@pytest.mark.parametrize("compiled_index", [0, 1])
def test_late_install_refuses_without_partially_patching_other_kernel(
    qsa_kernels, compiled_index
):
    _module, kernels, _capability = qsa_kernels
    before = {name: kernel.src for name, kernel in kernels.items()}
    kernel = list(kernels.values())[compiled_index]
    kernel.device_caches[0] = object()  # Detect existence without a real CUDA binder.
    with pytest.raises(RuntimeError, match="already compiled"):
        patch_module.patch_qsa_cache_hint(_config(), torch.device("cuda:0"))
    assert before == {name: item.src for name, item in kernels.items()}
    assert all(not hasattr(item, "_fl_qsa_original_src") for item in kernels.values())


@pytest.mark.parametrize("changed_structure", ["policy", "load_target"])
def test_upstream_contract_drift_is_atomic(qsa_kernels, changed_structure):
    _module, kernels, _capability = qsa_kernels
    kernel = list(kernels.values())[1]
    tree = ast.parse(kernel.src)
    if changed_structure == "policy":
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "eviction_policy":
                node.value = ast.Constant(value="evict_last")
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "keys":
                node.id = "changed_keys"
    kernel._unsafe_update_src(ast.unparse(tree))
    before = {name: item.src for name, item in kernels.items()}
    with pytest.raises(RuntimeError, match="revalidate"):
        patch_module.patch_qsa_cache_hint(_config(), torch.device("cuda:0"))
    assert before == {name: item.src for name, item in kernels.items()}


@pytest.mark.parametrize(
    "reason", ["other_model", "sm80", "sm100", "triton37", "no_flagtree", "cpu", "hip"]
)
def test_unaffected_models_and_stacks_keep_original_jit(
    qsa_kernels, monkeypatch, reason
):
    _module, kernels, capability = qsa_kernels
    before = {name: kernel.src for name, kernel in kernels.items()}
    config, device = _config(), torch.device("cuda:2")
    if reason == "other_model":
        config.architecture = "Qwen3ForCausalLM"
    elif reason in ("sm80", "sm100"):
        capability.return_value = (8, 0) if reason == "sm80" else (10, 0)
    elif reason == "triton37":
        monkeypatch.setattr(triton, "__version__", "3.7.1")
    elif reason == "no_flagtree":

        def missing(_name):
            raise patch_module.PackageNotFoundError("flagtree")

        monkeypatch.setattr(patch_module, "version", missing)
    elif reason == "cpu":
        device = torch.device("cpu")
    else:
        monkeypatch.setattr(torch.version, "hip", "test-rocm")
    assert patch_module.patch_qsa_cache_hint(config, device) is False
    assert before == {name: kernel.src for name, kernel in kernels.items()}
    if reason in ("other_model", "triton37", "no_flagtree", "cpu", "hip"):
        capability.assert_not_called()
