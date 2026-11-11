"""Required worker patches fail closed even when import-time patching is tolerant."""

import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_function(relative, name, **namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    scope = {"logger": logging.getLogger(__name__), **namespace}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope[name]


@pytest.mark.parametrize("failure", ["topk", "logprobs"])
def test_sampler_strict_failure(monkeypatch, failure):
    topk = SimpleNamespace(compiled_random_sample=lambda: None)
    logprobs = SimpleNamespace(batched_count_greater_than=lambda: None)
    delattr(
        topk if failure == "topk" else logprobs,
        "compiled_random_sample" if failure == "topk" else "batched_count_greater_than",
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.ops.topk_topp_sampler", topk)
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.ops.logprobs", logprobs)
    patch = load_function(
        "vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py",
        "patch_sampler_compile_for_iluvatar",
    )
    patch()  # Import-time compatibility remains tolerant.
    with pytest.raises(RuntimeError, match="sampler patch failed") as exc:
        patch(strict=True)
    assert isinstance(exc.value.__cause__, AttributeError)


def test_inductor_strict_failure(monkeypatch):
    import triton.backends

    monkeypatch.setattr(
        triton.backends,
        "backends",
        {
            "iluvatar": SimpleNamespace(
                compiler=SimpleNamespace(
                    supports_target=lambda target: target.backend == "corex"
                )
            )
        },
    )
    monkeypatch.setitem(sys.modules, "torch._inductor.runtime.triton_heuristics", None)
    patch = load_function(
        "vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py",
        "patch_torch_inductor_for_iluvatar",
    )
    patch()
    with pytest.raises(RuntimeError, match="Inductor target patch failed"):
        patch(strict=True)


@pytest.mark.parametrize("failing", ["inductor", "sampler"])
def test_worker_propagates_required_patch_failure(monkeypatch, failing):
    patches = {name: Mock() for name in ("inductor", "sampler")}
    patches[failing].side_effect = RuntimeError("required patch failed")
    module = SimpleNamespace(
        patch_torch_inductor_for_iluvatar=patches["inductor"],
        patch_sampler_compile_for_iluvatar=patches["sampler"],
    )
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar", module
    )
    for submodule, function in (
        ("moe_router", "patch_moe_router"),
        ("vision_precision", "patch_vision_precision"),
        ("mrope", "patch_mrope"),
    ):
        monkeypatch.setitem(
            sys.modules,
            f"vllm_fl.dispatch.backends.vendor.iluvatar.{submodule}",
            SimpleNamespace(**{function: Mock()}),
        )
    patch = load_function(
        "vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py",
        "apply_worker_patches",
        current_platform=SimpleNamespace(vendor_name="iluvatar"),
        patch_autotune_memory=Mock(),
        patch_v2_uva=Mock(),
        patch_torch_inductor_for_iluvatar=patches["inductor"],
        patch_sampler_compile_for_iluvatar=patches["sampler"],
    )
    with pytest.raises(RuntimeError, match="required patch failed"):
        patch()
    patches[failing].assert_called_once_with(strict=True)


@pytest.mark.parametrize(
    "registered", [{}, {"iluvatar": SimpleNamespace(compiler=None)}]
)
def test_inductor_strict_rejects_unknown_backend(monkeypatch, registered):
    import triton.backends

    monkeypatch.setattr(triton.backends, "backends", registered)
    patch = load_function(
        "vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py",
        "patch_torch_inductor_for_iluvatar",
    )
    with pytest.raises(RuntimeError, match="Cannot inspect Iluvatar Triton backends"):
        patch(strict=True)


def test_sampler_strict_success_is_idempotent(monkeypatch):
    eager = lambda: None

    def compiled():
        pass

    compiled.__wrapped__ = eager
    modules = []
    for name, attr in (
        ("topk_topp_sampler", "compiled_random_sample"),
        ("logprobs", "batched_count_greater_than"),
    ):
        module = SimpleNamespace(**{attr: compiled})
        monkeypatch.setitem(sys.modules, f"vllm.v1.sample.ops.{name}", module)
        modules.append((module, attr))
    patch = load_function(
        "vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py",
        "patch_sampler_compile_for_iluvatar",
    )
    patch(strict=True)
    patch(strict=True)
    for module, attr in modules:
        assert getattr(module, attr) is eager


@pytest.mark.parametrize("vendor", ["iluvatar", "nvidia", "ascend", ""])
def test_public_worker_hook_dispatches_only_iluvatar(monkeypatch, vendor):
    apply = Mock()
    key = "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar"
    monkeypatch.setitem(sys.modules, key, SimpleNamespace(apply_worker_patches=apply))
    hook = load_function(
        "vllm_fl/worker/worker.py",
        "_apply_worker_vendor_patches",
        current_platform=SimpleNamespace(vendor_name=vendor),
    )
    hook()
    assert apply.call_count == (1 if vendor == "iluvatar" else 0)


def test_public_worker_hook_propagates_failure(monkeypatch):
    apply = Mock(side_effect=RuntimeError("required vendor patch failed"))
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar",
        SimpleNamespace(apply_worker_patches=apply),
    )
    hook = load_function(
        "vllm_fl/worker/worker.py",
        "_apply_worker_vendor_patches",
        current_platform=SimpleNamespace(vendor_name="iluvatar"),
    )
    with pytest.raises(RuntimeError, match="required vendor patch failed"):
        hook()
