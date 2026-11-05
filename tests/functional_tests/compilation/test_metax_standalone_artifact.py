# Copyright (c) 2026 BAAI. All rights reserved.

"""Exercise the MetaX artifact's native persistence and GPU replay contracts."""

import importlib
import os
import subprocess
import sys

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "metax", reason="MetaX patch")


def _patch():
    return importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.metax.patches.fix_standalone_compile"
    )


def _function(x, y):
    return torch.sin(x) + y * 2, torch.sum(x, dim=1)


def _inputs(device, dtype):
    backing = torch.linspace(-1, 1, 9 * 19, device=device, dtype=dtype)
    x = backing.reshape(9, 19)[1:9, 1:17:2]
    y = torch.linspace(0, 1, 64, device=device, dtype=dtype).reshape(8, 8)
    assert not x.is_contiguous() and x.storage_offset() != 0
    return x, y


def _check(actual, x, y):
    # Independent eager CPU reference, with no calls into the compiler adapter.
    expected = (torch.sin(x.cpu()) + y.cpu() * 2, torch.sum(x.cpu(), dim=1))
    tolerance = 0.03 if x.dtype == torch.bfloat16 else 1e-4
    assert len(actual) == len(expected)
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            output.cpu(), reference, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("format", ["binary", "unpacked"])
@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", torch.float32), ("cuda", torch.float32), ("cuda", torch.bfloat16)],
)
def test_compile_save_load_and_fresh_process(tmp_path, format, device, dtype):
    patch = _patch()
    native = importlib.import_module("torch._inductor.standalone_compile")
    x, y = _inputs(device, dtype)
    graph = torch.fx.symbolic_trace(_function)
    cache = importlib.import_module("torch._functorch._aot_autograd.autograd_cache")
    cache.check_cacheable(graph)
    artifact = patch.standalone_compile(graph, [x, y])
    assert isinstance(artifact, native.CompiledArtifact)
    _check(artifact(x, y), x, y)
    x.add_(0.25)
    y.sub_(0.5)
    _check(artifact(x, y), x, y)
    path = tmp_path / "artifact"
    artifact.save(path=str(path), format=format)
    restored = native.CompiledArtifact.load(path=str(path), format=format)
    _check(restored(x, y), x, y)

    inputs = tmp_path / "inputs.pt"
    outputs = tmp_path / "outputs.pt"
    torch.save((x.cpu(), y.cpu()), inputs)
    code = """
import importlib, sys, torch
import vllm_fl
from vllm.platforms import current_platform
assert current_platform.vendor_name == 'metax'
native = importlib.import_module('torch._inductor.standalone_compile')
x, y = torch.load(sys.argv[3], weights_only=True)
backing = torch.empty((9, 19), device=sys.argv[5], dtype=x.dtype)
view = backing[1:9, 1:17:2]
view.copy_(x)
x, y = view, y.to(sys.argv[5])
artifact = native.CompiledArtifact.load(path=sys.argv[1], format=sys.argv[2])
first = tuple(t.cpu() for t in artifact(x, y))
x.add_(0.5)
second = tuple(t.cpu() for t in artifact(x, y))
torch.save((first, second), sys.argv[4])
"""
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "fresh-inductor")
    env["TRITON_CACHE_DIR"] = str(tmp_path / "fresh-triton")
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(path),
            format,
            str(inputs),
            str(outputs),
            device,
        ],
        cwd="/",
        env=env,
        check=True,
        timeout=180,
    )
    first, second = torch.load(outputs, weights_only=True)
    _check(first, x.cpu(), y.cpu())
    _check(second, x.cpu() + 0.5, y.cpu())


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_artifact_capture_and_changed_input_replay(dtype):
    x, y = _inputs("cuda", dtype)
    artifact = _patch().standalone_compile(torch.fx.symbolic_trace(_function), [x, y])
    for _ in range(3):
        artifact(x, y)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = artifact(x, y)
    for _ in range(2):
        x.add_(0.25)
        y.sub_(0.125)
        for output in captured:
            output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check(captured, x, y)
