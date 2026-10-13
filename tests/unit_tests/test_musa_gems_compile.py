# Copyright (c) 2026 BAAI. All rights reserved.
"""Real CPU Dynamo/dispatcher checks for the MUSA FlagGems op boundary.

Dispatch registration and the exact tensor-device guard are redirected to CPU
under a private test namespace; unmodified wrappers separately check CPU fallback.
The CPU kernel doubles execute the actual torch metadata/device-index helpers
that failed in MUSA CI; GPU kernel correctness still needs hardware acceptance.
"""

import ast
import builtins
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes
from torch._subclasses.fake_tensor import FakeTensorMode

ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "vllm_fl/dispatch/backends/flaggems/impl"


def rms_kernel(x, residual, weight, epsilon):
    # This exact helper is intentionally skipped by Dynamo in torch 2.9/2.10.
    torch._utils._get_device_index(x.device, allow_cpu=True)
    x = x.contiguous()
    if residual is None:
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
            * weight
        ).to(x.dtype)
    residual = residual.contiguous()
    added = x.float() + residual.float()
    residual.copy_(added)
    x.copy_(
        added * torch.rsqrt(added.square().mean(-1, keepdim=True) + epsilon) * weight
    )
    return x, residual


def silu_kernel(x, y):
    _, dtype = elementwise_dtypes(
        x, y, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    return (x.float() / (1 + torch.exp(-x.float())) * y).to(dtype)


@pytest.fixture(scope="module")
def boundary():
    state = NS(
        platform=NS(device_type="musa"), registrations=[], rms_calls=0, silu_calls=0
    )

    def rms(*args):
        state.rms_calls += 1
        return rms_kernel(*args)

    def silu_out(x, y, out):
        state.silu_calls += 1
        out.copy_(silu_kernel(x, y))
        return out

    rms_module = NS(gems_rms_forward=rms, use_c_extension=False)
    silu_module = NS(gems_silu_and_mul=silu_kernel)
    out_module = NS(silu_and_mul_out=silu_out)

    def register(
        name, fn, mutates_args=None, fake_impl=None, target_lib=None, dispatch_key=None
    ):
        assert dispatch_key == "PrivateUse1"
        state.registrations.append((name, tuple(mutates_args or ())))
        target_lib.define(
            name + torch.library.infer_schema(fn, mutates_args=mutates_args or [])
        )
        target_lib.impl(name, fn, "CPU")
        target_lib._register_fake(name, fake_impl)

    runtime = NS()
    loaded_modules = []
    fake_modules = {}
    for name, attrs in {
        "flag_gems": {},
        "flag_gems.modules": {},
        "flag_gems.fused": {},
        "flag_gems.modules.normalization": vars(rms_module),
        "flag_gems.modules.activation": vars(silu_module),
        "flag_gems.fused.silu_and_mul": vars(out_module),
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        module.__path__ = []
        fake_modules[name] = module
    module_patch = patch.dict(sys.modules, fake_modules)
    module_patch.start()

    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.utils.torch_utils":
            return NS(direct_register_custom_op=register)
        if name == "vllm.platforms":
            return NS(current_platform=state.platform)
        if name == "flag_gems.modules.normalization":
            return fake_modules[name]
        if name == "flag_gems.modules.activation":
            return fake_modules[name]
        if name == "flag_gems.fused.silu_and_mul":
            return fake_modules[name]
        if name == "musa_ops" and level == 1:
            return runtime
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(path, *, namespace_replace=False, cpu_tensor=False):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if namespace_replace:

            class TestNamespace(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if node.value == "vllm_fl_musa":
                        node.value = "musa_boundary_cpu_test"
                    return node

                def visit_Attribute(self, node):
                    self.generic_visit(node)
                    if node.attr == "vllm_fl_musa":
                        node.attr = "musa_boundary_cpu_test"
                    return node

            tree = TestNamespace().visit(tree)
        if cpu_tensor:

            class CPUTensorGuard(ast.NodeTransformer):
                count = 0

                def visit_Compare(self, node):
                    self.generic_visit(node)
                    if (
                        ast.dump(node.left)
                        == ast.dump(ast.parse("x.device.type", mode="eval").body)
                        and len(node.ops) == 1
                        and isinstance(node.ops[0], ast.Eq)
                        and len(node.comparators) == 1
                        and isinstance(node.comparators[0], ast.Constant)
                        and node.comparators[0].value == "musa"
                    ):
                        node.comparators[0].value = "cpu"
                        self.count += 1
                    return node

            guard = CPUTensorGuard()
            tree = guard.visit(tree)
            assert guard.count == 1
        name = "musa_boundary_contract_" + path.stem + ("_cpu" if cpu_tensor else "")
        module = ModuleType(name)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": imports}
        sys.modules[name] = module
        loaded_modules.append(name)
        exec(compile(tree, str(path), "exec"), vars(module))
        return module

    runtime = load(IMPL / "musa_ops.py", namespace_replace=True)
    activation = load(IMPL / "activation.py", cpu_tensor=True)
    norm = load(IMPL / "normalization.py", cpu_tensor=True)
    state.original_activation = load(IMPL / "activation.py").silu_and_mul_flaggems
    state.original_norm = load(IMPL / "normalization.py").rms_norm_flaggems
    state.runtime = runtime
    state.rms_module = fake_modules["flag_gems.modules.normalization"]
    state.activation = activation.silu_and_mul_flaggems
    state.norm = norm.rms_norm_flaggems
    yield state
    for name in loaded_modules:
        sys.modules.pop(name)
    module_patch.stop()


@pytest.fixture(autouse=True)
def isolated_compile(boundary):
    boundary.platform.device_type = "musa"
    torch._dynamo.reset()
    with torch.no_grad():
        yield
    torch._dynamo.reset()


def compiled(fn, graphs):
    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    return torch.compile(fn, backend=backend, fullgraph=True)


def rms_oracle(x, weight, epsilon):
    rows = x.tolist()
    return torch.tensor(
        [
            [
                v * w / math.sqrt(sum(t * t for t in row) / len(row) + epsilon)
                for v, w in zip(row, weight.tolist())
            ]
            for row in rows
        ],
        dtype=x.dtype,
    )


@pytest.mark.parametrize(
    "op,pattern", [("norm", "_get_device_index"), ("silu", "non-Tensor")]
)
def test_original_python_kernel_hits_real_dynamo_limitation(op, pattern):
    fn = (
        (lambda x: rms_kernel(x, None, torch.ones(x.shape[-1]), 1e-6))
        if op == "norm"
        else (lambda x: silu_kernel(x, x))
    )
    with pytest.raises(torch._dynamo.exc.Unsupported, match=pattern):
        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2, 4))


def test_rms_fullgraph_uses_one_opaque_node_and_original_kernel(boundary):
    weight = torch.tensor([0.5, 1.5, 0.25, 2.0])
    obj = NS(weight=weight, variance_epsilon=1e-6)
    graphs = []
    fn = compiled(lambda x: boundary.norm(obj, x), graphs)
    before = boundary.rms_calls
    for x in (torch.arange(1.0, 9.0).reshape(2, 4), torch.full((2, 4), -3.0)):
        result = fn(x)
        torch.testing.assert_close(result, rms_oracle(x, weight, 1e-6))
        assert result.data_ptr() != x.data_ptr()
    assert len(graphs) == 1
    assert boundary.rms_calls == before + 2
    assert any(
        "musa_boundary_cpu_test.rms_norm" in str(n.target)
        for n in graphs[0].graph.nodes
    )


@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_fused_rms_preserves_mutation_alias_and_contiguous_copy(
    boundary, noncontiguous, backend
):
    x = torch.arange(1.0, 9.0).reshape(2, 4)
    residual = torch.full((2, 4), 0.5)
    if noncontiguous:
        x = x.t().contiguous().t()
        residual = residual.t().contiguous().t()
    original_x, original_r = x.clone(), residual.clone()
    obj = NS(weight=torch.tensor([0.5, 1.0, 1.5, 2.0]), variance_epsilon=1e-6)
    fn = torch.compile(
        lambda a, b: boundary.norm(obj, a, b), backend=backend, fullgraph=True
    )
    y, r = fn(x, residual)
    torch.testing.assert_close(y, rms_oracle(original_x + original_r, obj.weight, 1e-6))
    torch.testing.assert_close(r, original_x + original_r)
    assert y.is_contiguous() and r.is_contiguous()
    if noncontiguous:
        torch.testing.assert_close(x, original_x)
        torch.testing.assert_close(residual, original_r)
        assert y.data_ptr() != x.data_ptr() and r.data_ptr() != residual.data_ptr()
    else:
        assert y is x and r is residual


def test_silu_fullgraph_out_contract_and_no_reused_output_alias(boundary):
    graphs = []
    fn = compiled(lambda x: boundary.activation(None, x), graphs)
    x = torch.arange(-8.0, 8.0).reshape(2, 8)
    before = boundary.silu_calls
    first = fn(x)
    snapshot = first.clone()
    second = fn(x + 1)
    expected = torch.tensor(
        [
            [v / (1 + math.exp(-v)) * w for v, w in zip(row[:4], row[4:])]
            for row in x.tolist()
        ]
    )
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(first, snapshot)
    assert first.data_ptr() != second.data_ptr()
    assert first.data_ptr() != x.data_ptr()
    assert len(graphs) == 1 and boundary.silu_calls == before + 2
    assert any(
        "musa_boundary_cpu_test.silu_and_mul" in str(n.target)
        for n in graphs[0].graph.nodes
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fake_outputs_have_exact_shape_dtype_device_and_stride(boundary, dtype):
    with FakeTensorMode():
        x = torch.empty(4, 2, dtype=dtype).t()
        weight = torch.empty(4, dtype=dtype)
        y = boundary.runtime.rms_norm(x, None, weight, 1e-6)
        assert (y.shape, y.dtype, y.device) == (x.shape, x.dtype, x.device)
        assert y.is_contiguous()
        r = torch.empty_like(x)
        a, b = boundary.runtime.rms_norm(x, r, weight, 1e-6)
        assert a.is_contiguous() and b.is_contiguous()
        z = boundary.runtime.silu_and_mul(x)
        expected = torch.empty_like(x[..., :2])
        assert (z.shape, z.dtype, z.device, z.stride()) == (
            expected.shape,
            expected.dtype,
            expected.device,
            expected.stride(),
        )


def test_registration_schema_records_exact_mutations(boundary):
    assert boundary.registrations == [
        ("rms_norm", ()),
        ("fused_add_rms_norm", ("x", "residual")),
        ("silu_and_mul", ("out",)),
    ]
    schema = str(torch.ops.musa_boundary_cpu_test.fused_add_rms_norm.default._schema)
    assert "Tensor(a0!) x" in schema and "Tensor(a1!) residual" in schema
    assert schema.endswith("-> ()")


@pytest.mark.parametrize("device", ["cuda", "npu", "cpu"])
def test_non_musa_compilation_does_not_route_to_new_boundary(boundary, device):
    boundary.platform.device_type = device
    obj = NS(weight=torch.ones(4), variance_epsilon=1e-6)
    fn = torch.compile(lambda x: boundary.norm(obj, x), backend="eager", fullgraph=True)
    with pytest.raises(torch._dynamo.exc.Unsupported, match="_get_device_index"):
        fn(torch.ones(2, 4))


@pytest.mark.parametrize(
    "op,pattern", [("norm", "_get_device_index"), ("silu", "non-Tensor")]
)
def test_cpu_tensor_on_musa_platform_keeps_original_route(boundary, op, pattern):
    obj = NS(weight=torch.ones(4), variance_epsilon=1e-6)
    original = (
        (lambda x: boundary.original_norm(obj, x))
        if op == "norm"
        else (lambda x: boundary.original_activation(None, x))
    )
    fn = torch.compile(original, backend="eager", fullgraph=True)
    with pytest.raises(torch._dynamo.exc.Unsupported, match=pattern):
        fn(torch.ones(2, 4))


def test_eager_musa_retains_original_gems_path(boundary):
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    obj = NS(weight=torch.ones(4), variance_epsilon=1e-6)
    with torch.enable_grad():
        y = boundary.norm(obj, x)
        y.sum().backward()
    assert x.grad is not None
    assert len(boundary.registrations) == 3


@pytest.mark.parametrize(
    "op,pattern", [("norm", "_get_device_index"), ("silu", "non-Tensor")]
)
def test_grad_enabled_compile_keeps_original_kernel_contract(boundary, op, pattern):
    obj = NS(weight=torch.ones(4, requires_grad=True), variance_epsilon=1e-6)
    with torch.enable_grad():
        x = torch.ones(2, 4, requires_grad=True)
        original = (
            (lambda t: boundary.norm(obj, t))
            if op == "norm"
            else (lambda t: boundary.activation(None, t))
        )
        fn = torch.compile(original, backend="eager", fullgraph=True)
        with pytest.raises(torch._dynamo.exc.Unsupported, match=pattern):
            fn(x)
    assert len(boundary.registrations) == 3


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_both_inference_contexts_use_compiled_boundary(boundary, context):
    with context():
        obj = NS(weight=torch.ones(4), variance_epsilon=1e-6)
        graphs = []
        fn = compiled(lambda x: boundary.norm(obj, x), graphs)
        x = torch.arange(1.0, 9.0).reshape(2, 4)
        torch.testing.assert_close(fn(x), rms_oracle(x, obj.weight, 1e-6))
        assert len(graphs) == 1


def test_dynamic_batch_uses_one_symbolic_graph(boundary):
    obj = NS(weight=torch.ones(4), variance_epsilon=1e-6)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    fn = torch.compile(
        lambda x: boundary.norm(obj, x), backend=backend, fullgraph=True, dynamic=True
    )
    for size in (2, 5, 3):
        x = torch.arange(1.0, size * 4 + 1).reshape(size, 4)
        torch.testing.assert_close(fn(x), rms_oracle(x, obj.weight, 1e-6))
    assert len(graphs) == 1


def test_enabled_c_extension_keeps_its_original_entry_point(boundary, monkeypatch):
    monkeypatch.setattr(boundary.rms_module, "use_c_extension", True)

    def native_rms(x, residual, weight, epsilon):
        return x + weight

    monkeypatch.setattr(boundary.rms_module, "gems_rms_forward", native_rms)
    graphs = []
    obj = NS(weight=torch.ones(4), variance_epsilon=1e-6)
    fn = compiled(lambda x: boundary.norm(obj, x), graphs)
    x = torch.arange(1.0, 9.0).reshape(2, 4)
    torch.testing.assert_close(fn(x), x + 1)
    assert len(graphs) == 1
    assert not any(
        "musa_boundary_cpu_test" in str(n.target) for n in graphs[0].graph.nodes
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
def test_half_inputs_and_weight_dtype_keep_output_contract(
    boundary, dtype, weight_dtype
):
    x = torch.arange(1.0, 9.0).reshape(2, 4).to(dtype)
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=weight_dtype)
    obj = NS(weight=weight, variance_epsilon=1e-6)
    norm = torch.compile(
        lambda t: boundary.norm(obj, t), backend="eager", fullgraph=True
    )
    y = norm(x)
    assert y.dtype == dtype
    torch.testing.assert_close(y, rms_oracle(x, weight, 1e-6))
    silu = torch.compile(
        lambda t: boundary.activation(None, t), backend="eager", fullgraph=True
    )
    z = silu(x)
    assert z.dtype == dtype
    torch.testing.assert_close(z, silu_kernel(x[..., :2], x[..., 2:]))


def test_noncontiguous_silu_matches_preserve_format_output(boundary):
    x = torch.arange(1.0, 33.0).reshape(8, 4).t()
    expected = silu_kernel(x[..., :4], x[..., 4:])
    fn = torch.compile(
        lambda t: boundary.activation(None, t), backend="eager", fullgraph=True
    )
    actual = fn(x)
    torch.testing.assert_close(actual, expected)
    assert actual.stride() == torch.empty_like(x[..., :4]).stride()
