# Copyright (c) 2026 BAAI. All rights reserved.
"""Real CPU Dynamo/dispatcher checks for the MUSA FlagGems op boundary.

Dispatch registration and the exact tensor-device guard are redirected to CPU
under a private test namespace; unmodified wrappers separately check CPU fallback.
Pure tensor reference kernels make fallback and gradient assertions independent
of Dynamo version-specific errors. Separate controls retain the device-index
limitation and execute dtype metadata behind the opaque dispatcher boundary.
GPU kernel correctness still needs hardware acceptance.
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


def silu_reference(x, y):
    return (x.float() / (1 + torch.exp(-x.float())) * y).to(x.dtype)


def silu_kernel(x, y):
    # The metadata helper executes inside the opaque CPU kernel, never in FX.
    _, dtype = elementwise_dtypes(
        x, y, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    return (x.float() / (1 + torch.exp(-x.float())) * y).to(dtype)


@pytest.fixture(scope="module")
def boundary():
    state = NS(platform=NS(device_type="musa"), registrations=[], kernel_calls=[])

    def silu_out(x, y, out):
        out.copy_(silu_kernel(x, y))
        return out

    rms_module = NS(gems_rms_forward=rms_kernel, use_c_extension=False)
    silu_module = NS(gems_silu_and_mul=silu_reference)
    out_module = NS(silu_and_mul_out=silu_out)

    def register(
        name, fn, mutates_args=None, fake_impl=None, target_lib=None, dispatch_key=None
    ):
        assert dispatch_key == "PrivateUse1"
        state.registrations.append((name, tuple(mutates_args or ())))
        target_lib.define(
            name + torch.library.infer_schema(fn, mutates_args=mutates_args or [])
        )

        def counted_kernel(*args, **kwargs):
            # Only dispatcher execution mutates this spy; tracing never sees it.
            state.kernel_calls.append(name)
            return fn(*args, **kwargs)

        target_lib.impl(name, counted_kernel, "CPU")
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
    state.silu_module = fake_modules["flag_gems.modules.activation"]
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


def test_device_index_limitation_is_hidden_by_opaque_boundary(boundary, monkeypatch):
    def guarded_rms(x, residual, weight, epsilon):
        torch._utils._get_device_index(x.device, allow_cpu=True)
        return rms_kernel(x, residual, weight, epsilon)

    obj = NS(weight=torch.ones(4), variance_epsilon=1e-6)
    x = torch.arange(1.0, 9.0).reshape(2, 4)
    with pytest.raises(torch._dynamo.exc.Unsupported):
        torch.compile(
            lambda t: guarded_rms(t, None, obj.weight, obj.variance_epsilon),
            backend="eager",
            fullgraph=True,
        )(x)
    torch._dynamo.reset()
    monkeypatch.setattr(boundary.rms_module, "gems_rms_forward", guarded_rms)
    graphs = []
    actual = compiled(lambda t: boundary.norm(obj, t), graphs)(x)
    torch.testing.assert_close(actual, rms_oracle(x, obj.weight, obj.variance_epsilon))
    assert len(graphs) == 1
    assert any(
        "musa_boundary_cpu_test.rms_norm" in str(n.target)
        for n in graphs[0].graph.nodes
    )


def test_dtype_metadata_kernel_preserves_tensor_math():
    x = torch.arange(-8.0, 8.0).reshape(2, 8)
    # Do not assume this helper must fail tracing on every supported Torch version.
    torch.testing.assert_close(silu_kernel(x, x + 1), silu_reference(x, x + 1))


def test_rms_fullgraph_uses_one_opaque_node_and_original_kernel(boundary):
    weight = torch.tensor([0.5, 1.5, 0.25, 2.0])
    obj = NS(weight=weight, variance_epsilon=1e-6)
    graphs = []
    fn = compiled(lambda x: boundary.norm(obj, x), graphs)
    before = boundary.kernel_calls.count("rms_norm")
    for x in (torch.arange(1.0, 9.0).reshape(2, 4), torch.full((2, 4), -3.0)):
        result = fn(x)
        torch.testing.assert_close(result, rms_oracle(x, weight, 1e-6))
        assert result.data_ptr() != x.data_ptr()
    assert len(graphs) == 1
    assert boundary.kernel_calls.count("rms_norm") == before + 2
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
    before = boundary.kernel_calls.count("silu_and_mul")
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
    assert (
        len(graphs) == 1 and boundary.kernel_calls.count("silu_and_mul") == before + 2
    )
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


def assert_plain_graph(graphs):
    assert len(graphs) == 1
    assert not any(
        "musa_boundary_cpu_test" in str(n.target) for n in graphs[0].graph.nodes
    )


def reference_and_wrapper(boundary, op, obj, *, unmodified=False):
    if op == "norm":
        wrapper = boundary.original_norm if unmodified else boundary.norm
        return lambda x: rms_kernel(
            x, None, obj.weight, obj.variance_epsilon
        ), lambda x: wrapper(obj, x)
    wrapper = boundary.original_activation if unmodified else boundary.activation
    return lambda x: silu_reference(x[..., :2], x[..., 2:]), lambda x: wrapper(None, x)


def assert_value_and_grad(actual, expected, inputs):
    torch.testing.assert_close(actual, expected)
    if torch.is_grad_enabled():
        actual_grads = torch.autograd.grad(actual.sum(), inputs)
        expected_grads = torch.autograd.grad(expected.sum(), inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)


def check_original_compile(boundary, op, *, unmodified=False, grad_enabled=False):
    with torch.set_grad_enabled(grad_enabled):
        weight = torch.tensor([0.5, 1.5, 0.25, 2.0], requires_grad=grad_enabled)
        obj = NS(weight=weight, variance_epsilon=1e-6)
        reference, original = reference_and_wrapper(
            boundary, op, obj, unmodified=unmodified
        )
        x = torch.arange(1.0, 9.0).reshape(2, 4).requires_grad_(grad_enabled)
        inputs = (x, weight) if op == "norm" else (x,)
        graphs = []
        calls = list(boundary.kernel_calls)
        assert_value_and_grad(compiled(original, graphs)(x), reference(x), inputs)
        assert_plain_graph(graphs)
        assert boundary.kernel_calls == calls


@pytest.mark.parametrize("device", ["cuda", "npu", "cpu"])
@pytest.mark.parametrize("op", ["norm", "silu"])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_non_musa_compilation_does_not_route_to_new_boundary(
    boundary, device, op, grad_enabled
):
    boundary.platform.device_type = device
    check_original_compile(boundary, op, grad_enabled=grad_enabled)


@pytest.mark.parametrize("op", ["norm", "silu"])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_cpu_tensor_on_musa_platform_keeps_original_route(boundary, op, grad_enabled):
    check_original_compile(boundary, op, unmodified=True, grad_enabled=grad_enabled)


@pytest.mark.parametrize("op", ["norm", "silu"])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_eager_musa_retains_original_gems_path(boundary, monkeypatch, op, grad_enabled):
    with torch.set_grad_enabled(grad_enabled):
        weight = torch.tensor([0.5, 1.5, 0.25, 2.0], requires_grad=grad_enabled)
        obj = NS(weight=weight, variance_epsilon=1e-6)
        reference, original = reference_and_wrapper(boundary, op, obj)
        x = torch.arange(1.0, 9.0).reshape(2, 4).requires_grad_(grad_enabled)
        inputs = (x, weight) if op == "norm" else (x,)
        calls = list(boundary.kernel_calls)
        assert_value_and_grad(original(x), reference(x), inputs)
        assert boundary.kernel_calls == calls
    sentinel = torch.ones(2, 4)
    if op == "norm":
        monkeypatch.setattr(
            boundary.rms_module, "gems_rms_forward", lambda *args: sentinel
        )
    else:
        monkeypatch.setattr(
            boundary.silu_module, "gems_silu_and_mul", lambda *args: sentinel
        )
    obj = NS(weight=torch.ones(4), variance_epsilon=1e-6)
    _, original = reference_and_wrapper(boundary, op, obj)
    calls = list(boundary.kernel_calls)
    with torch.set_grad_enabled(grad_enabled):
        assert original(torch.ones(2, 4)) is sentinel
    assert boundary.kernel_calls == calls
    assert len(boundary.registrations) == 3


@pytest.mark.parametrize("op", ["norm", "silu"])
def test_grad_enabled_compile_keeps_original_kernel_contract(boundary, op):
    check_original_compile(boundary, op, grad_enabled=True)
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
