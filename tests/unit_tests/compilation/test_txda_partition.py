# Copyright (c) 2026 BAAI. All rights reserved.
"""TXDA native boundaries against real FX nodes, without importing vLLM."""

import operator
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import fx

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/compilation/txda_partition.py"
_HELPERS = runpy.run_path(str(_SOURCE))
is_boundary = _HELPERS["is_txda_native_graph_boundary"]
make_predicate = _HELPERS["make_txda_split_predicate"]


def _node(op, target):
    graph = fx.Graph()
    value = graph.placeholder("value")
    return graph.create_node(op, target, (value,))


@pytest.mark.parametrize("method", ["float", "to", "bfloat16", "type_as", "mm"])
def test_native_methods_form_eager_boundaries(method):
    assert is_boundary(_node("call_method", method))


@pytest.mark.parametrize(
    "target",
    [
        operator.matmul,
        torch.matmul,
        torch.mm,
        torch.bmm,
        torch.addmm,
        torch.nn.functional.linear,
        torch.ops.aten.to,
        torch.ops.aten.to.dtype,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.linear.default,
    ],
)
def test_native_function_aliases_form_eager_boundaries(target):
    assert is_boundary(_node("call_function", target))


@pytest.mark.parametrize(
    "target", [operator.mul, operator.add, torch.ops.aten.mul.Tensor]
)
def test_gems_elementwise_nodes_remain_available_for_capture(target):
    assert not is_boundary(_node("call_function", target))


def test_metadata_and_view_nodes_are_not_new_boundaries():
    graph = fx.Graph()
    value = graph.placeholder("value")
    assert not is_boundary(value)
    value.meta["example_value"] = torch.ones(2, 4)
    assert not is_boundary(graph.call_method("view", (value, -1)))
    assert not is_boundary(graph.output(value))


@pytest.mark.parametrize("device", ["cuda", "npu", "musa", "cpu"])
def test_other_platforms_keep_the_original_predicate(device):
    node = _node("call_method", "float")
    original = Mock(return_value=False)
    operations = ["vllm::unified_attention"]
    assert not make_predicate(original, device)(node, operations)
    original.assert_called_once_with(node, operations)


def test_upstream_attention_boundaries_and_split_list_are_preserved():
    node = _node("call_function", operator.mul)
    original = Mock(return_value=True)
    operations = ["vllm::unified_attention"]
    assert make_predicate(original, "txda")(node, operations)
    original.assert_called_once_with(node, operations)


def test_real_dynamo_native_cast_matmul_then_gems_region():
    observed = []
    predicate = make_predicate(lambda node, ops: False, "txda")

    def backend(graph, example_inputs):
        for node in graph.graph.nodes:
            if node.op in ("call_method", "call_function"):
                observed.append((node.op, node.target, predicate(node, [])))
        return graph.forward

    def function(value, weight):
        product = torch.matmul(value.float(), weight.to(torch.float32))
        return (product * 2.0).to(torch.bfloat16)

    compiled = torch.compile(function, backend=backend, fullgraph=True)
    value = torch.ones((2, 4), dtype=torch.bfloat16)
    weight = torch.eye(4, dtype=torch.bfloat16)
    for scale in (1, 3):
        torch.testing.assert_close(
            compiled(value * scale, weight), function(value * scale, weight)
        )

    assert observed == [
        ("call_method", "float", True),
        ("call_method", "to", True),
        ("call_function", torch.matmul, True),
        ("call_function", operator.mul, False),
        ("call_method", "to", True),
    ]


@pytest.mark.parametrize("target", [torch.sin, torch.nn.functional.relu, torch.clone])
def test_unknown_native_compute_is_eager(target):
    node = _node("call_function", target)
    assert is_boundary(node)
    assert make_predicate(lambda node, ops: False, "txda")(node, [])


@pytest.mark.parametrize(
    "method", ["reshape", "flatten", "contiguous", "copy_", "add_"]
)
def test_methods_that_may_copy_or_mutate_are_eager(method):
    assert is_boundary(_node("call_method", method))


def test_output_mutations_are_not_capture_safe():
    graph = fx.Graph()
    x = graph.placeholder("x")
    output = graph.placeholder("output")
    assert is_boundary(graph.call_function(torch.add, (x, 1), {"out": output}))
    assert is_boundary(
        graph.call_function(torch.ops.aten.add.out, (x, 1), {"out": output})
    )
    assert is_boundary(
        graph.call_function(torch.ops.aten.mul.out, (x, 1), {"out": output})
    )


@pytest.mark.parametrize("example", [torch.ones(2, 3), None])
def test_tensor_or_unknown_getitem_is_rejected_even_after_partitioning(example):
    graph = fx.Graph()
    x = graph.placeholder("x")
    computed = graph.call_function(operator.mul, (x, 2))
    if example is not None:
        computed.meta["example_value"] = example
    indexed = graph.call_function(operator.getitem, (computed, 0))
    graph.output(indexed)
    module = fx.GraphModule({}, graph)
    assert is_boundary(indexed)
    valid, reason = _HELPERS["validate_txda_graph_module"](module)
    assert not valid
    assert "getitem" in reason


@pytest.mark.parametrize("metadata_key", ["example_value", "val"])
def test_proven_sequence_getitem_is_only_metadata(metadata_key):
    graph = fx.Graph()
    pair = graph.placeholder("pair")
    pair.meta[metadata_key] = (torch.ones(2), torch.zeros(2))
    selected = graph.call_function(operator.getitem, (pair, 1))
    graph.output(selected)
    module = fx.GraphModule({}, graph)
    assert not is_boundary(selected)
    assert _HELPERS["validate_txda_graph_module"](module) == (True, "")


def test_opaque_module_calls_are_eager():
    assert is_boundary(_node("call_module", "opaque_native_extension"))


def test_transparent_callable_validation():
    module = fx.symbolic_trace(lambda x: (x * 2 + 1).transpose(0, 1))
    next(iter(module.graph.nodes)).meta["example_value"] = torch.ones(2, 4)
    validate = _HELPERS["validate_txda_graph_callable"]
    assert validate(module) == (True, "")
    assert validate(module.forward) == (True, "")
    opaque = lambda *args: module(*args)
    opaque.graph = module.graph
    assert not validate(opaque)[0]
    native = fx.symbolic_trace(lambda x: x.float())
    assert not validate(native)[0]


def test_safe_views_do_not_admit_potential_copy_reshape():
    safe = fx.symbolic_trace(lambda x: x.view(-1).unsqueeze(0).expand(2, -1))
    unsafe = fx.symbolic_trace(lambda x: x.transpose(0, 1).reshape(-1))
    for module in (safe, unsafe):
        next(iter(module.graph.nodes)).meta["example_value"] = torch.ones(2, 4)
    validate = _HELPERS["validate_txda_graph_module"]
    assert validate(safe) == (True, "")
    assert not validate(unsafe)[0]


def test_installation_is_explicit_scoped_and_idempotent(monkeypatch):
    original = Mock(return_value=False)
    original_split = Mock()
    original_codegen = Mock()
    backends = SimpleNamespace(
        should_split=original,
        split_graph=original_split,
        _decompose_size_nodes=Mock(),
        generate_execution_code=original_codegen,
    )
    compilation = ModuleType("vllm.compilation")
    compilation.backends = backends
    monkeypatch.setitem(sys.modules, "vllm.compilation", compilation)
    install = _HELPERS["install_txda_graph_partition"]
    install("cuda")
    assert backends.should_split is original
    assert backends.split_graph is original_split
    assert backends.generate_execution_code is original_codegen
    install("txda")
    installed = backends.should_split
    installed_split = backends.split_graph
    installed_codegen = backends.generate_execution_code
    assert installed_codegen is not original_codegen
    assert installed_split is not original_split
    assert installed is not original
    install("txda")
    assert backends.should_split is installed
    assert backends.split_graph is installed_split
    assert backends.generate_execution_code is installed_codegen
    assert installed(_node("call_method", "float"), [])


def test_custom_attribute_method_cannot_hide_native_work():
    class Opaque:
        def add(self, value):
            return value

    root = torch.nn.Module()
    root.opaque = Opaque()
    graph = fx.Graph()
    x = graph.placeholder("x")
    obj = graph.get_attr("opaque")
    result = graph.call_method("add", (obj, x))
    graph.output(result)
    module = fx.GraphModule(root, graph)
    valid, reason = _HELPERS["validate_txda_graph_module"](module)
    assert not valid
    assert "get_attr" in reason


def test_tensor_attribute_remains_valid():
    root = torch.nn.Module()
    root.register_buffer("weight", torch.ones(2))
    graph = fx.Graph()
    weight = graph.get_attr("weight")
    result = graph.call_method("mul", (weight, 2))
    graph.output(result)
    assert _HELPERS["validate_txda_graph_module"](fx.GraphModule(root, graph)) == (
        True,
        "",
    )


@pytest.mark.parametrize("kind", ["forward", "hook", "pre_hook"])
def test_forward_overrides_or_hooks_cannot_bypass_graph_validation(kind):
    module = fx.symbolic_trace(lambda x: x * 2)
    if kind == "forward":
        module.forward = lambda x: x.float()
    elif kind == "hook":
        module.register_forward_hook(lambda module, args, output: output)
    else:
        module.register_forward_pre_hook(lambda module, args: args)
    assert not _HELPERS["validate_txda_graph_module"](module)[0]


def test_explicit_as_strided_cannot_read_beyond_staged_logical_storage():
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["example_value"] = torch.ones(2)
    assert is_boundary(graph.call_method("as_strided", (x, (4,), (1,))))
    assert is_boundary(graph.call_function(torch.as_strided, (x, (4,), (1,))))
    assert is_boundary(
        graph.call_function(torch.ops.aten.as_strided.default, (x, (4,), (1,)))
    )


@pytest.mark.parametrize("kind", ["class_forward", "class_call", "compiled_call"])
def test_opaque_class_execution_cannot_bypass_safe_fx_nodes(kind):
    module = fx.symbolic_trace(lambda x: x * 2)
    if kind == "class_forward":
        type(module).forward = lambda self, x: x.float()
    elif kind == "class_call":
        type(module).__call__ = lambda self, x: x.float()
    else:
        module._compiled_call_impl = lambda x: x.float()
    assert not _HELPERS["validate_txda_graph_module"](module)[0]


def test_unknown_call_wrapper_cannot_hide_native_work():
    module = fx.symbolic_trace(lambda x: x * 2)

    class OpaqueCall:
        cls = type(module)
        cls_call = None

        def __call__(self, module, value):
            return value.float()

    type(module)._wrapped_call = OpaqueCall()
    assert not _HELPERS["validate_txda_graph_module"](module)[0]


def _split_with_producer_getitem_policy(graph, splitting_ops):
    """Exercise vLLM's getitem-to-producer policy with real FX partitioning."""
    from torch.fx.passes.split_module import split_module

    assignments = {}
    partition = 0
    predicate = make_predicate(lambda node, ops: False, "txda")
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and node.args[0].op != "placeholder"
        ):
            assignments[node] = assignments[node.args[0]]
            continue
        if predicate(node, splitting_ops):
            partition += 1
            assignments[node] = partition
            partition += 1
        else:
            assignments[node] = partition
    return split_module(
        graph, None, lambda node: assignments[node], keep_original_order=True
    )


def test_dynamic_tensor_index_no_longer_creates_partition_cycle():
    graph = fx.symbolic_trace(lambda x: (x + 1)[(x + 1).argmax()])
    # Reuse the same producer, as Dynamo does.
    add_nodes = [n for n in graph.graph.nodes if n.target is operator.add]
    add_nodes[1].replace_all_uses_with(add_nodes[0])
    graph.graph.erase_node(add_nodes[1])
    graph.recompile()
    with pytest.raises(RuntimeError, match="cycle exists between partitions"):
        _split_with_producer_getitem_policy(graph, [])
    split = _HELPERS["make_txda_graph_splitter"](_split_with_producer_getitem_policy)
    result = split(graph, [])
    for values in ([1.0, 3.0, -2.0], [5.0, 1.0, 4.0]):
        value = torch.tensor(values)
        torch.testing.assert_close(result(value), graph(value))
    assert any(n.target is operator.getitem for n in graph.graph.nodes)


@pytest.mark.parametrize(
    "value,index",
    [
        ((10, 20, 30), 1),
        ([10, 20, 30], slice(0, 3, 2)),
        (torch.arange(12).reshape(3, 4), (slice(None), slice(1, 4, 2))),
        (torch.arange(12).reshape(3, 4), torch.tensor([2, 0])),
    ],
)
def test_eager_indexing_preserves_python_and_tensor_semantics(value, index):
    graph = fx.Graph()
    value_node = graph.placeholder("value")
    index_node = graph.placeholder("index")
    item = graph.call_function(operator.getitem, (value_node, index_node))
    item.meta["test_metadata"] = "retained"
    graph.output(item)
    module = fx.GraphModule({}, graph)
    module.meta["test_graph_metadata"] = "retained"
    original = Mock(side_effect=lambda gm, ops: gm)
    rewritten = _HELPERS["make_txda_graph_splitter"](original)(module, ["attention"])
    assert rewritten is not module
    assert rewritten.meta == module.meta
    new_item = next(n for n in rewritten.graph.nodes if n.op == "call_function")
    assert new_item.meta == item.meta
    assert is_boundary(new_item)
    assert item.target is operator.getitem
    assert original.call_args.args[1] == ["attention"]
    result = rewritten(value, index)
    expected = operator.getitem(value, index)
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(result, expected)
    else:
        assert result == expected


def test_index_rewrite_shares_tensor_attributes_without_copying_storage():
    root = torch.nn.Module()
    root.register_buffer("weight", torch.arange(4))
    graph = fx.Graph()
    index = graph.placeholder("index")
    weight = graph.get_attr("weight")
    result = graph.call_function(operator.getitem, (weight, index))
    graph.output(result)
    module = fx.GraphModule(root, graph)
    rewritten = _HELPERS["make_txda_graph_splitter"](lambda gm, ops: gm)(module, [])
    assert rewritten.weight is module.weight
    assert rewritten(2).item() == 2


def test_graph_without_indexing_keeps_original_identity():
    graph = fx.symbolic_trace(lambda x: x * 2 + 1)
    original = Mock(return_value="split-result")
    assert _HELPERS["make_txda_graph_splitter"](original)(graph, []) == "split-result"
    original.assert_called_once_with(graph, [])


def test_size_decomposition_runs_before_index_rewrite_on_cloned_ir():
    graph = fx.symbolic_trace(lambda value: value.size()[0])

    def decompose(module):
        size = next(n for n in module.graph.nodes if n.target == "size")
        getitem = next(n for n in module.graph.nodes if n.target is operator.getitem)
        assert getitem.args == (size, 0)
        getitem.replace_all_uses_with(3)
        module.graph.erase_node(getitem)
        module.graph.erase_node(size)

    original = Mock(side_effect=lambda module, ops: module)
    result = _HELPERS["make_txda_graph_splitter"](original, decompose)(graph, [])
    assert result(torch.ones(3, 4)) == 3
    assert any(n.target is operator.getitem for n in graph.graph.nodes)


def test_eager_codegen_adapter_preserves_methods_globals_and_kwargs():
    native = fx.symbolic_trace(lambda value, scale=2: value.float() * scale)
    root = torch.nn.Module()
    root.native = native
    graph = fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_module("native", (value,), {"scale": 3})
    graph.output(result)
    module = fx.GraphModule(root, graph)
    original = Mock(return_value="generated-code")
    adapt = _HELPERS["make_txda_codegen_adapter"](original)
    assert adapt(module) == "generated-code"
    assert not isinstance(module.native, fx.GraphModule)
    assert module.native.graph is native
    value = torch.tensor([1, 2], dtype=torch.bfloat16)
    torch.testing.assert_close(module(value), value.float() * 3)
    wrapped = module.native
    adapt(module)
    assert module.native is wrapped


def test_codegen_adapter_keeps_compiled_piecewise_callables():
    original_graph = fx.symbolic_trace(lambda value: value * 2)
    root = torch.nn.Module()
    root.compiled = original_graph
    graph = fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_module("compiled", (value,))
    graph.output(result)
    module = fx.GraphModule(root, graph)
    compiled = Mock(side_effect=lambda value: value * 2)
    # vLLM shadows the original registered FX module with PiecewiseBackend.
    module.__dict__["compiled"] = compiled
    _HELPERS["make_txda_codegen_adapter"](lambda graph: None)(module)
    assert module.compiled is compiled
    assert module._modules["compiled"] is original_graph
    value = torch.tensor([1.0, 3.0])
    torch.testing.assert_close(module(value), value * 2)


def test_eager_codegen_adapter_retains_mutation_and_output_aliases():
    graph = fx.symbolic_trace(lambda value: value.add_(1))
    region = _HELPERS["_TxdaEagerRegion"](graph)
    value = torch.tensor([2.0, 3.0])
    result = region(value)
    assert result is value
    torch.testing.assert_close(value, torch.tensor([3.0, 4.0]))
