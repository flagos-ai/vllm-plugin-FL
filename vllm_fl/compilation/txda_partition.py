# Copyright (c) 2026 BAAI. All rights reserved.
"""Conservative TXDA boundaries for add/mul piecewise graph capture.

TXDA native operations are not generally captured, and the current SDK cannot
replay a graph with multiple kernels. Only these known elementwise operations
and metadata-only views may enter the single-kernel graph sequence wrapper.
"""

import builtins
import operator
from copy import deepcopy
from functools import wraps
from types import CodeType, MethodType

import torch
from torch import fx

_ELEMENTWISE_FUNCTIONS = frozenset({operator.add, operator.mul, torch.add, torch.mul})
_ELEMENTWISE_METHODS = frozenset({"add", "mul"})
_VIEW_METHODS = frozenset(
    {
        "view",
        "view_as",
        "transpose",
        "permute",
        "t",
        "unsqueeze",
        "squeeze",
        "expand",
        "expand_as",
        "detach",
    }
)
_METADATA_METHODS = frozenset(
    {"size", "dim", "numel", "stride", "storage_offset", "is_contiguous"}
)
_VIEW_FUNCTIONS = frozenset(
    {
        torch.transpose,
        torch.permute,
        torch.t,
        torch.unsqueeze,
        torch.squeeze,
        torch.detach,
    }
)
_SAFE_ATEN_OPS = frozenset(
    {
        "aten::add",
        "aten::mul",
        "aten::view",
        "aten::_unsafe_view",
        "aten::transpose",
        "aten::permute",
        "aten::t",
        "aten::unsqueeze",
        "aten::squeeze",
        "aten::expand",
        "aten::detach",
        "aten::alias",
        "aten::sym_size",
        "aten::sym_stride",
        "aten::sym_numel",
        "aten::sym_storage_offset",
        "aten::size",
        "aten::stride",
        "aten::numel",
        "aten::dim",
        "aten::storage_offset",
        "aten::is_contiguous",
    }
)
_MISSING = object()


def _example_value(value):
    if isinstance(value, fx.Node):
        return value.meta.get("example_value", value.meta.get("val", _MISSING))
    return value


def _safe_constant(value) -> bool:
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        return True
    if value is None or type(value) in (
        bool,
        int,
        float,
        str,
        torch.dtype,
        torch.device,
    ):
        return True
    return type(value) in (tuple, torch.Size) and all(
        _safe_constant(item) for item in value
    )


def _attribute_value(node: fx.Node):
    module = node.graph.owning_module
    if module is None:
        return _MISSING
    value = module
    for name in str(node.target).split("."):
        if not isinstance(value, torch.nn.Module):
            return _MISSING
        fields = vars(value)
        value = fields.get(name, _MISSING)
        if value is _MISSING:
            for category in ("_parameters", "_buffers", "_modules"):
                if name in fields.get(category, {}):
                    value = fields[category][name]
                    break
    return value


def _tensor_receiver(value) -> bool:
    example = _example_value(value)
    if isinstance(example, torch.Tensor):
        return True
    if not isinstance(value, fx.Node):
        return type(value) in (torch.Tensor, torch.nn.Parameter)
    if value.op == "get_attr":
        return type(_attribute_value(value)) in (torch.Tensor, torch.nn.Parameter)
    if value.meta.get("tensor_meta") is not None or value.type is torch.Tensor:
        return True
    if value.op == "call_function" and value.target in _ELEMENTWISE_FUNCTIONS:
        return any(_tensor_receiver(arg) for arg in value.args)
    if value.op == "call_method" and value.target in (
        _ELEMENTWISE_METHODS | _VIEW_METHODS
    ):
        return bool(value.args) and _tensor_receiver(value.args[0])
    return False


def _safe_operand(value) -> bool:
    if not isinstance(value, fx.Node):
        return _safe_constant(value)
    if value.op == "get_attr":
        return _safe_constant(_attribute_value(value))
    example = _example_value(value)
    if example is not _MISSING:
        return isinstance(example, torch.Tensor) or _safe_constant(example)
    # Inputs without metadata are checked against tensors/immutable scalars by
    # the graph wrapper's staging helper before any captured call executes.
    return value.op == "placeholder" or is_txda_graph_safe_node(value)


def _is_sequence_extraction(node: fx.Node) -> bool:
    if len(node.args) != 2:
        return False
    sequence, index = map(_example_value, node.args)
    # Tensor indexing can launch native device work. Tuple/list/Size extraction
    # only selects an existing Python object and does not execute a device op.
    return isinstance(sequence, (tuple, list, torch.Size)) and isinstance(
        index, (int, slice)
    )


def is_txda_graph_safe_node(node: fx.Node) -> bool:
    """Allow only known add/mul computations and metadata-only operations."""
    if node.op in ("placeholder", "output"):
        return True
    if node.op == "get_attr":
        return _safe_constant(_attribute_value(node))
    if node.op == "call_method":
        return (
            node.target in (_ELEMENTWISE_METHODS | _VIEW_METHODS | _METADATA_METHODS)
            and bool(node.args)
            and _tensor_receiver(node.args[0])
            and all(_safe_operand(value) for value in node.args[1:])
            and all(_safe_operand(value) for value in node.kwargs.values())
        )
    if node.op != "call_function":
        return False
    target = node.target
    if node.kwargs.get("out") is not None:
        return False
    if target is operator.getitem:
        return _is_sequence_extraction(node)
    if target is builtins.getattr:
        return (
            len(node.args) == 2
            and isinstance(_example_value(node.args[0]), torch.Tensor)
            and node.args[1] in {"shape", "dtype", "device", "ndim"}
        )
    if isinstance(target, torch._ops.OpOverloadPacket):
        return target._qualified_op_name in _SAFE_ATEN_OPS
    if isinstance(target, torch._ops.OpOverload):
        return not target._schema.is_mutable and target._schema.name in _SAFE_ATEN_OPS
    return (target in _ELEMENTWISE_FUNCTIONS or target in _VIEW_FUNCTIONS) and (
        all(_safe_operand(value) for value in node.args)
        and all(_safe_operand(value) for value in node.kwargs.values())
    )


def is_txda_native_graph_boundary(node: fx.Node) -> bool:
    """Keep every other computation eager, including opaque custom operators."""
    return node.op in ("call_function", "call_method", "call_module") and not (
        is_txda_graph_safe_node(node)
    )


def make_txda_split_predicate(original, device_type: str):
    """Preserve vLLM boundaries and extend them only for TXDA.

    vLLM imports its predicate into compilation.backends by name. Install this
    function at that consumer when enabling the limited TXDA graph path. The
    graph wrapper must also validate each resulting region: upstream assigns
    non-placeholder getitem nodes to their producer without consulting this
    predicate, and native Tensor indexing must never slip into capture.
    """

    @wraps(original)
    def should_split(node, splitting_ops):
        return original(node, splitting_ops) or (
            device_type == "txda" and is_txda_native_graph_boundary(node)
        )

    return should_split


def _transparent_fx_forward(graph) -> bool:
    from torch.fx._lazy_graph_module import _LazyGraphModule
    from torch.fx.graph_module import _WrappedCall

    if type(graph).__bases__ not in ((fx.GraphModule,), (_LazyGraphModule,)):
        return False
    if "forward" in vars(graph) or graph._forward_hooks or graph._forward_pre_hooks:
        return False
    if graph._compiled_call_impl is not None:
        return False
    # Accessing code materializes a normal LazyGraphModule's generated forward.
    generated = compile(graph.code, "<txda-fx-validation>", "exec", dont_inherit=True)
    expected = next(
        (
            value
            for value in generated.co_consts
            if isinstance(value, CodeType) and value.co_name == "forward"
        ),
        None,
    )
    actual = getattr(type(graph).forward, "__code__", None)
    if expected is None or actual is None:
        return False
    if any(
        getattr(actual, field) != getattr(expected, field)
        for field in ("co_code", "co_names", "co_varnames", "co_consts")
    ):
        return False
    call_code = getattr(type(graph).__call__, "__code__", None)
    known_calls = [
        value
        for value in fx.GraphModule.recompile.__code__.co_consts
        if isinstance(value, CodeType) and value.co_name == "call_wrapped"
    ]
    wrapped = type(graph)._wrapped_call
    return (
        type(wrapped) is _WrappedCall
        and call_code in known_calls
        and wrapped.cls is type(graph)
        and wrapped.cls_call is None
    )


def validate_txda_graph_module(graph) -> tuple[bool, str]:
    """Validate every FX node, including getitem nodes moved by vLLM's splitter."""
    if not isinstance(graph, fx.GraphModule):
        return False, "expected a torch.fx.GraphModule"
    if not _transparent_fx_forward(graph):
        return (
            False,
            "FX callable overrides or hooks are not supported for TXDA capture",
        )
    for node in graph.graph.nodes:
        if not is_txda_graph_safe_node(node):
            return False, f"unsupported TXDA graph node {node.op}: {node.target}"
    return True, ""


def validate_txda_graph_callable(runnable) -> tuple[bool, str]:
    """Accept only transparent GraphModules or their bound forward methods.

    Adapters such as vLLM PiecewiseBackend must be checked by their owner and
    pass both their graph and their compiled range callables through this API.
    Attribute names alone do not establish that an opaque callable is safe.
    """
    if isinstance(runnable, fx.GraphModule):
        return validate_txda_graph_module(runnable)
    if (
        isinstance(runnable, MethodType)
        and isinstance(runnable.__self__, fx.GraphModule)
        and runnable.__func__ is type(runnable.__self__).forward
    ):
        return validate_txda_graph_module(runnable.__self__)
    return False, "opaque callable is not a validated FX GraphModule"


def _txda_eager_getitem(value, index):
    return operator.getitem(value, index)


def make_txda_graph_splitter(original, decompose_sizes=None):
    """Keep indexing in graph order instead of moving it to its producer.

    vLLM moves every operator.getitem to the producer's partition, assuming
    tuple extraction. Tensor indexing may also depend on an index computed in
    a later partition, creating a dependency cycle. Keep the same operation
    behind an eager helper so vLLM applies its normal sequential partitioning.
    The TXDA eager compiler accepts tuple inputs across these boundaries.
    """

    @wraps(original)
    def split_graph(graph, splitting_ops):
        if not any(
            node.op == "call_function" and node.target is operator.getitem
            for node in graph.graph.nodes
        ):
            return original(graph, splitting_ops)
        # Clone only IR, keeping parameters/modules shared. vLLM retains the
        # incoming graph separately; do not rewrite that diagnostic artifact.
        copied_module = fx.GraphModule(graph, deepcopy(graph.graph))
        copied_module.meta = graph.meta.copy()
        # The upstream size pass recognizes operator.getitem when replacing
        # size()[dim]. Preserve that transformation before hiding indexing.
        if decompose_sizes is not None:
            decompose_sizes(copied_module)
        for node in copied_module.graph.nodes:
            if node.op == "call_function" and node.target is operator.getitem:
                node.target = _txda_eager_getitem
        copied_module.recompile()
        return original(copied_module, splitting_ops)

    return split_graph


class _TxdaEagerRegion(torch.nn.Module):
    """Keep native FX execution out of vLLM's restricted inline codegen."""

    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def forward(self, *args, **kwargs):
        return self.graph(*args, **kwargs)


def make_txda_codegen_adapter(original):
    """Bind eager FX regions as callables, retaining their native operations.

    vLLM's stitching codegen only inlines call_function nodes and imports
    torch/operator. Native methods and plugin indexing helpers must retain
    the original FX forward and its globals. Compiled pieces have already
    been replaced by PiecewiseBackend/GraphWrapper and are left untouched.
    """

    @wraps(original)
    def generate_execution_code(split_graph):
        for name, _ in list(split_graph.named_children()):
            module = getattr(split_graph, name)
            if isinstance(module, fx.GraphModule):
                setattr(split_graph, name, _TxdaEagerRegion(module))
        return original(split_graph)

    return generate_execution_code


def install_txda_graph_partition(device_type: str) -> None:
    """Install once, only when PlatformFL enables the limited TXDA graph path."""
    if device_type != "txda":
        return
    from vllm.compilation import backends

    original = backends.should_split
    if getattr(original, "_vllm_fl_txda_partition", False) is not True:
        predicate = make_txda_split_predicate(original, device_type)
        predicate._vllm_fl_txda_partition = True
        backends.should_split = predicate
    original_split = backends.split_graph
    if getattr(original_split, "_vllm_fl_txda_partition", False) is not True:
        splitter = make_txda_graph_splitter(
            original_split, backends._decompose_size_nodes
        )
        splitter._vllm_fl_txda_partition = True
        backends.split_graph = splitter
    original_codegen = backends.generate_execution_code
    if getattr(original_codegen, "_vllm_fl_txda_partition", False) is not True:
        codegen = make_txda_codegen_adapter(original_codegen)
        codegen._vllm_fl_txda_partition = True
        backends.generate_execution_code = codegen
