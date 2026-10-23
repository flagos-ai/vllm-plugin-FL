"""Generic runtime-kernel to operator identity rules."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

OperatorIdentity = tuple[str, ...]


@dataclass(frozen=True)
class KernelSignature:
    callable_name: str
    symbol: str
    template_args: tuple[str, ...]


@dataclass(frozen=True)
class KernelGroupingRule:
    name: str
    match: Callable[[KernelSignature], bool]
    key: Callable[[KernelSignature], OperatorIdentity]


_COLLECTIVE_MARKERS = (
    "all_gather",
    "all_reduce",
    "allgather",
    "allreduce",
    "cross_device_reduce",
    "reduce_scatter",
    "reducescatter",
    "all_to_all",
    "alltoall",
)
_COMMUNICATION_BACKEND_MARKERS = (
    "flagcx",
    "nccl",
    "rccl",
    "symm_mem",
)
_DISTRIBUTED_NAMESPACE_MARKERS = (
    "c10d::",
    "collective",
    "distributed::",
    "processgroup",
)
_COMPUTE_MARKERS = (
    "fused",
    "fusion",
    "gemm",
    "matmul",
    "norm",
)
_STAGED_KERNEL = re.compile(
    r"^(?P<family>.+?)(?:_stage|::stage)_?\d+(?:_[A-Za-z0-9]+)*$",
    re.IGNORECASE,
)
_MOE_ALIGN_STAGE_KERNEL = re.compile(
    r"^moe_align_block_size_stage_?\d+(?:_[A-Za-z0-9]+)*$",
    re.IGNORECASE,
)

# These are concrete operations, not generic launch wrappers. Only their
# template specialization changes across the observed runtime kernels.
_DIRECT_KERNEL_FAMILIES = frozenset(
    {
        "cublasLt::splitKreduce_kernel",
        "marlin_moe_wna16::Marlin",
    }
)
_GENERIC_LAUNCH_WRAPPERS = frozenset({"cutlass::device_kernel"})

# Add a rule here only when the discarded arguments are known specializations.
# The rules are kernel-family based and contain no model-specific names.
KERNEL_GROUPING_RULES = (
    KernelGroupingRule(
        "deep_gemm_callable",
        lambda s: s.symbol.startswith("deep_gemm::"),
        lambda s: ("kernel_family", s.symbol),
    ),
    KernelGroupingRule(
        "direct_kernel_specializations",
        lambda s: s.symbol in _DIRECT_KERNEL_FAMILIES and bool(s.template_args),
        lambda s: ("kernel_family", s.symbol),
    ),
)


def _combined_name(operator_name: str, kernel_name: str) -> str:
    return f"{operator_name} {kernel_name}".lower()


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def is_fused_communication_compute(operator_name: str, kernel_name: str) -> bool:
    """Return whether a collective also performs non-communication compute."""
    combined = _combined_name(operator_name, kernel_name)
    return _contains_any(combined, _COLLECTIVE_MARKERS) and _contains_any(
        combined, _COMPUTE_MARKERS
    )


def is_pure_communication(operator_name: str, kernel_name: str) -> bool:
    """Classify communication from generic collective/backend name markers."""
    if is_fused_communication_compute(operator_name, kernel_name):
        return False
    combined = _combined_name(operator_name, kernel_name)
    is_collective = _contains_any(
        combined, _COLLECTIVE_MARKERS + _COMMUNICATION_BACKEND_MARKERS
    )
    is_distributed_broadcast = "broadcast" in combined and _contains_any(
        combined,
        _DISTRIBUTED_NAMESPACE_MARKERS + _COMMUNICATION_BACKEND_MARKERS,
    )
    return is_collective or is_distributed_broadcast


def kernel_callable_identity_name(kernel_name: str) -> str:
    """Return a callable identity, preserving semantic template types."""
    name = kernel_name.strip()
    if name.startswith("void "):
        name = name[5:].lstrip()
    if name.endswith(")"):
        depth = 0
        for index in range(len(name) - 1, -1, -1):
            character = name[index]
            if character == ")":
                depth += 1
            elif character == "(":
                depth -= 1
                if depth == 0:
                    name = name[:index].rstrip()
                    break

    template_start = name.find("<")
    if template_start >= 0:
        prefix = re.sub(r"_rank_\d+$", "", name[:template_start])
        return prefix + name[template_start:]
    return re.sub(r"_rank_\d+$", "", name)


def kernel_signature(kernel_name: str) -> KernelSignature:
    """Split only top-level C++ template arguments, retaining nested types."""
    callable_name = kernel_callable_identity_name(kernel_name)
    template_start = callable_name.find("<")
    if template_start < 0:
        return KernelSignature(callable_name, callable_name, ())

    symbol = callable_name[:template_start]
    arguments: list[str] = []
    start = template_start + 1
    angle_depth = 1
    paren_depth = bracket_depth = brace_depth = 0
    for index in range(start, len(callable_name)):
        character = callable_name[index]
        if character == "<":
            angle_depth += 1
        elif character == ">":
            angle_depth -= 1
            if angle_depth == 0:
                if index != len(callable_name) - 1:
                    break
                arguments.append(callable_name[start:index].strip())
                return KernelSignature(callable_name, symbol, tuple(arguments))
        elif character == "(":
            paren_depth += 1
        elif character == ")":
            paren_depth -= 1
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth -= 1
        elif character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
        elif (
            character == ","
            and angle_depth == 1
            and paren_depth == bracket_depth == brace_depth == 0
        ):
            arguments.append(callable_name[start:index].strip())
            start = index + 1

    # An unfamiliar demangled form must not be grouped on a partial parse.
    return KernelSignature(callable_name, callable_name, ())


def specialization_identity(signature: KernelSignature) -> OperatorIdentity | None:
    matches = [rule for rule in KERNEL_GROUPING_RULES if rule.match(signature)]
    if len(matches) > 1:
        raise ValueError(
            f"overlapping kernel grouping rules: {signature.callable_name}"
        )
    return matches[0].key(signature) if matches else None


def staged_kernel_family_name(kernel_name: str) -> str | None:
    """Return the common callable for a numbered multi-stage kernel family."""
    callable_name = kernel_callable_identity_name(kernel_name)
    if _MOE_ALIGN_STAGE_KERNEL.fullmatch(callable_name):
        return None
    match = _STAGED_KERNEL.fullmatch(callable_name)
    return match.group("family") if match else None


def _operator_kind(source_operator: str, kernel_name: str) -> str:
    if source_operator.startswith("aten::"):
        return "aten"
    if source_operator.startswith("triton_") or kernel_name.startswith("triton_"):
        return "triton_compiled"
    if source_operator == "null":
        return "unattributed"
    if "::" in source_operator:
        return "custom"
    return "runtime_operator"


def operator_descriptor(
    source_operator: str, kernel_name: str
) -> tuple[str, str, OperatorIdentity]:
    """Map an operator/kernel pair without model- or kernel-specific tables."""
    callable_name = kernel_callable_identity_name(kernel_name)
    if callable_name.startswith(("nvjet_tst_", "nvjet_tss_")):
        return (
            "aten::mm",
            "aten",
            ("operator", "aten", "aten::mm"),
        )
    if kernel_name.strip().startswith("void deep_gemm::"):
        signature = kernel_signature(kernel_name)
        return (
            source_operator,
            _operator_kind(source_operator, kernel_name),
            specialization_identity(signature) or ("kernel_callable", callable_name),
        )
    if _MOE_ALIGN_STAGE_KERNEL.fullmatch(callable_name):
        return (
            callable_name,
            "custom",
            ("moe_align_block_size_stage", callable_name),
        )
    if is_fused_communication_compute(source_operator, kernel_name):
        return (
            source_operator,
            "fused_communication_compute",
            ("fused_communication_compute", callable_name),
        )
    if is_pure_communication(source_operator, kernel_name):
        return (
            source_operator,
            "communication",
            ("communication_kernel", callable_name),
        )

    staged_family = staged_kernel_family_name(kernel_name)
    if staged_family is not None:
        return (
            source_operator,
            _operator_kind(source_operator, kernel_name),
            ("staged_kernel_family", staged_family),
        )

    operator_kind = _operator_kind(source_operator, kernel_name)
    if operator_kind in {"aten", "runtime_operator"}:
        identity = ("operator", operator_kind, source_operator)
    elif operator_kind == "triton_compiled":
        identity = ("compile_kernel", kernel_name)
    else:
        signature = kernel_signature(kernel_name)
        identity = specialization_identity(signature)
        if identity is None and operator_kind == "custom":
            if signature.symbol not in _GENERIC_LAUNCH_WRAPPERS:
                identity = ("api_kernel_family", source_operator, signature.symbol)
            else:
                identity = ("api_kernel_callable", source_operator, callable_name)
        if identity is None:
            identity = ("kernel_callable", callable_name)
    return source_operator, operator_kind, identity
