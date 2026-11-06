"""Auditable rules for FlagOS operator coverage classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TRITON_OPERATOR_KINDS = frozenset({"torch_compile", "triton_compiled"})

# Explicit fused implementation to baseline API mappings. A rule is usable
# only when its evidence token is present in the captured enable-op list.
FLAGGEMS_FUSED_API_RULES = {
    "flag_gems.fused.fused_moe.": frozenset({"vllm::moe_forward_shared"}),
    "flag_gems.fused.moe_align_block_size.": frozenset(
        {"_moe_C::moe_align_block_size", "moe_align_block_size"}
    ),
    "flag_gems.fused.topk_softmax.": frozenset({"_moe_C::topk_softmax"}),
}


@dataclass(frozen=True)
class CoverageDecision:
    covered: bool | None
    flagos_type: str
    evidence: str


@dataclass(frozen=True)
class FlagOSEvidence:
    aten_apis: frozenset[str]
    fused_apis: frozenset[str]
    lines: tuple[str, ...]
    available: bool = True


def registered_aten_apis(
    module_name: str, callable_name: str, registry: dict
) -> set[str]:
    """Resolve a recorded FlagGems callable through its ATen registrations."""
    module = f"flag_gems.ops.{module_name}"
    return {
        "aten::" + registration[0].split(".", 1)[0]
        for registration in registry.get(callable_name, ())
        if getattr(registration[1], "__module__", None) == module
    }


def read_flagos_evidence(path: Path | None) -> FlagOSEvidence:
    """Parse runtime evidence using FlagGems' actual ATen registration table."""
    if path is None:
        return FlagOSEvidence(frozenset(), frozenset(), (), available=False)
    from flag_gems import FULL_CONFIG_BY_FUNC

    lines = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    )
    aten_apis: set[str] = set()
    for line in lines:
        match = re.search(r"flag_gems\.ops\.([^.]+)\.([A-Za-z0-9_]+)", line)
        if not match:
            continue
        module_name, callable_name = match.groups()
        aten_apis.update(
            registered_aten_apis(module_name, callable_name, FULL_CONFIG_BY_FUNC)
        )

    fused_apis: set[str] = set()
    for token, api_names in FLAGGEMS_FUSED_API_RULES.items():
        if any(token in line for line in lines):
            fused_apis.update(api_names)
    return FlagOSEvidence(frozenset(aten_apis), frozenset(fused_apis), lines)


def is_triton_operator(
    operator_names: set[str], operator_kinds: set[str], kernel_names: set[str]
) -> bool:
    return bool(operator_kinds & TRITON_OPERATOR_KINDS) or any(
        name.startswith("triton_") for name in operator_names | kernel_names
    )


def classify_coverage(
    operator_names: set[str],
    operator_kinds: set[str],
    kernel_names: set[str],
    evidence: FlagOSEvidence,
) -> CoverageDecision:
    """Classify one normalized baseline operator.

    The project-specific policy counts every observed Triton operator in the
    numerator. Other operators require runtime evidence. Communication is not
    covered unless a future FlagCX evidence rule is added here.
    """
    if "communication" in operator_kinds:
        return CoverageDecision(
            False,
            "none",
            "communication is in the denominator; no FlagCX evidence",
        )
    if is_triton_operator(operator_names, operator_kinds, kernel_names):
        return CoverageDecision(
            True,
            "triton",
            "policy: all observed Triton operators count as FlagOS coverage",
        )
    matched_aten = sorted(operator_names & set(evidence.aten_apis))
    if matched_aten:
        return CoverageDecision(
            True,
            "flaggems",
            "flaggems_enable_oplist: " + ",".join(matched_aten),
        )
    matched_fused = sorted(operator_names & set(evidence.fused_apis))
    if matched_fused:
        return CoverageDecision(
            True,
            "flaggems",
            "flaggems fused mapping: " + ",".join(matched_fused),
        )
    if "aten" in operator_kinds:
        if not evidence.available:
            return CoverageDecision(
                None,
                "",
                "FlagGems enable-op evidence was not provided",
            )
        return CoverageDecision(
            False,
            "none",
            "ATen API is absent from flaggems_enable_oplist",
        )
    return CoverageDecision(
        None,
        "",
        "insufficient evidence to determine FlagOS coverage",
    )
