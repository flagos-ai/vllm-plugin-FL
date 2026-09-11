#!/usr/bin/env python3
"""Extract rank-scoped runtime kernel summaries from PyTorch traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
MetadataKey = tuple[str, str | None, str | None, str]
MappingKey = tuple[str, str | None, str, str, str | None]
OperatorIdentity = tuple[str, ...]

VOCAB_MASK_COMPILE_FUNCTION = (
    "vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask"
)
VOCAB_MASK_TRITON_KERNELS = frozenset(
    {
        "triton_poi_fused___and_____or___add_bitwise_not_ge_lt_mul_sub_0",
        "triton_poi_fused___and_____or___add_ge_lt_mul_sub_0",
        "triton_poi_fused___and_____or___bitwise_not_ge_lt_1",
        "triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0",
        "triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0",
        "triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1",
    }
)

MOE_ALIGN_BLOCK_SIZE_KERNEL = re.compile(
    r"^moe_align_block_size_stage\d+(?:_[A-Za-z0-9]+)*$"
)
MOE_ALIGN_BLOCK_SIZE_OPERATOR = "moe_align_block_size"

PURE_COMMUNICATION_OPERATORS = frozenset(
    {
        "_C_custom_ar::all_reduce",
        "symm_mem::one_shot_all_reduce",
        "symm_mem::one_shot_all_reduce_",
        "symm_mem::two_shot_all_reduce_",
        "symm_mem::two_shot_all_reduce_out",
        "symm_mem::multimem_all_reduce_",
        "symm_mem::multimem_one_shot_all_reduce",
    }
)
FUSED_COMMUNICATION_COMPUTE_OPERATORS = frozenset(
    {
        "vllm::flashinfer_trtllm_fused_allreduce_norm",
    }
)


def is_fused_communication_compute(operator_name: str, kernel_name: str) -> bool:
    if operator_name in FUSED_COMMUNICATION_COMPUTE_OPERATORS:
        return True
    lowered = kernel_name.lower()
    return any(
        token in lowered
        for token in (
            "allreduce_fusion_kernel",
            "fused_all_gather_matmul",
            "fused_all_gather_scaled_matmul",
            "fused_matmul_reduce_scatter",
            "fused_scaled_matmul_reduce_scatter",
        )
    )


def is_pure_communication(operator_name: str, kernel_name: str) -> bool:
    if operator_name in PURE_COMMUNICATION_OPERATORS:
        return True
    if kernel_name.startswith("ncclDevKernel_"):
        return True
    lowered = kernel_name.lower()
    return any(
        token in lowered
        for token in (
            "cross_device_reduce_",
            "one_shot_all_reduce_kernel",
            "two_shot_all_reduce_kernel",
            "multimem_all_reduce_kernel",
        )
    )


def kernel_callable_identity_name(kernel_name: str) -> str:
    """Return a demangled kernel's namespace-qualified callable name."""
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

    normalized: list[str] = []
    template_depth = 0
    for character in name:
        if character == "<":
            template_depth += 1
        elif character == ">" and template_depth:
            template_depth -= 1
        elif template_depth == 0:
            normalized.append(character)
    return "".join(normalized) if template_depth == 0 else name


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        prefix = source.read(4096)
        source.seek(0)
        if '"traceEvents"' not in prefix:
            yield from json.load(source).get("traceEvents", [])
            return
        for line in source:
            if '"traceEvents"' in line:
                break
        else:
            raise ValueError(f"traceEvents not found: {path}")
        event_lines: list[str] = []
        for line in source:
            if not event_lines:
                if line.startswith("  {"):
                    event_lines.append(line)
                elif line.lstrip().startswith("]"):
                    return
                continue
            event_lines.append(line)
            if line.startswith("  }"):
                encoded = "".join(event_lines).rstrip().removesuffix(",")
                yield json.loads(encoded)
                event_lines.clear()


def trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.pt.trace.json.gz"))
    files.extend(sorted(path.rglob("*.pt.trace.json")))
    return files


def runtime_trace_files(path: Path) -> list[Path]:
    return [
        file for file in trace_files(path) if not file.name.startswith("graph_capture_")
    ]


def rank_in_filename(file: Path) -> int:
    match = re.search(r"(?:^|_)rank_?(\d+)(?:[._]|$)", file.name)
    return int(match.group(1)) if match else -1


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def decode(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def duration_ns(event: dict[str, Any]) -> int:
    return round(float(event.get("dur", 0.0)) * 1000)


def ns_to_us(value: int) -> float:
    return value / 1000


def format_percent(value_ns: int, total_ns: int) -> str:
    percent = value_ns / total_ns * 100 if total_ns else 0.0
    if percent < 0.001:
        return "<0.001%"
    return f"{percent:.3f}%"


def metadata(event: dict[str, Any]) -> MetadataKey:
    args = event.get("args", {})
    has_shapes = "Input Dims" in args
    has_dtypes = "Input type" in args
    shapes = canonical(args["Input Dims"]) if has_shapes else None
    dtypes = canonical(args["Input type"]) if has_dtypes else None
    status = (
        "shape_and_dtype"
        if has_shapes and has_dtypes
        else "shape_only"
        if has_shapes
        else "dtype_only"
        if has_dtypes
        else "no_input_metadata"
    )
    return str(event.get("name", "")), shapes, dtypes, status


def external_id(event: dict[str, Any]) -> int | str | None:
    value = event.get("args", {}).get("External id")
    return value if isinstance(value, (int, str)) else None


def mapping(
    event_external_id: int | str | None,
    cpu_by_external_id: dict[int | str, set[MetadataKey]],
) -> dict[str, Any]:
    if event_external_id is None:
        return {
            "mapping_status": "missing_external_id",
            "operator": None,
            "input_shapes": None,
            "input_dtypes": None,
        }
    candidates = sorted(
        cpu_by_external_id.get(event_external_id, set()),
        key=lambda item: (
            item[0],
            item[1] or "",
            item[2] or "",
            item[3],
        ),
    )
    if not candidates:
        return {
            "mapping_status": "no_cpu_op_match",
            "operator": None,
            "input_shapes": None,
            "input_dtypes": None,
        }
    if len(candidates) > 1:
        candidate_rows = [
            {
                "operator": item[0],
                "input_shapes": decode(item[1]),
                "input_dtypes": decode(item[2]),
                "metadata_status": item[3],
            }
            for item in candidates
        ]
        names = {item[0] for item in candidates}
        return {
            "mapping_status": (
                "shape_ambiguous" if len(names) == 1 else "operator_ambiguous"
            ),
            "operator": next(iter(names)) if len(names) == 1 else None,
            "input_shapes": None,
            "input_dtypes": None,
            "candidate_operators": candidate_rows,
        }

    operator, shapes, dtypes, metadata_status = candidates[0]
    status_by_metadata = {
        "shape_and_dtype": "operator_shape_matched",
        "shape_only": "operator_matched_dtype_missing",
        "dtype_only": "operator_matched_shape_missing",
        "no_input_metadata": "operator_matched_metadata_missing",
    }
    return {
        "mapping_status": status_by_metadata[metadata_status],
        "operator": operator,
        "input_shapes": decode(shapes),
        "input_dtypes": decode(dtypes),
    }


def mapping_key(link: dict[str, Any]) -> MappingKey:
    candidates = link.get("candidate_operators")
    return (
        link["mapping_status"],
        link["operator"],
        canonical(link["input_shapes"]),
        canonical(link["input_dtypes"]),
        canonical(candidates) if candidates is not None else None,
    )


def variant_row(
    key: MappingKey,
    count: int,
    time_ns: int,
) -> dict[str, Any]:
    status, _operator, shapes, dtypes, candidates = key
    row = {
        "mapping_status": status,
        "input_shapes": decode(shapes),
        "input_dtypes": decode(dtypes),
        "kernel_event_count": count,
        "kernel_time_us": ns_to_us(time_ns),
    }
    if candidates is not None:
        row["candidate_operators"] = decode(candidates)
    return row


def collect_runtime(
    files: list[Path], rank: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cpu_event_count = 0
    cpu_metadata_count: Counter[str] = Counter()
    cpu_operator_names: set[str] = set()

    category_count: Counter[str] = Counter()
    category_ns: Counter[str] = Counter()
    kernel_count: Counter[str] = Counter()
    kernel_ns: Counter[str] = Counter()
    kernel_status_count: dict[str, Counter[str]] = defaultdict(Counter)
    kernel_status_ns: dict[str, Counter[str]] = defaultdict(Counter)
    kernel_variant_count: dict[str, Counter[MappingKey]] = defaultdict(Counter)
    kernel_variant_ns: dict[str, Counter[MappingKey]] = defaultdict(Counter)

    for trace in files:
        cpu_by_external_id: dict[int | str, set[MetadataKey]] = defaultdict(set)
        for event in iter_events(trace):
            if event.get("cat") != "cpu_op":
                continue
            cpu_event_count += 1
            key = metadata(event)
            cpu_operator_names.add(key[0])
            cpu_metadata_count[key[3]] += 1
            event_external_id = external_id(event)
            if event_external_id is not None:
                cpu_by_external_id[event_external_id].add(key)

        for event in iter_events(trace):
            category = str(event.get("cat", ""))
            if category not in GPU_CATEGORIES:
                continue
            event_ns = duration_ns(event)
            category_count[category] += 1
            category_ns[category] += event_ns
            if category != "kernel":
                continue

            name = str(event.get("name", ""))
            link = mapping(external_id(event), cpu_by_external_id)
            link_key = mapping_key(link)
            status = link["mapping_status"]
            kernel_count[name] += 1
            kernel_ns[name] += event_ns
            kernel_status_count[name][status] += 1
            kernel_status_ns[name][status] += event_ns
            kernel_variant_count[name][link_key] += 1
            kernel_variant_ns[name][link_key] += event_ns

    kernel_total_ns = category_ns["kernel"]
    ordered_names = sorted(kernel_count, key=lambda name: (-kernel_ns[name], name))
    kernel_summary: dict[str, Any] = {}
    kernel_report: dict[str, Any] = {}
    per_kernel_count_matches = True
    per_kernel_time_matches = True
    per_kernel_status_count_matches = True
    per_kernel_status_time_matches = True

    for name in ordered_names:
        total_ns = kernel_ns[name]
        kernel_summary[name] = {
            "total_call_count": kernel_count[name],
            "total_time_us": ns_to_us(total_ns),
            "percent": format_percent(total_ns, kernel_total_ns),
        }

        status_breakdown = {
            status: {
                "kernel_event_count": kernel_status_count[name][status],
                "kernel_time_us": ns_to_us(kernel_status_ns[name][status]),
            }
            for status in sorted(kernel_status_count[name])
        }
        operator_variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
        unattributed_variants: list[dict[str, Any]] = []
        for key in sorted(
            kernel_variant_count[name],
            key=lambda item: (
                -kernel_variant_ns[name][item],
                item[0],
                item[1] or "",
                item[2],
                item[3],
                item[4] or "",
            ),
        ):
            row = variant_row(
                key,
                kernel_variant_count[name][key],
                kernel_variant_ns[name][key],
            )
            operator = key[1]
            if operator is None:
                unattributed_variants.append(row)
            else:
                operator_variants[operator].append(row)

        kernel_report[name] = {
            "mapping_status_breakdown": status_breakdown,
            "operator_variants": dict(sorted(operator_variants.items())),
            "unattributed_variants": unattributed_variants,
        }
        variant_count = sum(kernel_variant_count[name].values())
        variant_ns = sum(kernel_variant_ns[name].values())
        status_count = sum(kernel_status_count[name].values())
        status_ns = sum(kernel_status_ns[name].values())
        per_kernel_count_matches &= variant_count == kernel_count[name]
        per_kernel_time_matches &= variant_ns == total_ns
        per_kernel_status_count_matches &= status_count == kernel_count[name]
        per_kernel_status_time_matches &= status_ns == total_ns

    summary_count = sum(item["total_call_count"] for item in kernel_summary.values())
    summary_ns = sum(kernel_ns.values())
    report_count = sum(
        sum(kernel_variant_count[name].values()) for name in kernel_report
    )
    report_ns = sum(sum(kernel_variant_ns[name].values()) for name in kernel_report)
    status_count = sum(
        sum(kernel_status_count[name].values()) for name in kernel_report
    )
    status_ns = sum(sum(kernel_status_ns[name].values()) for name in kernel_report)
    summary = {
        "scope": {
            "rank": rank,
            "phase": "runtime_only",
            "graph_capture_included": False,
            "timing_denominator": "sum_of_rank0_runtime_kernel_durations",
        },
        "runtime_trace_files": [str(file) for file in files],
        "cpu_operator_event_count": cpu_event_count,
        "unique_cpu_operator_names": len(cpu_operator_names),
        "cpu_operator_event_count_by_metadata_status": dict(
            sorted(cpu_metadata_count.items())
        ),
        "gpu_event_count": sum(category_count.values()),
        "gpu_event_count_by_category": dict(sorted(category_count.items())),
        "gpu_activity_us_by_category": {
            key: ns_to_us(value) for key, value in sorted(category_ns.items())
        },
        "kernel_event_count": category_count["kernel"],
        "unique_kernel_names": len(kernel_summary),
        "kernel_time_total_us": ns_to_us(kernel_total_ns),
        "kernel_mapping_event_count_by_status": {
            status: sum(rows[status] for rows in kernel_status_count.values())
            for status in sorted(
                {status for rows in kernel_status_count.values() for status in rows}
            )
        },
        "kernel_mapping_time_us_by_status": {
            status: ns_to_us(sum(rows[status] for rows in kernel_status_ns.values()))
            for status in sorted(
                {status for rows in kernel_status_ns.values() for status in rows}
            )
        },
        "conservation": {
            "kernel_key_sets_match": set(kernel_summary) == set(kernel_report),
            "kernel_event_count_in_trace": category_count["kernel"],
            "kernel_event_count_in_summary": summary_count,
            "kernel_event_count_in_report": report_count,
            "kernel_event_count_matches": (
                category_count["kernel"]
                == summary_count
                == report_count
                == status_count
            ),
            "kernel_time_us_in_trace": ns_to_us(kernel_total_ns),
            "kernel_time_us_in_summary": ns_to_us(summary_ns),
            "kernel_time_us_in_report": ns_to_us(report_ns),
            "kernel_time_matches": (
                kernel_total_ns == summary_ns == report_ns == status_ns
            ),
            "per_kernel_call_count_matches": per_kernel_count_matches,
            "per_kernel_time_matches": per_kernel_time_matches,
            "per_kernel_status_call_count_matches": per_kernel_status_count_matches,
            "per_kernel_status_time_matches": per_kernel_status_time_matches,
        },
    }
    return kernel_summary, kernel_report, summary


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summary_csv_rows(
    kernel_report: dict[str, Any], kernel_total_ns: int
) -> list[dict[str, Any]]:
    relation_count: Counter[tuple[str | None, str]] = Counter()
    relation_ns: Counter[tuple[str | None, str]] = Counter()
    for kernel_name, report in kernel_report.items():
        for operator, variants in report["operator_variants"].items():
            for row in variants:
                key = (operator, kernel_name)
                relation_count[key] += row["kernel_event_count"]
                relation_ns[key] += round(row["kernel_time_us"] * 1000)
        for row in report["unattributed_variants"]:
            key = (None, kernel_name)
            relation_count[key] += row["kernel_event_count"]
            relation_ns[key] += round(row["kernel_time_us"] * 1000)

    ordered = sorted(
        relation_count,
        key=lambda key: (
            key[0] is None,
            key[0] or "",
            -relation_ns[key],
            key[1],
        ),
    )
    return [
        {
            "operator_name": operator if operator is not None else "null",
            "kernel_name": kernel_name,
            "kernel_call_count": relation_count[(operator, kernel_name)],
            "kernel_time_us": ns_to_us(relation_ns[(operator, kernel_name)]),
            "percent": format_percent(
                relation_ns[(operator, kernel_name)], kernel_total_ns
            ),
        }
        for operator, kernel_name in ordered
    ]


def operator_descriptor(
    source_operator: str, kernel_name: str
) -> tuple[str, str, OperatorIdentity | None]:
    if MOE_ALIGN_BLOCK_SIZE_KERNEL.fullmatch(kernel_name):
        return (
            MOE_ALIGN_BLOCK_SIZE_OPERATOR,
            "custom",
            ("custom_group", MOE_ALIGN_BLOCK_SIZE_OPERATOR),
        )
    if is_fused_communication_compute(source_operator, kernel_name):
        return (
            source_operator,
            "fused_communication_compute",
            ("fused_communication_compute", kernel_name),
        )
    if is_pure_communication(source_operator, kernel_name):
        return (
            source_operator,
            "communication",
            ("communication_kernel", kernel_callable_identity_name(kernel_name)),
        )
    if kernel_name in VOCAB_MASK_TRITON_KERNELS:
        return (
            VOCAB_MASK_COMPILE_FUNCTION,
            "torch_compile",
            ("compile_kernel", kernel_name),
        )
    if source_operator == "null":
        if kernel_name.startswith("nvjet_tst_"):
            return (
                "null",
                "unattributed_nvjet",
                ("unattributed_nvjet",),
            )
        return (
            "null",
            "unattributed",
            ("kernel_callable", kernel_callable_identity_name(kernel_name)),
        )
    if source_operator.startswith("aten::"):
        operator_kind = "aten"
    elif source_operator.startswith("triton_"):
        operator_kind = "triton_compiled"
    elif "::" in source_operator:
        operator_kind = "custom"
    else:
        operator_kind = "runtime_operator"
    if operator_kind in {"aten", "runtime_operator"}:
        identity = ("operator", operator_kind, source_operator)
    elif operator_kind == "triton_compiled":
        identity = ("compile_kernel", kernel_name)
    else:
        identity = ("kernel_callable", kernel_callable_identity_name(kernel_name))
    return (source_operator, operator_kind, identity)


def operator_list_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations: dict[tuple[str, str, str], OperatorIdentity | None] = {}
    for summary_row in summary_rows:
        kernel_name = summary_row["kernel_name"]
        operator_name, operator_kind, identity = operator_descriptor(
            summary_row["operator_name"], kernel_name
        )
        relation = (operator_name, operator_kind, kernel_name)
        previous = relations.setdefault(relation, identity)
        if previous != identity:
            raise RuntimeError(f"conflicting operator identities for {relation}")

    identities = sorted(
        {identity for identity in relations.values() if identity is not None},
        key=lambda identity: (
            identity[:2] != ("operator", "aten"),
            identity[0] == "communication_kernel",
            identity[0].startswith("unattributed"),
            identity,
        ),
    )
    operator_ids = {
        identity: index for index, identity in enumerate(identities, start=1)
    }
    rows: list[dict[str, Any]] = []
    for relation, identity in sorted(
        relations.items(),
        key=lambda item: (
            item[1] is None,
            operator_ids[item[1]] if item[1] is not None else 0,
            item[0][0] == "null",
            item[0][0],
            item[0][2],
            item[0][1],
        ),
    ):
        operator_name, operator_kind, kernel_name = relation
        rows.append(
            {
                "operator_id": (
                    operator_ids[identity] if identity is not None else "null"
                ),
                "operator_name": operator_name,
                "operator_kind": operator_kind,
                "kernel_name": kernel_name,
            }
        )
    return rows


def details_csv_rows(kernel_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kernel_name, report in kernel_report.items():
        variants: list[tuple[str | None, dict[str, Any]]] = []
        for operator, operator_variants in report["operator_variants"].items():
            variants.extend((operator, row) for row in operator_variants)
        variants.extend((None, row) for row in report["unattributed_variants"])
        variants.sort(
            key=lambda item: (
                -item[1]["kernel_time_us"],
                item[1]["mapping_status"],
                item[0] or "",
                canonical(item[1]["input_shapes"]),
                canonical(item[1]["input_dtypes"]),
                canonical(item[1].get("candidate_operators")),
            )
        )
        for variant_index, (operator, row) in enumerate(variants, start=1):
            rows.append(
                {
                    "operator_name": operator if operator is not None else "null",
                    "kernel_name": kernel_name,
                    "variant_index": variant_index,
                    "mapping_status": row["mapping_status"],
                    "input_shapes": canonical(row["input_shapes"]),
                    "input_dtypes": canonical(row["input_dtypes"]),
                    "candidate_operators": canonical(row.get("candidate_operators")),
                    "kernel_event_count": row["kernel_event_count"],
                    "kernel_time_us": row["kernel_time_us"],
                }
            )
    return rows


def csv_us_to_ns(value: str) -> int:
    return int(Decimal(value) * 1000)


def validate_csv_outputs(
    kernel_time_path: Path,
    kernel_shape_dtype_path: Path,
    operator_list_path: Path,
    kernel_summary: dict[str, Any],
) -> dict[str, bool]:
    with kernel_time_path.open(encoding="utf-8", newline="") as source:
        summary_rows = list(csv.DictReader(source))
    with kernel_shape_dtype_path.open(encoding="utf-8", newline="") as source:
        details_rows = list(csv.DictReader(source))
    with operator_list_path.open(encoding="utf-8", newline="") as source:
        operator_rows = list(csv.DictReader(source))

    expected_keys = list(kernel_summary)
    summary_keys = list(dict.fromkeys(row["kernel_name"] for row in summary_rows))
    details_keys = list(dict.fromkeys(row["kernel_name"] for row in details_rows))
    details_count: Counter[str] = Counter()
    details_ns: Counter[str] = Counter()
    for row in details_rows:
        details_count[row["kernel_name"]] += int(row["kernel_event_count"])
        details_ns[row["kernel_name"]] += csv_us_to_ns(row["kernel_time_us"])

    summary_count: Counter[str] = Counter()
    summary_ns: Counter[str] = Counter()
    for row in summary_rows:
        summary_count[row["kernel_name"]] += int(row["kernel_call_count"])
        summary_ns[row["kernel_name"]] += csv_us_to_ns(row["kernel_time_us"])
    expected_count = {
        name: values["total_call_count"] for name, values in kernel_summary.items()
    }
    expected_ns = {
        name: csv_us_to_ns(str(values["total_time_us"]))
        for name, values in kernel_summary.items()
    }
    summary_relations = [
        (row["operator_name"], row["kernel_name"]) for row in summary_rows
    ]
    details_relations = {
        (row["operator_name"], row["kernel_name"]) for row in details_rows
    }
    operator_relations = [
        (row["operator_name"], row["operator_kind"], row["kernel_name"])
        for row in operator_rows
    ]
    expected_operator_rows = operator_list_rows(summary_rows)
    normalized_operator_rows = [
        {
            **row,
            "operator_id": (
                row["operator_id"]
                if row["operator_id"] == "null"
                else int(row["operator_id"])
            ),
        }
        for row in operator_rows
    ]
    identity_to_id: dict[tuple[str, ...], str] = {}
    id_to_identity: dict[str, tuple[str, ...]] = {}
    operator_ids_stable = True
    operator_classification_matches = True
    for row in operator_rows:
        operator_name = row["operator_name"]
        operator_kind = row["operator_kind"]
        kernel_name = row["kernel_name"]
        operator_id = row["operator_id"]
        expected_name, expected_kind, identity = operator_descriptor(
            operator_name, kernel_name
        )
        operator_classification_matches &= (
            expected_name == operator_name and expected_kind == operator_kind
        )
        if identity is None:
            operator_ids_stable &= operator_id == "null"
            continue
        previous_id = identity_to_id.setdefault(identity, operator_id)
        previous_identity = id_to_identity.setdefault(operator_id, identity)
        operator_ids_stable &= (
            previous_id == operator_id and previous_identity == identity
        )
    nvjet_rows = [
        row for row in operator_rows if row["operator_kind"] == "unattributed_nvjet"
    ]
    communication_rows = [
        row for row in operator_rows if row["operator_kind"] == "communication"
    ]
    seen_communication = False
    communication_rows_last = True
    seen_non_aten = False
    aten_rows_first = True
    for row in operator_rows:
        is_communication = row["operator_kind"] == "communication"
        is_aten = row["operator_kind"] == "aten"
        if seen_non_aten and is_aten:
            aten_rows_first = False
        seen_non_aten |= not is_aten
        seen_communication |= is_communication
        if seen_communication and not is_communication:
            communication_rows_last = False

    custom_ids_by_identity: dict[str, set[str]] = defaultdict(set)
    for row in operator_rows:
        if (
            row["operator_kind"] == "custom"
            and row["operator_name"] != MOE_ALIGN_BLOCK_SIZE_OPERATOR
        ):
            identity = kernel_callable_identity_name(row["kernel_name"])
            custom_ids_by_identity[identity].add(row["operator_id"])
    custom_parameter_variants_grouped = all(
        len(operator_ids) == 1 for operator_ids in custom_ids_by_identity.values()
    )
    kernel_ids_by_callable: dict[str, set[str]] = defaultdict(set)
    for row in operator_rows:
        if row["operator_kind"] in {
            "custom",
            "unattributed",
        }:
            identity = kernel_callable_identity_name(row["kernel_name"])
            kernel_ids_by_callable[identity].add(row["operator_id"])
    kernel_specializations_grouped = all(
        len(operator_ids) == 1 for operator_ids in kernel_ids_by_callable.values()
    )
    compile_kernels_by_id: dict[str, set[str]] = defaultdict(set)
    compile_ids_by_kernel: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in operator_rows:
        if row["operator_kind"] in {"torch_compile", "triton_compiled"}:
            compile_kernels_by_id[row["operator_id"]].add(row["kernel_name"])
            compile_ids_by_kernel[(row["operator_kind"], row["kernel_name"])].add(
                row["operator_id"]
            )
    compile_kernels_separate = all(
        len(kernel_names) == 1 for kernel_names in compile_kernels_by_id.values()
    ) and all(len(operator_ids) == 1 for operator_ids in compile_ids_by_kernel.values())
    moe_align_rows = [
        row
        for row in operator_rows
        if MOE_ALIGN_BLOCK_SIZE_KERNEL.fullmatch(row["kernel_name"])
    ]
    return {
        "csv_kernel_key_sets_match": (
            set(summary_keys) == set(expected_keys)
            and set(details_keys) == set(expected_keys)
        ),
        "csv_kernel_event_count_matches": (
            dict(summary_count) == expected_count
            and dict(details_count) == expected_count
        ),
        "csv_kernel_time_matches": (
            dict(summary_ns) == expected_ns and dict(details_ns) == expected_ns
        ),
        "csv_summary_relations_unique": (
            len(summary_relations) == len(set(summary_relations))
        ),
        "csv_summary_detail_relations_match": (
            set(summary_relations) == details_relations
        ),
        "csv_operator_list_relations_unique": (
            len(operator_relations) == len(set(operator_relations))
        ),
        "csv_operator_list_rows_match_expected": (
            normalized_operator_rows == expected_operator_rows
        ),
        "csv_operator_list_kernel_sets_match": (
            {row["kernel_name"] for row in operator_rows}
            == {row["kernel_name"] for row in summary_rows}
        ),
        "csv_operator_ids_present": all(
            re.fullmatch(r"[1-9]\d*", row["operator_id"]) is not None
            for row in operator_rows
        ),
        "csv_operator_ids_stable": operator_ids_stable,
        "csv_operator_classification_matches": operator_classification_matches,
        "csv_aten_rows_first": aten_rows_first,
        "csv_communication_rows_numbered": all(
            re.fullmatch(r"[1-9]\d*", row["operator_id"]) is not None
            for row in communication_rows
        ),
        "csv_communication_rows_last": communication_rows_last,
        "csv_custom_parameter_variants_grouped": (custom_parameter_variants_grouped),
        "csv_kernel_specializations_grouped": kernel_specializations_grouped,
        "csv_compile_kernels_separate": compile_kernels_separate,
        "csv_moe_align_block_size_grouped": (
            len({row["operator_id"] for row in moe_align_rows}) <= 1
            and all(
                row["operator_name"] == MOE_ALIGN_BLOCK_SIZE_OPERATOR
                and row["operator_kind"] == "custom"
                for row in moe_align_rows
            )
        ),
        "csv_unattributed_nvjet_grouped": (
            len({row["operator_id"] for row in nvjet_rows}) <= 1
            and all(row["kernel_name"].startswith("nvjet_tst_") for row in nvjet_rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime_files = [
        file
        for file in runtime_trace_files(args.runtime)
        if rank_in_filename(file) == args.rank
    ]
    if not runtime_files:
        raise FileNotFoundError(
            f"no rank-{args.rank} runtime profiler traces under {args.runtime}"
        )

    kernel_summary, kernel_report, summary = collect_runtime(runtime_files, args.rank)
    failed_checks = [
        key
        for key, value in summary["conservation"].items()
        if isinstance(value, bool) and not value
    ]
    if failed_checks:
        raise RuntimeError(f"kernel conservation failed: {summary['conservation']}")

    kernel_time_path = args.output_dir / "kernel_time.csv"
    kernel_shape_dtype_path = args.output_dir / "kernel_shape_dtype.csv"
    operator_list_path = args.output_dir / "operator_list.csv"
    summary_rows = summary_csv_rows(
        kernel_report, round(summary["kernel_time_total_us"] * 1000)
    )
    write_csv(
        kernel_time_path,
        [
            "operator_name",
            "kernel_name",
            "kernel_call_count",
            "kernel_time_us",
            "percent",
        ],
        summary_rows,
    )
    write_csv(
        kernel_shape_dtype_path,
        [
            "operator_name",
            "kernel_name",
            "variant_index",
            "mapping_status",
            "input_shapes",
            "input_dtypes",
            "candidate_operators",
            "kernel_event_count",
            "kernel_time_us",
        ],
        details_csv_rows(kernel_report),
    )
    write_csv(
        operator_list_path,
        ["operator_id", "operator_name", "operator_kind", "kernel_name"],
        operator_list_rows(summary_rows),
    )
    summary["conservation"].update(
        validate_csv_outputs(
            kernel_time_path,
            kernel_shape_dtype_path,
            operator_list_path,
            kernel_summary,
        )
    )
    failed_checks = [
        key
        for key, value in summary["conservation"].items()
        if isinstance(value, bool) and not value
    ]
    if failed_checks:
        raise RuntimeError(f"CSV conservation failed: {summary['conservation']}")
    write_json(args.output_dir / "summary.json", summary)
    for obsolete in (
        "kernel_summary.json",
        "kernel_report.json",
        "kernel_report.csv",
        "kernel_summary.csv",
        "kernel_details_report.csv",
        "non_kernel_gpu_activity.json",
        "operator_index.json",
    ):
        (args.output_dir / obsolete).unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
