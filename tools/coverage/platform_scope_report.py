#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Generate the platform-applicable coverage.py JSON from the raw full-scope report.

This is the tier-1 statistic of the 0.5.0-dev plan (单平台适用代码行覆盖率):
FlagCICD should display the line coverage of the code that is *applicable*
to a given platform, not the whole-package scope (which charges every
platform with the other vendor directories it can never execute).

Two filters are applied, in this order:

1. Directory level (automatic, from ``tools/coverage/scope_<platform>.json``)
   - files under any ``reporting_exclude`` vendor prefix are dropped from
     both numerator and denominator;
   - everything else is kept in full: the platform's own vendor directory
     plus all public code.

2. Line level (hybrid files, plan section 3.2 hard requirement)
   - public files such as ``vllm_fl/platform.py``, ``vllm_fl/worker/worker.py``
     or ``vllm_fl/ops/fused_moe/fused_moe_utils.py`` mix common code with
     branches keyed on ``vendor_name`` / ``device_type`` (``if vendor_name ==
     "hygon": ...``). Fully automating per-line ownership of arbitrary code
     is not tractable, so the tool combines:
     a) an AST pass over the hybrid file that classifies each ``if``/``elif``
        guard by its vendor literals and marks foreign-only branch bodies
        not-applicable — i.e. "other vendors' exclusive code is not counted
        in this platform's numerator or denominator" (plan 3.2). The
        classification rules:
        - ``attr == "X"`` / ``attr in (...)`` with X foreign, no local literal:
          body runs only on foreign platforms -> foreign;
        - the negated forms (``!=``, ``not in``) invert: a negated foreign
          literal runs the body on every other platform including ours ->
          keep; a negated LOCAL literal runs the body only on foreign
          platforms -> foreign;
        - ``and``-conjunctions are foreign when any conjunct is foreign and
          none is local: the AND holds only on the foreign platform then;
        - device_type literals resolve through VENDOR_DEVICE_TYPES; a
          device_type shared by the local vendor and any foreign vendor
          (e.g. "cuda" for kunlunxin/iluvatar/hygon) counts as neutral so a
          ``device_type == "cuda" and vendor_name == "iluvatar"`` guard is
          correctly foreign;
        - guards naming no vendor at all stay (common code).
     b) an optional hand-maintained override table
        (``--line-map scope_line_overrides.json``, ``file -> keep-lines``)
        for cases the AST pass cannot decide. The shipped
        ``scope_line_overrides.json`` documents the hybrid files and the
        manually annotated exceptions.

   Direction note: for platform P the excluded branch bodies belong to the
   OTHER vendors. E.g. on kunlunxin, hygon/iluvatar/thead/nvidia-only branch
   lines leave the report while kunlunxin branches stay (kunlunxin's
   device_type is "cuda", so ``device_type == "cuda"`` guards stay too);
   on thead, kunlunxin-only branch lines leave. Common code always stays.

The output is a standard coverage.py ``json`` report structure: per file
``executed_lines`` / ``summary`` / ``missing_lines`` with the file summary and
the grand totals recomputed from the filtered statement sets, so the
dashboard number always matches the report details. The raw full-scope JSON
stays untouched (and is preserved as a CI artifact).

Self-check: after building each file entry the tool asserts
``summary.missing_lines == len(missing_lines)`` and
``num_statements == len(executed | missing)`` — the lesson learned from
merge_coverage.py, where a range-based statement reconstruction under-reported
on files whose physical line count exceeds the statement count.

Usage::

    python tools/coverage/platform_scope_report.py \
        --report coverage-kunlunxin.json \
        --platform kunlunxin \
        --output coverage-kunlunxin-applicable.json \
        --commit-sha <git sha>
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

# Public files that mix common code with per-vendor branches (plan 3.2).
# The AST pass below is generic for any file it parses, but these are the
# files the plan names plus the ones the vendor-guard census found; keeping
# the list explicit makes a new hybrid file a deliberate addition.
KNOWN_HYBRID_FILES = (
    "vllm_fl/platform.py",
    "vllm_fl/__init__.py",
    "vllm_fl/utils.py",
    "vllm_fl/worker/worker.py",
    "vllm_fl/worker/model_runner.py",
    "vllm_fl/ops/fused_moe/fused_moe_utils.py",
    "vllm_fl/compilation/graph.py",
    "vllm_fl/dispatch/builtin_ops.py",
)

# vendor_name -> device_type, mirror of vllm_fl/utils.py VENDOR_DEVICE_MAP.
# Device-type comparisons are classified through this map: a guard on
# ``device_type == X`` is local for every platform whose vendor maps to X,
# foreign when only foreign vendors map to X, and shared (neutral) when both
# sides map to it — "cuda" is the notable shared case.
VENDOR_DEVICE_TYPES = {
    "nvidia": "cuda",
    "ascend": "npu",
    "iluvatar": "cuda",
    "metax": "cuda",
    "mthreads": "musa",
    "musa": "musa",
    "sunrise": "ptpu",
    "hygon": "cuda",
    "amd": "cuda",
    "thead": "cuda",
    "tsingmicro": "txda",
    "txda": "txda",
    "gcu": "gcu",
    "enflame": "gcu",
    "kunlunxin": "cuda",
}

# Vendor-directory name -> vendor_name literals used in guards. The "cuda"
# directory backs NVIDIA hardware (its Backend.vendor is "nvidia"), so it
# maps to nvidia; kunlunxin/iluvatar/hygon share the torch.cuda API but live
# in their own directories and are never called "cuda" in guards.
_DIR_TO_VENDOR_NAMES = {
    "cuda": ("nvidia",),
    "musa": ("musa", "mthreads"),
    "txda": ("txda", "tsingmicro"),
    "gcu": ("gcu", "enflame"),
}

# Vendor literals that may appear as bare device_type values in guards
# (``.device.type == "npu"`` on a torch device). These classify through the
# same vendor map: npu=ascend, musa=mthreads, gcu, txda=tsingmicro, ptpu=sunrise.
_DEVICE_TYPE_VENDORS = {
    dt: [
        v
        for v, d in VENDOR_DEVICE_TYPES.items()
        if d == dt and v in _DIR_TO_VENDOR_NAMES
    ]
    or [v for v, d in VENDOR_DEVICE_TYPES.items() if d == dt]
    for dt in {"cuda", "npu", "musa", "ptpu", "txda", "gcu", "cpu"}
}


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm_key(path_key: str) -> str:
    """Normalize a coverage.py file key to a repo-relative posix path."""
    p = path_key.replace("\\", "/")
    idx = p.find("vllm_fl/")
    if idx != -1:
        return p[idx:]
    return p


def _scope_vendor_names(scope: dict, platform: str) -> tuple[set[str], set[str]]:
    """Return (foreign vendor names, local vendor names) for this platform.

    Foreign: every vendor whose directory reporting_exclude lists (minus the
    platform itself). Local: the platform's own vendor name(s), including
    directory aliases (gcu -> enflame, musa -> mthreads, txda -> tsingmicro).
    """
    foreign: set[str] = set()
    for prefix in scope.get("reporting_exclude", []):
        parts = prefix.rstrip("/").split("/")
        name = parts[-1] if parts else ""
        if name in _DIR_TO_VENDOR_NAMES or name in VENDOR_DEVICE_TYPES:
            foreign.update(_DIR_TO_VENDOR_NAMES.get(name, (name,)))

    local: set[str] = {platform}
    local.update(_DIR_TO_VENDOR_NAMES.get(platform, (platform,)))
    foreign -= local
    return foreign, local


class _GuardClassifier:
    """Classify an ``if`` guard as foreign / local / shared / neutral.

    foreign  — the body can only execute on foreign platforms (exclude it);
    local    — the guard names our own vendor (body may run here, keep);
    shared   — the guard names a device_type both we and foreign vendors use
               (no decision by itself; the verdict is up to conjunction);
    neutral  — no vendor information (common code, keep).
    """

    def __init__(self, foreign_vendors: set[str], local_vendors: set[str]):
        self.foreign = foreign_vendors
        self.local = local_vendors
        # device_type -> foreign-local sharing
        self.foreign_dtypes = {
            VENDOR_DEVICE_TYPES[v] for v in foreign_vendors if v in VENDOR_DEVICE_TYPES
        }
        self.local_dtypes = {
            VENDOR_DEVICE_TYPES[v] for v in local_vendors if v in VENDOR_DEVICE_TYPES
        }
        self.shared_dtypes = self.foreign_dtypes & self.local_dtypes
        self.foreign_only_dtypes = self.foreign_dtypes - self.local_dtypes

    # -- literal classification ------------------------------------------

    def _classify_literal(self, literal: str, *, device_ctx: bool) -> str:
        """Classify one vendor/device literal without negation context."""
        if literal in self.local:
            return "local"
        if literal in self.foreign:
            # A vendor literal that doubles as a shared device_type ("cuda"
            # is not in the vendor sets here since dir->vendor mapping
            # resolves it to nvidia) still classifies by vendor.
            return "foreign"
        if device_ctx and literal in self.shared_dtypes:
            return "shared"
        if device_ctx and literal in self.foreign_only_dtypes:
            return "foreign"
        if device_ctx and literal in self.local_dtypes:
            return "local"
        return "neutral"

    # -- comparison classification ---------------------------------------

    def _classify_compare(self, node: ast.Compare) -> str:
        """Classify ``x == "lit"`` / ``x in ("a", "b")`` / negated variants."""
        target = _target_name(node.left)
        if target is None:
            return "neutral"
        low = target.lower()
        # Vendor guards read .vendor_name / vendor_name / _env_vendor_name();
        # device guards read .device_type / .device.type / device_type.
        is_vendor_ctx = "vendor" in low
        is_device_ctx = "device_type" in low or low.endswith("type") or low == "device"
        if not (is_vendor_ctx or is_device_ctx):
            return "neutral"

        literals: list[tuple[str, bool]] = []  # (literal, negated)
        for op, comp in zip(node.ops, node.comparators):
            negated = isinstance(op, (ast.NotEq,)) or (
                isinstance(op, ast.NotIn) if hasattr(ast, "NotIn") else False
            )
            if isinstance(comp, (ast.Tuple, ast.Set, ast.List)):
                for elt in comp.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        literals.append((elt.value, negated))
            elif isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                literals.append((comp.value, negated))

        if not literals:
            return "neutral"

        verdicts: set[str] = set()
        for literal, negated in literals:
            v = self._classify_literal(literal, device_ctx=is_device_ctx)
            if v == "neutral":
                continue
            if negated:
                # ``!= "X"`` inverts ownership: a body guarded by
                # ``vendor != foreign`` runs for us; ``vendor != local``
                # runs only for foreign platforms.
                v = {"foreign": "local", "local": "foreign"}.get(v, v)
            verdicts.add(v)

        if "foreign" in verdicts:
            return "foreign"
        if "local" in verdicts:
            return "local"
        if "shared" in verdicts:
            return "shared"
        return "neutral"

    # -- expression classification ---------------------------------------

    def classify(self, test: ast.expr) -> str:
        """Classify a full guard expression."""
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            # AND: every conjunct must hold. Any foreign conjunct with no
            # local conjunct means the whole guard can only hold on a
            # foreign platform.
            verdicts = [self.classify(v) for v in test.values]
            if "foreign" in verdicts and "local" not in verdicts:
                return "foreign"
            if "local" in verdicts:
                return "local"
            if "shared" in verdicts:
                return "shared"
            return "neutral"
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            # OR: the guard holds if ANY disjunct holds. A foreign disjunct
            # does not make the body foreign (ours may satisfy another
            # disjunct); a local disjunct keeps the body possibly ours.
            verdicts = [self.classify(v) for v in test.values]
            if "local" in verdicts or "shared" in verdicts:
                return "local"
            return "neutral"
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self.classify(test.operand)
            if inner == "foreign":
                return "local"
            if inner == "local":
                return "foreign"
            return inner
        if isinstance(test, ast.Compare):
            return self._classify_compare(test)
        if isinstance(test, ast.NamedExpr):
            return self.classify(test.value)
        # Fall back: scan every comparison in the expression.
        verdicts = set()
        for node in ast.walk(test):
            if isinstance(node, ast.Compare):
                verdicts.add(self._classify_compare(node))
        if (
            "foreign" in verdicts
            and "local" not in verdicts
            and "shared" not in verdicts
        ):
            return "foreign"
        if "local" in verdicts:
            return "local"
        return "neutral"


def _target_name(target: ast.expr) -> str | None:
    """Last attribute/name of an expression such as ``cls.vendor_name``.

    ``getattr(current_platform, "vendor_name", "")`` resolves to
    "vendor_name" through the first string argument.
    """
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Call):
        if _target_name(target.func) == "getattr" and target.args:
            arg = target.args[0] if len(target.args) < 2 else target.args[1]
            # getattr(obj, "vendor_name", default): the name is args[1].
            if len(target.args) >= 2:
                arg = target.args[1]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    return arg.value
            return _target_name(target.func)
        return _target_name(target.func)
    if isinstance(target, ast.Subscript):
        return _target_name(target.value)
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value
    return None


class _ForeignBranchCollector(ast.NodeVisitor):
    """Collect line ranges of branch bodies guarded by foreign vendors only."""

    def __init__(self, classifier: _GuardClassifier):
        self.classifier = classifier
        self.excluded_lines: set[int] = set()

    def _mark(self, stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            for node in ast.walk(stmt):
                if hasattr(node, "lineno"):
                    end = getattr(node, "end_lineno", None) or node.lineno
                    self.excluded_lines.update(range(node.lineno, end + 1))

    def visit_If(self, node: ast.If) -> None:
        if self.classifier.classify(node.test) == "foreign":
            self._mark(node.body)
            # An else-arm of a foreign-only guard is the "everyone else" path
            # (including us): keep. elif arms classify independently below.
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # ``"A" if vendor == "hygon" else "B"``: only the true value is
        # foreign-owned; the false value is the common path.
        if self.classifier.classify(node.test) == "foreign":
            for n in ast.walk(node.body):
                if hasattr(n, "lineno"):
                    end = getattr(n, "end_lineno", None) or n.lineno
                    self.excluded_lines.update(range(n.lineno, end + 1))
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        for case in node.cases:
            strings = set()
            for pat_node in ast.walk(case.pattern):
                if isinstance(pat_node, ast.MatchValue):
                    v = pat_node.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        strings.add(v.value)
            if (
                strings & self.classifier.foreign
                and not strings & self.classifier.local
            ):
                self._mark(case.body)
        self.generic_visit(node)


def foreign_branch_lines(
    source: str, foreign_vendors: set[str], local_vendors: set[str]
) -> set[int]:
    """Return the line numbers of foreign-vendor-only branch bodies."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    collector = _ForeignBranchCollector(
        _GuardClassifier(foreign_vendors, local_vendors)
    )
    collector.visit(tree)
    return collector.excluded_lines


def applicable_file_entry(
    info: dict,
    rel: str,
    line_map: dict | None,
    foreign_vendors: set[str],
    local_vendors: set[str],
    source_dir: Path | None,
) -> tuple[dict | None, dict | None]:
    """Build the applicable per-file report entry and its filter stats."""
    executed = set(info.get("executed_lines", []))
    missing = set(info.get("missing_lines", []))
    statements = executed | missing
    num_statements = info.get("summary", {}).get("num_statements", len(statements))
    if len(statements) < num_statements:
        # Producer truncated the line lists (e.g. the merge_coverage.py-era
        # combined reports under-report missing_lines). Keep the declared
        # denominator by padding the statement set with 1..N minus executed —
        # the merge_coverage.py lesson: never ship summary counts that
        # disagree with the line lists, and never let padding swallow
        # executed lines (the union always stays a superset of executed).
        statements |= set(range(1, num_statements + 1)) - executed

    keep: set[int] | None = None  # None = keep everything (no line filtering)
    applied = None

    own = (line_map or {}).get(rel)
    if own is not None:
        # Hand-maintained override: explicit keep-line set.
        keep = set(int(x) for x in own)
        applied = "line-map"
    elif rel in KNOWN_HYBRID_FILES:
        source_path = source_dir / rel if source_dir else None
        if source_path is not None and source_path.is_file():
            try:
                source = source_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                source = ""
            if source:
                foreign_lines = foreign_branch_lines(
                    source, foreign_vendors, local_vendors
                )
                if foreign_lines:
                    keep = statements - foreign_lines
                    applied = "ast-vendor-guard"

    stats = None
    if keep is not None:
        raw_statements = len(statements)
        raw_covered = len(executed)
        removed = statements - keep
        executed &= keep
        statements &= keep
        stats = {
            "raw_statements": raw_statements,
            "applicable_statements": len(statements),
            "raw_covered": raw_covered,
            "applicable_covered": len(executed),
            "lines_removed": len(removed),
            "removed_lines": sorted(removed),
            "mechanism": applied,
        }

    # Recompute missing from the final statement set so the summary can never
    # disagree with the line lists — mandatory after any padding/filtering
    # (merge_coverage.py lesson: the old range-padding shipped files whose
    # summary.missing_lines and len(missing_lines) diverged).
    missing = statements - executed
    if not statements:
        return None, stats

    covered = len(executed)
    pct = (100.0 * covered / len(statements)) if statements else 0.0
    summary = {
        "covered_lines": covered,
        "num_statements": len(statements),
        "percent_covered": round(pct, 4),
        "missing_lines": len(missing),
    }

    # Self-check: the merge_coverage.py lesson.
    assert summary["missing_lines"] == len(missing), (rel, summary, len(missing))
    assert summary["num_statements"] == len(statements), rel
    assert summary["covered_lines"] == len(executed), rel
    assert not (set(executed) & set(missing)), rel

    entry = {
        "executed_lines": sorted(executed),
        "summary": summary,
        "missing_lines": sorted(missing),
    }
    # Preserve the auxiliary per-file keys coverage.py exports (functions,
    # classes, excluded_lines) so consumers reading them from the raw report
    # keep working; they are informational and do not affect the totals.
    for aux in ("excluded_lines", "functions", "classes"):
        if aux in info:
            entry[aux] = info[aux]
    return entry, stats


def applicable_report(
    report: dict,
    scope: dict,
    platform: str,
    commit_sha: str,
    line_map: dict | None,
    source_dir: Path | None,
) -> tuple[dict, dict]:
    """Return (applicable report, console stats)."""
    exclude = tuple(scope.get("reporting_exclude", []))
    foreign_vendors, local_vendors = _scope_vendor_names(scope, platform)

    files_out: dict[str, dict] = {}
    dropped_files = 0
    hybrid_filtered: dict[str, dict] = {}

    for key, info in report.get("files", {}).items():
        rel = _norm_key(key)
        if any(rel.startswith(prefix) for prefix in exclude):
            dropped_files += 1
            continue

        entry, stats = applicable_file_entry(
            info, rel, line_map, foreign_vendors, local_vendors, source_dir
        )
        if entry is None:
            dropped_files += 1
            continue
        if stats is not None:
            hybrid_filtered[rel] = stats
        files_out[rel] = entry

    total_stmts = sum(v["summary"]["num_statements"] for v in files_out.values())
    total_covered = sum(v["summary"]["covered_lines"] for v in files_out.values())
    overall = (100.0 * total_covered / total_stmts) if total_stmts else 0.0

    meta = dict(report.get("meta", {}))
    meta.update(
        {
            "platform": platform,
            "source_commit": commit_sha,
            "caliber": "platform-applicable",
            "reporting_exclude_count": len(exclude),
            "hybrid_files_line_filtered": sorted(hybrid_filtered),
        }
    )

    out = {
        "meta": meta,
        "totals": {
            "covered_lines": total_covered,
            "num_statements": total_stmts,
            "percent_covered": round(overall, 4),
            "missing_lines": total_stmts - total_covered,
        },
        "files": files_out,
    }

    # Global self-check: totals must equal the per-file sums exactly.
    assert out["totals"]["missing_lines"] == sum(
        v["summary"]["missing_lines"] for v in files_out.values()
    )

    stats_out = {
        "dropped_files": dropped_files,
        "kept_files": len(files_out),
        "hybrid_filtered": hybrid_filtered,
    }
    return out, stats_out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report", required=True, type=Path, help="raw full-scope coverage.py JSON"
    )
    parser.add_argument(
        "--platform", required=True, help="platform name, e.g. kunlunxin"
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=None,
        help="scope_<platform>.json; defaults to tools/coverage/scope_<platform>.json next to this script",
    )
    parser.add_argument(
        "--line-map",
        type=Path,
        default=None,
        help=(
            "optional JSON {rel_file: [keep line numbers]}; "
            "defaults to tools/coverage/scope_line_overrides.json if present"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--commit-sha", default="", help="git SHA the report was measured on"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "repo root to read hybrid-file sources for the AST branch pass; "
            "defaults to two levels above this script"
        ),
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    scope_path = args.scope or (here / f"scope_{args.platform}.json")
    line_map_path = args.line_map or (here / "scope_line_overrides.json")
    source_dir = args.source_dir or here.parent.parent

    report = _load(args.report)
    scope = _load(scope_path)
    if scope.get("platform") not in (None, args.platform):
        print(
            f"[applicable] warning: scope file is for {scope.get('platform')}, "
            f"requested {args.platform}",
            file=sys.stderr,
        )
    line_map = _load(line_map_path) if line_map_path.is_file() else None
    if isinstance(line_map, dict) and "overrides" in line_map:
        # scope_line_overrides.json is self-describing: {schema_version, description,
        # hybrid_files_documented, overrides: {file: [keep lines]}}.
        line_map = line_map.get("overrides") or None

    out, stats = applicable_report(
        report, scope, args.platform, args.commit_sha, line_map, source_dir
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    t = out["totals"]
    print(
        f"[applicable] {args.platform}: {t['covered_lines']}/{t['num_statements']} "
        f"= {t['percent_covered']}% "
        f"({stats['kept_files']} files kept, {stats['dropped_files']} dropped) "
        f"-> {args.output}"
    )
    for rel, h in stats["hybrid_filtered"].items():
        print(
            f"[applicable]   hybrid {rel} [{h['mechanism']}]: "
            f"{h['raw_statements']} -> {h['applicable_statements']} stmts "
            f"({h['lines_removed']} foreign-vendor branch lines removed), "
            f"covered {h['raw_covered']} -> {h['applicable_covered']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
