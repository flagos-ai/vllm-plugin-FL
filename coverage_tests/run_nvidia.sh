#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
#
# CI-4 supplementary coverage round. Every phase records its own log, exit
# code, pytest pass/skip/deselect counts and wall-clock duration; a failing
# phase never aborts the remaining phases or the coverage reports. The script
# exits non-zero only after every report has been written.
set -uo pipefail

results=coverage-results
mkdir -p "$results"/{unit,functional,supplementary}
rm -f "$results"/{baseline,combined}{,-full,-scope}.json
rm -f "$results"/{unit,functional,supplementary}.{log,exit}
rm -f "$results/summary.txt"

export COVERAGE_FILE="$PWD/$results/.coverage"
export COVERAGE_RCFILE="$PWD/coverage_tests/.coveragerc"
sha="$(git rev-parse HEAD)"

overall_status=0

report() {
    local label="$1"
    local scope="$2"
    python .github/scripts/coverage_scope.py \
        --platform nvidia --scope "$scope" --source-sha "$sha" \
        --input coverage-nvidia.json \
        --full-output "$results/$label-full.json" \
        --effective-output "$results/$label.json" \
        --metadata-output "$results/$label-scope.json"
}

# Aggregate the pytest "X passed, Y skipped, Z deselected ... in Ns" summary
# lines of a phase log into a one-line counter string.
counts_from_log() {
    python - "$1" <<'PY'
import re
import sys

totals = {key: 0 for key in ("passed", "failed", "errors", "skipped", "deselected")}
found = False
pattern = re.compile(
    r"(\d+) (passed|failed|error|errors|skipped|deselected|xfailed|xpassed)\b"
)
with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
    for line in handle:
        stripped = line.strip()
        if " in " not in stripped:
            continue
        if not (stripped.startswith("=") or stripped[:1].isdigit()):
            continue
        matches = pattern.findall(stripped)
        if not matches:
            continue
        found = True
        for number, key in matches:
            key = "errors" if key == "error" else key
            if key in totals:
                totals[key] += int(number)
if not found:
    print("counts=n/a")
else:
    print(" ".join(f"{key}={value}" for key, value in totals.items()))
PY
}

summary_row() {
    local phase="$1" status="$2" rc="$3" duration="$4" counts="$5"
    printf '%-14s %-4s exit=%-3s %6ss  %s\n' \
        "$phase" "$status" "$rc" "$duration" "$counts" >>"$results/summary.txt"
}

run_phase() {
    local phase="$1"
    shift
    local log="$results/$phase.log"
    local started ended duration rc status counts
    started=$(date +%s)
    echo "==> [$phase] $*"
    "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    ended=$(date +%s)
    duration=$((ended - started))
    printf '%s\n' "$rc" >"$results/$phase.exit"
    counts=$(counts_from_log "$log")
    if [ "$rc" -eq 0 ]; then
        status=PASS
    else
        status=FAIL
        overall_status=1
    fi
    echo "[ci] $phase: $status (exit $rc, ${duration}s) $counts"
    summary_row "$phase" "$status" "$rc" "$duration" "$counts"
}

run_report() {
    local label="$1"
    local scope="$2"
    local rc
    report "$label" "$scope"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        summary_row "report:$label" PASS 0 0 "-"
    else
        summary_row "report:$label" FAIL "$rc" 0 "-"
        overall_status=1
    fi
}

{
    echo "supplementary coverage run"
    echo "sha: $sha"
    printf '%-14s %-4s %-8s %7s  %s\n' phase status exit elapsed counts
} >"$results/summary.txt"

run_phase unit \
    python tests/run.py --platform nvidia --device a100 --scope unit \
    --output-dir "$results/unit"
run_phase functional \
    python tests/run.py --platform nvidia --device a100 --scope functional \
    --output-dir "$results/functional"

if [ -f coverage-nvidia.json ]; then
    run_report baseline unit,functional
else
    echo "coverage-nvidia.json missing; skipping baseline report"
    summary_row "report:baseline" SKIP "-" 0 "no coverage json"
fi

run_phase supplementary \
    python -m coverage_tests.run --output-dir "$results/supplementary"

if [ -f coverage-nvidia.json ]; then
    run_report combined unit,functional,supplementary
else
    echo "coverage-nvidia.json missing; skipping combined report"
    summary_row "report:combined" SKIP "-" 0 "no coverage json"
fi

echo
echo "==> phase summary"
cat "$results/summary.txt"
exit "$overall_status"
