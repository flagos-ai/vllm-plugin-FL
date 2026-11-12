#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
set -euo pipefail

results=coverage-results
mkdir -p "$results"/{unit,functional,supplementary}
rm -f "$results"/{baseline,combined}{,-full,-scope}.json
rm -f "$results"/{unit,functional,supplementary}.log

export COVERAGE_FILE="$PWD/$results/.coverage"
export COVERAGE_RCFILE="$PWD/coverage_tests/.coveragerc"
sha="$(git rev-parse HEAD)"

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

python tests/run.py --platform nvidia --device a100 --scope unit \
    --output-dir "$results/unit" 2>&1 | tee "$results/unit.log"
python tests/run.py --platform nvidia --device a100 --scope functional \
    --output-dir "$results/functional" 2>&1 | tee "$results/functional.log"
report baseline unit,functional

python -m coverage_tests.run --output-dir "$results/supplementary" \
    2>&1 | tee "$results/supplementary.log"
report combined unit,functional,supplementary
