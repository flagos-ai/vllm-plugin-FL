# Coverage tooling

Tools around the coverage.py JSON reports produced by `tests/run.py`
(`--cov-report=json:coverage-<platform>.json`). All of them are REPORT-time:
the measurement scope stays `pyproject.toml [tool.coverage.run]`
(`source = vllm_fl`) and is never changed.

## Calibers

| caliber | what it is | where |
| --- | --- | --- |
| raw full-scope | every line of `vllm_fl`, all 14 vendor directories included | `coverage-<platform>.json`, uploaded as a GitHub artifact |
| platform-applicable | raw minus the other vendors' directories, minus their exclusive branch lines in public hybrid files | `coverage-<platform>-applicable.json`, uploaded to FlagCICD (kunlunxin first) |
| multi-platform aggregate | union of executed lines across all platforms over the full scope | `aggregate` reports (see A4) |

The raw JSON is preserved unchanged as a CI artifact; every derived caliber
can be recomputed from it.

## platform_scope_report.py — platform-applicable report

Implements the plan's tier-1 statistic (单平台适用代码行覆盖率): the line
coverage of the code applicable to one platform. Two filters:

1. **Directory level** (from `scope_<platform>.json`, `reporting_exclude`):
   the 12 other vendor directories drop out of numerator AND denominator.
   Everything else — including the platform's own vendor directory and all
   public code — stays.

2. **Line level** (plan 3.2, hybrid files): public files that mix common
   code with `if vendor_name == "X":` branches (`vllm_fl/platform.py`,
   `vllm_fl/worker/worker.py`, `vllm_fl/ops/fused_moe/fused_moe_utils.py`,
   ...) are filtered line-by-line. An AST pass classifies each `if`/`elif`
   guard:
   - `attr == "foreign-vendor"` / `attr in ("foreign", ...)` (no local
     literal in the same guard) → the whole branch body leaves the report;
   - negated guards invert: `vendor_name != "iluvatar"` runs the body for
     every other platform including ours → kept;
   - `and`-conjunctions are foreign when any conjunct is foreign and none
     is local, so `device_type == "cuda" and vendor_name == "iluvatar"`
     is foreign on kunlunxin while `device_type == "cuda"` alone is
     *shared* (kunlunxin, iluvatar, hygon, nvidia, thead all use the
     torch.cuda API) and never excludes anything by itself;
   - guards naming no vendor stay (common code).

   Direction: for platform P the excluded lines belong to the OTHER
   vendors. On kunlunxin, hygon/iluvatar/thead/nvidia-only branch lines
   leave the report; on thead, kunlunxin-only lines leave. Common lines
   and P's own vendor branches always stay.

### Usage

```bash
python tools/coverage/platform_scope_report.py \
    --report coverage-kunlunxin.json \
    --platform kunlunxin \
    --output coverage-kunlunxin-applicable.json \
    --commit-sha <git sha>
```

`--scope` defaults to `tools/coverage/scope_<platform>.json`;
`--line-map` defaults to `tools/coverage/scope_line_overrides.json`;
`--source-dir` defaults to the repo root (two levels above the script) —
the AST pass reads the hybrid files' current sources from there.

The output is a standard coverage.py JSON structure (`meta` / `files` /
`totals`, per file `executed_lines` / `summary` / `missing_lines`) with the
per-file summaries and grand totals recomputed from the filtered statement
sets, so the dashboard number always equals the report details. `meta`
records `platform`, `source_commit` and `caliber:
"platform-applicable"`.

Self-checks (hard asserts, the `merge_coverage.py` lesson):
`summary.missing_lines == len(missing_lines)`,
`num_statements == len(executed | missing)`, and the grand totals equal
the per-file sums.

### Known numbers (validation, kunlunxin)

Using the kunlunxin raw full-scope report measured at 9aec64d2:

- raw full-scope: 6743/25410 = 26.54%
- directory level (existing `scope_kunlunxin.json` caliber): 5316/15033
  — matches the known 15033 denominator exactly
- platform-applicable: **5224/14798 = 35.3021%**
  (235 foreign-vendor branch statements and 92 of their covered lines
  removed from the six hybrid files)

The 10512/15033 = 69.93% figure quoted for kunlunxin is the
baseline+supplementary *combined* caliber on the same denominator; the
applicable tool preserves the declared denominators, so applying it to a
combined report yields the same 15033-based denominator minus the removed
foreign-branch lines.

## scope_line_overrides.json — manual line-level exception table

Optional override table the tool loads by default. Format:

```json
{
  "schema_version": 1,
  "overrides": {
    "vllm_fl/platform.py": [10, 11, 12, 15]
  }
}
```

`overrides.<file>` REPLACES the AST verdict for that file with an explicit
keep-line set (repo-relative path → applicable line numbers). Only add an
entry when the AST pass misclassifies a region; cite the reason in the
`hybrid_files_documented` section. When the file exists but `overrides` is
empty, the AST pass decides everything (current state).

## merge_coverage.py — baseline+supplementary union

Statement-level union of the core baseline report and the supplementary
report of the same platform/commit. See its module docstring.

## area_report.py / delta_report.py

Per-area breakdowns and plan-vs-actual deltas over the combined reports.

## Dashboard validation procedure (FlagCICD)

After the kunlunxin switch ships in `_unit_test.yml`:

1. Run the kunlunxin unit-test job on a PR; confirm the
   `Compute platform-applicable coverage` step logs
   `applicable-caliber: kunlunxin platform-applicable coverage = <cov>/<stmts> (<pct>%)`
   and the job summary shows the same numbers.
2. Open the FlagCICD dashboard for that pipeline run and check:
   - the coverage metric posts without errors (`post-pytest-report@v2`
     reads `coverage-kunlunxin-applicable.json`);
   - the dashboard's displayed percentage and line counts equal the
     step's logged numbers (totals are recomputed from the per-file
     lists, so they must match to the digit);
   - the raw `coverage-kunlunxin.json` still lands in the GitHub
     artifact (unchanged caliber).
3. Compare against a local run on the same commit:
   ```bash
   python tools/coverage/platform_scope_report.py \
       --report coverage-kunlunxin.json --platform kunlunxin \
       --output /tmp/check.json --commit-sha <sha>
   ```
   `meta.caliber == "platform-applicable"`,
   `meta.source_commit == <sha>`, identical totals.
4. Once the dashboard is verified, flip the remaining platforms by
   removing the `inputs.platform == 'kunlunxin'` condition on
   `report_path` in `_unit_test.yml` (the applicable step already runs
   for every platform).
