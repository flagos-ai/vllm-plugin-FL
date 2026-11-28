<<<<<<< HEAD
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
=======
# vllm-plugin-FL coverage tooling

Reporting-time tools around coverage.py JSON reports for the
multi-platform supplementary coverage effort
(kunlunxin / hygon / iluvatar / thead). All tools are stdlib-only and
consume the JSON output of ``--cov-report=json``.

| tool | purpose |
|---|---|
| `merge_coverage.py` | statement-level union of one platform's core baseline + supplementary-only report |
| `aggregate_coverage.py` | **multi-platform aggregate** — union of executed lines across platforms over the full `vllm_fl` denominator |
| `area_report.py` | per-area (module prefix) numbers from one report |
| `delta_report.py` | per-area core → combined deltas |
| `scope_<platform>.json` | reporting-time scope config per platform (measurement scope never changes) |

## Multi-platform aggregate (`aggregate_coverage.py`)

### Formula

```
aggregate % = |∪_p executed_lines(p)| / total vllm_fl executable statements
```

* **Numerator** — the union of every participating platform's executed
  lines. A statement executed by several platforms counts **once**; the
  aggregate rewards complementary coverage, not repetition.
* **Denominator** — the complete `vllm_fl` executable-statement count
  (25,410 statements at the 0.5.0-dev frozen revision), identical for
  every input because all platforms measure the same tree.

Per file, the statement set is reconstructed as
`executed ∪ missing` — the pattern already validated in
`merge_coverage.py` — so a statement that one platform's report lists as
missing but another executes is still aggregated correctly, and
`missing = statements − executed` stays exact in the output.

### Consistency prerequisites

The aggregate is only meaningful when all inputs share:

1. **the same source commit** — the workflow checks out one frozen
   revision for every platform leg (`fetch-depth: 0`, SHA recorded in
   `meta.commit_sha`);
2. **the same coverage configuration** — `pyproject.toml`
   `[tool.coverage.*]` (measurement scope `source=vllm_fl`, never
   per-platform);
3. **the same caliber** — every input must be the same kind of report
   (e.g. all core baselines, or all combined core+supplementary).

The tool validates per-file `num_statements` across inputs. Under the
prerequisites above the counts are identical; any file where they differ
is listed in `meta.inconsistent_files`, the result is marked incomplete,
and a warning is printed — **it does not fail the run**, because the
aggregate is a reporting artifact, not a gate.

### Incomplete semantics

`--expected-platforms` names the platforms a complete aggregate should
have (default `kunlunxin,hygon,iluvatar,thead`). Any expected platform
without an input report is recorded in `meta.missing_platforms`,
`meta.complete` is set to `false`, and the tool prints a loud warning:

```
========================================================================
[aggregate] WARNING: AGGREGATE RESULT IS INCOMPLETE
[aggregate] missing platform reports: iluvatar
[aggregate] the union below is a FLOOR for the plan's aggregate target,
           not the target number itself
========================================================================
```

The (incomplete) aggregate is still produced: in CI, a single failed
platform leg must not hide the other platforms' numbers. An incomplete
aggregate is a **floor** — it can only understate the true union, never
overstate it — so it is safe to display but must not be read as the
plan's aggregate acceptance number.
>>>>>>> f1c89a64 (docs(coverage): aggregate README and workflow job-summary fix)

### Usage

```bash
<<<<<<< HEAD
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
=======
python tools/coverage/aggregate_coverage.py \
    --inputs coverage-kunlunxin.json coverage-hygon.json \
             coverage-iluvatar.json coverage-thead.json \
    --platforms kunlunxin,hygon,iluvatar,thead \
    --expected-platforms kunlunxin,hygon,iluvatar,thead \
    --commit-sha <sha> \
    --output coverage-aggregate.json \
    --summary-md coverage-aggregate.md
```

* `--platforms` may be omitted when inputs are named
  `coverage-<platform>(-<tag>).json`; the platform is derived from the
  file name.
* `--summary-md` writes a job-summary table: per-platform
  statements/covered/% plus the aggregate row and the completeness status.
* The output is a standard coverage.py-shaped JSON (`meta` + `files` +
  `totals`) with `meta.caliber = "multi-platform-aggregate"`,
  `meta.participating_platforms`, and the completeness fields described
  above.
* The tool self-checks the same invariant as `merge_coverage.py` —
  every file's `summary.missing_lines == len(missing_lines)` — before
  writing the output; a violation aborts with exit 3.

### CI integration

`.github/workflows/coverage_supplementary_multi.yml` runs an
`aggregate` job (`needs: coverage`, `if: always()`, plain
`ubuntu-latest` — only JSON artifacts are processed, no device). It
downloads the per-platform artifacts, builds the aggregate inputs,
runs the tool, and uploads `coverage-aggregate.json` +
`coverage-aggregate.md` as the `coverage-supplementary-multi-aggregate`
artifact with the summary table appended to the job summary. Because
the job runs `if: always()`, a failed or cancelled platform leg still
yields an aggregate — explicitly marked incomplete.
>>>>>>> f1c89a64 (docs(coverage): aggregate README and workflow job-summary fix)
