# MUSA four-route Gate CI runner

The MUSA workflow invokes this runner after ordinary wheel installation. The
GitHub CI job has not been executed in this validation record. Separate
isolated model runs and their actual results are recorded in the
[MUSA deployment guide](musa.md#validation-record). Local consumer tests use
labelled filesystem fixtures and do not certify GPU behavior.

The runner executes the committed `tools/adaptation-gate-cases` without editing
its 19 files during execution. It runs `test_text.py`, `test_image.py`, and
`test_mix_text_image.py` separately for each route and continues after an
ordinary pytest failure. The four sequential routes are:

| Route | Model directory | Mode | Port |
| --- | --- | --- | --- |
| 27b-eager | `/data/models/Qwen3.6-27B` | eager | 18062 |
| 27b-graph | `/data/models/Qwen3.6-27B` | graph | 18063 |
| 35b-a3b-eager | `/data/models/Qwen3.6-35B-A3B` | eager | 18064 |
| 35b-a3b-graph | `/data/models/Qwen3.6-35B-A3B` | graph | 18065 |

Each route contains five original scenarios and 26 logical requests; the full
matrix contains 20 scenarios and 104 logical requests. The consumer requires
the original ten checks for each output. The original long-text request keeps
its 512-token/256-character overrides. SDK retries remain unchanged, so the
wire-attempt count is reported as unknown.

## Environment and source binding

Run inside an already isolated Linux CI container with exactly four logical
MUSA devices and finite container CPU, memory and PID limits. This runner does
not allocate a container, choose physical devices, install dependencies, or
signal another job. Normal Python must provide the installed plugin, actual
vLLM 0.28.0, TorchMUSA, FlagGems, Triton, pytest, OpenAI and HTTPX. It requires
ordinary installed RECORD entries; an editable installation that does not
record the imported source bodies is rejected.

`preflight.py --create-source-binding` records actual committed plugin source,
the original 19 Git LF Gate blobs, the generated version file when present,
imported module source, distribution metadata, and corresponding RECORD
bodies. Supply the exact candidate commit as `--expected-commit`; it must match
the installed source. No version string is treated as a candidate identity.
The same binding is rechecked before every route. This observes selected vendor
source/metadata, not every vendor binary or every ABI.

Before binding creation, configure three dedicated writable caches on a mount
that permits native shared-library loading. The effective APIs are
`triton.knobs.cache.dir`, `flag_gems.utils.code_cache.cache_dir_path()` and
`torch._inductor.runtime.cache_dir_utils.cache_dir()`. Preflight checks those
actual API results, mount flags, and a new write/read/delete file. It preserves
inherited loader/vendor settings; the three explicit cache paths are task
configuration. Errors propagate into the preflight report.

Example preparation after an ordinary no-dependency wheel installation:

```bash
tools="$GITHUB_WORKSPACE/.github/scripts/musa"
kit="$GITHUB_WORKSPACE/tools/adaptation-gate-cases"
cache_root="/root/.cache/musa-gate-$run_id"
mkdir -p "$cache_root/triton" "$cache_root/flaggems" "$cache_root/torchinductor"
TRITON_CACHE_DIR="$cache_root/triton" \
FLAGGEMS_CACHE_DIR="$cache_root/flaggems" \
TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor" \
  "$PYTHON" -I -B "$tools/preflight.py" \
    --repo "$GITHUB_WORKSPACE" --expected-commit "$CANDIDATE_COMMIT" \
    --gate-dir "$kit" --cache-root "$cache_root" \
    --create-source-binding "$RUNNER_TEMP/source-binding-$run_id.json" \
    --output "$RUNNER_TEMP/preflight-binding-$run_id.json"
```

The source binding and output filenames must be fresh. `run_id` is a supplied
32-character UUID hex and `CANDIDATE_COMMIT` is the full checked-out Git OID.
The example does not create observer pins or an authorization receipt.

## Execution and evidence

The workflow can call `bash .github/scripts/musa/run-adaptation-gate.sh` with no
arguments. `gate_entry.py` then uses `MUSA_CI_PYTHON` (or `PYTHON`/`python`),
`GITHUB_WORKSPACE` (or the current repository), `MUSA_CI_EXPECTED_COMMIT`
(or `GITHUB_SHA`/actual `git rev-parse HEAD`), and a fresh UUID run ID. It creates
exclusive input/evidence directories under `MUSA_CI_RESULT_ROOT`
(default`$RUNNER_TEMP/musa-gate`, or`/tmp/musa-gate`) and task caches under
`MUSA_CI_CACHE_ROOT` (default`/root/.cache/musa-gate`). Model and port overrides
are `MUSA_GATE_MODEL_27B`, `MUSA_GATE_MODEL_35B_A3B` and `MUSA_CI_PORTS`. It performs a
bounded actual source/cache/device binding, then creates readonly observer
configuration using only original `case_utils.scenario_requests`; it sends no
request and imports no unused original test. The installed observer wheel must
exist before this entry. All preparation and routes share the shell outer
deadline. Failure to prepare creates an original entry report and does not
start a model.

The explicit argument form remains available:

```bash
bash "$tools/run-adaptation-gate.sh" \
  --repo "$GITHUB_WORKSPACE" --expected-commit "$CANDIDATE_COMMIT" \
  --gate-dir "$kit" --source-binding "$RUNNER_TEMP/source-binding-$run_id.json" \
  --cache-root "$cache_root" --run-id "$run_id" \
  --output-root "$RUNNER_TEMP/musa-gate-$run_id" \
  --observer-config "$ACTUAL_OBSERVER_CONFIG" \
  --observer-source-sha256 "$ACTUAL_OBSERVER_SOURCE_SHA256"
```

Model paths and the four ports have explicit CLI overrides. The server uses
TP4, bf16, length32768, sequences8, batch tokens4096 and memory utilization0.6;
its local-media whitelist is the original Gate images directory. Eager routes
add `--enforce-eager`. Every route passes `--shutdown-timeout 60` so the API
uses drain mode and gives real workers time to close their observer streams.
Graph routes preserve the existing plugin policy,
including the MUSA FULL-to-PIECEWISE downgrade; the runner does not promote FULL.

Deadlines are startup30s, readiness600s, each original pytest1900s, all clients
in a route5800s, and route6900s. Shell whole-entry timeout is29400s with180s
termination grace. Shutdown sends TERM to the held, verified own leader PID;
the API server has90s to finish its60s engine drain and real worker cleanup.
Other owned actors retain15s TERM grace. A verified own-group KILL15 fallback is
recorded as failure. Unknown identity never permits signalling another process.
Every actually sampled child lifetime is retained and rechecked even if it
changes its session. Cleanup scope is these owned process leases in the already
isolated CI job container; it is not a whole-host foreign-job inventory.
Any incomplete preflight, pytest or server cleanup marks remaining entries or
routes `NOT_RUN`; ordinary quality failures continue after complete cleanup.
Leader PID/starttime/session/group, actual argv, relevant task environment,
readiness attempts, complete raw stdout/stderr, JUnit, original JSON hashes,
actual exits or unknowns, and independent process attestations are retained.
Each route writes `logs/server.pid`; an intentional stop is distinguished from
an observed natural server exit. Logs are checked against a64MiB cap; an
oversize/incomplete fingerprint fails acceptance and preserves the raw file.

Only the spawned server session and the three separately spawned pytest client
leases are eligible process identities for observer joins. No full environment,
unknown process command line, credential, or model weight is exported.

## Graph observer integration

Only Gate server processes enable `VLLM_PLUGINS=fl,musa_gate_observer`; each
original pytest invocation adds `-p musa_gate_observer.pytest_plugin`.
Preflight telemetry remains disabled. Producer configuration is readonly and
contains the actual source pins, exact run ID, complete four-route request
matrix, and all 19 Gate fingerprints. Its `trace_directory` must be this new
output root's `observer-traces` directory. It contains no future actual PIDs or
fabricated results.

After all routes stop, the independent summarizer is called once with
`--trace-dir`, `--expected-manifest`, four `--route` arguments, and an exclusive
output. Aggregation verifies its source/run binding, actual invocation exit,
raw input hashes, original JSON hashes, and independently captured API, TP4
worker and three pytest-client PID lifetimes. Every graph logical request must
join to capture/replay on ranks0,1,2,3. Missing, unsupported, incomplete or
unknown evidence fails the full Gate even if business responses pass.

The observer adapter consumes the companion's flat schema:33 input fingerprints
(config plus32 process traces),104 logical joins,52 graph joins and20 original
scenario-file claims. It rejects `route_only` observations and requires all
full-matrix completion flags. Actual installed source/API pins and real graph
capture/replay observations remain required. Local consumer fixtures do
not close that gap. Run without observer options to retain business failures
and diagnostics, but the full matrix will intentionally fail graph acceptance.

## Local checks

`.github/scripts/musa/tests/test_aggregate_gate.py` checks the report consumer,
route admission and owned-child identity contracts with filesystem fixtures.
`test_shutdown_window.py` exercises actual four-route CLI construction and the
owned-process close path with a CPU clock/process fixture. The 26 CPU CI helper
tests pass after the shutdown-window fix; these tests do not execute real
workers or signals on the MUSA host.
`tools/musa-gate-observer/tests/test_summarize.py` checks full-matrix and
single-route observer consumption;
`tools/musa-gate-observer/tests/test_producer_contract.py` checks producer
wrappers with Mock dependencies. Syntax checks and these fixtures do not
execute the CI runner, preflight, model or actual graph capture/replay.

The targeted `tests/unit_tests/test_musa_accelerator_patch.py` CPU Mock
regression passed 48 cases after adding the MUSA synchronization binding.
That result covers callable validation, default/explicit device forwarding,
vendor error propagation and idempotence. Separately, candidate
`f867afb0c7f026bcabc8f80c2a68277a993ae800` completed the isolated CPU09 ordinary
three-wheel installation and source/metadata/RECORD binding with unchanged
vendor packages. Its actual 27B graph runtime30 and 35B-A3B graph runtime27 reached
readiness and recorded native capture/replay, then each failed with a
`sample_tokens` RPC timeout. Each route has 26 failed logical attempts,
three original pytest exits of 1 and five failed real JUnit testcases (2/2/1).
The original request errors retain `checks={}`; unavailable quality checks
were not synthesized. Both unchanged offline partial-route observer consumers
exited 1 on error events, leaving partial and full-matrix completion false.
The preserved eager runs runtime28/29 have 26/26 and25/26 quality successes,
respectively, but both original consumers exit1 because four worker traces lack
`telemetry_end`. The 35B single-text answer lacks literal `limitations`, failing
only the original semantic/order checks. Resource cleanup supplies no observer
end event. These runs used the upstream default abort shutdown with timeout0.
The graceful-window reruns preserve every request, check and observer contract.
Runtime31 adds only `--shutdown-timeout 60` and passes the complete 27B eager
single route: 26/26 quality successes, three pytest exits0, five JUnit passes,
eight real activation/end-bounded traces with no errors, all26 original
response-to-engine joins, and unchanged partial-route consumer exit0. The API
reports drain mode and all workers exiting gracefully; owned cleanup passes.
Runtime32 closes all eight real process traces normally with the same60s drain,
but its original 35B quality remains25/26: the1,578-character single-text
response lacks literal `limitations`, failing only semantic/order checks.
Three original pytest exits are1/0/0 and four of five JUnit testcases pass;
the unchanged consumer exits1 at joins for `original logical quality failure`.
The job exits1 despite API0 and successful owned resource cleanup. No consumer,
original quality threshold or vendor source was changed to pass either route.

All four current routes retain20 original reports and104 logical results:
52 text and52 image, with51 quality successes and53 failures. The12 original
pytest invocations produce20 real JUnit testcases, with9 passes,11 failures and
no error or skip testcase. The original full consumer runs once in default full
mode using all32 byte-verified, unfiltered current trace copies (runtime30/27/31/32),
without substituting old28/29 or selecting a partial route. It exits1 at events
on an actual graph API error. It stops early; this is no claim of complete
parsing or acceptance of all32 files. All full-matrix flags remainfalse, and
all208 required graph request/rank joins have not been verified complete.
Full four-route Gate acceptance is FAIL. Actual manual runs do not establish
execution of the GitHub CI job, which remains NOT_RUN. Automatic MUSA discovery
remains disabled in `.github/configs/platforms.yml` (`musa.enabled: false`).
Historical f80 and initial eager failures are retained in the
[deployment validation record](musa.md#validation-record).


## CI application wiring

The workflow invokes the no-argument shell after setup. The loader's validated
`MUSA_GATE_MODEL_27B` and `MUSA_GATE_MODEL_35B_A3B` variables are the same names
read by `gate_entry.py`. `MUSA_CI_RESULT_ROOT` uses the workflow artifact root;
both its exclusive run-ID output and preparation inputs are uploaded.
Setup installs the verified empty vLLM wheel, the current FL wheel and the
independent observer wheel with no index or dependency replacement. Preflight
pins current candidate Git bytes, including the changed Gate README.

Local syntax and stdlib fixtures do not certify Linux process cleanup, vendor
ABI, graph capture/replay or model acceptance. The CI environment must execute
the actual preflight, all original requests and the complete observer joins.
