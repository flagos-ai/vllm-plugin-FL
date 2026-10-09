# MUSA four-route Gate CI draft

This directory is a local draft. Nothing here has run a vendor import, device
probe, model, build, installation, SSH command, or CI job. The local consumer
tests use labelled filesystem fixtures and do not certify GPU behavior.

The runner executes the original `tools/adaptation-gate-cases` without modifying
its 19 files. It runs `test_text.py`, `test_image.py`, and
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
are `MUSA_CI_MODEL_27B`, `MUSA_CI_MODEL_35B` and `MUSA_CI_PORTS`. It performs a
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
add `--enforce-eager`. Graph routes preserve the existing plugin policy,
including the MUSA FULL-to-PIECEWISE downgrade; the draft does not promote FULL.

Deadlines are startup30s, readiness600s, each original pytest1900s, all clients
in a route5800s, and route6900s. Shell whole-entry timeout is29400s with65s termination
grace. Shutdown first sends TERM to the held, verified own leader PID and waits
15s for normal engine/worker shutdown. A verified own-group KILL15 fallback is
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

`tests/run-consumer-checks.py` performs only AST/memory compilation and pure
stdlib report-consumer filesystem fixtures. `consumer-result01` preserves the
first13-test result; `consumer-result02` preserves the14-test result after
adding the three original client-process requirement. `consumer-result03`
preserves14 passing fixtures for the companion observer's actual flat summary,
including canonical full input/output row hashes and PID-namespace joins.
`consumer-result04` adds original-row/namespace/raw-trace counterexamples.
`consumer-result05` records23 passing consumer/admission/known-child reader
fixtures. No runner, preflight or configuration producer was executed by these
checks. The final filenames and byte hashes are recorded
in `draft-files.json` when this draft is frozen for review.


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
