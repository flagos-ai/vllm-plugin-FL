# MUSA four-route Gate CI runner

The MUSA workflow invokes this runner after ordinary wheel installation. The
GitHub CI job has not been executed in this validation record. Separate
isolated model runs and their actual results are recorded in the
[MUSA deployment guide](musa.md#validation-record). Local consumer tests use
labelled filesystem fixtures and do not certify GPU behavior.

## Ordinary image and explicit CI dispatch

The [ordinary-image and dispatch instructions](musa.md#ordinary-stack-image-and-ci-dispatch)
describe the six-wheel `/opt/musa/venv` runtime, immutable image binding and
per-run four-device allocation. CI defaults to the published image, its pinned
empty wheel and the `latest` manifest under `/opt/musa`, with literal whole-file
SHA values from `.github/configs/musa.yml`. The existing Job 4e is enabled with
four-card defaults and checks the executing container's device mapping and idle
resources before Torch initialization. A per-run allocation remains an
optional JSON override for the actual executing runner. The workflow entry is
`ci.yml`; a dispatch
selects `platform=musa` and either inventory-only `musa_probe_only=true` or a
full run using the committed defaults or an explicit `musa_allocation` override.
The inventory job reads
sanitized resource observations without a GPU container or Torch import;
its success is not inference or Gate acceptance.

The committed-source FL/observer ordinary builds and local image CPU
installation checks passed. The ordinary image is published, with a successful
normal Docker push and verified manifest/config identity. Use the immutable
RepoDigest in the deployment guide. The new image's full MUSA GPU CI has not run;
the executing Job 4e's complete suite and original Gate remain the acceptance
criteria.
The inventory-only [run 38108902958](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/38108902958)
completed successfully at `2b6bee63ad1b852021a9e166195aa3fac75443f1`.
Lint, ordinary build, discovery and runner probe passed; `test-musa` was
skipped. The CPU07 results below remain separate private model validation
and are not substituted as this workflow's hardware result.

CI setup rebuilds FL and observer from its actual checkout commit into a fresh
venv and explicitly reinstalls the three pinned image dependency wheels.
The nested venv does not inherit the image's `/opt/musa/venv` packages.
Source binding and vendor/RECORD checks remain mandatory. The full workflow
sequences unit, functional, serial e2e and benchmark stages before the original
Gate; MUSA platform concurrency serializes competing uses of the allocation.
The final acceptance record must identify the actual workflow, checkout and
published image RepoDigest, together with all four original route results.

## CPU07 NDmul safe512 candidate

Follow the current [installation and run instructions](musa.md#cpu07-ndmul-safe512-candidate).
The source recipe is `latest-bfc6-ndsafe512`, whereas the CI interface is
`MUSA_CI_DEPENDENCY_PROFILE=latest` with three actual SHA-bound dependency
wheels and preserved source/repack lineage. CI setup also requires committed
FL/observer source matching `MUSA_CI_EXPECTED_COMMIT` and `GITHUB_SHA` when set;
the local CPU07 ordinary repack does not create a new Git commit. CPU07 installed a musafix2 Gems
ordinary repack; it did not run the new source-build command. Its other six
wheel bodies and vendor/image inputs were preserved.

Graph routes retain default asynchronous scheduling, TP4, bf16, 4096 batch
tokens, 8 sequences and the unchanged 32768-token model limit. They explicitly
use capture sizes `[1,2,4,8,16,256,1024,4096]` and GPU memory utilization 0.92.
Eager routes retain `--enforce-eager` and utilization 0.6. No graph mode is
forced; original inputs, retries, check names and thresholds are unchanged.

Sparse buckets cover every integer batch count through 4096. The actual
graph runs below completed with this memory budget; that observation does
not guarantee fit for another model, input envelope or shared environment.

| CPU07 route / actual evidence | Original quality | JUnit / complete telemetry |
| --- | --- | --- |
| 35B-A3B eager / runtime68r2 | 26/26; 260/260 checks true | 5 passing cases; 8 clean ends |
| 27B eager / runtime70 | 26/26; 260/260 checks true | 5 passing cases; 8 clean ends |
| 35B-A3B graph / runtime81 | 26/26; 260/260 checks true | 5 passing cases; 8 clean ends |
| 27B graph / runtime80 | 26/26; 260/260 checks true | 5 passing cases; 8 clean ends |

**The CPU07 four-route Gate passed for these exact runs.** All 104 original
logical outcomes passed all 1040 original checks, with zero false or absent
checks. The 12 JUnit XML files contain 20 passing cases, zero failures, errors
or skips. All 32 process streams have normal final ends with no errors,
drops, cap or I/O failures; all four owned jobs stopped with PID 0, no OOM
and empty cleanup errors.

The unchanged strict consumer ran once in full-matrix scope and exited 0.
It accepted all 32 byte-identical original streams (465,877,818 bytes), all
104 logical results and 20 scenarios, and certified 52 graph-request joins
and 208 request/rank replay joins. Its five full-matrix completion flags are
true and errors are empty. The consumer's separate source/GPU/ABI/model/Gate
acceptance flags remain false by design; the combined Gate conclusion uses
the independently checked original quality, JUnit and lifecycle evidence.
No cases, retries, check names or thresholds were changed.

This establishes the recorded CPU07 model/observer scope, not a compiler
root-cause fix, a general memory-fit guarantee or GitHub runner qualification.
This CPU07 record does not qualify the new image or GitHub hardware workflow.
Current ordinary-image/CI progress is recorded [above](#ordinary-image-and-explicit-ci-dispatch).
The finite evidence budgets are 2 GiB per stream and 12 GiB total with
streaming aggregation hashes; they are not a process-memory bound.

The capacity patch's aggregate-gate suite passed 31 CPU tests. The applied
graph-coverage/memory argv fixture passed two CPU tests, checking actual four
route arguments, full capture padding through 4096, eager/graph budgets and
the shutdown contract. These are CPU fixtures, not a GitHub workflow claim.

### Earlier CPU07 trials

Runtime71 passed all quality checks with maximum capture size 16, but three
short image requests fell back to NONE during mixed-prefill batches and had
no replay join on any rank; the original full consumer correctly failed.
Runtime76 used dense 13-size capture and utilization 0.9: it recorded 25/26
quality passes, 258 true and two false checks, four passing/one failing JUnit
cases and eight clean ends. Its long-text output lacked the original required
literal `limitations`; that failure is preserved.

Runtime74 at 0.6 and runtime77 at 0.9 failed before READY with no available KV
cache blocks. Runtime78 used the sparse eight sizes at 0.9 but had 0.51 GiB
of available KV memory versus 0.57 GiB required at the original 32768-token
limit (estimated supported length 28224). These startup failures have no
model client or executed quality cases and remain model Gate NOT_RUN.
Runtime79 was unexecuted. The final graph runs changed only the budget to
0.92; their actual successes do not erase those original failure receipts.

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

## Historical CPU06 musafix1 dependency profile

The CPU06 private environment used FlagGems `bfc6c49d2c4d06e5deac9784f203730c8372a84a` and FlagTree `69148faf5cc03edb0a5737abd4b75b691132cea9`. The [portable recipe](../../tools/musa_dependency_recipe/README.md) supplies an explicit latest Gems profile and Tree header-patch manifest, while retaining the historical fixed profile. PR #6894 contributes the reviewed split-KV production changes; the separate device checks tracked by [FlagGems #6973](https://github.com/flagos-ai/FlagGems/issues/6973) are not a model graph-cause claim or a Gate prerequisite. [FlagTree #1344](https://github.com/flagos-ai/FlagTree/issues/1344) records the optional OFF/OFF header build guard, not general GPU compatibility.

| Scope | Actual result |
| --- | --- |
| Gems/Tree CPU source builds | Ordinary wheels and complete RECORD checks passed; explicit patched source profiles |
| CPU06 private installation | Seven ordinary wheels verified; only Gems/Tree replace CPU05; other five wheel bodies and vendor fingerprints unchanged |
| CONFIG06 | Seven metadata rows, original 19 Gate files, four routes × 26 requests and four ranks bound successfully |
| Current captured-input implementation | Strong capture-input references and checked copies before replay; 34 CPU contract cases passed |
| CPU06 original four-route GPU Gate | 87/104 quality passes; strict full-matrix consumer FAIL; 35B-A3B eager remains unresolved |
| Historical migrated baseline | 35/104 quality passes; full Gate FAIL |

The latest original routes used the same CPU06 dependency profile and unchanged Gate cases:

| Route / actual run | Quality | JUnit | Saved observer evidence |
| --- | --- | --- | --- |
| 27B graph / runtime61r2 | 26/26; 260 checks true | 5 PASS | All eight normal ends; no errors or drops |
| 27B eager / runtime62 | 26/26; 260 checks true | 5 PASS | All eight normal ends; no errors or drops |
| 35B-A3B graph / runtime63 | 26/26; 260 checks true | 5 PASS | All eight normal ends; no errors or drops |
| 35B-A3B eager / runtime64 | 9/26; 90 checks true, 170 absent | 2 PASS / 3 FAIL | Four worker ends missing; API/client error ends |

All 104 original logical case outcomes were recorded: 87 passed, and 17 returned exceptions in runtime64. The 870 evaluated checks are true; 170 required checks are absent, not evaluated false. The 12 JUnit files contain 17 passing and three failing cases, with no errors or skips. The three passing routes have complete saved raw observer evidence and empty cleanup errors. Runtime64 retains its cleanup error despite the later owned stop/PID 0. Its first observed native error was reported while autotuning BF16 ND multiplication at `N=1179648`, `BLOCK_SIZE=1024`; the later vision `cu_seqlens.tolist()` stack is an asynchronous error boundary, not proof of the responsible kernel. No runtime64 shape/stride metadata was recorded.

The unchanged default full-matrix consumer ran once over byte-identical copies of all 32 streams (506,767,609 bytes). It exited 1 with `ContractError: unsupported/error/unknown event`; its fail-fast summary does not certify all 32 streams or complete graph joins. **The historical CPU06 musafix1 full Gate is FAIL.** These results do not replace the historical matrices below.

The CPU06 Tree wheel used explicit TRITON_BUILD_FLAGPRISM=OFF/TRITON_BUILD_PROTON=OFF plus the header guard, not a pristine default build. CPU checks do not establish native ABI, graph numerical or model acceptance. The observed TP4 communication route uses MCCL; FlagCX was not installed. Diagnostics remain disabled by default, automatic discovery remains `musa.enabled: false`, and GitHub CI is NOT_RUN. Requests, retries, check names/thresholds and strict observer joins are unchanged. Detailed run/candidate boundaries are centralized in [the runner validation history](#validation-history-and-candidate-boundaries); successful resource release never repairs failed quality, missing end events or cleanup errors.

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

Observer raw files have finite 2 GiB per-file and 12 GiB total budgets. The CI aggregator now verifies their whole SHA256 with streaming reads using those same limits; overflow or incomplete evidence still fails. These are evidence-retention limits, not a proof that every workload fits or a bound on consumer RSS. The separate server-log and summary limits remain unchanged.

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
owned-process close path with a CPU clock/process fixture. The aggregate-gate suite
passed 31 CPU tests after the streaming observer-capacity change;
these tests do not execute real workers or signals on the MUSA host.
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
At that validation snapshot, full four-route Gate acceptance was FAIL. The
manual runs did not establish execution of GitHub CI, which was NOT_RUN;
automatic discovery was disabled (`musa.enabled: false`).
Historical f80 and initial eager failures are retained in the
[deployment validation record](musa.md#validation-record).


## CI application wiring

The workflow invokes the no-argument shell after setup. The loader's validated
`MUSA_GATE_MODEL_27B` and `MUSA_GATE_MODEL_35B_A3B` variables are the same names
read by `gate_entry.py`. `MUSA_CI_RESULT_ROOT` uses the workflow artifact root;
both its exclusive run-ID output and preparation inputs are uploaded.
Setup installs the verified empty vLLM wheel, the current FL wheel and the
independent observer wheel without index access. The default dependency profile
retains vendor Gems/Tree; the explicit SHA-bound latest profile adds only its
three ordinary dependency wheels in the private venv. See the
[profile configuration](musa.md#select-the-ci-dependency-profile). No profile
changes original requests, checks, observer joins, Torch or SDK. Preflight
pins current candidate Git bytes, including the changed Gate README.

## Validation history and candidate boundaries

This section preserves earlier immutable run records. The current pinned profile and its acceptance status are summarized above; historical source counts and artifact hashes do not identify the later HEAD builds.

The portable [dependency recipe](../../tools/musa_dependency_recipe/README.md) includes a separately pinned Tree compatibility patch for base `ba25a9551b2e5157d843affe6f4c2cada1dcfde8`. The candidate's explicit `0.7.0+mthreads3.6.gitba25a955.profilefix1` MUSA core build disables Prism and Proton and changes one header directive, preventing disabled Proton headers from being included. Preserve that base-plus-patch lineage and actual build receipt in the SHA-bound latest dependency manifest; a base commit alone must not imply an unmodified wheel. The patched ordinary wheel build, new private seven-wheel installation and CPU config37 now have successful actual CPU receipts, complete RECORD/direct-URL/static-origin checks and stopped owned containers with PID 0. Tree post-build source verification confirmed the declared one-header change and 6461 unchanged originals. The artifact is `0.7.0+mthreads3.6.gitba25a955.profilefix1`, `cp310-cp310-linux_x86_64`, 180,144,953 bytes, SHA256 `e11252f44e39742553ca52b267d617cae666ee9129da57fae91f76d774a72019`; [FlagTree #1344](https://github.com/flagos-ai/FlagTree/issues/1344) records its build guard. Diagnostics remain disabled by default. [FlagGems #6973](https://github.com/flagos-ai/FlagGems/issues/6973) separately records the public-source accelerator checks; issue state is not a Gate prerequisite and no model graph cause is established. This build-boundary result makes no target-import, default-build, GPU-fault, ABI or full-Gate success claim.

Runtime37 passed the native device/Tensor probe, but an additional owner caused the before-serve resource guard to stop before API/client/Gate execution. Its zero-request result is **NOT_RUN**, not model or graph-quality FAIL. Owned resources closed at PID 0 with no cleanup errors, and complete original records remain.

The second-node fixture migration passed whole-archive and extraction checks for 9073 regular files plus four symlinks. A separate actual fixed-image CPU container reread every regular file through EOF/SHA256, matched 185 present vendor files and one expected absent metadata path, and checked the existing configuration/install pins. Child/job/owned stop exited 0, PID 0, no OOM, cleanup errors empty; only its unique stopped container object was subsequently removed. These are CPU/material-provenance checks and do not count as GPU execution or original logical quality results.

Runtime42 reached API readiness on the migrated latest environment, and all three original pytest groups exited 1. The five original reports contain 26 logical results, all failed (0/26); exceptions left their `checks` dictionaries empty, so this is not a count of 260 evaluated false booleans. JUnit records five failures, zero errors and zero skips.

The complete original kernel journal contains eight explicit `GPU_CDM_STREAM` submission response FAULT records: two for each owned TP0, TP1, TP2 and TP3 worker at 08:26:39-40 UTC. The full original records establish faults for all four workers, extending the earlier bounded sample. These precede the server's 08:28:13 UTC EngineCore `get_response`/shared-memory `acquire_read` and `sample_tokens` RPC timeout, followed by an MCCL unhandled MUSA error. No particular kernel, FlagGems or FlagTree cause has been identified.

Each of the four worker traces recorded 455 capture-success events, 195 replay-success events and four scheduler-return events. These are observed call/return boundaries, not proof of native asynchronous completion or successful request/rank joins. All eight observer streams have end events, but each worker records a shutdown MUSA synchronization error; end presence does not establish observer acceptance.

The natural API process exited 0, but graceful shutdown recorded an assertion that owned worker/engine/client processes remained. Its cleanup-error list is nonempty. The exact owned container was subsequently stopped with exit 137, PID 0, no OOM and stop exit 0. That final resource release does not erase the earlier cleanup error or make the route pass. The historical runtime42 27B graph result is **FAIL**. At that snapshot, no graph repair had been established. The original four-route requests, thresholds and strict consumer remain unchanged.

Runtime43r2 completed the unchanged 27B eager route: **PASS**, 26/26 logical results and all 260 original quality checks passed, independently recomputed from the saved responses. All three original pytest groups exited 0 and all five JUnit cases passed, with zero errors or skips. The eight observer streams have 2023 events, activation first, normal `telemetry_end` last and no errors; the unchanged strict consumer returned 0 with `--partial-route qwen36-27b-eager` and `route_observer_complete: true`. Eager graph capture/replay is not applicable. The API, clients and job exited 0; cleanup errors were empty, and the exact owned container was stopped with PID 0, no OOM and stop exit 0. Earlier runtime43 attempts stopped before container creation and remain preserved launch failures, not model-quality results. This is a complete single-route result; the consumer's full-matrix flag remains false.

Runtime44 executed the unchanged 35B-A3B eager route: **FAIL**, 9/26 logical quality passes. The nine text responses passed all 90 original checks; the 17 image/mixed requests returned exceptions with empty checks. The three pytest groups exited 0/1/1, and JUnit records two passes and three failures. The first observed MUSA illegal-memory error occurred during the vision attention path (`cu_seqlens` conversion to a CPU list), with later worker stacks reporting the same asynchronous error. This is an observed failure boundary, not identification of the kernel that caused it. All four worker traces lack `telemetry_end`; three contain scheduler exceptions. The strict partial consumer exited 1 at the API error and did not parse all eight streams to completion. Cleanup retains the graceful-API remaining-process assertion despite the subsequent successful owned stop and PID 0.

Runtime45 executed the unchanged 35B-A3B graph route: **FAIL**, 0/26 logical quality passes and empty checks for all 26 exception results. All three pytest groups exited 1 and all five JUnit cases failed. Its first EngineCore shared-memory/RPC timeout followed 46 recorded output tokens. The captured live stacks show all four worker MainThreads waiting in `prepare_inputs_event.synchronize()` and the TP0 output-copy thread waiting in its output event's `synchronize()`; the other three output-copy threads were idle in the queue. These are wait sites, not proof of a stream mismatch or a particular kernel/package cause. A recorded HTTP response was status 500 and does not establish quality acceptance. The four worker traces lack end events. The strict partial consumer exited 1 at the API error before complete parsing; graceful cleanup again retains the remaining-process assertion. The final exact owned stop exited 0 with PID 0 and no OOM.

The historical cf174/`ba25a9551b2e5157d843affe6f4c2cada1dcfde8` baseline uses the original `f867afb` FL source and the corrected `cf174` Gems/explicit-profile Tree dependencies described above:

| Original route | Runtime | Logical quality passes | Result |
| --- | --- | --- | --- |
| 27B eager | 43r2 | 26/26 | PASS, complete single-route observer |
| 27B graph | 42 | 0/26 | FAIL, four-worker submission faults/RPC timeout |
| 35B-A3B eager | 44 | 9/26 | FAIL, vision-path illegal-memory observation |
| 35B-A3B graph | 45 | 0/26 | FAIL, event waits/RPC timeout |

All four routes executed all 20 original scenario reports and 104 logical requests: **35 quality passes and 69 failures**. The saved responses' 350 evaluated original checks are true; 690 required checks are absent because 69 requests returned exceptions. Those absent checks are not 690 evaluated false booleans. The 12 original JUnit files contain 20 cases: seven pass, 13 fail, zero errors and zero skips. All 32 original streams were independently read as raw evidence. A raw whole-file read does not establish successful parsing by the strict consumer. The passing eager consumer used `--partial-route`; the failed 35B consumers stopped at their first API error. The unchanged default full-matrix consumer has now run once with all four routes and byte-identical copies of all 32 streams. It exited 1 on the 27B graph API exception with an events ContractError. Its fail-fast summary records the eight 27B eager streams and the next API stream; this prefix does not establish successful parsing of all 32 streams or complete graph request/rank joins. **The historical cf174/`ba25a9551b2e5157d843affe6f4c2cada1dcfde8` baseline full Gate is FAIL**; no aggregation or resource stop can turn the failed quality/observer/cleanup records into a pass.

A separate native capture-stream FL candidate completed its own CPU-only ordinary seven-wheel installation and static source/RECORD/direct-URL checks, with vendor fingerprints unchanged. Its repacked FL wheel changes only `vllm_fl/compilation/graph.py` and its RECORD; the original distribution version stays `0.0.0`, and the preserved `f867afb` commit identifies the historical base, not the changed member. The candidate obtains the native MUSA current stream for capture while retaining the existing path on other backends.

Runtime48 executed this candidate's unchanged 27B graph route: **FAIL**, 0/26 quality passes, 26 API exceptions with empty checks, three pytest exits of 1 and five JUnit failures, with zero errors or skips. Each worker recorded 455 captures, 390 replay method returns and seven scheduler returns before the EngineCore shared-memory/RPC timeout; the scheduler dump recorded five output tokens. Those Python/native-method returns do not prove native graph completion. All eight streams have end events, but all four worker ends retain shutdown MUSA synchronization errors. The unchanged strict partial consumer exited 1 at the API exception before accepting all eight streams. Graceful shutdown retained the remaining-process cleanup assertion; the later exact owned stop returned 0 with PID 0, no OOM and container exit 137.

The capture getter change did not make this route pass or establish the cause of the baseline faults. This is a separate candidate result and does not overwrite the baseline 35/104 matrix. The other candidate routes and a full candidate matrix have not run. The separately installed diagnostic plugin remains disabled by default.

A second candidate, CPU02, retains the native MUSA capture-stream getter and adds MUSA-only `capture_error_mode="thread_local"`. Its separate private environment completed seven ordinary-wheel installation and installed payload/RECORD/direct-URL verification; the six dependency wheels and image/vendor fingerprints remained unchanged. The FL wheel again changes only `vllm_fl/compilation/graph.py` and its RECORD, retaining distribution version `0.0.0` and the historical `f867afb` source base. The 12 GraphWrapper CPU contract cases passed, including stream selection and capture-keyword forwarding. Those CPU results establish installation and Python behavior, not native capture compatibility or model acceptance.

Runtime50 attempted this candidate's unchanged TP4 27B graph route with default asynchronous scheduling. Startup failed during capture: the first native fatal at **2026-10-10 14:29:30.317 UTC** was the MCCL watchdog path `WorkMCCL::finishedGPUExecutionInternal` → `watchdogHandler` → `mcclCommWatchdog`, reporting `operation not permitted when stream is capturing`. Engine initialization cancellation followed. There was no READY, API trace or client execution, and none of the 26 original requests ran: **Startup FAIL / model Gate NOT_RUN**, rather than 0/26 quality results. The four worker traces record captures but zero replay, scheduler-return or end events; they do not satisfy the eight-stream observer contract or establish native graph completion.

The original job exited 1 and retained the graceful-shutdown error `API exited unsuccessfully before cleanup`. The subsequent exact owned stop returned 0; final inspection recorded PID 0, no OOM and container exit 137. These resource records do not erase the startup or cleanup failure. `thread_local` did not make this configuration pass; the evidence does not establish a generally unsupported MCCL graph mode or a responsible kernel/package. The baseline remains 35/104, and the other candidate routes and full candidate matrix have not run. Requests, quality thresholds, observer/consumer rules, disabled automatic discovery and GitHub CI NOT_RUN remain unchanged. The diagnostic plugin stays disabled by default.

A third candidate, CPU03, keeps the native MUSA capture-stream getter with the original global capture mode and adds MUSA-only `vllm::all_reduce` partitioning in `platform.py`. It does not include the failed `thread_local` experiment. The separately installed ordinary FL wheel changes exactly `vllm_fl/compilation/graph.py`, `vllm_fl/platform.py` and RECORD; its other 242 payload members match the original wheel. Distribution version `0.0.0` and the historical `f867afb` source base remain explicit. Seven ordinary wheels passed installed payload/RECORD/direct-URL checks, and the six dependency wheels plus image/vendor fingerprints stayed unchanged. The owned CPU installer and config containers completed successfully and were removed after stopped-PID-0 inspection; their private environment and evidence are preserved.

All 18 CPU configuration/FX contract cases passed. They exercise the real configuration initializer, operator partitioning and graph-wrapper selection with explicit compiler/native-graph test doubles; they establish the Python splitting contract, not native MCCL capture/replay or a repaired model Gate.

The first runtime54 harness stopped before model startup because its Docker create label used `gate-runtime50` while the ownership check required `gate-runtime54`. Its model Gate is NOT_RUN; the failed owned object was removed with the original evidence retained. Runtime54r2 then reached READY with TP4/PIECEWISE and `vllm::all_reduce` in `splitting_ops`. All 26 original requests ran: **1/26 quality PASS, 25 FAIL**, with all three pytest groups exiting 1 and all five JUnit cases failing. The recorded checks contain 77 true, 23 false and 160 absent values; exceptions with empty checks are not fabricated boolean failures. The one passing request is `text_concurrent_8/text-03`.

The observed native error appeared during vision processing: a `mul_generic_nd_kernel` autotuning launch with N=1105920 preceded the illegal-memory observation at SDPA `cu_seqlens.tolist()`. Asynchronous error reporting does not identify the responsible kernel. This candidate progressed beyond the earlier capture startup failure, but it did not repair model quality or establish native correctness.

The original observer also exhausted its separate 64 MiB per-process budget. Each of the four worker files preserves 1351 capture records, 32359 replay records and 168 scheduler returns, then stops without `telemetry_end`; API and three client traces have normal ends. Repeated cap-error reporting produced a large stderr log, and the four worker files together exceed the old consumer's 256 MiB total input budget used for that run. These complete original failure artifacts are retained: Python capture/replay returns do not prove native completion, and this run cannot satisfy the eight-stream observer contract. A separate ordinary observer wheel now uses immutable profile references expanded before the original strict checks, finite 2 GiB per-stream/12 GiB total input budgets and bounded overflow reporting. Overflow remains a failure, and the capacity change does not convert this original run into a PASS.

The job exited 1 and retained the graceful-shutdown assertion that owned worker/engine/client processes remained after natural API exit. Exact owned stop returned 0, with final PID 0, no OOM and container exit 137; those resource records do not erase the Gate or telemetry failures. At that runtime54r2 snapshot, other candidate3 routes/full matrix were NOT_RUN. The subsequent candidate3 eager runtime55 returned 10/26 quality passes (nine text plus one image), 100 true checks and 160 absent checks; its pytest exits were 0/1/1 and JUnit had three passes/two failures. Four worker ends were absent, and the vision illegal-memory observation followed ND-mul metadata N=1290240. The original cleanup failure is retained; asynchronous error exposure does not identify the responsible kernel. The original requests and quality thresholds, baseline **35/104**, disabled automatic discovery and GitHub CI NOT_RUN remain unchanged.

The subsequent private candidate uses actual public source builds at FlagGems `bfc6c49d2c4d06e5deac9784f203730c8372a84a` and FlagTree `69148faf5cc03edb0a5737abd4b75b691132cea9`. The corrected Gems ordinary wheel has explicit archive version `5.4.0+archive.gbfc6c49d2c4d.pr6894.musafix1`; its fresh-master comparison retains exactly the four reviewed production changes plus version packaging. Tree is the explicit MUSA Prism-OFF/Proton-OFF header-patched profile, version `0.7.0+mthreads3.6.git69148faf.profilefix1`. The new Tree receipt verifies the 6463-file original archive and declared header change; it does not supply the older `ba25a9551b2e5157d843affe6f4c2cada1dcfde8` build's separate post-build all-other-files proof. These CPU build receipts do not establish default-profile, native ABI or model acceptance.

CPU06 privately installed seven ordinary wheels and verified complete wheel/installed `RECORD` files, payloads, direct URLs and selected module origins. Only Gems and Tree replace the CPU05 dependency wheels; the other five wheel bodies, including FL, observer and diagnostics, are unchanged. Vendor/image fingerprints remained unchanged. CONFIG06 successfully bound seven metadata rows, the original 19 Gate files, the same four routes and 26 requests per route to a fresh UUID. Both owned CPU jobs completed with stop exit 0, PID 0, no OOM and empty cleanup errors; their unique container objects were removed while the environment, configs and evidence were retained. These are actual installation/configuration results, separate from the then-pending CPU06 104-request GPU Gate, whose completed results are preserved above. The optional diagnostics plugin remains disabled by default.

The current MUSA GraphWrapper retains capture input tensors strongly, then validates the complete pytree, tensor metadata and alias relationships before copying compatible new inter-piece input values into the captured storage before replay. Non-MUSA behavior retains its existing path. All 34 CPU contract cases passed for this implementation; CPU05 repacked only `graph.py` plus RECORD, preserving its other 243 wheel members, and CPU06 reuses that wheel unchanged. These checks establish Python input-lifetime/staging behavior, not native graph numerical or model acceptance.

Tiny58r3 ran the four-rank GraphWrapper-to-MCCL-to-GraphWrapper reproduction: eager completed 240 checked iterations, whereas all 240 graph replay iterations failed the independent numerical oracle. The recorded capture input weak references were dead and incoming inter-piece pointers differed from captured pointers. This exposes the captured-input lifetime boundary in that reproduction, without proving the cause of an earlier model fault. Tiny58r4 and Tiny58r5 stopped in wrappers before numerical rank execution and remain numerical NOT_RUN; they are not successful retests of the input change.

Runtime60 used the earlier cf174/`ba25a9551b2e5157d843affe6f4c2cada1dcfde8` dependencies and CPU04 observer/diagnostics with the unchanged 27B eager Gate. All 26 logical outputs passed all 260 original checks, the three pytest groups exited 0 and all five JUnit cases passed. Its strict route remains FAIL: the graceful-shutdown wrapper still required the old CPU03 venv executable, rejected the actual CPU04 API argv before sending TERM, and left the API and four worker end events absent. The original cleanup error and incomplete streams are preserved; the later successful owned stop/PID-0 release and removal do not repair that evidence. This is a functional 26/26 result with a wrapper cleanup failure, not a model-quality failure or complete Gate PASS.

The observed TP4 communication route is MCCL; FlagCX was not installed for these runs. The CPU06 graph route and its four-route GPU matrix were pending at that historical snapshot; their completed 87/104 quality result and strict failure are preserved above. The original request bodies, timeouts, retries, ten check names and thresholds, strict observer joins and failure handling remain unchanged. At that historical snapshot, automatic discovery was `musa.enabled: false` and GitHub CI was NOT_RUN; neither CPU success nor a diagnostic reproduction certified all 104 requests.


Keep the original 104 requests, 20 reports, 32 observer streams and strict full consumer; no CPU record, component-only suite or earlier matrix substitutes for those executions. Automatic MUSA discovery is now enabled in Job 4e; acceptance requires its actual new-image suite and complete original Gate results.

Local syntax and stdlib fixtures do not certify Linux process cleanup, vendor
ABI, graph capture/replay or model acceptance. The CI environment must execute
the actual preflight, all original requests and the complete observer joins.
