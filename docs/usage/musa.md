# MUSA deployment and adaptation Gate

This branch uses vLLM **0.28.0**. Build the upstream `empty` wheel and the FL
plugin as ordinary wheels in a private environment that inherits the MUSA
image's vendor packages. The older MUSA guide using vLLM 0.24.0 describes the
same installation pattern; its vLLM version and image are separate baselines.

## Environment observed during this upgrade

| Component | Observed version or configuration |
|---|---|
| Accelerator | Four allocated Moore Threads S5000 devices, TP4 |
| Upstream | vLLM 0.28.0, commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| Empty wheel | `vllm 0.28.0+empty`; SHA256 `2306c2ea119182a5690eb05776cdffa4ba097a92ba13114a6cb49722d0b53fec` |
| Historical candidate | `f80f78c466f4d55feed60120d1d6848f259991b9` |
| Runtime-tested candidate | `f867afb0c7f026bcabc8f80c2a68277a993ae800` |
| Vendor image ID | `sha256:863675633eb0ffd7055d60351ca9f208f271a60dc7cae811a7c68548627fdde9` |
| MUSA SDK / driver | 4.3.5 / 3.3.5 |
| PyTorch / TorchMUSA | 2.9.0 / 2.9.0 |
| FlagGems | `5.3.2.post1.dev22+gb1f939eb5.d20260804` |
| FlagTree | `0.6.1a2+mthreads3.6` |
| Communication | Actual four-rank initialization used MCCL |
| Resource limits | 24 CPUs, 256 GiB memory, 4096 PIDs, private IPC, 16 GiB shared memory |
| JIT cache | Private 8 GiB tmpfs with `rw,exec,nosuid,nodev` |

These values identify the environment that was inspected. They do not identify
a published v0.28.0 MUSA image. The existing CI tag
`harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.20.2-musa-ci` remains a vendor
baseline; it must be validated with the selected empty wheel and candidate.
`MUSA_CI_IMAGE` can select another validated image or digest.

FlagGems and FlagTree were present and imported from the vendor image. No
usable FlagCX library/wrapper pair was found in the bounded directories
inspected in that environment, and `FLAGCX_PATH` was unset. This is a statement
about that inspection, not every directory on a host. The observed MCCL route
does not require a guessed `FLAGCX_PATH`. To exercise FlagCX, install and verify
its matching library and wrapper separately and record that route's result.

## Build without changing vendor packages

Start from a MUSA image with compatible TorchMUSA, FlagTree, FlagGems, and the
build/client dependencies already installed. Network downloads are unnecessary
in the isolated installation step. Produce the empty wheel once from the exact
upstream source revision:

```bash
git -C /path/to/vllm rev-parse HEAD
# Required result: 2cf0a6915ce544dc493a0990f2ea38d81601128a
mkdir /path/to/new-empty-wheelhouse
cd /path/to/vllm
VLLM_TARGET_DEVICE=empty python -m pip wheel \
  --no-index --no-deps --no-build-isolation \
  --wheel-dir /path/to/new-empty-wheelhouse .
sha256sum /path/to/new-empty-wheelhouse/*.whl
```

Verify that the wheel METADATA names `vllm` with version `0.28.0+empty`.
Supply its **actual** SHA256 when using another independently built wheel;
do not copy the observed hash above onto different bytes.

Build the plugin and install into a fresh venv. Use `--system-site-packages` to
inherit vendor dependencies, and `--ignore-installed` to ensure vLLM and FL are
installed in the private venv despite any inherited distributions:

```bash
cd /path/to/vllm-plugin-FL
python -I -B -m venv --system-site-packages --without-pip /path/to/new-musa-venv
PY=/path/to/new-musa-venv/bin/python
mkdir /path/to/new-plugin-wheelhouse
"$PY" -I -B -m pip --isolated wheel --no-index --no-deps --no-build-isolation \
  --wheel-dir /path/to/new-plugin-wheelhouse .
"$PY" -I -B -m pip --isolated install --ignore-installed --no-index --no-deps \
  --no-cache-dir --no-compile \
  /path/to/verified-vllm-0.28.0+empty.whl \
  /path/to/verified-vllm_plugin_fl.whl
```

Replace the wheel placeholders with real complete wheel filenames. Do not set
`VLLM_VENDOR=cuda` for this MUSA build. Do not uninstall TorchMUSA, Torch,
FlagTree/Triton or FlagGems to make pip dependency resolution succeed. Import
provenance must show `vllm` and `vllm_fl` inside the new venv and vendor modules
in the image's existing site-packages. A successful wheel build alone does not
prove accelerator inference.

## Allocate devices and executable caches

`MTHREADS_VISIBLE_DEVICES` selects physical devices when the MUSA runtime
creates the container. `MUSA_VISIBLE_DEVICES` selects the device ordinals
visible to processes inside that container. Validate the actual native PCI
mapping before inference; the two masks are not interchangeable.

For the inspected allocation, physical devices `4,5,6,7` became logical devices
`0,1,2,3`. Their respective PCI bus IDs were `0000:83:00.0`, `0000:a3:00.0`,
`0000:c3:00.0`, `0000:e3:00.0`. Use the allocation assigned to your runner;
do not copy these host-specific numbers onto a different machine.

Configure all three installed cache APIs before importing the model:

```bash
export GEMS_VENDOR=mthreads
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1
export VLLM_NO_USAGE_STATS=1
export TRITON_CACHE_DIR=/root/.cache/triton
export FLAGGEMS_CACHE_DIR=/root/.cache/flaggems
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
"$PY" -I -B .github/scripts/musa/cache_check.py
```

The effective cache mount must permit execution. On the inspected container,
`/tmp` had `noexec`; putting generated shared libraries there produced loader
errors. Triton, FlagGems and TorchInductor each need their supported cache
variable pointing to the private executable mount. The cache check records
actual module paths/hashes, effective directories, mount flags, and a temporary
write/read/remove check. Keep `nosuid,nodev` on the cache mount. Do not change
global loader paths or modify vendor Python files to redirect these caches.

## Run the complete Gate

The unchanged [adaptation cases](../../tools/adaptation-gate-cases/README.md)
contain five scenarios per route: one long text, eight concurrent text, one
image, eight concurrent image, and eight mixed requests. Each route has
26 logical SDK requests; all four routes have 104, comprising 52 text and
52 image requests. SDK retries may produce additional HTTP attempts.

| Model | Execution | Example distinct port |
|---|---|---:|
| Qwen3.6-27B | eager | 18062 |
| Qwen3.6-27B | graph | 18063 |
| Qwen3.6-35B-A3B | eager | 18064 |
| Qwen3.6-35B-A3B | graph | 18065 |

Use TP4 and read-only model files. The server must permit the **existing** Gate
`images` directory through `--allowed-local-media-path`; do not regenerate the
PNG fixtures during acceptance. Eager uses `--enforce-eager`. Graph acceptance
requires actual MUSA capture and replay associated with original requests,
engine IDs, batches, ranks, and graph generations. Merely removing the eager
flag or seeing a capture progress message is insufficient.

Execute all three original pytest files even when an earlier file fails. Keep
the original messages, SDK timeout/retry behavior, sampling, response reports,
and all ten quality checks. A missing scenario or graph observation fails the
Gate. Preserve complete server/client stdout, stderr, actual exits, readiness,
all 20 scenario JSON reports and provenance. Unknown exits must remain unknown.

For a remote run, use tmux with a distinct session name and an outer deadline
so SSH disconnection does not terminate the test. Start one route at a time,
then stop only its owned process/container and confirm it has exited before
the next route. Avoid clearing caches or stopping processes belonging to other
users. A ready endpoint or a single successful text request is a smoke result,
not a complete Gate result.

## Existing CI integration

The MUSA jobs use `.github/configs/musa.yml` and
`.github/scripts/musa/{check,setup}.sh`. The loader resolves explicit repository
variables into concrete Docker options; it does not leave shell `$VARIABLE`
references in GitHub's container options.

| Repository variable | Required value |
|---|---|
| `MUSA_CI_RUNTIME` | Installed Docker MUSA runtime name, for example `mthreads` |
| `MUSA_CI_PHYSICAL_DEVICES` | Exactly four unique physical device IDs assigned to the runner |
| `MUSA_CI_LOGICAL_DEVICES` | `0,1,2,3` after validated runtime remapping |
| `MUSA_CI_PCI_BUS_IDS` | Four canonical PCI IDs in that logical order |
| `MUSA_CI_EMPTY_WHEEL` | Complete absolute `/data/...whl` path |
| `MUSA_CI_EMPTY_WHEEL_SHA256` | Actual 64-character lowercase SHA256 |
| `MUSA_GATE_MODEL_27B` | Complete absolute `/data/.../Qwen3.6-27B` directory |
| `MUSA_GATE_MODEL_35B_A3B` | Complete absolute `/data/.../Qwen3.6-35B-A3B` directory |
| `MUSA_CI_IMAGE` | Optional validated image tag or digest |

The `/data` bind is read-only. Setup builds and installs ordinary wheels into
a fresh private venv under `RUNNER_TEMP`; it does not edit inherited vendor
packages. The existing unit, functional, e2e and benchmark jobs retain their
interfaces. The graph functional helper now exercises TorchMUSA rather than
skipping the MUSA platform. Its small tensor test is additional component
validation; it does not replace model Gate acceptance.

The dedicated self-hosted Gate job has a 500-minute outer budget covering all
four 6900-second route limits and setup/cleanup/artifacts. The runner keeps
individual readiness and pytest deadlines within that budget. This uses the
[job timeout setting](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idtimeout-minutes),
which is separate from the individual step timeout setting.

MUSA discovery remains disabled in `.github/configs/platforms.yml` until the
runner's image, allocation, wheel, models and full Gate are validated. Configure
the variables above and validate the actual runner before changing that switch.

## Validation record

The following historical runs use candidate
`f80f78c466f4d55feed60120d1d6848f259991b9`. Its isolated wheel build, installation,
static source binding and unchanged vendor comparison passed. Its first full
27B eager attempt stopped before client tests because an owned-process FD
snapshot was incomplete. The container was stopped with PID 0 and no OOM;
that attempt is retained as a failure. The subsequent 27B eager run passed
all five scenarios, 26 logical requests and all ten checks per response, with
three pytest exits of 0. Device/cache/probe and job exits were 0; process
monitoring and cleanup had no errors, with no OOM or PID limit events. Its
container stopped with PID 0. The observed server exit of 137 followed explicit
container shutdown and is not a natural-exit result. The full collection has
357 files, 23,418,880 bytes, SHA256
`72f4e2a8e299c3e8d9299d13108f2efcba2d1e8678531d9f16cb0f1c4da0d204`.
The 35B-A3B eager route completed with **25/26 responses passing** and pytest
exits `[1, 0, 0]`. Its long-text answer omitted the required literal
`limitations`, so the original `expected_semantics` and `expected_order` checks
failed despite coherent text. The other four scenarios passed. No original
quality check or fixture was weakened. There were no OOM/PID-limit events or
monitoring errors, and the owned container stopped with PID 0. The complete
collection has 370 files, 31,795,200 bytes, SHA256
`edca6091fbe9581dac77424fc49f66f0a90ad581703314b45529c995a310cf45`.

The 27B graph route **failed during startup**, before readiness or the
original client. It executed **zero original logical requests**. The first
worker exception was `profile_cudagraph_memory` in `model_runner.py:6953`,
calling `torch.accelerator.synchronize()` and raising
`RuntimeError: Backend doesn't support synchronizing all streams on device.`
The profiling cleanup at line 6803 repeated that error. The API exited
naturally with code 1. Its cleanup report preserves that unsuccessful exit;
the later explicit stop of the owned container returned 0, with final
container exit 137, PID 0 and no OOM. Process monitoring reported no errors or
PID-limit events. Four source-verified worker observer activations do not
prove successful capture/replay associated with any original request. The
complete collection has 471 files, 8,325,120 bytes, SHA256
`705f89cbfd9c8f8e61b7a270efda680082ce1a500024b6151df48945e143b0c5`.

The prepared f80 35B-A3B graph attempt remains **NOT_RUN / ON_HOLD** after that
startup failure. The historical matrix is therefore:

| Route | Original requests executed | Result |
|---|---:|---|
| 27B eager | 26 | PASS: 26/26, all original checks |
| 27B graph | 0 | FAIL: startup synchronization error |
| 35B-A3B eager | 26 | FAIL: 25/26; long text omits `limitations` |
| 35B-A3B graph | 0 | NOT_RUN / ON_HOLD |

The MUSA backend now binds `torch.accelerator.synchronize` to the real
`torch_musa.synchronize` alongside the existing allocator API bindings. It
validates all required vendor callables before changing the bindings and
preserves default/explicit device forwarding, vendor exceptions and
idempotence. The targeted CPU Mock regression passed 48 cases with process
exit 0. No graph-disable workaround or vendor file edit is used. The
historical f80 results, including the 35B eager quality failure, remain
recorded and do not certify a later candidate.

Candidate `f867afb0c7f026bcabc8f80c2a68277a993ae800` was then built and installed
in the isolated CPU09 environment as three ordinary wheels: `vllm
0.28.0+empty`, `vllm-plugin-fl 0.0.0` and `musa-gate-observer 0.1.0`. The current
FL wheel is 542,550 bytes, SHA256
`f3a90b458382c16483da92e763cc4b68ebf1195f069ceed2670e84d7d8423b55`;
the observer wheel is 24,375 bytes, SHA256
`0d9ad617f2204ea53ba6210c12f4529b7cef4b86c76d729cbcac5a54f158ae17`.
Installed metadata, payload, RECORD, direct wheel origins and candidate source
binding were verified. Inherited vendor files and metadata were unchanged.
The successful installation and static binding do not establish GPU or model
acceptance.

The subsequent 27B graph run, runtime30, reached readiness with the actual
TorchMUSA synchronization binding and began generating tokens. It then stopped
progressing, and EngineCore terminated after `sample_tokens` RPC timed out at
23:17:11 CST on 2026-10-09. Later worker shutdown reported native MUSA
`unknown error` or `operation not permitted` at the cleanup synchronization
point. Those later errors do not establish the underlying cause of the stall.
The old generic unsupported-all-streams startup error was not the failure in
this run.

All three original pytest invocations finished with exit 1. Their real JUnit
files contain two text, two image and one mixed testcase, all five failed.
All five original JSON reports were retained: **26 logical attempts, 0 quality
successes**, comprising 13 text and 13 image inputs. Each failed request has
`APIConnectionError` and the original `checks={}`; the ten quality checks were
unavailable, and no check result was synthesized. SDK retries and original
request settings were unchanged; 26 is not a count of wire attempts.

The eight complete traces contain one API, four TP workers and three original
pytest clients. Actual native graph capture/replay events were recorded, but
the unchanged offline consumer exited 1 because of error events. It marked the
partial route and every full-matrix/acceptance flag false. Capture/replay
counts include warmup and do not prove successful joins for all 26 original
requests. The API's recorded natural exit was 0 despite the failed inference;
that exit does not make the Gate pass. The later owned-container stop returned
0; fresh inspection shows container exit 137, PID 0, no OOM and an empty
cleanup-error list. The complete collection contains 566 payload files plus
its manifest, 88,893,440 tar bytes, SHA256
`3051c1d68a4d99a7d939a5f8214d7ca2386497c5b6b9fd7cf65117fad94ed42c`.

The 35B-A3B graph run, runtime27, also reached readiness and generated tokens,
then terminated with `sample_tokens` RPC timeout at 23:47:46 CST on
2026-10-09. Its later native MUSA errors were reported during worker cleanup;
the specific cause of the stalled native work remains unestablished. All
three original pytest exits were 1, with all five real JUnit testcases failed.
Its five JSON reports retain **26 logical attempts and 0 quality successes**,
13 text and 13 image inputs, and `APIConnectionError` rows with `checks={}`.
Its eight complete process traces include actual graph events, but the
unchanged partial-route observer consumer exited 1 on error events and kept
all completion/acceptance flags false. The natural API exit was 0; the owned
container stop returned 0, with final container exit 137, PID 0, no OOM and
an empty cleanup-error list. Its complete collection has 599 payload files
plus its manifest, 83,261,440 tar bytes, SHA256
`e7c4501dcfd525017e1bbaa42a51b6f688cdaa02fdfa75fa049fa62cdb95f59a`.

The preserved first 27B eager run, runtime28, completed all five original
scenarios with **26/26 quality successes**, all ten checks per response, and
13 text plus 13 image inputs. Its three original pytest exits were 0 and its
five real JUnit testcases (2/2/1) passed. The actual server command contains
`--enforce-eager`, uses port 18062, and records compilation and graph modes
`NONE`. The job and the owned graceful API shutdown returned 0. The owned
container stop returned 0; fresh inspection shows container exit 137, PID 0,
no OOM and an empty cleanup-error list. Its complete collection has 411
payload files plus its manifest, 55,470,080 tar bytes, SHA256
`eda15a016c9c3f545dde384bb5c66a4966dfdb839c884f8cf9f7824415825ab1`.

The **complete runtime28 route was FAIL** because observer closure did not
pass. API and all three client traces have normal ending events, but the four
worker traces each contain activation and 234 scheduler begin/return pairs,
ending at `scheduler_return` without `telemetry_end`. The unchanged original
consumer exited 1 with `ContractError: activation first and telemetry_end last
required`; partial and full-matrix completion remained false. Successful
process cleanup does not replace these missing events. The original traces
were retained unchanged, and the consumer's closure requirement was preserved.

The preserved first 35B-A3B eager run, runtime29, produced **25/26 quality
successes** with original pytest exits `[1, 0, 0]` and four of five real JUnit
testcases passing. Only `text_single` failed: its response omitted the required
literal `limitations`, so `expected_semantics` and `expected_order` were false;
the other eight checks, including minimum length, passed. All 26 requests had
responses and all ten original checks; the remaining 25 quality results passed.
Four workers each ended at `scheduler_return` after 297 begin/return pairs,
without `telemetry_end`. The unchanged original consumer exited 1 on the missing
boundary. The API exited 0 after an owned SIGTERM; the container stop returned 0,
with final exit 137, PID 0, no OOM and no cleanup errors. Its complete collection
has 424 payload files plus its manifest, 58,460,160 tar bytes, SHA256
`1563385462dacbe8b9f58e3e8be161c3ae185f58e2759d817a55e5bdfa97e4df`.

The missing worker ends in runtime28/29 accompanied the upstream default
`--shutdown-timeout 0`: the API selected abort mode and immediately killed the
EngineCore tree. The later eager reruns add **only `--shutdown-timeout 60`** to
the server command, allowing drain mode and real worker cleanup. The original
requests, SDK retries, quality thresholds, model, TP4, observer and vendor code
remain unchanged. Historical traces and failed consumer results are preserved;
no ending event is synthesized or closure requirement relaxed.

The 27B eager rerun, runtime31, is a **complete single-route PASS**. All five
original scenarios pass, with 26/26 quality successes, 260 true original checks,
13 text plus 13 image inputs, three pytest exits of 0 and five passing real JUnit
testcases. Its actual command includes `--enforce-eager`, port 18062 and
`--shutdown-timeout 60`, with compilation and graph modes `NONE`. The eight
source-bound traces contain 2,023 events; each of the four workers records
activation, 234 scheduler begin/return pairs and a real final `telemetry_end`
with no errors. The unchanged original partial-route consumer exits 0 with
`route_observer_complete=true`, all 26 original response-to-engine joins, and no
errors. Its scope remains `route_only`; all full-matrix and acceptance flags
remain false. Independent reads verify the original returned rows, all 260
quality checks and joins to each of the four schedulers.

Runtime31 records drain mode with timeout 60s and `all workers exited
gracefully`. Its job and API exit 0. The owned container stop returns 0, with
final container exit 137, PID 0, no OOM and no cleanup errors. The complete
collection has 411 payload files plus its manifest, 55,316,480 tar bytes,
SHA256 `418d8d9a533700688a056052fc5f0f57adfe1b008627a853d71ab07e1c014632`.
The 35B-A3B eager rerun, runtime32, is **FAIL on original quality**, with
**25/26 quality successes**. All 26 requests have responses and ten original
checks, with 13 text and 13 image inputs. The single-text response contains
1,578 characters but omits the required literal `limitations`; only
`expected_semantics` and `expected_order` fail. Its remaining eight checks and
the other 25 requests pass. Independent execution of the original pure check
functions against the saved responses confirms 258 true and two false checks.
The three original pytest exits are `[1, 0, 0]`, and four of five real JUnit
testcases pass. No original prompt, token limit, threshold or fixture was changed.

Runtime32's real observer endings now pass the boundary requirement: its eight
complete traces contain 2,527 events, and each of the four workers has
activation, 297 scheduler begin/return pairs and a final `telemetry_end` with no
errors. The unchanged original partial-route consumer still exits 1, now at
`joins` with `ContractError: original logical quality failure`. Its route and
full-matrix completion flags remain false. The actual server uses
`--enforce-eager`, port 18064 and `--shutdown-timeout 60`, with compilation and
graph modes `NONE`. It records drain mode and all workers exiting gracefully.
The job exits 1; the API exits 0 and the owned container stop returns 0. Fresh
inspection shows container exit 137, PID 0, no OOM and no cleanup errors. This
resource success does not override the quality failure. The complete collection
has 420 payload files plus its manifest, 57,692,160 tar bytes, SHA256
`67a05fca76319677f74b325da36ae00069b82f39e075cc2a5c0f140f733c3392`.

| Route | Current original logical attempts | Current result |
|---|---:|---|
| 27B eager, runtime31 | 26 | PASS: 26/26 quality and complete single-route observer |
| 27B graph, runtime30 | 26 | FAIL: 0/26 quality successes; RPC timeout |
| 35B-A3B eager, runtime32 | 26 | FAIL: 25/26 quality; literal `limitations` missing |
| 35B-A3B graph, runtime27 | 26 | FAIL: 0/26 quality successes; RPC timeout |

All four current routes were executed, retaining **20 original JSON reports
and 104 logical requests**: 52 text and 52 image inputs, **51 quality successes
and 53 failures**. Their 12 original pytest invocations produce 20 real JUnit
testcases: nine pass, 11 fail, with no error or skip testcase. Failed graph
requests retain `APIConnectionError` and `checks={}`; unavailable quality checks
were not fabricated. These counts describe logical results, not SDK wire attempts.

The unchanged original full consumer was executed once in default full-matrix
mode over byte-verified copies of all 32 unfiltered traces from runtime30/27/31/32;
old runtime28/29 results were not substituted. It exited 1 at `events` with
`ContractError: unsupported/error/unknown event`. Its first rejected event is
the runtime30 graph API's actual `api_exception`, status `error`, at sequence 7.
The consumer stops at that error; verified copies do not mean it parsed all
32 streams to completion. All full-matrix completion and acceptance flags remain
false, and the required 208 graph request/rank joins are not verified complete.

Full four-route Gate acceptance is **FAIL**. The CI runner's shutdown-window
fix is applied locally and its 26 CPU helper tests pass. The GitHub CI job has
not executed, and automatic MUSA discovery remains disabled in
`.github/configs/platforms.yml` (`musa.enabled: false`).

The dedicated job also builds `tools/musa-gate-observer` as a third independent,
ordinary wheel, without index access, dependencies, or build isolation. It
verifies the observer's local metadata, wheel payload and RECORD; vendor
packages remain inherited. Only the Gate server and its three separate original
pytest processes activate the observer. The [CI runner contract](musa-ci-gate-runner.md)
describes its full-matrix aggregation and cleanup gates. A partial observer
diagnostic cannot pass the complete CI matrix.

The origin check pins all 19 Gate files to the **current checked-out candidate**.
This includes the updated Gate README. The request implementation, original
three tests and eight PNG fixtures retain their own committed-byte identities;
the 19-file table is not claimed to be identical to a historical upstream kit.

The validated Debian-based vendor image has no `ensurepip`. The venv therefore
uses `--without-pip`; `python -I -B -m pip --isolated` reuses the inherited
protected pip module without replacing it or writing vendor packages.
