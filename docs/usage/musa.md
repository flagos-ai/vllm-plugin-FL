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
| Plugin candidate | `f80f78c466f4d55feed60120d1d6848f259991b9` |
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

As of the current work, candidate f80's isolated wheel build, installation,
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

Both graph routes are **NOT_RUN** in this record. Full four-route acceptance
has not passed; the 35B eager quality failure remains a failed Gate route.
Update this record with actual outcomes as those routes finish.

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
