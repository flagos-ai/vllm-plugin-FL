# Enflame S60 Deployment Guide

The pinned image and deployment instructions below describe the published vLLM 0.32.0 Enflame image.
For the image build requirements, see the
[image recipe](../../docker/enflame/README.md).

This guide starts an OpenAI-compatible vLLM service on Enflame S60 GCUs using
the dependency stack validated by this repository. Use the pinned image and
install the plugin from the revision you intend to deploy; the image does not
contain each candidate's plugin changes.

## Published image and expected software stack

| Component | Version or configuration |
|---|---|
| Prebuilt image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-enflame-ci` |
| Image manifest | `sha256:732e3d505e89203215727189b0820dd601dc426c4b05092259bdabeec7c68b71` |
| Accelerator / host driver | Enflame S60 / `1.10.6` |
| vLLM | `0.32.0+empty` (`VLLM_TARGET_DEVICE=empty`) |
| Python | `3.12.8` |
| PyTorch | `2.11.0+cpu` with `torch-gcu==2.11.0+3.8.20260713` |
| FlagTree | `0.7.0+enflame3.6`, the Enflame-specific release |
| Triton GCU | `3.6.0+1.0.20260722`; FlagTree provides the `triton` Python namespace |
| FlagGems | `5.5.0.dev0` |
| TOPS SDK / TOPS ATen | `3.8.20260714` / `3.8.20260710` |
| FlagCX / ECCL | `0.13.0` / `3.7.1.2` |
| Preinstalled FlagGems-vllm | `0.3.0.dev1`; local wheel provenance, public source SHA unverified |
| OpenAI SDK / pandas | `3.28.0` / `3.0.6` |
| FlagGems-vllm setup pin | Official `flagos-ai/FlagGems-vllm`, commit `93fd1b0c1dc3b084dbf6872630fc9bb6113218ed` (`0.3.0.dev1`); verify the resolved installation separately |

### Provenance limits and required deployment record

The recipe pins official vLLM commit
`5df63a7203cdb9328d824a27b682e77c7b9928be` and FlagGems tag
`v5.5.0.dev0` (peeled commit `2bcfe2656dad0b654077fcfe7d66e6a93542afab`).
These are source/build requirements, not a complete source bill of materials.
[Verified image-layer package records](../validation/pr675-image-732e3d5/README.md)
now tie the SDK, FlagCX, ECCL and compiler/operator package metadata to the
published digest. The SDK layer and every later overlay were downloaded and
SHA256 verified; later overlays upgrade Triton GCU to the July 22 build and
supply FlagGems 5.5 and vLLM 0.32. Inherited image labels still mention 0.28.
The records include metadata hashes and available wheel hashes/direct URLs.
Public Git SHAs for local/binary builds and hashes/paths of communication
libraries actually loaded at runtime remain unverified. Image provenance is
therefore incomplete; do not infer those missing records from a separately
modified development container.

For the exact digest and plugin SHA being validated, retain the image inspection,
`python -m pip freeze --all`, `python -m pip inspect`, the vendor SDK package
manifest, compiler paths/versions, and hashes/paths of the loaded FlagCX and
communication shared libraries. Retain source SHAs for source builds and wheel
URLs plus SHA256 for binary installs. A local `file://` direct URL without a Git
revision does not establish public source provenance. Check the preinstalled
stack before setup; do not reinstall already matching dependencies merely to
produce a version listing. Hardware results must identify any actual setup
changes in addition to the image digest.

The `+cpu` PyTorch version is the vendor package's version string; execution
uses `torch_gcu` and the `gcu` device. Do not replace it with a generic PyTorch
wheel. Likewise, a generic FlagTree build or standalone upstream Triton wheel
is not interchangeable with this Enflame compiler stack.

Keep the plugin and its
[Enflame dispatch policy](../../vllm_fl/dispatch/config/enflame.yaml) on the same
revision. The policy selects the GCU attention backend and retains its tested
FlagGems exclusions. Build instructions and the rationale for the temporary
vendor adapters are in the [image recipe](../../docker/enflame/README.md).

## 1. Check the host and choose devices

Run Docker commands on the S60 host. It needs the compatible Enflame driver,
Docker, and tmux. Check accelerator memory and process ownership before choosing
two available devices for tensor parallelism:

```bash
efsmi
docker version
tmux -V
```

The checked-in CI configuration uses devices `2,3`; those IDs are an example,
not a guarantee that they are free. Do not stop another user's service to make
room. The examples below run one model and execution mode at a time on two GCUs.

Start a unique tmux session before cloning, pulling the image, or running setup
and validation. Run the following host steps inside that session so an SSH
disconnect does not interrupt them:

```bash
tmux new-session -s "enflame-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

## 2. Prepare the checkout and models

Use a private checkout at the branch, tag, or commit you intend to deploy.
The example selects the vLLM 0.32 adaptation branch `0.5.0-dev`; before the
Enflame PR is merged, select its reviewed commit instead of assuming the target
branch contains the adaptation. Keep model weights on the host and mount them read-only. The native test YAMLs
expect the following two directories under `/data/models/Qwen` in the container:

```bash
git clone --branch 0.5.0-dev https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL

export REPO_DIR="$PWD"
export MODEL_ROOT=/data/models/Qwen
export TOPS_VISIBLE_DEVICES=2,3  # Replace with the available device pair.

test -f "$MODEL_ROOT/Qwen3.6-27B/config.json"
test -f "$MODEL_ROOT/Qwen3.6-35B-A3B/config.json"
git rev-parse HEAD
```

`MODEL_ROOT` must point to the actual host directory containing complete model
weights, tokenizer files, and configuration. The commands do not download
missing models. Record the selected plugin commit with the validation results.

## 3. Pull the image and create a private container

The digest pins the dependency image even if its tag changes. Authenticate with
`docker login harbor.baai.ac.cn` only if required by your registry access.

```bash
export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-enflame-ci@sha256:732e3d505e89203215727189b0820dd601dc426c4b05092259bdabeec7c68b71
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'

export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export CONTAINER="vllm-fl-enflame-$RUN_ID"
export LOG_DIR="$PWD/enflame-logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# Choose one of the four combinations in the table below.
export MODEL_NAME=Qwen3.6-27B
export MODE=eager
export PORT=8000

docker run --detach \
  --name "$CONTAINER" \
  --hostname vllm-plugin-fl \
  --privileged \
  --ipc=host \
  --ulimit nofile=65535:65535 \
  -p "$PORT:8000" \
  -v "$REPO_DIR:/workspace/vllm-plugin-FL" \
  -v "$MODEL_ROOT:/data/models/Qwen:ro" \
  -v "$LOG_DIR:/logs" \
  -e VLLM_PLUGINS=fl \
  -e TOPS_VISIBLE_DEVICES="$TOPS_VISIBLE_DEVICES" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=99999 \
  -e TORCH_GCU_ENABLE_INT64_AND_UINT64=1 \
  -e ENABLE_I64_CHECK=0 \
  -e TORCHGCU_INDUCTOR_ENABLE=1 \
  -e MODEL_NAME="$MODEL_NAME" \
  -e MODE="$MODE" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc 'exec sleep infinity'
```

The privileged container exposes the vendor devices, matching the existing S60
CI configuration. Keep Docker's private PID namespace: do not add `--pid=host`,
because the test runner performs process cleanup. The model mount uses the
container paths from the native YAMLs regardless of the host's storage location.

| Model | `MODEL_NAME` | `MODE` |
|---|---|---|
| Qwen3.6-27B eager | `Qwen3.6-27B` | `eager` |
| Qwen3.6-27B graph | `Qwen3.6-27B` | `graph` |
| Qwen3.6-35B-A3B eager | `Qwen3.6-35B-A3B` | `eager` |
| Qwen3.6-35B-A3B graph | `Qwen3.6-35B-A3B` | `graph` |

## 4. Verify and install the selected plugin

Continue in the tmux session created above and enter the private container:

```bash
docker exec -it "$CONTAINER" bash
```

Inside the container:

```bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
bash .github/scripts/enflame/check.sh 2>&1 | tee /logs/check.log
bash .github/scripts/enflame/setup.sh 2>&1 | tee /logs/setup.log
```

`check.sh` requires at least two visible GCUs and creates a small GCU tensor.
The image supplies vLLM, flaggems-vllm, the vendor runtime, and test dependencies.
`setup.sh` reinstalls only the selected plugin checkout with
`--no-build-isolation --no-deps -e .`, then checks imports and device availability.
It reuses the image's installed dependencies without reinstalling them.

Do not resolve runtime dependencies again with a generic `pip install -U`: a
dependency can overwrite FlagTree's Triton files while leaving its metadata
intact. The checkout's [compiler guard](../../docker/enflame/verify_compiler.py)
and [index_fill guard](../../docker/enflame/verify_index_fill.py) are available
for diagnosing the image stack; import success is not hardware Gate acceptance.

## 5. Start and query the service

Continue inside the container. These examples use the native TP2 model settings
and the S60 overrides: `max_model_len=8192`, memory utilization `0.90`, and the
GCU collective path. Graph mode omits `--enforce-eager`.

```bash
case "$MODEL_NAME" in
  Qwen3.6-27B|Qwen3.6-35B-A3B) ;;
  *) echo "Unsupported example MODEL_NAME: $MODEL_NAME" >&2; exit 2 ;;
esac
mode_args=()
case "$MODE" in
  eager) mode_args+=(--enforce-eager) ;;
  graph) ;;
  *) echo "MODE must be eager or graph" >&2; exit 2 ;;
esac

vllm serve "/data/models/Qwen/$MODEL_NAME" \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --compilation-config '{"custom_ops":["none","+SiluAndMul"],"cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  "${mode_args[@]}" 2>&1 | tee "/logs/$MODEL_NAME-$MODE-server.log"
```

Cold-cache compilation and graph capture can take substantial time. The native
S60 inference jobs allow 120 minutes; adaptation Gate jobs allow 180 minutes. Follow the service log and wait for startup to
finish before sending requests. In a second host terminal, use the host port
selected above (8000 in this example):

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

Check the returned model name and response, and retain server errors if any.
An HTTP response is a deployment smoke check; it does not establish full model
or adaptation Gate coverage. Stop this example server with Ctrl-C in its tmux
session before using the same devices for another model, mode, or test.

## 6. Run the repository validation

With the example server stopped, run these commands inside the private
container and retain the generated reports and logs:

```bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
python tests/run.py --platform enflame --device s60 --scope unit \
  --output-dir /logs/unit 2>&1 | tee /logs/unit-console.log
python tests/run.py --platform enflame --device s60 --scope functional \
  --output-dir /logs/functional 2>&1 | tee /logs/functional-console.log
python tests/run.py --platform enflame --device s60 --scope e2e --task inference \
  --model qwen3_6 --case 27b_tp2_eager --output-dir /logs/27b-eager \
  2>&1 | tee /logs/27b-eager-console.log
python tests/run.py --platform enflame --device s60 --scope e2e --task inference \
  --model qwen3_6 --case 27b_tp2_graph --output-dir /logs/27b-graph \
  2>&1 | tee /logs/27b-graph-console.log
python tests/run.py --platform enflame --device s60 --scope e2e --task serving \
  --model qwen3_6 --case 35b_a3b_tp2_eager --output-dir /logs/35b-eager \
  2>&1 | tee /logs/35b-eager-console.log
python tests/run.py --platform enflame --device s60 --scope e2e --task serving \
  --model qwen3_6 --case 35b_a3b_tp2_graph --output-dir /logs/35b-graph \
  2>&1 | tee /logs/35b-graph-console.log
```

Each invocation writes to a separate output directory so its JSON/XML reports
are preserved instead of overwritten by the next run.

The inference smokes are retained separately. Run the full adaptation Gate for
all four combinations (each has five scenarios, 26 requests and 260 quality
checks), retaining each directory and reporting failures and skips explicitly:

```bash
for case_name in 27b_tp2_eager 27b_tp2_graph 35b_a3b_tp2_eager 35b_a3b_tp2_graph; do
  python tests/run.py --platform enflame --device s60 --scope e2e \
    --task adaptation_gate --model qwen3_6 --case "$case_name" \
    --output-dir "/logs/gate-$case_name"
done
```

The S60 overrides in `tests/platforms/enflame.yaml` apply only to these four
Qwen3.6 cases: memory utilization 0.90 reserves device headroom, max sequences
64 bounds concurrent scheduling, and max batched tokens 4096 bounds prefill
workspace independently of the 8192 context limit. The explicit SiLU custom
operator and capture sizes [1,2,4,8,16,32] constrain graph compilation; eager
cases still set enforce_eager. These are capacity/dispatch choices, not relaxed
quality thresholds. They are not defaults for other models. A historical
32768-context, 256-sequence capacity run is not equivalent to these settings.
Record the effective ModelConfig engine/CLI arguments, plugin SHA, model
revision, actual runner class, environment, commands and all raw JSON/logs.
Do not count zero cases, skipped scenarios or queued jobs as Gate acceptance.

The image manifest was verified in the registry. That observation does not
establish hardware acceptance of this image or an arbitrary plugin checkout. Re-run the relevant tests
and original Gate for the exact revision you deploy.

## Numeric and sampling validation boundaries

The GCU RMSNorm adapter targets vLLM 0.32's CUDA provider, not its Python
`forward_native` formula. In the pinned upstream
[`csrc/layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/csrc/layernorm_kernels.cu),
the fused residual sum is stored as `scalar_t` before variance accumulation;
normalization and weight multiplication use FP32 before the final output cast.
This explains a deliberate rounding-contract difference; it does not establish
bitwise equivalence across reduction orders. Acceptance still requires an
independent CUDA-provider reference on identical saved inputs plus S60 eager,
compile, graph-capture and replay outputs. Native-reference differences and
provider-reference differences must be reported separately.

The V2 `compute_topk_scores` compatibility shim copies GCU logits to CPU when
that function requests top-k, including positive returned top-logprobs and any
warmup invocation of that function. It does not change general `torch.topk`.
Only selected indices return to GCU; unused values remain on CPU. The full
batch-by-vocabulary readback, synchronization and CPU sorting still occur.
No S60 service-cost acceptance is claimed for this fallback. Compare logprobs
disabled/enabled for the same prompts, output lengths and scheduling settings,
across batch sizes and actual model vocabulary sizes; retain request-level ITL,
aggregate output-token throughput, host CPU load and transfer measurements.
Report the temporary fallback as unvalidated for production capacity until that
comparison or a validated official device-operator replacement is available.

## Limitations and troubleshooting

- **Device unavailable:** check `efsmi`, selected device ownership, the host
  driver, container device access, and `TOPS_VISIBLE_DEVICES`. Package import
  success alone does not establish GCU availability.
- **Compiler guard fails:** retain its output and recreate the container from
  the pinned image. Do not bypass the guard or install a generic Triton wheel.
- **Out of memory:** first check that the selected pair is available and that
  the S60 `0.90` budget is applied. Changing context length, concurrency,
  precision, or tensor parallelism requires separate validation of that new
  configuration; it is not a passing result for the original Gate.
- **Runner or speculative-decoding error:** Enflame does not force the runner environment variable. When unset, vLLM
  0.32 selects the runner from Triton availability and configuration; eligible
  configurations can select `GPUModelRunnerFLV2` automatically. Explicit `0`
  selects the legacy runner and `1` requests V2. Record the actual runner
  class from startup logs; a V1 result is not evidence for V2.
  Validate the selected runner and execution mode on hardware. Hybrid models with speculative decoding and
  `mamba_cache_mode=align` are rejected because the fused Mamba state-copy path
  does not yet translate its indirect pointers to GCU addresses. Other
  speculative/cache combinations still need their own hardware validation.
- **Other dependency versions or optional C++ operators:** the compatibility
  adapters target the fixed stack above. The Python FlagGems wheel does not
  establish coverage of the optional native C++ extension; see the
  [image recipe](../../docker/enflame/README.md) before changing that stack.

After saving results and confirming that this private container has no active
tests or service, stop it from the host with `docker stop "$CONTAINER"`. Keep
its logs and result files; do not stop a shared container or another task.
