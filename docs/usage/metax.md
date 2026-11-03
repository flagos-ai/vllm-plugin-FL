# MetaX deployment on vLLM 0.32

The `0.5.0-dev` branch targets vLLM **0.32.0**. The published CI image below
provides the validated runtime dependencies. Its build replaces the selected
MetaX FlagOS base's older vLLM package with an ordinary 0.32 empty wheel.
Keep the image's torch, FlagTree/Triton, FlagGems and MACA attention libraries
together; mcoplib is optional. Install the checked-out FL plugin with the setup
step below.

The base below is listed in the [FlagOS resource download catalog](https://flagos.io/resourcedownload)
and the [official MetaX model card](https://modelscope.cn/models/FlagRelease/MiniCPM-OCR-1.0-BF16-metax-FlagOS).
Its preinstalled vLLM is **0.24.0**; pulling the base alone does not install the
0.32 adaptation.

## Runtime and build

| Component | Configuration |
|---|---|
| Published CI image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-metax-ci@sha256:734c6f7d620e2ca3be4fa382a9e2cd6e1d345a310e2e502f11317b5521b886ae` |
| Base image | `harbor.baai.ac.cn/flagrelease-public/minicpm-ocr-1.0-metax-c550-gems5.5.0-pluginfl-0601420-vllm0.24.0-maca3.8.1.3:20261021` |
| Base digest | `sha256:7c810404a21cb60d0a6d1f532baca82d1dea8ee428231f7fb066d609aacbadcf` |
| Native runtime | Ubuntu `24.04.4`, Python `3.12.13`, MACA SDK `3.8.1.3`; validation host driver `3.8.1` |
| PyTorch | `2.10.0+metax3.8.1.0`; `torch.version.git_version=f0eb8bf5e72d15327617097aa264c899b3581986` |
| FlagGems | `5.5.0.dev0`; vendor distribution does not embed a Git SHA; package fingerprint `9bebc2d5c987466aefc61a4280b7e1cf9935626aa4c4e1f5c07319f582e8ad53` |
| FlagTree / imported Triton | `0.7.0+metax3.6` / `3.6.0`; vendor distribution does not embed a Git SHA; package fingerprint `4028e7bc5821c15002a78ea1e02f242994bb8c260c51afe85c55978f8ecf3928` |
| FlagGems-vllm | Image-pinned `0.3.0.dev1`; no embedded Git SHA; package fingerprint `9d66bf7f643b2209e774312f171a064c28644df94345b9e76f6476c1d3b428c5` |
| vLLM | `0.32.0+empty`, built from [`5df63a7203cdb9328d824a27b682e77c7b9928be`](https://github.com/vllm-project/vllm/commit/5df63a7203cdb9328d824a27b682e77c7b9928be); no NVIDIA `vllm._C` |
| MACA FlashAttention | `2.6.3+metax3.8.1.0torch2.10`; no embedded Git SHA; package fingerprint `8ed84bd3a348d70d534093b2612c35c0b44763e86ba91375fa48f3e3b781942d` |
| Native FLA / FlashMLA / FlashInfer | `0.5.0+metax3.8.1.0torch2.10` / `1.0.1+metax3.8.1.0torch2.10` / `0.2.6+metax3.8.1.0torch2.10` |
| Python interoperability pins | Transformers `5.19.0`, tokenizers `0.23.3`, depyf `0.20.0`; vendor XGrammar `0.2.3+flagos` and compressed-tensors `0.17.0+flagos` retained |
| Build dependency resolver | uv `0.13.4`, native-package overrides and explicit transitive Triton exclusion when FlagTree owns the import namespace |
| Optional FlagCX | `v0.13.0`, [`0beba7aa7e76cc0885a9929f1975857bcd310d5e`](https://github.com/flagos-ai/FlagCX/commit/0beba7aa7e76cc0885a9929f1975857bcd310d5e); not bundled in the CI image |
| Plugin runner | `ModelRunnerFL`, `VLLM_USE_V2_MODEL_RUNNER=0` |
| Attention | MACA FlashAttention; separate MACA FlashAttention MLA prefill |
| Communication | MCCL; pass `--disable-custom-all-reduce` |
| Default graph mode | `PIECEWISE`; explicit eager / `NONE` is preserved |
| Execution RPC watchdog | Set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` before starting the service; the upstream 300-second default can expire during C550 cold GEMM tuning |

Package fingerprints are SHA256 hashes of sorted `relative_path<TAB>file_sha256<LF>`
entries, covering module files and distribution native libraries while excluding
Python bytecode caches. They identify the tested vendor packages when their
source commit was not recorded. They are not inferred Git commits.
The vendor API reference is the MetaX v0.32 development tree at
[`440ec38e95083b8f143e03d7cc1f3c7ca1b8981e`](https://github.com/MetaX-MACA/vLLM-metax/commit/440ec38e95083b8f143e03d7cc1f3c7ca1b8981e).

Pull the published 0.32 CI runtime:

```bash
export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-metax-ci@sha256:734c6f7d620e2ca3be4fa382a9e2cd6e1d345a310e2e502f11317b5521b886ae
docker pull "$IMAGE"
```

The published digest was verified against the tested local image and pulled
successfully from Harbor. CI installs its checked-out FL plugin source into
this dependency image.

For a local build, use the repository's build script:

```bash
bash docker/build.sh --platform metax --target ci \
  --image-name vllm-plugin-fl --image-tag 0.32.0-metax-ci
export IMAGE=vllm-plugin-fl:0.32.0-metax-ci
```

The build installs vLLM non-editably in empty mode, retains the base's native
stack, and installs Python dependencies for 0.32. Do not install the default
PyPI torch, Triton or NVIDIA vLLM wheel into this image.

## FlagGems and FlagTree requirements

Use the recorded vendor packages to reproduce this configuration. The build's
native-stack overrides preserve them, and `setup.sh` preserves the image's
FlagGems-vllm `0.3.0.dev1` rather than replacing it with the generic installer
version. These are tested versions, not claims about the minimum supported
versions of every upstream package.

The FlagGems-vllm 0.3 cache writer handles block gaps but assumes compact token
and head strides. The plugin uses a stride-aware, C550-safe Triton writer for
vLLM 0.32's packed K/V layout. An upstream FlagGems-vllm writer needs equivalent
page/head-stride support before replacing this fallback. No new FlagGems or
FlagTree upstream commit is required for the tested configuration.

FlagTree supplies the `triton` namespace through `/opt/flagtree` in this image's
`PYTHONPATH`. Preserve that entry when adding paths for optional components:

```bash
export PYTHONPATH="/path/to/extra/source${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import triton; print(triton.__version__, triton.__file__)'
```

The expected import is Triton `3.6.0` from `/opt/flagtree`. The plugin keeps
legacy Triton 3.0 AST workarounds disabled on this compiler. The local cache and
GDN launch changes account for MACA's resource limits without replacing the
FlagTree compiler.

Cold tuning can also occur on the first request for a new matrix shape. Set the
positive execution-RPC timeout shown above before the service imports vLLM.
This setting is separate from HTTP client timeouts. The CI setup script exports
it for later job steps while retaining an explicit user value; it does not
require another FlagGems or FlagTree revision.

## Start an isolated container

Check `mx-smi` and select idle cards before starting. The host needs Docker,
`/dev/dri` and `/dev/mxcd`. Set `MODEL_ROOT` to a directory containing complete
checkpoints; the examples use `Qwen3.6-35B-A3B` directly under that directory.

```bash
export MODEL_ROOT=/data/models
export CONTAINER=metax-vllm032
docker run -d --name "$CONTAINER" \
  --hostname vllm-plugin-fl --ipc=host --shm-size=64g \
  --device /dev/dri:/dev/dri --device /dev/mxcd:/dev/mxcd \
  -e GEMS_VENDOR=metax -e VLLM_PLUGINS=fl \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e MACA_VISIBLE_DEVICES=0,1,2,3 \
  -v "$MODEL_ROOT":/data/models:ro \
  -v "$PWD":/src \
  --entrypoint bash "$IMAGE" -lc 'sleep infinity'

docker exec "$CONTAINER" bash -lc '
  cd /src
  bash .github/scripts/metax/setup.sh'
```

The setup script installs the checked-out plugin, preserves the FlagOS image's
recorded FlagGems-vllm version, verifies vLLM, and removes legacy
`zz_fl_*.pth` boot hooks. No separate shim installer is needed.

## Run inference or serving

Start with a small model and eager execution, then test TP and graph execution.
Long-running remote commands, including installation, tests and serving, should
run in a named `tmux` session.

```bash
docker exec "$CONTAINER" bash -lc '
  vllm serve /data/models/Qwen3.6-35B-A3B \
    --served-model-name metax-qwen36 \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-batched-tokens 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.70 \
    --disable-custom-all-reduce \
    --enforce-eager'
```

Remove `--enforce-eager` to exercise the plugin's PIECEWISE graph default.
Keep the same model, prompts and sampling parameters when comparing modes.
Port 8000 in this example is inside the container:

```bash
docker exec "$CONTAINER" curl http://127.0.0.1:8000/v1/models
docker exec "$CONTAINER" curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"metax-qwen36","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Validation

The adaptation has CPU regression coverage for attention selection and cache
updates, MLA prefill, GDN output/state layout, MCCL/CUDA runtime ABI, native
operator ownership, version-gated shims and delayed MoE patch installation.
The GPU test independently compares the chunk-state kernel against a Torch
reference for variable sequence lengths and both natural/base-2 exponentials.

Validated on `bm-zksg-wx-zone1-d-mc550-64g-2-126` with the runtime above:

| Check | Result |
|---|---|
| Local CPU regression scope, including FlagCX policy and cross-platform contracts | 309 passed; 73 accelerator cases skipped locally |
| Real allocator/cache writer/attention | 24 passed; exact K/V writes, maximum attention error 0.0078125 |
| FlagOS operators, Gemma RMSNorm/MRoPE, full GDN and convolution/preparation | 33 GPU cases passed |
| vLLM runner selection | 80 passed in the GPU image |
| Model runner / compatibility symbols | 12 passed in the GPU image |
| MoE router lifecycle and padded softmax/sigmoid | 22 CPU contracts and four GPU numerical cases passed |
| Qwen3.5-4B, TP1 eager | Correct text answers across four batches, including long/short mixing and repeated requests |
| Qwen3.5-4B, TP1 PIECEWISE | Captured sizes 1/2/4; all nine requests match eager token for token |
| Qwen3.6-35B-A3B, TP4 eager | HTTP model discovery, France/arithmetic/repeated text and generated-image recognition passed |
| Qwen3.6-35B-A3B, TP4 PIECEWISE | Captured sizes 1/2/4 with MoE/MCCL; HTTP text matches eager and image smoke facts pass |
| Original 35B CI request settings | Text and image substring matchers pass |

The cache regression uses actual allocator views, nonzero layer offsets,
528-to-16 block subdivision, both supported cache layouts, and the model's
16-query-head / 4-KV-head / 256-head-dimension geometry. It checks every written
K/V element before comparing prefill, decode and mixed attention outputs with
an independent float32 reference. The eager model check retains the default
multimodal profiling and compares France/arithmetic/greeting answers with a
Transformers reference using the same checkpoint and chat template.
The TP4 HTTP check uses the serving options shown above, temperature zero and
`chat_template_kwargs={"enable_thinking": false}`; the image contains
"Hello VLM" in a blue rectangle. It verifies the visible text, color and shape
independently of the wording of the description.
The TP4 image descriptions differ between execution modes: the graph response
identifies the small yellow text as white. The smoke checks OCR and the blue
rectangle, not detailed visual accuracy. The original CI request enables
thinking and exhausts its 256-token budget on the image; its substring checks
pass on the generated draft. The explicit no-thinking requests above produce
complete answers.

```bash
# Complete MetaX unit scope in the GPU container.
python tests/run.py --platform metax --device c550 --scope unit

# Focused kernel and patch checks.
python -m pytest \
  tests/unit_tests/attention/test_metax_chunk_delta_h.py \
  tests/unit_tests/attention/test_metax_paged_attention_numerics.py \
  tests/unit_tests/attention/test_metax_gdn_numerics.py \
  tests/unit_tests/attention/test_metax_gdn_prep_numerics.py \
  tests/unit_tests/ops/test_metax_qwen35_numerics.py \
  tests/unit_tests/dispatch/test_metax_patches.py -v -s

# Existing serving acceptance case.
FL_TEST_PLATFORM=metax FL_TEST_DEVICE=c550 \
FL_TEST_MODEL=qwen3_6 FL_TEST_CASE=35b_a3b_tp4_eager \
python -m pytest tests/e2e_tests/serving/test_serving_smoke.py -v -s
```

Results from PR [#554](https://github.com/flagos-ai/vllm-plugin-FL/pull/554)
belong to vLLM 0.28. They do not establish a 0.32 inference or graph pass.

## Original adaptation Gate

Run the original [adaptation Gate](../../tools/adaptation-gate-cases/README.md)
for both requested checkpoints and both execution modes. Each row below runs
`text_single` (one request), `text_concurrent_8` (eight), `image_single` (one),
`image_concurrent_8` (eight), and `mixed_concurrent_8` (four text plus four
image). The full matrix has 20 scenarios and 104 requests. A row passes only
when all five scenarios pass and all 26 responses pass every quality check.

| Model | Mode | Original Gate result |
|---|---|---|
| Qwen3.6-27B | eager | 5/5 scenarios, 26/26 requests passed |
| Qwen3.6-27B | PIECEWISE graph | 5/5 scenarios, 26/26 requests passed with execution-RPC watchdog 1800 seconds; capture sizes 1/2/4/8 |
| Qwen3.6-35B-A3B | eager | 5/5 scenarios, 26/26 requests passed |
| Qwen3.6-35B-A3B | PIECEWISE graph | 5/5 scenarios, 26/26 requests passed; actual capture sizes 1/2/4/8 |

The recorded 35B runs use TP4, BF16, `max_model_len=32768`, eight scheduler
slots, `max_num_batched_tokens=1024` and memory utilization 0.70. The original
client sends temperature-zero, no-thinking requests, with its 512-token long
text budget and 128-token default budget unchanged. The test helper allowed
240 health checks at ten-second intervals and confirmed readiness before
starting `run_test.sh`. `SERVICE_TIMEOUT=2400` and `REQUEST_TIMEOUT=900`
configure readiness and per-request wall-clock limits; prompts and quality
thresholds are unchanged. This is a separate acceptance scope from the earlier
HTTP smoke.
The first requests for new prompt or batch shapes can trigger additional
FlagGems GEMM tuning after graph capture. The recorded 27B graph service took
about 738 seconds to initialize; its first long-text Gate request completed in
about 210 seconds with all quality checks passing.
That initial run then hit the upstream 300-second internal RPC deadline during
image GEMM tuning, closing the engine and leaving later requests unavailable.
The complete five-scenario rerun with the documented 1800-second execution-RPC
timeout passed all 26 requests. It reused compilation caches (about 22 seconds
to initialize); its single image took about 114 seconds and concurrent image
requests about 281–286 seconds. The client timeout remained 900 seconds.
Across the four successful model/mode rows, all 20 scenarios and all 104
responses passed every original quality check. The initial failed reports and
server trace are retained separately.

The Gate README permits a custom server command. Inside the container, start
one service in its own terminal or remote `tmux` session:

```bash
export GATE_DIR=/src/tools/adaptation-gate-cases
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export MODEL_PATH=/data/models/Qwen3.6-35B-A3B
export SERVED_MODEL_NAME=qwen
export MODE=eager PORT=18320
export RESULTS_DIR=/work/gate-results/35b-eager
export SERVER_PID_FILE="$RESULTS_DIR/server.pid"
mkdir -p "$RESULTS_DIR"
mode_args=()
if [[ "$MODE" == eager ]]; then mode_args=(--enforce-eager); fi
printf '%s\n' "$$" > "$SERVER_PID_FILE"
exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 32768 --max-num-batched-tokens 1024 --max-num-seqs 8 \
  --gpu-memory-utilization 0.70 --disable-custom-all-reduce \
  --allowed-local-media-path "$GATE_DIR/images" \
  "${mode_args[@]}"
```

In another container terminal or `tmux` session, run all original files:

```bash
unset MAX_TOKENS
MODEL_PATH=/data/models/Qwen3.6-35B-A3B SERVED_MODEL_NAME=qwen PORT=18320 \
  RESULTS_DIR=/work/gate-results/35b-eager \
  SERVER_PID_FILE=/work/gate-results/35b-eager/server.pid \
  SERVICE_TIMEOUT=2400 REQUEST_TIMEOUT=900 \
  bash /src/tools/adaptation-gate-cases/run_test.sh
```

Stop that service before changing modes. Repeat with `MODE=graph`, port 18321
and a distinct `35b-graph` results directory, then repeat both modes with a
complete Qwen3.6-27B checkpoint. Each row produces five JSON reports containing
the actual prompts, outputs, checks and request counts. Keep all reports even
when one scenario fails; `run_test.sh` runs all three pytest files.

## Optional FlagCX validation

The CI image does not include FlagCX. The optional configuration uses the
v0.13.0 source/core at the commit in the runtime table, without installing its
Torch backend plugin. The tested native library SHA256 is
`33b3c77248366928f89b6b75ad6b5eb579126512d39381ebd59a4c245896f0ad`,
and `plugin/interservice/flagcx_wrapper.py` has SHA256
`a791806a0ab36da872cd200b5466c36001ef798c8fb444fc62b1d1f6805ca5cf`.
The native core links the image's `/opt/maca/lib/libmccl.so`.

Use a matching source tree containing `build/lib/libflagcx.so` and the Python
interservice wrapper. The pinned upstream source documents the MetaX build as
`make USE_METAX=1`, with MACA development headers and libraries. Mount the
completed tree read-only when starting an isolated container:

```bash
# Add to the docker run options before --entrypoint:
-v "$FLAGCX_DIR:/opt/FlagCX:ro" \
-e FLAGCX_PATH=/opt/FlagCX \
```

Set this environment before the new Python process starts. If extending the
import/library paths manually inside the container, retain the image's paths:

```bash
export FLAGCX_PATH=/opt/FlagCX
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export PYTHONPATH="$FLAGCX_PATH${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$FLAGCX_PATH/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The core-only configuration uses Gloo for Torch control/bootstrap groups and
`CommunicatorFL` / `PyFlagcxCommunicator` for device collectives. The plugin
requires successful MetaX FlagCX initialization instead of silently falling
back. It uses vLLM's opaque collective operators in this configuration so
Dynamo does not trace ctypes; default MCCL execution is unchanged. Group APIs
pass the communicator for the new FlagCX ABI and retain legacy signatures.

Run the real device tests on free cards, using two or four processes:

```bash
cd /src
torchrun --standalone --nproc-per-node=2 --module pytest --noconftest \
  tests/functional_tests/distributed/test_metax_flagcx_collectives.py \
  -q -s -p no:cacheprovider
```

The suite has four unique cases per topology: FP32/BF16 eager collectives and
FP32/BF16 graph replay. It verifies all-reduce, all-gather, reduce-scatter,
broadcast, all-gatherv, reduce-scatterv and point-to-point exchange against CPU
references. Graph cases change inputs for three replays. Both the two-rank and
four-rank runs passed all four cases on every rank; the communicator must be
available and enabled. A single-process skip is not a FlagCX validation result.
Qwen3.5-4B TP2 eager and PIECEWISE serving also passed HTTP text, repeated-request
and image smoke checks with the enabled FlagCX communicator. The PIECEWISE run
captured graph sizes 1/2/4 and recorded successful native FlagCX calls during
capture on both workers. Its first profile took about 376 seconds and engine
initialization about 689 seconds on the tested cold cache. Worker stacks showed
FlagGems GEMM autotuning in this stage. Allow sufficient startup time for a cold
cache.

To reproduce the serving scope inside the container, use a complete 4B checkpoint:

```bash
export SMALL_MODEL_PATH=/data/models/Qwen3.5-4B
vllm serve "$SMALL_MODEL_PATH" \
  --served-model-name metax-flagcx --host 127.0.0.1 --port 18020 \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 2048 --max-num-batched-tokens 512 --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4]}' \
  --enforce-eager
```

Remove `--enforce-eager` for the recorded PIECEWISE configuration. Use the same
temperature-zero, no-thinking API requests as the HTTP example above, with
`model=metax-flagcx`. Keep this serving check separate from the original Gate.
Pipeline/expert parallelism and FlagCX KV transfer are outside this test scope.

## Compatibility boundaries

- `mcoplib` is optional. Native operators retain ownership when present;
  otherwise attention cache/merge use bundled Triton kernels and the top-k
  bridge preserves bias, renormalization and graph-padding semantics in Torch.
- Unquantized MoE requires the FlagGems backend and an OOT policy allowing
  `fused_moe`. A native MoE fallback has not been validated; disabling this
  backend reports an explicit unsupported-policy error.
- The FlagGems-vllm 0.3 cache writer accepts gaps between blocks but assumes
  contiguous tokens and heads inside each block. vLLM 0.32's interleaved K/V
  cache views use the fully stride-aware Triton writer instead, including
  allocator offsets and the 528-token manager / 16-token kernel subdivision.
- Legacy torch 2.8 compilation shims and Triton 3.0 AST workarounds are enabled
  only on their matching runtimes. The modern torch 2.10 / Triton 3.6 compiler
  keeps its native implementation.
- The MetaX sampler fallback and the third-party `triton_kernels` restriction
  remain enabled, matching the vendor's 0.32 implementation.
- The FLA chunk-state kernel retains single-stage autotuning for C550's shared
  memory limit, with the 0.32 masked-pointer and `USE_EXP2` behavior.
- Dense FlashMLA requires the vendor scheduler-object ABI, including
  `FlashMLASchedMeta`. Older tuple-based libraries are rejected with a dependency
  error. Sparse MLA and FP8 MLA caches are not supported by this backend.
- Explicit upstream V2 runner selection is rejected. FULL graphs and additional
  speculative/quantized configurations require separate hardware validation.
- All changes are installed through the FL plugin lifecycle. Do not edit
  installed vLLM sources or copy CUDA extensions from another environment.
