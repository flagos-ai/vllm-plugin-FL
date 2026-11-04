# Ascend 910C: vLLM 0.32.0 empty

The image keeps the CANN 9.0.0 and torch_npu 2.10 runtime from the earlier
910C environment:

```text
quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
```

The Dockerfile removes its vLLM 0.20.2 installation and installs upstream
`v0.32.0` with `VLLM_TARGET_DEVICE=empty`. The resulting package version is
`0.32.0+empty`; no NVIDIA vLLM extension is required.

| Component | Version / source |
| --- | --- |
| Python | 3.11.15, aarch64 |
| PyTorch / torch_npu | 2.10.0+cpu / 2.10.0 |
| vLLM | 0.32.0+empty (`v0.32.0`) |
| FlagTree | 0.6.2a1+ascend3.5, supplying Triton 3.5.1 |
| FlagGems | v5.4.0, `6ed2f39071ef85db26b7e64f0652a162f434959c` |
| cann-shmem | 1.6.0 (FlagTree TLE import dependency) |
| NumPy / PyYAML | 2.2.6 / 6.0.1 |

FlagTree supplies the Ascend Triton compiler; FlagGems is pinned to official
v5.4.0 for compatibility with the NumPy 2 dependencies required by vLLM 0.32. `cann-shmem` is distributed through the
[official CANN package index](https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/).
Standalone `triton` / `triton-ascend` distributions must be removed before
installing FlagTree because they write to the same Python module directory.

Build isolation must remain disabled for vLLM and the plugin: upstream build
requirements target another PyTorch version. Runtime constraints preserve the
CPU PyTorch and NPU adapter while installing vLLM's common dependencies.
FlagGems is installed without its hardware extras, which specify an older stack.

The selected NumPy version satisfies the vLLM/OpenCV and FlagGems requirements.
The inherited CANN profiler can still report optional dependency conflicts;
video processing and profiler integrations are outside the text/image gate.

## Build and CI

Build the image from the repository root:

```bash
docker/build.sh \
  --platform ascend \
  --target ci \
  --image-name harbor.baai.ac.cn/plugin/vllm-plugin-fl
```

The shared CI currently uses the published
`ascend-vllm0.28.0-a3-ci-20260922-r5-gas-clean` image. Build and publish a
replacement tag before updating `ci_image` in `.github/configs/ascend.yml`;
changing the configuration does not publish it.
`.github/scripts/ascend/setup.sh` checks the runtime versions and rejects an
old image instead of silently testing another vLLM version.

The CI image also pins the pytest plugins used by the shared workflows. The
setup script performs only the editable plugin overlay and version checks, so
test startup does not depend on PyPI availability on the NPU runner.

The Dockerfile prepares the runtime and test dependencies, but does not install
this repository's wheel. For a self-contained image, install the plugin
non-editably from a checkout and publish the resulting image separately. The
shared CI overlays the pull-request checkout with an editable install so that
each run tests the submitted source.

For a deployable image, install a wheel from the selected immutable plugin
revision without replacing the prepared dependencies:

```bash
python -m pip wheel --no-build-isolation --no-deps --wheel-dir /tmp/vllm-fl-wheels .
python -m pip install --no-deps --force-reinstall /tmp/vllm-fl-wheels/vllm_plugin_fl-*.whl
```

## Inference settings

Select the NPUs assigned to the job and load the CANN environment before
starting inference. Replace the example device ID with the device IDs assigned
on your host:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export ASCEND_RT_VISIBLE_DEVICES=0
export GLOO_SOCKET_IFNAME=lo
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
```

`GLOO_SOCKET_IFNAME=lo` avoids hostname resolution delays for a single host;
multi-host jobs must use an interface reachable by all ranks.

Ascend defaults to `ModelRunnerFL` (`VLLM_USE_V2_MODEL_RUNNER=0`). The upstream
V2 runner requires CUDA/UVA operations unavailable on NPU. For the graph
configuration, operator whitelist and complete Qwen3.6 gate procedure, see
[Ascend graph with vLLM 0.32](../../docs/ascend-vllm032-graph.md).
A small eager control can be started with:

```bash
vllm serve /models/Qwen3-0.6B \
  --host 127.0.0.1 --port 7023 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 2048 --gpu-memory-utilization 0.3 \
  --enforce-eager --no-enable-chunked-prefill \
  --no-async-scheduling --no-enable-prefix-caching
```

Run long installations and inference jobs in a named `tmux` session. The
driver must be usable inside the container. The validated container uses
`--privileged`; keep process device selection explicit and adapt the driver
mounts to your host.

The multimodal smoke example recreates the reference document's 300x200
white image, blue rectangle and yellow "Hello VLM" text. It checks the
capital-of-France answer and the image description, writing outputs and
token IDs to JSON:

```bash
python examples/ascend_multimodal_smoke.py \
  --model /models/Qwen3.6-27B \
  --memory 0.8 --max-model-len 4096 \
  --output qwen27.json
```

The example uses TP2 and disables graph capture, chunked prefill, prefix
caching and async scheduling. Its lower defaults (`--memory 0.5`,
`--max-model-len 2048`) also support the 27B development check when assigned
cards have less free memory. Allocate sufficient free memory before using
the larger settings or the 35B-A3B model.

The Ascend operator policy retains native torch_npu implementations for
known issues in this FlagGems revision: `argmax` registration, vision GELU
compilation, GDN state-reset `index_put`, and non-contiguous `index` reads.
The latter is required for correct visual RoPE caches. GDN prefill also
bridges the upstream state layout and preserves the channel-contiguous
convolution output expected by the model.

CANN 9.0's ATB RoPE kernel is used only for FP16 and BF16 inputs. FP32 RoPE
runs through the reference implementation before ATB is launched, avoiding a
deferred device error that would otherwise surface during a later operation.
The FlagGems adapter also accepts the six-argument Ascend RoPE interface in the
pinned `qwen-vllm_for_ascend` revision.

Unquantized modular MoE experts use native `torch_npu.npu_grouped_matmul`.
The legacy Ascend `fused_experts_impl` patch does not cover vLLM's
modular entry point. Generic Triton MoE kernels failed on 910C with both
UB overflow and invalid memory accesses after reducing the tile size.
The native bridge preserves expert mapping, biases and the prepare stage's
input-side router weighting. Other quantized MoE formats and MoE LoRA remain
unvalidated.

The earlier vLLM 0.28 validation of DeepSeek-V4-Flash ModelSlim checkpoints used the plugin's
`fl_modelslim_w8a8` quantization config. On 910C, the adapter uses BF16
short-window attention and a native torch_npu W8A8 MoE path. The validated
configuration uses TP8, one output projection group per rank, and a maximum
sequence length equal to the model's 128-token sliding window:

```bash
vllm serve /models/DeepSeek-V4-Flash-w8a8-mtp \
  --tensor-parallel-size 8 --dtype bfloat16 \
  --quantization fl_modelslim_w8a8 \
  --max-model-len 128 --max-num-seqs 1 \
  --max-num-batched-tokens 128 --gpu-memory-utilization 0.85 \
  --enforce-eager
```

Long-context, MTP speculative decoding, graph capture, and performance tuning
for this fallback path remain outside the validated configuration.

The vLLM Triton top-k/top-p sampler is disabled on Ascend. FlagTree 0.6.2a1
cannot lower its large-batch kernel on 910C, including the profiling batch
used by the OpenAI server. Sampling uses vLLM's PyTorch implementation; this
allows the normal server batch-size default without changing request behavior.

## Validation scope

The shared CI runs setup, unit and functional tests, offline and serving
inference, and a benchmark smoke test on the configured 910C runner. See the
[Ascend platform cases](../../tests/platforms/ascend.yaml) for the active
matrix and [PR #487](https://github.com/flagos-ai/vllm-plugin-FL/pull/487)
for the earlier vLLM 0.28 validation. The vLLM 0.32 graph scope is documented
in the graph guide above; video processing remains unvalidated.

## Model provisioning

Provision model weights before starting CI. The mount in
[the Ascend CI configuration](../../.github/configs/ascend.yml) must expose the
model paths declared by the model YAML files. For the current matrix, the
container layout is:

```text
/data/models/
└── Qwen/
    ├── Qwen3-0.6B/
    ├── Qwen3.6-27B/
    └── Qwen3.6-35B-A3B/
```

E2E jobs expect these model directories to be provisioned before CI starts.
The workflow validates the selected model paths with the shared
`.github/scripts/generate_matrix.py --check-models` check before running
`tests/run.py`.

To avoid occupying a scarce NPU runner during development, validate changes
on the host with the same image, setup script, and `tests/run.py` command
first. Ascend is enabled in the automatic PR platform registry. Manual runs of
the same `CI` workflow also use that registry to select enabled platforms.
