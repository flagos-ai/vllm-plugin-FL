# Ascend 910C: vLLM 0.28.0 empty

The image keeps the CANN 9.0.0 and torch_npu 2.10 runtime from the earlier
910C environment:

```text
quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
```

The Dockerfile removes its vLLM 0.20.2 installation and installs upstream
`v0.28.0` with `VLLM_TARGET_DEVICE=empty`. The resulting package version is
`0.28.0+empty`; no NVIDIA vLLM extension is required.

| Component | Version / source |
| --- | --- |
| Python | 3.11.15, aarch64 |
| PyTorch / torch_npu | 2.10.0+cpu / 2.10.0 |
| vLLM | 0.28.0+empty (`v0.28.0`) |
| FlagTree | 0.6.2a1+ascend3.5, supplying Triton 3.5.1 |
| FlagGems | `3b406c36212744b98b9720bf6d0a5387c09fe96b` |
| cann-shmem | 1.6.0 (FlagTree TLE import dependency) |
| NumPy | 1.26.4 (FlagGems requirement) |

FlagTree and the FlagGems `qwen-vllm_for_ascend` branch follow the
[official Ascend manual](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend),
checked on 2026-09-10. `cann-shmem` is distributed through the
[official CANN package index](https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/).
Standalone `triton` / `triton-ascend` distributions must be removed before
installing FlagTree because they write to the same Python module directory.

Build isolation must remain disabled for vLLM and the plugin: upstream build
requirements target another PyTorch version. Runtime constraints preserve the
CPU PyTorch and NPU adapter while installing vLLM's common dependencies.
FlagGems is installed without its hardware extras, which specify an older stack.

The upstream dependency metadata currently conflicts: OpenCV 4.13 requires
NumPy >= 2 on Python 3.11, while this FlagGems revision pins NumPy 1.26.4.
The image retains FlagGems's pin after installing vLLM common requirements;
`pip check` will report this conflict. Video processing is not validated by
the text/image inference checks. The inherited CANN profiler may also report
optional dependency conflicts.

## Build and CI

Build the image from the repository root:

```bash
docker/build.sh \
  --platform ascend \
  --target ci \
  --image-name harbor.baai.ac.cn/flagscale/vllm-plugin-fl
```

Build and publish `ascend-vllm0.28.0-a3-ci` before dispatching manual CI.
The configuration names this target image; changing the configuration does
not publish it. `.github/scripts/ascend/setup.sh` checks the runtime versions
and rejects an old image instead of silently testing vLLM 0.20.2.

Install the plugin checkout without replacing the prepared dependencies:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

## Inference settings

Select the NPUs assigned to the job and load the CANN environment before
starting inference. For the single-host 910C_174 development environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export ASCEND_RT_VISIBLE_DEVICES=14,15
export GLOO_SOCKET_IFNAME=lo
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
```

`GLOO_SOCKET_IFNAME=lo` avoids hostname resolution delays for a single host;
multi-host jobs must use an interface reachable by all ranks.

Ascend defaults to `ModelRunnerFL` (`VLLM_USE_V2_MODEL_RUNNER=0`). The upstream
V2 runner requires CUDA/UVA operations unavailable on NPU. This stack also
requires eager execution: the platform rejects non-eager configuration early
because vLLM 0.28's compilation and graph paths still contain CUDA-only
assumptions. Use:

```bash
vllm serve /models/Qwen3-0.6B \
  --host 127.0.0.1 --port 7023 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 2048 --gpu-memory-utilization 0.3 \
  --enforce-eager --no-enable-chunked-prefill \
  --no-async-scheduling --no-enable-prefix-caching
```

Run long installations and inference jobs in a named `tmux` session. The
driver must be usable inside the container; on 910C_174 the two-device
unprivileged configuration failed initialization, while the approved
privileged task container initialized correctly. Keep process device
selection set explicitly when using a privileged container.

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
convolution output expected by vLLM 0.28.

CANN 9.0's ATB RoPE kernel is used only for FP16 and BF16 inputs. FP32 RoPE
runs through the reference implementation before ATB is launched, avoiding a
deferred device error that would otherwise surface during a later operation.
The FlagGems adapter also accepts the six-argument Ascend RoPE interface in the
pinned `qwen-vllm_for_ascend` revision.

Unquantized modular MoE experts use native `torch_npu.npu_grouped_matmul`.
The legacy Ascend `fused_experts_impl` patch does not cover vLLM 0.28's
modular entry point. Generic Triton MoE kernels failed on 910C with both
UB overflow and invalid memory accesses after reducing the tile size.
The native bridge preserves expert mapping, biases and the prepare stage's
input-side router weighting. Quantized MoE and MoE LoRA remain unvalidated.

The vLLM Triton top-k/top-p sampler is disabled on Ascend. FlagTree 0.6.2a1
cannot lower its large-batch kernel on 910C, including the profiling batch
used by the OpenAI server. Sampling uses vLLM's PyTorch implementation; this
allows the normal server batch-size default without changing request behavior.

## Validation on 910C_174

The following checks ran in the prepared task container on NPUs 14 and 15
with the versions above and the default Ascend operator policy:

| Check | Settings | Result |
| --- | --- | --- |
| Qwen3-0.6B text | TP1 and TP2, BF16, eager, 2048 tokens, memory 0.3 | Correct arithmetic and Chinese capital answers |
| Qwen3 scheduling | TP2, chunked prefill, prefix caching, async scheduling | Long prompt crossed the prefill chunk; second prefix pass reused 384 cached tokens; async answers correct |
| Qwen3-4B OpenAI API | TP2, BF16, eager, streaming Chat API | `/v1/models` and streaming chat passed |
| Qwen3.6-27B text + image | TP2, BF16, eager, 4096 tokens, memory 0.8 | Paris; Hello VLM, yellow text, blue rectangle |
| Qwen3.6-35B-A3B text + image | TP2, BF16, eager, 4096 tokens, memory 0.8 | Paris; Hello VLM, blue rectangle; text color described ambiguously as white or pale yellow |
| Qwen3.6-35B-A3B OpenAI API | TP2, BF16, eager, text and generated image | `/v1/models`, Paris, Hello VLM and blue rectangle passed |
| Unit regression | Entire `tests/unit_tests` suite | 555 passed; 9 platform-specific tests skipped |
| Functional device checks | Ascend ops, HCCL helpers and raw `torch.npu.NPUGraph` primitives | All selected tests passed |

Full-model vLLM compilation and graph capture are unsupported and rejected
explicitly; the raw `torch.npu.NPUGraph` checks above do not exercise that
model path. Video and performance benchmarks remain unvalidated. The hybrid attention bridge
currently makes contiguous cache inputs for native attention kernels;
performance tuning is still needed. A clean Docker build was attempted with
the correct context and host networking, but 910C_174 could not resolve
`pypi.org`; image publication and the manual CI job therefore remain pending.

## Model provisioning

On 910C_174, models are stored under `/public-flash/models`. The Ascend CI
configuration mounts that directory read-only at `/data/models/Qwen`, matching
the model YAML files. The resulting container layout is:

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
first. When host validation passes, dispatch the `CI` workflow and select
`ascend` for the platform input. Ascend remains excluded from the automatic PR
platform registry.
