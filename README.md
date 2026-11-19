# vllm-plugin-FL

vllm-plugin-FL is a plugin for the [vLLM](https://github.com/vllm-project/vllm) inference/serving framework, built on FlagOS's unified multi-chip backend — including the unified operator library [FlagGems](https://github.com/flagos-ai/FlagGems) and the unified communication library [FlagCX](https://github.com/flagos-ai/FlagCX). It extends vLLM's capabilities and performance across diverse hardware environments. Without changing vLLM's original interfaces or usage patterns, the same command can run model inference/serving on different chips.

## Adaptation Gate

The adaptation gate validates text, image, and mixed text-image requests across
the required models in both eager and graph modes. See the
[adaptation gate cases](./tools/adaptation-gate-cases/README.md) for the test
matrix, scenarios, and usage.

## Version Compatibility

| vllm-plugin-FL Branch | Community vLLM Version |
|-----------------------|------------------------|
| `release/0.2` | [v0.20.2](https://github.com/vllm-project/vllm/tree/v0.20.2) |
| `main` | [v0.24.0](https://github.com/vllm-project/vllm/tree/v0.24.0) |

## Supported Models and Chips

In theory, vllm-plugin-FL can support all models available in vLLM, as long as no unsupported operators are involved. The tables below summarize the current support status of end-to-end verified models and chips, including both fully supported and in-progress ("Merging") entries.

### Supported Models

| Model | Status | Reference |
|-------|--------|-----------|
| Qwen3.5-397B-A17B | Supported | [example](./examples/qwen3_5_offline_inference.py) |
| Qwen3-Next-80B-A3B | Supported | [example](./examples/qwen3_next_offline_inference.py) |
| Qwen3.8-Flash-Next / Qwen4Exp | Supported | [implementation](./vllm_fl/models/qwen3_8_flash_next/) |
| Qwen3-4B | Supported | [example](./examples/offline_inference.py) |
| MiniCPM-o 4.5 | Supported | [example](./examples/minicpm/) |
| Unlimited-OCR | Ascend 910C BF16 | [example](./examples/unlimited_ocr/serve.py) / [serving and validation](#unlimited-ocr-on-ascend) |
| GLM-5 | Supported | [example](./examples/glm_5_offline_inference.py) |
| GLM-5.3 Flash | Supported | [implementation](./vllm_fl/models/glm5/model.py) |
| Qwen3.5-35B-A3B | Supported | [example](./examples/qwen3_5_offline_inference.py)  |
| BAAI/bge-m3 | Supported | [implementation](./vllm_fl/models/bge_m3.py) |
| MiniMax-M2.7 | Supported | [implementation](./examples/minimax_m27_offline_inference.py) |
| MiniMax-M3.1-Flash | Supported | [example](./examples/minimax_m31_offline_inference.py) / [deployment guide](./docs/minimax-m31.md) |
| HY4-Preview | Merging (NVIDIA) | [installation](#hy4-on-nvidia) |

### Supported Chips

| Chip Vendor | Status | Reference |
|-------------|--------|-----------|
| NVIDIA | Supported | - |
| Ascend | Supported | - |
| MetaX | Supported | - |
| T-Head | Supported | - |
| Iluvatar | Supported | - |
| Tsingmicro | Supported | - |
| Moore Threads | Supported | - |
| Hygon | Supported | - |
| Sunrise | Supported | - |
| ARM64 CPU | Supported | - |

## Quick Start

### Setup

1. Install vLLM

    For **NVIDIA** GPUs, install vLLM from the official [v0.24.0](https://github.com/vllm-project/vllm/tree/v0.24.0) release (optional if the correct version is already installed):
    ```sh
    pip install vllm==0.24.0
    ```

    For **non-NVIDIA** chips, install vLLM from source with the `empty` device target:
    ```sh
    git clone -b v0.24.0 https://github.com/vllm-project/vllm.git
    cd vllm
    VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation --no-deps .
    ```

2. Install vllm-plugin-FL

    2.1 Clone the repository:

    ```sh
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    ```

    2.2 Install
    ```sh
    cd vllm-plugin-FL
    pip install --no-build-isolation .
    # or editable install
    pip install --no-build-isolation -e .
    ```

    Qwen3.8-Flash-Next / Qwen4 uses the HC, PLE and QSA operators available
    on [FlagGems-vllm main](https://github.com/flagos-ai/FlagGems-vllm).
    Follow the shared [FlagGems-vllm installation instructions](docs/flaggems-vllm.md).

    On devices without native FP8 support, run standard vLLM FP8 checkpoints
    with `--dtype bfloat16`. Qwen3.8-Flash-Next / Qwen4 converts supported
    per-tensor and block-scaled FP8 weights to BF16 when loading. ModelOpt,
    MXFP8, and compressed-tensors FP8 formats are not covered by this fallback.
    BF16 weights need about twice the device memory of FP8 weights; an FP8 KV
    cache request is likewise served as BF16 and may need twice the memory.

    For CUDA-like devices, including CUDA and HIP/ROCm environments that use
    PyTorch's CUDA dispatch key, build the plugin native extension by setting
    `VLLM_VENDOR=cuda` during installation:
    ```sh
    cd vllm-plugin-FL
    VLLM_VENDOR=cuda pip install --no-build-isolation .
    # or editable install
    VLLM_VENDOR=cuda pip install --no-build-isolation -e .
    ```

    This builds and installs `vllm_fl._C`, which provides native C++ support
    required by some graph/custom-op paths, especially when vLLM is installed
    with `VLLM_TARGET_DEVICE=empty`.

    If `VLLM_VENDOR` is not set, vllm-plugin-FL is installed as a Python-only
    plugin and the native extension is skipped.

3. Install [FlagGems](https://flagos-ai.github.io/FlagGems/getting-started/install/)

    3.1 Install Build Dependencies

    ```sh
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    ```

    3.2 Install FlagGems

    ```sh
    git clone -b v5.3.4 https://github.com/flagos-ai/FlagGems
    cd FlagGems
    pip install --no-build-isolation .
    # or editable install
    pip install --no-build-isolation -e .
    ```

### Runtime compatibility hooks

The plugin installs runtime compatibility hooks through vLLM's plugin entry
points without modifying the installed vLLM package. Model-specific config and
model registrations are loaded only for their corresponding architectures.

Operator adapters use the plugin dispatch manager, so backend selection,
fallback, per-op policy, operator-list recording, and I/O diagnostics continue
to follow the common FlagOS controls.

On Iluvatar with vLLM 0.24, the FlagOS GPU runner binds all attention caches
within each decoder layer, including separate main and index caches. Binding
preserves layer order, cache order within a layer, tensor identity and shared
aliases. This cache initialization fix does not enable graph capture or establish
model or operator support; validate those separately in the target environment.

### Operator profiling

See the [operator profiling tools](./tools/operator_profile/README.md) for a
reproducible native-vLLM and plugin comparison workflow.

4. (Optional) Install [FlagCX](https://github.com/flagos-ai/FlagCX/blob/main/docs/getting_started.md#build-and-installation)

    4.1 Clone the repository:
    ```sh
    git clone -b v0.13.0 https://github.com/flagos-ai/FlagCX.git
    cd FlagCX
    git submodule update --init --recursive
    ```

    4.2 Build the library with different flags targeting to different platforms:
    ```sh
    make USE_NVIDIA=1
    ```

    4.3 Set environment
    ```sh
    export FLAGCX_PATH="$PWD"
    ```

    4.4 Installation FlagCX
    ```sh
    cd plugin/torch/
    FLAGCX_ADAPTOR=[xxx] pip install . --no-build-isolation
    # or editable install
    FLAGCX_ADAPTOR=[xxx] pip install -e . --no-build-isolation
    ```
    Note: [xxx] should be selected according to the current platform, e.g., nvidia, ascend, etc.


If there are multiple plugins in the current environment, you can specify use vllm-plugin-fl via VLLM_PLUGINS='fl'.

### Additional Steps for Ascend

1. Install [FlagTree](https://github.com/flagos-ai/flagtree/)

    ```sh
    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
    python3 -m pip install flagtree==0.6.1rc1+ascend3.5 $RES
    ```

    For other chips, please refer to [FlagTree](https://github.com/flagos-ai/flagtree/) for the corresponding version (e.g., `flagtree==0.6.1+iluvatar3.6`, `flagtree==0.6.1+metax3.6`, etc.).

2. Set required environment variable

    ```sh
    export TRITON_ALL_BLOCKS_PARALLEL=1
    ```

3. Select eager or full-decode graph execution

    For eager execution, add `enforce_eager=True` to the `LLM` constructor or
    pass `--enforce-eager` on the command line. For `FULL_DECODE_ONLY`, use:

    ```sh
    export VLLM_PLUGINS=fl
    export USE_FLAGGEMS=0

    vllm serve Qwen/Qwen3-0.6B \
        --dtype bfloat16 --tensor-parallel-size 1 \
        --max-model-len 2048 --max-num-batched-tokens 1024 --max-num-seqs 4 \
        --gpu-memory-utilization 0.3 \
        --no-enable-prefix-caching --no-async-scheduling \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
    ```

    This path uses the eager FX backend with NPU graph capture/replay. The
    standard Python-only FL installation above is sufficient. Use a compatible
    FlagGems/FlagTree installation when setting `USE_FLAGGEMS=1`.


### Unlimited-OCR on Ascend

Unlimited-OCR loads the original checkpoint through FL's model and multimodal
registration. NPU execution uses FL and torch-npu. The verified environment
is Ascend 910C with CANN 9.0.1, Python
3.12.13, PyTorch 2.10.0+cpu, torch-npu 2.10.0.post2, Transformers 5.13.0 and
torchvision 0.25.0+cpu.

Load the CANN toolkit environment. Use the source install above with
`VLLM_TARGET_DEVICE=empty` at the pinned vLLM 0.24 revision and install
Python-only FL in that environment.
[Pinned vLLM source](https://github.com/vllm-project/vllm/tree/ee0da84ab9e04ac7610e28580af62c365e898389)

From the repository root, pass the checkpoint path and one Ascend device ID
per tensor-parallel rank to the serving example.

```bash
python examples/unlimited_ocr/serve.py /path/to/Unlimited-OCR --devices 0,1 --tp 2
```

The example selects FL and the torch-npu path (`VLLM_PLUGINS=fl`,
`USE_FLAGGEMS=0`). Its defaults are BF16 weights and KV cache, TP2,
`FULL_DECODE_ONLY` with capture sizes 1, 2, 4, 8, 16, 32 and 64, synchronous
scheduling and chunked prefill. It uses a 32,768-token context, 64 active
sequences, a 1,024-token batch budget and 0.80 device-memory utilization.
R-SWA retains the full KV cache and uses a 2,048 MiB mask budget per rank.
Prefix caching and speculative decoding are disabled. For single-device
eager serving, use `--devices 0 --tp 1 --eager --max-num-seqs 4`.

To route the SiLU-and-multiply activation through the official FlagGems package,
set `USE_FLAGGEMS=1 VLLM_FL_FLAGOS_WHITELIST=silu_and_mul` when launching the
example. This selects the FL `default.flagos` activation and keeps R-SWA, RoPE
and MoE on their Ascend routes. The benchmark results below use the default
torch-npu configuration.

Send one image and a text prompt to `/v1/chat/completions` for
`Unlimited-OCR`. The example uses a single BOS, Gundam image preprocessing
(`image_size=640`, `base_size=1024`, `crop_mode=true`) and the FL NGram
logits processor. Both measurements below set `ngram_size=35` and
`window_size=128` per request.

#### Validation

The tables show the FL OCR baseline `8f36bf1` followed by the measured FL
revision `d5aacf5e2002`. The two builds used different pairs of Ascend 910C
devices. At `d5aacf5e2002`, 298 targeted tests passed, TP2 graph capture and
replay were verified on both ranks, and TP1 eager serving was verified with
one image.

Performance used EvalScope 1.14.1 `random_vl` with 1,024 streaming requests per run
and concurrency 64. Each build ran once per length with zero failures and a
mean of 1,024 output tokens. Each request included one synthetic 2,048 × 2,048
RGB image. Generation used a 1,024-token fixed length, seed 42, temperature 0
and disabled EOS and stop criteria. Server input counts include image and
template tokens. Total throughput counts server input and output tokens.

| Text tokens | Server input tokens | Output tokens/s | Total tokens/s | Mean TTFT (ms) | Mean TPOT (ms) |
|------------:|--------------------:|----------------:|---------------:|---------------:|---------------:|
| 1,024 | 2,938 | 1,826.93 → 1,806.14 | 7,068.64 → 6,988.20 | 2,884.50 → 2,989.70 | 32.17 → 32.46 |
| 4,096 | 6,010 | 1,531.12 → 1,509.97 | 10,517.45 → 10,372.19 | 2,168.76 → 2,353.82 | 39.54 → 39.94 |
| 16,384 | 18,298 | 811.07 → 805.77 | 15,304.17 → 15,204.11 | 13,349.97 → 13,599.42 | 65.11 → 65.38 |

OmniDocBench v1.5 used one request per page at concurrency 64 with the
`document parsing.` prompt, non-streaming generation, temperature 0, seed 1024
and an 8,192-token output cap. Official `quick_match` scoring covered all
1,355 Markdown outputs.
The CDM self-check returned F1 = 1.0. Both builds had zero HTTP failures.
Scoring included 14 → 16 length-truncated outputs and two empty Markdown
outputs per build. The edit metrics use a 0–1 scale (lower is better). The
other scores use a 0–100 scale (higher is better). Overall averages text
accuracy (`(1 - Text Edit) × 100`), Formula CDM and Table TEDS.

| Metric | Baseline | Measured FL revision |
|--------|---------:|---------------------:|
| Text Edit | 0.03970 | 0.03913 |
| Formula CDM | 91.685 | 91.240 |
| Table TEDS | 91.033 | 90.682 |
| Table TEDS-S | 94.640 | 93.946 |
| Reading-order Edit | 0.04419 | 0.04484 |
| Overall | 92.916 | 92.670 |

### HY4 on NVIDIA

HY4 is paired with this branch's vLLM 0.24.0 environment and the shared
[FlagGems-vllm integration](./docs/flaggems-vllm.md). Its MLA query concatenation
and graph-safe paged MQA logits APIs are now merged into upstream FlagGems-vllm.
The shared CI dependency now includes both APIs. Install the same pinned upstream
source using the integration instructions above; no separate HY4 upgrade is needed.

HY4 uses the public `flaggems_vllm` sparse indexer, cache and attention APIs;
query quantization uses general FlagGems. No model-specific dependency extra or
fork revision is required. The shared fused-operator rollout switch and its
default behavior are described in the integration documentation.
Runtime selection checks callable APIs, device kernels and the shared dispatch
policy before allocating model weights. A selected operator's execution error
propagates without retrying another implementation.

Use `hy4_safetensors` to load the preview checkpoint with expert slicing. The
portable path requires BF16 KV (`auto` with a BF16 model, or `bfloat16`); FP8 KV
requires the complete native sparse/cache/prefill path. Shared indexers require
pipeline parallel size 1. The [manual checkpoint smoke configuration](./tests/models/hy4/fp8_tp16.yaml)
requires 16 NVIDIA GPUs; the single-GPU dummy smoke checks model integration.
HY4-Preview currently targets NVIDIA H100, vLLM 0.24.0, the MXFP8 preview
checkpoint, BF16 activations and BF16 portable KV. Other chips, multimodal serving and portable FP8 KV
are not validated. Sparse indexer providers use explicit entry points while
preserving the real CUDA platform identity.
The portable sparse indexer checks the selected paged MQA logits provider's
graph capability before loading weights. Upstream FlagGems-vllm exposes
`fp8_fp4_paged_mqa_logits_graph` for static-grid CUDA Graph capture, reading
lengths and page mappings on the GPU on every replay. The shared dispatch
selects that public entry point when available; older operator builds without
the capability require `--enforce-eager`.

With the graph-safe operator build, use `FULL_DECODE_ONLY` with compilation
mode `NONE`, for example:

```bash
--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}'
```

The shared CachedOp prewarming from
[vllm-plugin-FL #552](https://github.com/flagos-ai/vllm-plugin-FL/pull/552) is
used after model loading. Whole-model Dynamo compilation remains limited
by the current FlagGems `libentry` implementation; CUDA Graph mode `NONE`
and eager execution remain available.

### Run a Task

#### Offline Batched Inference
With vLLM and vLLM-fl installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing). See the example script: [offline_inference](./examples/offline_inference.py). Or use blow python script directly.
```python
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    # Create an LLM.
    llm = LLM(model="Qwen/Qwen3-4B", max_num_batched_tokens=16384, max_num_seqs=2048)
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Advanced use

For dispatch environment variable usage, see [environment variables usage](./vllm_fl/dispatch/README.md#environment-variables).

### Using Cuda Communication library
If you want to use the original Cuda Communication, you can unset the following environment variables.
```sh
unset FLAGCX_PATH
```

### Using native CUDA operators
If you want to use the original CUDA operators, you can set the following environment variables.
```sh
export USE_FLAGGEMS=0
```
