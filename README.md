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
| `main` | [v0.28.0](https://github.com/vllm-project/vllm/tree/v0.28.0) |
| `0.5.0-dev` (NVIDIA CUDA adaptation) | [v0.32.0](https://github.com/vllm-project/vllm/tree/v0.32.0) |

## Supported Models and Chips

In theory, vllm-plugin-FL can support all models available in vLLM, as long as no unsupported operators are involved. The tables below summarize the current support status of end-to-end verified models and chips, including both fully supported and in-progress ("Merging") entries.

### Supported Models

| Model | Status | Reference |
|-------|--------|-----------|
| Qwen3.5-397B-A17B | Supported | [example](./examples/qwen3_5_offline_inference.py) |
| Qwen3-Next-80B-A3B | Supported | [example](./examples/qwen3_next_offline_inference.py) |
| Qwen3-4B | Supported | [example](./examples/offline_inference.py) |
| MiniCPM-o 4.5 | Supported | [example](./examples/minicpm/) |
| GLM-5 | Supported | [example](./examples/glm_5_offline_inference.py) |
| Qwen3.5-35B-A3B | Supported | [example](./examples/qwen3_5_offline_inference.py)  |
| BAAI/bge-m3 | Supported | [implementation](./vllm_fl/models/bge_m3.py) |
| MiniMax-M2.7 | Supported | [implementation](./examples/minimax_m27_offline_inference.py) |
| DeepSeek-V4-Flash (ModelSlim W8A8, Ascend TP8) | Supported | [Ascend guide](./docker/ascend/README.md) |

### Supported Chips

| Chip Vendor | Status | Reference |
|-------------|--------|-----------|
| NVIDIA | Supported | [deployment guide](./docs/usage/nvidia.md) |
| Ascend | Supported | [deployment guide](./docs/usage/ascend.md) |
| MetaX | Supported | - |
| T-Head | Supported | [deployment guide](./docs/usage/thead.md) |
| Iluvatar | Supported | [deployment guide](./docs/usage/iluvatar.md) |
| Tsingmicro | Supported | - |
| Moore Threads | Supported | - |
| Hygon | Supported | [deployment guide](./docs/usage/hygon.md) |
| Sunrise | Supported | - |
| ARM64 CPU | Supported | - |

## Quick Start

### Setup

For NVIDIA on `0.5.0-dev`, follow the [NVIDIA guide](./docs/usage/nvidia.md)
for the full official vLLM 0.32 CUDA image, pinned development dependencies,
upstream V1/V2 runners and regression commands. FlagGems remains enabled.
It replaces the previous NVIDIA 0.28 setup.
The generic setup below applies to the other backends; their 0.32 validation
is outside the NVIDIA adaptation. See also the
[T-Head PPU guide](./docs/usage/thead.md).

1. Install vLLM

    For **non-NVIDIA** chips, install vLLM from source with the `empty` device target:
    ```sh
    git clone -b v0.28.0 https://github.com/vllm-project/vllm.git
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
    pip install --no-build-isolation --no-deps .
    # or editable install
    pip install --no-build-isolation --no-deps -e .
    ```

    For **non-NVIDIA backends** that use PyTorch's CUDA dispatch key (such as
    HIP/ROCm) with an EMPTY vLLM build, build the plugin native extension by
    setting `VLLM_VENDOR=cuda` during installation:
    ```sh
    cd vllm-plugin-FL
    VLLM_VENDOR=cuda pip install --no-build-isolation --no-deps .
    # or editable install
    VLLM_VENDOR=cuda pip install --no-build-isolation --no-deps -e .
    ```

    This builds and installs `vllm_fl._C`, which provides native C++ support
    required by some graph/custom-op paths, especially when vLLM is installed
    with `VLLM_TARGET_DEVICE=empty`.

    If `VLLM_VENDOR` is not set, vllm-plugin-FL is installed as a Python-only
    plugin and the native extension is skipped.

    NVIDIA uses the official vLLM CUDA extensions and the Python-only plugin
    installation in the [NVIDIA guide](./docs/usage/nvidia.md); it does not
    require `VLLM_VENDOR=cuda`.

3. Install [FlagGems](https://flagos-ai.github.io/FlagGems/getting-started/install/)

    3.1 Install build dependencies

    ```sh
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    ```

    3.2 Install FlagGems for the selected backend

    ```sh
    git clone --branch master https://github.com/flagos-ai/FlagGems
    cd FlagGems
    pip install --no-build-isolation --no-deps .
    # or editable install
    pip install --no-build-isolation --no-deps -e .
    ```

Compatible explicit fused operators also require the pinned
[FlagGems-vllm package](./docs/flaggems-vllm.md). The NVIDIA CI image preinstalls it;
general ATen registration remains with FlagGems.

### Runtime compatibility hooks

The plugin installs runtime compatibility hooks through vLLM's plugin entry
points without modifying the installed vLLM package. Model configs and classes
that are native in vLLM 0.28.0 and Transformers 5 are not duplicated here.

Operator adapters use the plugin dispatch manager, so backend selection,
fallback, per-op policy, operator-list recording, and I/O diagnostics continue
to follow the common FlagOS controls.

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

1. Install the validated [FlagTree](https://github.com/flagos-ai/flagtree/)
   and FlagGems revisions

    ```sh
    python3 -m pip uninstall -y triton triton-ascend flagtree flag-gems
    python3 -m pip install --no-deps flagtree===0.6.2a1+ascend3.5 \
      --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
    python3 -m pip install --no-deps cann-shmem==1.6.0 \
      --index-url=https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/

    git clone --branch qwen-vllm_for_ascend https://github.com/flagos-ai/FlagGems.git
    cd FlagGems
    git checkout 3b406c36212744b98b9720bf6d0a5387c09fe96b
    python3 -m pip install --no-build-isolation --no-deps .
    ```

    This FlagTree wheel supplies Triton 3.5.1. See the
    [Ascend image guide](./docker/ascend/README.md) for the complete CANN,
    PyTorch, vLLM empty-package, and dependency constraints.

2. Enable eager execution

    Eager execution is the recommended setting for Ascend. Add `enforce_eager=True` to the `LLM` constructor or pass `--enforce-eager` on the command line. Graph mode is still experimental; see the tested cases in the [Ascend image guide](./docker/ascend/README.md).


### Run a Task

#### Offline Batched Inference
With vLLM and vLLM-fl installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing). See the example script: [offline_inference](./examples/offline_inference.py). Or use blow python script directly.
```python
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    # Create an LLM.
    llm = LLM(
        model="Qwen/Qwen3-4B",
        max_num_batched_tokens=16384,
        enforce_eager=current_platform.device_type == "npu",
    )
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

This single-prompt example uses vLLM's default `max_num_seqs`. Set it based on
the expected request concurrency and available device memory for your deployment.

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
