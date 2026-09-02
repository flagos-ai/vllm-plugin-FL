# vllm-plugin-FL

vllm-plugin-FL is a plugin for the [vLLM](https://github.com/vllm-project/vllm) inference/serving framework, built on FlagOS's unified multi-chip backend — including the unified operator library [FlagGems](https://github.com/flagos-ai/FlagGems) and the unified communication library [FlagCX](https://github.com/flagos-ai/FlagCX). It extends vLLM's capabilities and performance across diverse hardware environments. Without changing vLLM's original interfaces or usage patterns, the same command can run model inference/serving on different chips.

## Version Compatibility

| vllm-plugin-FL Branch | Community vLLM Version |
|-----------------------|------------------------|
| `release/0.2` | [v0.20.2](https://github.com/vllm-project/vllm/tree/v0.20.2) |
| `main` | [v0.28.0](https://github.com/vllm-project/vllm/tree/v0.28.0) |

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

    The tested **NVIDIA** baseline is the official CUDA 12.9 image:
    ```sh
    docker pull vllm/vllm-openai:v0.28.0-cu129
    ```

    For a non-container NVIDIA installation, install the matching official
    [vLLM v0.28.0](https://github.com/vllm-project/vllm/tree/v0.28.0)
    package first. The NVIDIA FlagTree step below intentionally replaces the
    standalone Triton package from vLLM while leaving vLLM itself unchanged.

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

    For CUDA-like devices, including CUDA and HIP/ROCm environments that use
    PyTorch's CUDA dispatch key, build the plugin native extension by setting
    `VLLM_VENDOR=cuda` during installation:
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

3. Install [FlagGems](https://flagos-ai.github.io/FlagGems/getting-started/install/)

    3.1 For NVIDIA, install the tested FlagTree/Triton runtime

    FlagTree and FlagGems are a matched runtime. vLLM 0.28.0 normally installs
    standalone Triton 3.7.1, but the NVIDIA baseline validated here uses
    FlagTree `0.6.2a1`, which provides Triton `3.6.0`. Do not install a second
    standalone Triton package on top of FlagTree.

    ```sh
    # Repeat this command until no standalone Triton distribution remains.
    python3 -m pip uninstall -y triton

    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
    python3 -m pip install flagtree===0.6.2a1 $RES
    ```

    3.2 Install build dependencies

    ```sh
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    ```

    3.3 Install the latest FlagGems master

    ```sh
    git clone --branch master https://github.com/flagos-ai/FlagGems
    cd FlagGems
    pip install --no-build-isolation --no-deps .
    # or editable install
    pip install --no-build-isolation --no-deps -e .
    ```

    The NVIDIA results in this repository were produced with FlagGems commit
    `34c6d2ce416d32a98f1dd2bb6e4cfb67c59342d9`
    (`5.3.5+g34c6d2ce4`). Re-run the blacklist and model tests when moving to a
    newer FlagGems master revision.

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

3. Enable eager execution

    Ascend requires eager execution. Add `enforce_eager=True` to the `LLM` constructor or pass `--enforce-eager` on the command line.


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
