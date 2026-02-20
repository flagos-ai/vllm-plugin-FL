# vllm-plugin-FL

vllm-plugin-FL is a plugin for the [vLLM](https://github.com/vllm-project/vllm) inference/serving framework, built on FlagOS's unified multi-chip backend — including the unified operator library [FlagGems](https://github.com/flagos-ai/FlagGems) and the unified communication library [FlagCX](https://github.com/flagos-ai/FlagCX). It extends vLLM's capabilities and performance across diverse hardware environments. Without changing vLLM's original interfaces or usage patterns, the same command can run model inference/serving on different chips.

## Supported Models and Chips

In theory, vllm-plugin-FL can support all models available in vLLM, as long as no unsupported operators are involved. The tables below summarize the current support status of end-to-end verified models and chips, including both fully supported and in-progress ("Merging") entries.

### Supported Models

| Model | Status | Reference |
|-------|--------|-----------|
| Qwen3.5 | Merging | [example](./examples/qwen3_5_offline_inference.py) |
| Qwen3-Next | Supported | [example](./examples/qwen3_next_offline_inference.py) |
| Qwen3-4B | Supported | [example](./examples/offline_inference.py) |
| MiniCPM | Supported | [example](./examples/minicpm/) |

### Supported Chips

| Chip Vendor | Status | Reference |
|-------------|--------|-----------|
| NVIDIA | Supported | - |
| Ascend | Merging | [PR #55](https://github.com/flagos-ai/vllm-plugin-FL/pull/55) |
| MetaX | Merging | [PR #47](https://github.com/flagos-ai/vllm-plugin-FL/pull/47) |
| Pingtouge | Merging | - |
| Iluvatar | Merging | [PR #58](https://github.com/flagos-ai/vllm-plugin-FL/pull/58) |
| Tsingmicro | Merging | [PR #52](https://github.com/flagos-ai/vllm-plugin-FL/pull/52) |

## Quick Start

### Setup

0. Install vllm from the official [v0.13.0](https://github.com/vllm-project/vllm/tree/v0.13.0) (optional if the correct version is installed) or from the fork [vllm-FL](https://github.com/flagos-ai/vllm-FL).


1. Install vllm-plugin-FL

    1.1 Clone the repository:

    ```sh
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    ```

    1.2 install
    ```sh
    cd vllm-plugin-FL
    pip install -r requirements.txt
    pip install --no-build-isolation .
    # or editble install
    pip install --no-build-isolation -e .
    ```

2. Install [FlagGems](https://github.com/flagos-ai/FlagGems/blob/master/docs/getting-started.md#quick-installation)

    2.1 Install Build Dependencies

    ```sh
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    ```

    2.2 Installation FlagGems

    ```sh
    git clone https://github.com/flagos-ai/FlagGems
    cd FlagGems
    pip install --no-build-isolation .
    # or editble install
    pip install --no-build-isolation -e .
    ```

3. Install [FlagCX](https://github.com/flagos-ai/FlagCX/blob/main/docs/getting_started.md#build-and-installation)

    3.1 Clone the repository:
    ```sh
    git clone https://github.com/flagos-ai/FlagCX.git
    cd FlagCX
    git checkout -b v0.9.0
    git submodule update --init --recursive
    ```

    3.2 Build the library with different flags targeting to different platforms:
    ```sh
    make USE_NVIDIA=1
    ```

    3.3 Set environment
    ```sh
    export FLAGCX_PATH="$PWD"
    ```

    3.4 Installation FlagCX
    ```sh
    cd plugin/torch/
    FLAGCX_ADAPTOR=[xxx] pip install . --no-build-isolation
    # or editable install
    FLAGCX_ADAPTOR=[xxx] pip install -e . --no-build-isolation
    ```
    Note: [xxx] should be selected according to the current platform, e.g., nvidia, ascend, etc.


If there are multiple plugins in the current environment, you can specify use vllm-plugin-fl via VLLM_PLUGINS='fl'.

### Run a Task

#### Offline Batched Inference
With vLLM and vLLM-fl installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing). See the example script: [offline_inference](./examples/offline_inference.py). Or use blow python script directly.
```python
from vllm import LLM, SamplingParams
import torch
from vllm.config.compilation import CompilationConfig


if __name__ == '__main__':
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
