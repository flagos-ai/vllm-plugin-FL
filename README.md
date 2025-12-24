# vllm-FL
A vLLM plugin built on the FlagOS unified multi-chip backend.

## Quick Start

### Setup

0. Install vllm from the official [v0.11.0](https://github.com/vllm-project/vllm/tree/v0.11.0) (optional if the correct version is installed) or from the fork [vllm-FL](https://github.com/flagos-ai/vllm-FL).

1. Clone the repository:

    ```sh
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    ```

2. Install FlagGems

    2.1 Install Build Dependencies

    ```sh
    pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
    ```

    2.3 Installation FlagGems

    ```shell
    git clone https://github.com/flagos-ai/FlagGems
    cd FlagGems
    pip install --no-build-isolation .
    # or editble install
    pip install --no-build-isolation -e .
    ```

3. Install FlagCX

    3.1 Clone the repository:
    ```sh
    git clone GitHub - flagos-ai/FlagCX
    git checkout -b v0.3.0
    ```

    3.2 Build the library with different flags targeting to different platforms:
    ```sh
    cd FlagCX
    make USE_NVIDIA=1
    ```

    3.3 Set environment
    ```sh
    export FLAGCX_PATH=${pwd}
    ```

    3.4 Installation FlagGems
    ```sh
    cd FlagCX/plugin/torch/
    python setup.py develop --adaptor nvidia/ascend
    ```

4. Install vllm-plugin-fl

```sh
pip install --no-build-isolation -e .
```

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