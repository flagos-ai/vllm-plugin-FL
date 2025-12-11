# vllm-FL
A vLLM plugin built on the FlagOS unified multi-chip backend.

## Quick Start
vllm-plugin-fl based on FlagGems and FlagCX. 


### Setup

0. Install vllm from the official v0.11.0 (optional if the correct version is installed) or from the fork vllm-FL(https://github.com/flagos-ai/vllm-FL).

1. Clone the repository:

    ```sh
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    ```

2. install FlagGems

    2.1 Install Build Dependencies

    ```sh
    pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
    ```

    2.3 Installation FlagGems

    ```shell
    git clone https://github.com/FlagOpen/FlagGems.git
    cd FlagGems
    pip install --no-build-isolation .
    # or editble install
    pip install --no-build-isolation -e .
    ```

3. install FlagCX

    3.1 Clone the repository:
    ```sh
    git clone https://github.com/flagos-ai/FlagCX
    git checkout -b v0.3.0
    ```
    
    3.2 Build the library with different flags targeting to different platforms:
    ```sh
    cd FlagCX
    make USE_NVIDIA=1
    ```

    3.3 set environment
    ```sh
    export FLAGCX_PATH=${pwd}
    ```

4. install vllm-plugin-fl

```sh
pip install --no-build-isolation -e .
```

If there are multiple plugins in the current environment, you can specify use vllm-plugin-fl via VLLM_PLUGINS='fl'.