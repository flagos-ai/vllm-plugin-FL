# MiniCPM-xx vLLM Deployment Guide

## 1. Environment Setup

### 1.1 Install vLLM

```bash
pip install vllm==0.13.0
```

For video inference, install the video module:
```bash
pip install vllm[video]
```

### 1.2 Install [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL)

#### 1.2.1 Clone the repository:

```sh
git clone https://github.com/flagos-ai/vllm-plugin-FL

cd vllm-plugin-FL
pip install -r requirements.txt
pip install --no-build-isolation .
# or editble install
pip install --no-build-isolation -e .
```

#### 1.2.2 Install [FlagGems](https://github.com/flagos-ai/FlagGems/blob/master/docs/getting-started.md#quick-installation)

```sh
pip install flag-gems==4.2.1rc0 [-i https://pypi.tuna.tsinghua.edu.cn/simple]
```

#### 1.2.3 [Opitonal] Install [FlagCX](https://github.com/flagos-ai/FlagCX/blob/main/docs/getting_started.md#build-and-installation)

```sh
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX
git checkout -b v0.7.0
git submodule update --init --recursive
make USE_NVIDIA=1 # NVIDIA GPU Platform
export FLAGCX_PATH="$PWD"

cd plugin/torch/
python setup.py develop --adaptor [xxx]
```
Note: [xxx] should be selected according to the current platform, e.g., nvidia, ascend, etc.


If there are multiple plugins in the current environment, you can specify use vllm-plugin-fl via VLLM_PLUGINS='fl'.


## 2. API Service Deployment

### 2.1 Launch API Service

```bash
vllm serve <model_path>  --dtype auto --max-model-len 2048 --api-key token-abc123 --gpu_memory_utilization 0.9 --trust-remote-code --max-num-batched-tokens 2048
```
**Parameter Description:**
- `<model_path>`: Specify the local path to your MiniCPM-V4.5 model
- `--api-key`: Set the API access key
- `--max-model-len`: Set the maximum model length
- `--gpu_memory_utilization`: GPU memory utilization rate

