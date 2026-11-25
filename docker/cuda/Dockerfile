# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BUILD_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.0.3:2.3.0
ARG FLAGOS_DAILY_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-daily/simple
# ARG FLAGOS_NVIDIA_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-nvidia/simple

# ---------- base stage ----------
FROM ${BUILD_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

ARG CUDA_VERSION=13.0.3
ARG VLLM_VERSION=0.32.0
ARG FLAGGEMS_VLLM_VERSION=0.3.0.dev20261119
ARG FLAGOS_DAILY_INDEX_URL

# System dependencies and CUDA development tools for JIT compilation
RUN CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvrtc-${CUDA_VERSION_DASH} \
        cuda-nvrtc-dev-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} \
        libcublas-dev-${CUDA_VERSION_DASH} \
        libcusolver-dev-${CUDA_VERSION_DASH} \
        libcusparse-dev-${CUDA_VERSION_DASH} \
        # Required by fastsafetensors
        libnuma-dev \
        # numactl CLI for NUMA binding at runtime
        numactl \
        # Required by DeepGEMM (deep_jit uses elfutils/libdwfl.h)
        libdw-dev \
        # Required by av/torchcodec/PyNvVideoCodec for video multimodal models
        ffmpeg \
        # Required by opencv-python (non-headless fallback) and X11-dependent libs
        libsm6 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Ensures nvcc/ptxas (installed above, needed at runtime for FlashInfer/
# DeepGEMM/EP kernel JIT compilation) are reachable on PATH. Harmless no-op
# if already set by FINAL_BASE_IMAGE.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install FlagGems-vLLM and vLLM
RUN PYTHONPATH=/opt/triton \
    pip install --no-cache-dir \
        "flaggems-vllm==${FLAGGEMS_VLLM_VERSION}" --extra-index-url "${FLAGOS_DAILY_INDEX_URL}" \
        "vllm==${VLLM_VERSION}"

# Install DeepGEMM
RUN PYTHONPATH=/opt/triton \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    bash -c 'git clone --recursive https://github.com/deepseek-ai/DeepGEMM /workspace/DeepGEMM && \
    cd /workspace/DeepGEMM && \
    ./install.sh'

WORKDIR /workspace


# ---------- dev stage ----------
FROM base AS dev

# Install dev/test tools
RUN PYTHONPATH=/opt/triton \
    pip install --no-cache-dir \
        pytest \
        "pytest-xdist>=3.0.0" \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        numpy \
        requests \
        decorator \
        modelscope \
        ruff \
        pre-commit \
        ninja \
        cmake


# ---------- test stage ----------
FROM base AS test

# Install vLLM-Plugin-FL wheel package for SVT

# Minimal test environment - inherits vLLM and FlagGems from base
LABEL maintainer="FlagOS Contributors" \
      description="FlagOS vLLM-Plugin-FL environment with FlagGems"
