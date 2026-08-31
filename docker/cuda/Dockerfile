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

ARG CUDA_VERSION=13.0.2
ARG UBUNTU_VERSION=24.04

# ---------- base stage ----------
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.6
ARG VLLM_VERSION=0.24.0

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        git \
        curl \
        ca-certificates \
        vim \
        wget \
        python3-pip \
        libibverbs-dev \
        gcc-11 \
        g++-11 \
        librdmacm-dev \
        libnuma-dev \
        software-properties-common \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Verify GCC version
RUN gcc --version

# Install pip via uv
RUN uv pip install --system --break-system-packages pip setuptools setuptools-scm wheel

ENV PATH="/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0

# ---------- dev stage ----------
FROM base AS dev

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.24.0

# Install vLLM
RUN uv pip install --system --break-system-packages \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        "vllm==${VLLM_VERSION}"

# Install dev tools
RUN uv pip install --system --break-system-packages \
        pytest \
        pytest-cov \
        pytest-json-report \
        ruff \
        pre-commit \
        ninja \
        cmake

# ---------- ci stage ----------
FROM base AS ci

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.24.0

# Install vLLM
RUN uv pip install --system --break-system-packages \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        "vllm[audio]==${VLLM_VERSION}"

# Install dev/test tools
RUN uv pip install --system --break-system-packages \
        pytest \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        numpy \
        requests \
        decorator \
        "modelscope>=1.18.1" \
        ruff \
        pre-commit \
        ninja \
        cmake
ARG FLAGGEMS_VERSION=v5.3.5

# Install FlagGems
RUN uv pip install --system --break-system-packages scikit-build-core==0.11 pybind11 \
    && git clone --branch ${FLAGGEMS_VERSION} --depth 1 https://github.com/flagos-ai/FlagGems /workspace/FlagGems \
    && touch /workspace/FlagGems/src/flag_gems/fused/DSA/__init__.py \
    && uv pip install --system --break-system-packages --no-build-isolation /workspace/FlagGems

# Install DeepGEMM
RUN git clone https://github.com/deepseek-ai/DeepGEMM.git /workspace/DeepGEMM \
    && cd /workspace/DeepGEMM \
    && git checkout 891d57b \
    && git submodule update --init --recursive \
    && TORCH_DEVICE_BACKEND_AUTOLOAD=0 uv pip install --system --break-system-packages --no-build-isolation .

ENV VLLM_PLUGINS=fl

WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.24.0

# Install vLLM
RUN uv pip install --system --break-system-packages \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        "vllm==${VLLM_VERSION}"

WORKDIR /workspace

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
