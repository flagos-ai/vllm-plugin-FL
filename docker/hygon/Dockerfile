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

ARG HYGON_BASE_IMAGE=harbor.baai.ac.cn/plugin/hygon-dtk26.04-tree0.6.0hcu3.6-triton3.6.0-cxnone-plugin0.3.0-vllm0.24.0-cp310-pt2100-x64:202609041747

# ---------- base stage ----------
FROM ${HYGON_BASE_IMAGE} AS base

ARG VLLM_VERSION=0.24.0
ARG PYTHON_VERSION=3.10
ARG UBUNTU_VERSION=22.04
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL

ENV EDEBIAN_FRONTEND=noninteractive \
    PYTORCH_MIOPEN_SUGGEST_NDHWC=1 \
    ALLREDUCE_STREAM_WITH_COMPUTE=1 \
    HIP_KERNEL_EVENT_SYSTENFENCE=1 \
    HIP_PATH=/opt/dtk/hip \
    CMAKE_PREFIX_PATH=/opt/hyhal:/opt/dtk/.hyhal/rocm_smi/:/opt/hyhal:/opt/dtk-26.04-DCC2602-0317/.hyhal/rocm_smi/: \
    NCCL_NCHANNELS_PER_PEER=2 \
    HYHAL_PATH=/opt/hyhal \
    PYTHONPATH=/usr/local/: \
    CPLUS_INCLUDE_PATH=/opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/dushmem/include:/opt/dtk/opencl/include:/opt/dtk/.hyhal/rocm_smi/include:/opt/dtk-26.04-DCC2602-0317/dcc/gcvm/include:/opt/dtk-26.04-DCC2602-0317/include:/opt/hyhal/include:/opt/dtk-26.04-DCC2602-0317/llvm/include:/opt/dtk-26.04-DCC2602-0317/dushmem/include:/opt/dtk-26.04-DCC2602-0317/opencl/include:/opt/dtk-26.04-DCC2602-0317/.hyhal/rocm_smi/include: \
    LIBRARY_PATH=/opt/hyhal/lib:/opt/dtk/opencl/lib:/opt/dtk/dushmem/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/hyhal/lib:/opt/dtk-26.04-DCC2602-0317/opencl/lib:/opt/dtk-26.04-DCC2602-0317/dushmem/lib:/opt/dtk-26.04-DCC2602-0317/.hyhal/rocm_smi/lib: \
    DTKROOT=/opt/dtk \
    ROCM_PATH=/opt/dtk \
    LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/lib64/:/opt/mpi/lib:/opt/hwloc/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk-26.04-DCC2602-0317/dcc/gcvm/lib:/opt/dtk-26.04-DCC2602-0317/hip/lib:/opt/dtk-26.04-DCC2602-0317/llvm/lib:/opt/dtk-26.04-DCC2602-0317/lib:/opt/dtk-26.04-DCC2602-0317/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk-26.04-DCC2602-0317/dushmem/lib:/opt/dtk-26.04-DCC2602-0317/opencl/lib:/opt/dtk-26.04-DCC2602-0317/.hyhal/rocm_smi/lib: \
    HIP_USE_GRAPH_QUEUE_POOL=1 \
    PATH=/usr/local/bin:/opt/mpi/bin:/opt/hwloc/bin/:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/dtk/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk/opencl/bin:/opt/dtk/.hyhal/rocm_smi/bin:/opt/dtk-26.04-DCC2602-0317/bin:/opt/dtk-26.04-DCC2602-0317/llvm/bin:/opt/dtk-26.04-DCC2602-0317/hip/bin:/opt/dtk-26.04-DCC2602-0317/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk-26.04-DCC2602-0317/opencl/bin:/opt/dtk-26.04-DCC2602-0317/.hyhal/rocm_smi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    C_INCLUDE_PATH=/opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/opencl/include:/opt/dtk/.hyhal/rocm_smi/include:/opt/dtk-26.04-DCC2602-0317/dcc/gcvm/include:/opt/dtk-26.04-DCC2602-0317/include:/opt/hyhal/include:/opt/dtk-26.04-DCC2602-0317/llvm/include:/opt/dtk-26.04-DCC2602-0317/opencl/include:/opt/dtk-26.04-DCC2602-0317/.hyhal/rocm_smi/include: \
    TRITON_HIP_CLANG_PATH=/opt/rocm/aillvm/bin/clang-18 \
    GEMS_VENDOR=hygon \
    VLLM_PLUGINS=fl \
    VLLM_VENDOR=cuda

# Prepare Python build tooling for editable source installs.
# RUN python -m pip install --upgrade \
#         ${INDEX_URL:+--index-url ${INDEX_URL}} \
#         ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
#         pip "setuptools<81" wheel setuptools-scm setuptools-rust

# Replace the base image's vLLM with official vLLM built in empty mode.
# RUN python -m pip uninstall -y vllm || true \
#     && python -m pip cache remove vllm || true \
#     && SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')" \
#     && find "${SITE_PACKAGES}" -maxdepth 1 \
#         \( -name "vllm" -o -name "vllm-*.dist-info" -o -name "vllm-*.egg-info" \) \
#         -exec rm -rf {} + 2>/dev/null || true \
#     && rm -rf /opt/vllm-src/vllm \
#     && mkdir -p /opt/vllm-src \
#     && git clone https://github.com/vllm-project/vllm.git /opt/vllm-src/vllm \
#     && cd /opt/vllm-src/vllm \
#     && git checkout "v${VLLM_VERSION}" \
#     && VLLM_TARGET_DEVICE=empty python -m pip install -v --no-build-isolation -e . --no-deps

WORKDIR /workspace

# ---------- ci stage ----------
FROM base AS ci

ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
ARG FLAGGEMS_VERSION=master

# Install dev/test tools in the image instead of every CI job.
RUN python -m pip install \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        pytest \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        numpy \
        requests \
        openai \
        decorator \
        pyyaml \
        sqlalchemy \
        setuptools-scm \
        scikit-build-core==0.11 \
        compressed-tensors==0.17.0 \
        pybind11 \
        ninja \
        cmake

# Install FlagGems.
# RUN rm -rf /opt/vllm-src/FlagGems \
#     && mkdir -p /opt/vllm-src \
#     && git clone https://github.com/flagos-ai/FlagGems /opt/vllm-src/FlagGems \
#     && cd /opt/vllm-src/FlagGems \
#     && git checkout "${FLAGGEMS_VERSION}" \
#     && python -m pip install \
#         ${INDEX_URL:+--index-url ${INDEX_URL}} \
#         ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
#         --no-build-isolation -e .


# Install FlagTree
# RUN rm -rf /opt/vllm-src/FlagTree \
#     && mkdir -p /opt/vllm-src \
#     && python3 -m pip uninstall -y triton \
#     && RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple" \
#     && python3.10 -m pip install flagtree===0.6.1+hcu3.6 $RES

ENV TRITON_HIP_CLANG_PATH=/opt/rocm/aillvm/bin/clang-18

WORKDIR /workspace

# ---------- dev stage ----------
FROM base AS dev

ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL

RUN python -m pip install \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        pytest \
        pytest-cov \
        pytest-json-report \
        ruff \
        pre-commit \
        ninja \
        cmake

WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release

WORKDIR /workspace
