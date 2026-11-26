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

ARG BUILD_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.10:2.3.0
ARG FLAGOS_DAILY_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-daily/simple
ARG FLAGOS_HYGON_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-hygon/simple
ARG UV_DEFAULT_INDEX=https://pypi.org/simple

# ---------- base stage ----------
FROM ${BUILD_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

ARG VLLM_VERSION=0.32.0+flagos
ARG FLAGGEMS_VLLM_VERSION=0.3.0.dev20261124
ARG FLAGOS_DAILY_INDEX_URL
ARG FLAGOS_HYGON_INDEX_URL
ARG UV_DEFAULT_INDEX

# System / host driver compatibility fix (base runtime already includes DTK 26.10).
# Host driver 6.4.2-V2.1.0 mounts libamd_smi.so without
# `amdsmi_get_gpu_enumeration_info`, causing `amdsmi==7.0.2` to fail on import.
RUN uv pip uninstall --python /flagos/bin/python amdsmi || true

# Ensures DTK / HyHAL libraries and FlagTree are reachable on LD_LIBRARY_PATH / PYTHONPATH.
ENV ROCM_PATH=/opt/dtk
ENV HIP_PATH=/opt/dtk/hip
ENV DTKROOT=/opt/dtk
ENV HYHAL_PATH=/opt/hyhal
ENV LD_LIBRARY_PATH="/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/lib:/opt/dtk/.hyhal/lib64"
ENV PYTHONPATH=/opt/flagtree

# Install FlagGems-vLLM and vLLM (constrained to keep base torch/triton/flagtree intact)
COPY docker/hygon/constraints.txt /opt/fl-ci/constraints.txt
RUN uv pip install --python /flagos/bin/python --no-cache \
        --constraint /opt/fl-ci/constraints.txt \
        --extra-index-url "${FLAGOS_DAILY_INDEX_URL}" \
        --extra-index-url "${FLAGOS_HYGON_INDEX_URL}" \
        --index-strategy unsafe-first-match \
        "flaggems-vllm==${FLAGGEMS_VLLM_VERSION}" \
        "vllm==${VLLM_VERSION}"

WORKDIR /workspace


# ---------- dev stage ----------
FROM base AS dev

ARG UV_DEFAULT_INDEX

# Install dev/test tools
RUN uv pip install --python /flagos/bin/python --no-cache \
        --constraint /opt/fl-ci/constraints.txt \
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

# Minimal test environment - inherits vLLM and FlagGems from base
LABEL maintainer="FlagOS Contributors" \
      description="FlagOS vLLM-Plugin-FL environment with FlagGems"
