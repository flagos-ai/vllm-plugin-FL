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

ARG BUILD_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-mthreads-musa5.3.0:2.3.0
ARG FLAGOS_DAILY_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-daily/simple
ARG FLAGOS_MTHREADS_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-mthreads/simple
# vllm==0.32.0+flagos (py3-none-any, torch/triton-free) is currently published
# only on the hygon index; drop this ARG once it lands on flagos-pypi-mthreads.
ARG FLAGOS_HYGON_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-hygon/simple
ARG UV_DEFAULT_INDEX=https://pypi.org/simple

# ---------- base stage ----------
FROM ${BUILD_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

ARG VLLM_VERSION=0.32.0+flagos
ARG FLAGGEMS_VLLM_VERSION=0.3.0.dev20261124
ARG FLAGOS_DAILY_INDEX_URL
ARG FLAGOS_MTHREADS_INDEX_URL
ARG FLAGOS_HYGON_INDEX_URL
ARG UV_DEFAULT_INDEX

# FlagTree (/opt/flagtree, provides the Triton modules) and the vendor Triton
# (/opt/triton) are not in site-packages. The base's zz-flagos.sh wires them
# into PYTHONPATH only for bash (via BASH_ENV); make them importable for
# non-interactive `sh` CI steps too (flag_gems imports triton at top level).
ENV PYTHONPATH=/opt/flagtree

# Install FlagGems-vLLM and vLLM (constrained to keep base
# torch/torch_musa/flagtree/flag-gems intact).
RUN mkdir -p /opt/fl-ci && printf '%s\n' \
        'flag-gems==5.4.0' \
        'flagcx==0.14.1+musa5.3.0' \
        'flagtree==0.7.0+mthreads3.6' \
        'numpy==1.26.4' \
        'torch==2.13.0.post1+musa20261110' \
        'torch-musa==2.13.0.post1+musa20261110' \
        'torchaudio==2.11.0.post2+musa20261110' \
        'torchvision==0.28.0.post1+musa5.3.0' \
        'triton-musa==3.6.0+musa5.3.0' \
        > /opt/fl-ci/constraints.txt && \
    uv pip install --python /flagos/bin/python --no-cache \
        --constraint /opt/fl-ci/constraints.txt \
        --extra-index-url "${FLAGOS_DAILY_INDEX_URL}" \
        # Check hygon before mthreads so uv finds the cp312 xgrammar wheel
        # instead of stopping on the cp310-only wheel in mthreads.
        --extra-index-url "${FLAGOS_HYGON_INDEX_URL}" \
        --extra-index-url "${FLAGOS_MTHREADS_INDEX_URL}" \
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
