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

ARG BUILD_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-metax-maca3.8.1.3:2.3.0
ARG FLAGOS_DAILY_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-daily/simple
ARG FLAGOS_METAX_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-metax/simple
ARG UV_DEFAULT_INDEX=https://pypi.org/simple

# ---------- base stage ----------
FROM ${BUILD_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

ARG VLLM_VERSION=0.32.0+flagos
ARG FLAGGEMS_VLLM_VERSION=0.3.0.dev20261126
ARG FLAGOS_DAILY_INDEX_URL
ARG FLAGOS_METAX_INDEX_URL
ARG UV_DEFAULT_INDEX

# The FlagOS 2.3.0 runtime exports MACA paths and PYTHONPATH=/opt/flagtree
# through /etc/profile.d via BASH_ENV. Docker RUN and other non-login shells
# do not source those files, so the MACA toolchain, FlagTree and the /flagos
# venv are pinned here to stay visible while building and running.
ENV MACA_PATH=/opt/maca \
    LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/mxgpu_llvm/lib" \
    PYTHONPATH=/opt/flagtree \
    PATH="/flagos/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/bin:${PATH}"

# Install FlagGems-vLLM and vLLM (constrained to keep base torch/flagtree intact)
RUN mkdir -p /opt/fl-ci && printf '%s\n' \
        'flag-gems==5.4.0' \
        'flagcx==0.14.0+maca3.8.1.3' \
        'flagtree==0.7.0+metax3.6' \
        'flash-attn==2.6.3+metax3.8.1.0torch2.10' \
        'numpy==1.26.4' \
        'torch==2.10.0+metax3.8.1.0' \
        'torchaudio==2.4.1+metax3.8.1.0' \
        'torchvision==0.25.0+metax3.8.1.0' \
        > /opt/fl-ci/constraints.txt && \
    uv pip install --python /flagos/bin/python --no-cache \
        --constraint /opt/fl-ci/constraints.txt \
        --extra-index-url "${FLAGOS_DAILY_INDEX_URL}" \
        --extra-index-url "${FLAGOS_METAX_INDEX_URL}" \
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
