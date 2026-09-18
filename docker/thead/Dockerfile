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

ARG THEAD_BASE_IMAGE=egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun/inference-xpu-pytorch:26.04-v2.1.0-vllm0.23.0-torch2.10-cu130-20260710

# ---------- base stage ----------
FROM ${THEAD_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    GEMS_VENDOR=thead \
    VLLM_PLUGINS=fl

# actions/checkout requires git before any platform setup script runs.
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#         build-essential \
#         ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---------- dev stage ----------
FROM base AS dev

RUN pip install --no-cache-dir \
        pytest \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        pre-commit \
        ruff

WORKDIR /workspace

# ---------- ci stage ----------
FROM base AS ci

ARG VLLM_VERSION=0.24.0
ARG FLAGGEMS_REF=v5.3.4
ARG INDEX_URL=

RUN pip install --no-cache-dir \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        setuptools-scm==10.0.5 \
        setuptools_rust \
        scikit-build-core==0.11 \
        pybind11 \
        ninja \
        cmake

# Replace the vLLM Python package from the vendor image with official vLLM in
# empty mode. The T-Head kernels and torch runtime remain provided by the base
# image. Install in editable mode from /workspace/vllm so changes are live.
RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm \
    && git -C /workspace/vllm checkout "v${VLLM_VERSION}" \
    && VLLM_TARGET_DEVICE=empty pip install -v \
        --no-build-isolation --no-deps /workspace/vllm

# FlagGems is pinned to the source revision already validated on T-Head PPU.
# --no-build-isolation reuses the setuptools/wheel/setuptools-scm already
# installed above instead of letting pip fetch build deps from the mirror.
RUN git clone https://github.com/FlagOpen/FlagGems.git /workspace/FlagGems \
    && git -C /workspace/FlagGems checkout ${FLAGGEMS_REF} \
    && GEMS_VENDOR=thead pip install -v --no-build-isolation \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        -e /workspace/FlagGems

# Install FlagTree (Triton backend for T-Head PPU).
# See https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ppu
ARG FLAGTREE_VERSION=0.6.2a2+ppu3.6
ARG FLAGOS_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-hosted/simple

RUN pip install --no-cache-dir \
        --index-url ${FLAGOS_INDEX_URL} \
        flagtree===${FLAGTREE_VERSION}

# Keep project-independent CI dependencies in the image. setup.sh only needs
# to install the checked-out vllm-plugin-FL package.
RUN pip install --no-cache-dir \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        pytest \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        requests \
        openai \
        decorator \
        pyyaml \
        sqlalchemy \
        "modelscope>=1.18.1"

WORKDIR /workspace
