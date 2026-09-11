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

ARG METAX_BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:vllm-metax-0.20.0-maca.ai3.7.0.107-torch2.8-py312-ubuntu22.04-amd64

# ---------- base stage ----------
FROM ${METAX_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    GEMS_VENDOR=metax \
    VLLM_PLUGINS=fl \
    PATH="/opt/conda/bin:${PATH}"

# actions/checkout requires git before any platform setup script runs. Build
# tools are installed here because both vLLM empty mode and FlagGems are fixed
# parts of the MetaX runtime rather than per-job CI dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---------- dev stage ----------
FROM base AS dev

RUN python -m pip install --no-cache-dir \
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
ARG FLAGGEMS_REF=c2d04fce89d8226a0e2a544fb23bd66b9f973a6c
ARG INDEX_URL=https://pypi.org/simple

RUN python -m pip install --no-cache-dir \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        wheel \
        setuptools-scm \
        setuptools_rust \
        scikit-build-core==0.11 \
        pybind11 \
        ninja \
        cmake

# Replace the vLLM Python package from the vendor image with official vLLM in
# empty mode. The MetaX kernels and torch runtime remain provided by the base
# image. Install in editable mode from /workspace/vllm so changes are live.
RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm \
    && git -C /workspace/vllm checkout "v${VLLM_VERSION}" \
    && VLLM_TARGET_DEVICE=empty python -m pip install -v \
        --no-build-isolation --no-deps -e /workspace/vllm

# FlagGems is pinned to the source revision already validated on MetaX C550.
# --no-build-isolation reuses the setuptools/wheel/setuptools-scm already
# installed above instead of letting pip fetch build deps from the mirror,
# which may not carry the exact versions FlagGems' pyproject.toml requires.
# INDEX_URL is passed explicitly so runtime deps (e.g. PyYAML==6.0.1) are
# resolved from public PyPI rather than the aliyun mirror in the base image.
RUN git clone https://github.com/FlagOpen/FlagGems.git /workspace/FlagGems \
    && git -C /workspace/FlagGems checkout ${FLAGGEMS_REF} \
    && GEMS_VENDOR=metax python -m pip install -v --no-build-isolation \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        -e /workspace/FlagGems


# Keep project-independent CI dependencies in the image. setup.sh only needs
# to install the checked-out vllm-plugin-FL package.
RUN python -m pip install --no-cache-dir \
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

# ---------- release stage ----------
FROM base AS release

WORKDIR /workspace
