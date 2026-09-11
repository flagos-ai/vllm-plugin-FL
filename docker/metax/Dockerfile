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

ARG METAX_BASE_IMAGE=harbor.baai.ac.cn/plugin/metax-maca3.7.0-treenone-triton3.0.0-cxnone-plugin0.2.0-vllm0.24.0-cp312-pt280-x64:202507290700

# ---------- base stage ----------
FROM ${METAX_BASE_IMAGE} AS base

ARG METAX_BASE_IMAGE
ARG VLLM_VERSION=0.24.0
ARG FLAGGEMS_REF=95ed7f9c43b23cba704baaabca4200978c0d6292
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=dev
ARG SOURCE_URL=https://github.com/flagos-ai/vllm-plugin-FL

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.description="MetaX CI image for vllm-plugin-FL" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${METAX_BASE_IMAGE}" \
      io.flagos.accelerator.backend="metax" \
      io.flagos.image.stage="${IMAGE_STAGE}" \
      io.flagos.vllm.version="${VLLM_VERSION}" \
      io.flagos.flaggems.revision="${FLAGGEMS_REF}"

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

RUN python -m pip install --no-cache-dir \
        wheel \
        setuptools-scm \
        scikit-build-core==0.11 \
        pybind11 \
        ninja \
        cmake

# The validated base image already contains vLLM 0.24.0 built with
# VLLM_TARGET_DEVICE=empty and the MetaX-tested FlagGems revision.

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

# Keep project-independent CI dependencies in the image. setup.sh only needs
# to install the checked-out vllm-plugin-FL package.
RUN python -m pip install --no-cache-dir \
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
