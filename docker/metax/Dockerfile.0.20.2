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

ARG METAX_BASE_IMAGE
ARG VLLM_VERSION=0.20.2
ARG FLAGGEMS_REF=3123859968915e361e8452dd796dc6b27c956324
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

# Replace the vLLM Python package from the vendor image with official vLLM in
# empty mode. The MetaX kernels and torch runtime remain provided by the base
# image. Install non-editably so /workspace mounts cannot hide the package.
RUN python -m pip uninstall -y vllm || true; \
    python -m pip cache remove vllm || true; \
    SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"; \
    find "${SITE_PACKAGES}" -maxdepth 1 \
        \( -name "vllm" -o -name "vllm-*.dist-info" -o -name "vllm-*.egg-info" \) \
        -exec rm -rf {} + 2>/dev/null || true; \
    rm -rf /tmp/vllm \
    && git clone --depth 1 --branch "v${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm.git /tmp/vllm \
    && VLLM_TARGET_DEVICE=empty python -m pip install --no-cache-dir \
        --no-build-isolation --no-deps /tmp/vllm \
    && rm -rf /tmp/vllm

# FlagGems is pinned to the source revision already validated on MetaX C550.
RUN rm -rf /tmp/FlagGems \
    && rm -f /tmp/FlagGems.tar.gz \
    && mkdir -p /tmp/FlagGems \
    && curl --fail --location --retry 5 --retry-all-errors \
        --retry-delay 5 --connect-timeout 30 --max-time 1200 \
        "https://codeload.github.com/FlagOpen/FlagGems/tar.gz/${FLAGGEMS_REF}" \
        --output /tmp/FlagGems.tar.gz \
    && tar -xzf /tmp/FlagGems.tar.gz --strip-components=1 -C /tmp/FlagGems \
    && python -m pip install --no-cache-dir \
        -r /tmp/FlagGems/requirements/requirements_metax.txt \
    && GEMS_VENDOR=metax python -m pip install --no-cache-dir \
        --no-build-isolation /tmp/FlagGems \
    && rm -rf /tmp/FlagGems /tmp/FlagGems.tar.gz \
    && python -m pip install --no-cache-dir \
        numpy==1.26.4 \
        opencv-python-headless==4.11.0.86 \
        outlines-core==0.2.14

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
