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

ARG HYGON_BASE_IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.20.0-ubuntu22.04-dtk26.04-py3.10-MiniCPM-V-4.6

# ---------- base stage ----------
FROM ${HYGON_BASE_IMAGE} AS base

ARG VLLM_VERSION=0.20.2
ARG PYTHON_VERSION=3.10
ARG UBUNTU_VERSION=22.04
ARG INDEX_URL
ARG EXTRA_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive
ENV DTK_HOME=/opt/dtk \
    ROCM_PATH=/opt/dtk \
    HIP_PATH=/opt/dtk/hip \
    HSA_PATH=/opt/dtk/hsa \
    HIP_CLANG_PATH=/opt/dtk/llvm/bin \
    DEVICE_LIB_PATH=/opt/dtk/amdgcn/bitcode \
    GEMS_VENDOR=hygon \
    VLLM_PLUGINS=fl
ENV PATH="/root/.local/bin:/opt/dtk/bin:/opt/dtk/hip/bin:/opt/dtk/llvm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/hyhal/lib/criu:/opt/hyhal/lib/rocprofiler:/opt/hyhal/lib:/usr/local/hyhal/lib:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/lib:/opt/dtk/aillvm/lib:/opt/dtk/hsa/lib:${LD_LIBRARY_PATH}"
ENV CMAKE_PREFIX_PATH="/opt/dtk:/opt/dtk/hip:/opt/dtk/hsa:/opt/hyhal:/usr/local/hyhal:${CMAKE_PREFIX_PATH}"

# Add the minimal Hygon runtime libraries staged by docker/build.sh.
COPY .hygon-runtime/ /

# Prepare Python build tooling for editable source installs.
RUN python -m pip install --upgrade \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        pip "setuptools<81" wheel setuptools-scm

# Replace the base image's vLLM with official vLLM built in empty mode.
RUN python -m pip uninstall -y vllm || true \
    && python -m pip cache remove vllm || true \
    && SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && find "${SITE_PACKAGES}" -maxdepth 1 \
        \( -name "vllm" -o -name "vllm-*.dist-info" -o -name "vllm-*.egg-info" \) \
        -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /opt/vllm-src/vllm \
    && mkdir -p /opt/vllm-src \
    && git clone https://github.com/vllm-project/vllm.git /opt/vllm-src/vllm \
    && cd /opt/vllm-src/vllm \
    && git checkout "v${VLLM_VERSION}" \
    && VLLM_TARGET_DEVICE=empty python -m pip install -v --no-build-isolation -e . --no-deps

WORKDIR /workspace

# ---------- ci stage ----------
FROM base AS ci

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG FLAGGEMS_VERSION=62d70b9e858ec407572153ee8cdf65cc24a637d5
ARG VLLM_PLUGIN_FL_VERSION=ffa2ee3eb3831f3873dd0966d12fc8e0b4e6e3d4

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
        pybind11 \
        ninja \
        cmake

# Install FlagGems from the pinned source commit.
RUN rm -rf /opt/vllm-src/FlagGems \
    && mkdir -p /opt/vllm-src \
    && git clone https://github.com/flagos-ai/FlagGems /opt/vllm-src/FlagGems \
    && cd /opt/vllm-src/FlagGems \
    && git checkout "${FLAGGEMS_VERSION}" \
    && python -m pip install \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        --no-build-isolation -e .

# Install vllm-plugin-FL from the pinned source commit.
RUN rm -rf /opt/vllm-src/vllm-plugin-FL \
    && mkdir -p /opt/vllm-src \
    && git clone https://github.com/flagos-ai/vllm-plugin-FL /opt/vllm-src/vllm-plugin-FL \
    && cd /opt/vllm-src/vllm-plugin-FL \
    && git checkout "${VLLM_PLUGIN_FL_VERSION}" \
    && CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/dtk;/opt/dtk/hip;/opt/dtk/hsa;/opt/hyhal;/usr/local/hyhal ${CMAKE_ARGS:-}" \
        VLLM_VENDOR=cuda python -m pip install \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        --no-build-isolation -e .

WORKDIR /workspace

# ---------- dev stage ----------
FROM base AS dev

ARG INDEX_URL
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
