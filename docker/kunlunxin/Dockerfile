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

ARG KUNLUNXIN_BASE_IMAGE=harbor.baai.ac.cn/plugin/xvllm-ubuntu2204-py310-torch29-0200:v20.0.10.0

# ---------- base stage ----------
FROM ${KUNLUNXIN_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG KUNLUNXIN_BASE_IMAGE
ARG VLLM_VERSION=0.20.2
ARG FLAGGEMS_REF=v5.0.0
ARG FLAGCX_REF=v0.13.0
ARG VLLM_PLUGIN_FL_REF=38e7dbc20197e2db742c4e4c9687d36ea4df9900
ARG DRIVER_VERSION=5.0.21.43
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=dev
ARG SOURCE_URL=https://github.com/flagos-ai/vllm-plugin-FL
ARG INDEX_URL
ARG EXTRA_INDEX_URL

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.description="Kunlunxin P800 CI image for vllm-plugin-FL" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${KUNLUNXIN_BASE_IMAGE}" \
      io.flagos.accelerator.backend="kunlunxin" \
      io.flagos.image.stage="${IMAGE_STAGE}" \
      io.flagos.vllm.version="${VLLM_VERSION}" \
      io.flagos.flaggems.revision="${FLAGGEMS_REF}" \
      io.flagos.flagcx.revision="${FLAGCX_REF}" \
      io.flagos.plugin.revision="${VLLM_PLUGIN_FL_REF}" \
      io.flagos.driver.version="${DRIVER_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    GEMS_VENDOR=kunlunxin \
    VLLM_PLUGINS=fl \
    FLAGCX_PATH=/workspace/FlagCX \
    USE_RESHAPE_AND_CACHE_FLASH=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
    PATH="/root/miniconda/envs/python310_torch29_cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/workspace/FlagCX/build/lib:/opt/kunlun/lib:${LD_LIBRARY_PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        ninja-build \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip config set global.index-url "${INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    && if [[ -n "${EXTRA_INDEX_URL:-}" ]]; then \
        python -m pip config set global.extra-index-url "${EXTRA_INDEX_URL}"; \
    fi

# Follow the Kunlunxin validation document source-install order.
RUN python -m pip uninstall -y vllm vllm_xpu || true \
    && rm -rf /workspace/vllm \
    && git clone --depth 1 --branch "v${VLLM_VERSION}" \
        https://gh-proxy.com/https://github.com/vllm-project/vllm.git /workspace/vllm \
    && cd /workspace/vllm \
    && git checkout "v${VLLM_VERSION}" \
    && python -m pip install setuptools_scm \
    && VLLM_TARGET_DEVICE=empty python -m pip install -v --no-build-isolation --no-deps . \
    && python -m pip uninstall -y numpy \
    && python -m pip install "numpy<2.0"

RUN rm -rf /workspace/vllm-plugin-FL \
    && git clone https://gh-proxy.com/https://github.com/flagos-ai/vllm-plugin-FL.git /workspace/vllm-plugin-FL \
    && cd /workspace/vllm-plugin-FL \
    && git checkout "${VLLM_PLUGIN_FL_REF}" \
    && python -m pip install --no-build-isolation -e .

RUN rm -rf /workspace/FlagGems \
    && git clone --depth 1 --branch "${FLAGGEMS_REF}" \
        https://gh-proxy.org/https://github.com/flagos-ai/FlagGems /workspace/FlagGems \
    && cd /workspace/FlagGems \
    && git checkout "${FLAGGEMS_REF}" \
    && python -m pip install -U scikit-build-core==0.11 \
        pybind11 \
        ninja \
        cmake \
    && python -m pip install sqlalchemy \
    && python -m pip install --no-build-isolation --no-deps .

RUN rm -rf "${FLAGCX_PATH}" \
    && git clone --depth 1 --branch "${FLAGCX_REF}" \
        https://gh-proxy.com/https://github.com/flagos-ai/FlagCX.git "${FLAGCX_PATH}" \
    && cd "${FLAGCX_PATH}" \
    && git checkout "${FLAGCX_REF}" \
    && git config --global url."https://gh-proxy.org/https://github.com/".insteadOf "https://github.com/" \
    && git submodule update --init --recursive \
    && CPATH="/usr/local/cuda-12.8/targets/x86_64-linux/include:/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/nvidia/nccl/include:/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch_xmlir/xre/include:${CPATH:-}" \
       LIBRARY_PATH="/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs:/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch_xmlir:/root/miniconda/envs/python310_torch29_cuda/xcudart/lib:${LIBRARY_PATH:-}" \
       CCL_HOME=/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch_xmlir/xccl \
       make USE_KUNLUNXIN=1 \
    && cd plugin/torch \
    && CPATH="/usr/local/cuda-12.8/targets/x86_64-linux/include:${CPATH:-}" \
       LIBRARY_PATH="/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}" \
       FLAGCX_ADAPTOR=klx python -m pip install -e . --no-build-isolation \
    && mkdir -p /opt/kunlun/lib \
    && cp /root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch_xmlir/libbkcl.so /opt/kunlun/lib/ \
    && patchelf --set-rpath /opt/kunlun/lib:/workspace/FlagCX/build/lib \
        /workspace/FlagCX/build/lib/libflagcx.so

WORKDIR /workspace

# ---------- dev stage ----------
FROM base AS dev

ARG INDEX_URL
ARG EXTRA_INDEX_URL

RUN python -m pip install --no-cache-dir \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        pytest \
        pytest-cov \
        pytest-json-report \
        pre-commit \
        ruff

WORKDIR /workspace

# ---------- ci stage ----------
FROM base AS ci

ARG INDEX_URL
ARG EXTRA_INDEX_URL

RUN python -m pip install --no-cache-dir \
        ${INDEX_URL:+--index-url ${INDEX_URL}} \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        pytest \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        requests \
        openai \
        pillow \
        decorator \
        pyyaml \
        sqlalchemy

RUN python - <<'PY'
import os
import sys

flagcx_path = os.environ["FLAGCX_PATH"]
if flagcx_path not in sys.path:
    sys.path.insert(0, flagcx_path)

import flag_gems
import flagcx  # noqa: F401
import torch
import vllm

print(f"vLLM: {vllm.__version__}")
print(f"FlagGems: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch: {torch.__version__}")
print("FlagCX import ok")
PY

WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release

WORKDIR /workspace
