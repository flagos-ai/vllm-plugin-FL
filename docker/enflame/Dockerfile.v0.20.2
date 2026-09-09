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

ARG ENFLAME_BASE_IMAGE=harbor.baai.ac.cn/plugin/enflame001-gems5.4.0-treenone-triton3.6.0-cxnone-plugin0.2.0-vllm0.20.2-cp312-pt211-x64-1.9.29:202607221058

# ---------- base stage ----------
FROM ${ENFLAME_BASE_IMAGE} AS base

ARG ENFLAME_BASE_IMAGE
ARG VLLM_VERSION=0.20.2
ARG DRIVER_VERSION=1.9.29
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=dev
ARG SOURCE_URL=https://github.com/flagos-ai/vllm-plugin-FL

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.description="Enflame S60 image for vllm-plugin-FL" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${ENFLAME_BASE_IMAGE}" \
      io.flagos.accelerator.backend="enflame" \
      io.flagos.image.stage="${IMAGE_STAGE}" \
      io.flagos.vllm.version="${VLLM_VERSION}" \
      io.flagos.driver.version="${DRIVER_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=99999 \
    TORCH_GCU_ENABLE_INT64_AND_UINT64=1 \
    ENABLE_I64_CHECK=0 \
    TORCHGCU_INDUCTOR_ENABLE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
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

RUN python -m pip install --no-cache-dir \
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

WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release

WORKDIR /workspace
