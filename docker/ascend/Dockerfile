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

# Keep the chip image aligned with the environment validated on Ascend 910C.
ARG ASCEND_BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.20.2rc1-a3

# ---------- base stage ----------
FROM ${ASCEND_BASE_IMAGE} AS base

ARG FLAGGEMS_VERSION=3e6528cf04f5f964a7b0fa6628de6f0410dbfd02

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl \
    VLLM_FL_PLATFORM=ascend \
    HF_ENDPOINT=https://hf-mirror.com \
    FL_MODEL_BASE_PATH=/data/models

RUN python -m pip install --upgrade pip "setuptools>=77.0.3" \
    && python -m pip install --no-cache-dir \
        scikit-build-core==0.11 \
        pybind11 \
        ninja \
        cmake \
    && git clone https://github.com/flagos-ai/FlagGems.git /workspace/FlagGems \
    && git -C /workspace/FlagGems checkout "${FLAGGEMS_VERSION}" \
    && python -m pip install --no-build-isolation /workspace/FlagGems \
    && python -m pip cache purge

WORKDIR /workspace

# ---------- dev stage ----------
FROM base AS dev

RUN python -m pip install --no-cache-dir \
        pytest \
        pytest-cov \
        pytest-json-report \
        ruff \
        pre-commit

# ---------- ci stage ----------
FROM dev AS ci

RUN python -m pip install --no-cache-dir \
        pytest-timeout \
        pytest-metadata \
        numpy==1.26.4 \
        requests \
        openai \
        decorator \
        pyyaml \
        "modelscope>=1.18.1" \
        "huggingface_hub>=0.34"

# ---------- release stage ----------
FROM base AS release

WORKDIR /workspace
