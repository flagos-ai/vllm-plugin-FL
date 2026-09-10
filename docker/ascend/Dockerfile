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

# Reuse its CANN 9.0 / torch_npu 2.10 runtime, then replace vLLM with empty.
ARG ASCEND_BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.20.2rc1-a3

# ---------- base stage ----------
FROM ${ASCEND_BASE_IMAGE} AS base

ARG VLLM_VERSION=0.28.0
ARG FLAGTREE_VERSION=0.6.2a1+ascend3.5
# Official FlagTree Ascend manual's qwen-vllm_for_ascend branch.
ARG FLAGGEMS_VERSION=3b406c36212744b98b9720bf6d0a5387c09fe96b

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl \
    VLLM_FL_PLATFORM=ascend \
    HF_ENDPOINT=https://hf-mirror.com \
    FL_MODEL_BASE_PATH=/data/models

COPY runtime-constraints.txt /workspace/ascend-runtime-constraints.txt

RUN python -m pip uninstall -y vllm vllm-ascend triton triton-ascend flagtree flag-gems \
    && python -m pip install --upgrade pip "setuptools>=77.0.3,<80" \
    && python -m pip install --no-cache-dir \
        scikit-build-core==0.12.2 \
        "setuptools-scm>=8" \
        "setuptools-rust>=1.9" \
        wheel \
        jinja2 \
        pybind11 \
        ninja \
        cmake \
    && git clone --depth 1 --branch "v${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm.git /workspace/vllm \
    && VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE="${VLLM_VERSION}+empty" \
        python -m pip install --no-build-isolation --no-deps /workspace/vllm \
    && python -m pip install --no-cache-dir \
        -c /workspace/ascend-runtime-constraints.txt \
        -r /workspace/vllm/requirements/common.txt \
    && python -m pip install --no-cache-dir --no-deps \
        "flagtree===${FLAGTREE_VERSION}" \
        --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
    && python -m pip install --no-cache-dir --no-deps cann-shmem==1.6.0 \
        --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/ \
    && git clone https://github.com/flagos-ai/FlagGems.git /workspace/FlagGems \
    && git -C /workspace/FlagGems checkout "${FLAGGEMS_VERSION}" \
    && python -m pip install --no-cache-dir \
        packaging==26.0 PyYAML==6.0.3 sqlalchemy==2.0.48 numpy==1.26.4 \
    && python -m pip install --no-build-isolation --no-deps /workspace/FlagGems \
    && python -c "from importlib.metadata import version; assert version('vllm') == '${VLLM_VERSION}+empty'; assert version('flagtree') == '${FLAGTREE_VERSION}'" \
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
