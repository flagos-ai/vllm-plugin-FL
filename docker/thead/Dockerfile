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

# FlagOS Runtime 2.3.0 for T-Head PPU 2.1.0.
ARG THEAD_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0@sha256:35fcbd0c3777f0af1f936fc725a91be76279ea77849d192ea11d6f90c12172bc
FROM ${THEAD_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG INDEX_URL=https://pypi.org/simple
ARG THEAD_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-thead/simple
ARG VLLM_REQUIREMENT=vllm==0.32.0+flagos
ARG FLAGGEMS_REQUIREMENT=flag-gems==5.5.0.dev20261117
ARG FLAGGEMS_VLLM_REQUIREMENT=flaggems-vllm==0.3.0.dev20261117
ARG FLASH_ATTN_REQUIREMENT=flash-attn-3==2.8.2+v0.1.0.ppu2.1.1
ARG DAILY_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-daily/simple

ENV FLAGCX_PATH=/flagos/lib/python3.12/site-packages/flagcx \
    VLLM_PLUGINS=fl \
    VLLM_TARGET_DEVICE=empty \
    VLLM_VENDOR="" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    TORCH_DEVICE_BACKEND_AUTOLOAD=1 \
    GEMS_VENDOR=thead \
    USE_FLAGGEMS=1

# Keep the PPU Torch ABI in the dependency resolution.
RUN PYTHONPATH=/opt/triton \
    python -m pip install --no-cache-dir \
        --index-url "${INDEX_URL}" --extra-index-url "${THEAD_INDEX_URL}" \
        "${VLLM_REQUIREMENT}" \
        "torch==2.10.0+v0.1.0.ppu2.1.1" "torchvision==0.25.0+v0.1.0.ppu2.1.1" \
        "torchaudio==2.10.0+v0.1.0.ppu2.1.1" "numpy==1.26.4" \
        "modelscope==1.35.3" "setuptools==80.10.2" "setuptools-scm>=8" \
        wheel packaging pybind11 sqlalchemy \
    && python -m pip install --no-cache-dir --no-deps \
        --index-url "${THEAD_INDEX_URL}" --extra-index-url "${DAILY_INDEX_URL}" \
        "${FLAGGEMS_REQUIREMENT}" "${FLAGGEMS_VLLM_REQUIREMENT}" \
        "flagcx==0.14.1+ppu2.1.0" "${FLASH_ATTN_REQUIREMENT}"

WORKDIR /workspace

FROM base AS ci
ARG PLUGIN_SOURCE_SHA
ARG SOURCE_DATE_EPOCH
LABEL org.flagos.vllm-plugin-fl.revision="${PLUGIN_SOURCE_SHA}"

RUN --mount=type=bind,from=plugin_source,target=/src,rw \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="0.5.0.dev0+g${PLUGIN_SOURCE_SHA}" \
        python -m pip install --no-deps --no-build-isolation --no-cache-dir /src \
    && python -m pip install --no-cache-dir --index-url "${INDEX_URL}" \
        pytest pytest-cov pytest-timeout pytest-json-report "pytest-xdist>=3.0.0" \
        requests openai pillow decorator \
    && python -m pip check
