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

# FlagOS Runtime 2.3.0 for Ascend 910C: CANN 9.2.1, Python 3.12 (/flagos),
# Torch 2.13 + torch-npu, FlagGems 5.4.0, FlagTree 0.7.1a1 and FlagCX are
# preserved from the base; vLLM and FlagGems-vLLM are installed on top,
# constrained to keep the native stack intact.
ARG BUILD_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.2.1-910c:2.3.0
ARG FLAGOS_DAILY_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-daily/simple
ARG FLAGOS_ASCEND_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-ascend/simple
ARG UV_DEFAULT_INDEX=https://pypi.org/simple

# ---------- base stage ----------
FROM ${BUILD_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

ARG VLLM_VERSION=0.32.0+flagos
ARG FLAGGEMS_VLLM_VERSION=0.3.0.dev20261125
ARG FLAGOS_DAILY_INDEX_URL
ARG FLAGOS_ASCEND_INDEX_URL
ARG UV_DEFAULT_INDEX

# GitHub Actions runs bare steps with `sh`, which ignores the image's
# BASH_ENV=/etc/bash_env.sh, losing the CANN/ATB environment exported by
# /etc/profile.d/vendor.sh (verified: direct exec fails with
# libascend_hal.so not found). Mirror the vendor.sh path exports here; bash
# shells re-source vendor.sh and land on the same values. Driver paths
# (/usr/local/Ascend/driver) are volume-mounted at run time.
ENV ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.2.1
ENV ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1
ENV ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.2.1/opp
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.1
ENV TOOLCHAIN_HOME=/usr/local/Ascend/cann-9.2.1/toolkit
ENV ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
# Full 13-segment CANN/ATB/driver library path (snapshot of vendor.sh in
# the runtime image).
ENV LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/cann-9.2.1/lib64:/usr/local/Ascend/cann-9.2.1/lib64/plugin/opskernel:/usr/local/Ascend/cann-9.2.1/lib64/plugin/nnengine:/usr/local/Ascend/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/cann-9.2.1/devlib
# FlagTree (Triton) ships in /opt/flagtree; keep the CANN python and op_impl
# paths reachable the way zz-flagos.sh does for bash shells.
ENV PYTHONPATH=/opt/flagtree:/usr/local/Ascend/cann-9.2.1/python/site-packages:/usr/local/Ascend/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe

# Install FlagGems-vLLM and vLLM (constrained to keep the base
# torch/torch-npu/FlagGems/FlagTree stack intact).
RUN mkdir -p /opt/fl-ci && printf '%s\n' \
        'cann-shmem==1.6.0' \
        'flag-gems==5.4.0' \
        'flagcx==0.14.1+cann9.2.1' \
        'flagtree==0.7.1a1+ascend3.5' \
        'numpy==1.26.4' \
        'torch==2.13.0+cpu' \
        'torch-npu==2.13.0' \
        'torchaudio==2.11.0+cpu' \
        'torchvision==0.28.0+cpu' \
        > /opt/fl-ci/constraints.txt && \
    uv pip install --python /flagos/bin/python --no-cache \
        --constraint /opt/fl-ci/constraints.txt \
        --extra-index-url "${FLAGOS_DAILY_INDEX_URL}" \
        --extra-index-url "${FLAGOS_ASCEND_INDEX_URL}" \
        --index-strategy unsafe-first-match \
        "flaggems-vllm==${FLAGGEMS_VLLM_VERSION}" \
        "vllm==${VLLM_VERSION}"

WORKDIR /workspace


# ---------- dev stage ----------
FROM base AS dev

ARG UV_DEFAULT_INDEX

# Install dev/test tools
RUN uv pip install --python /flagos/bin/python --no-cache \
        --constraint /opt/fl-ci/constraints.txt \
        pytest \
        "pytest-xdist>=3.0.0" \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        numpy \
        requests \
        decorator \
        modelscope \
        ruff \
        pre-commit \
        ninja \
        cmake


# ---------- test stage ----------
FROM base AS test

# Minimal test environment - inherits vLLM and FlagGems from base
LABEL maintainer="FlagOS Contributors" \
      description="FlagOS vLLM-Plugin-FL environment with FlagGems"
