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

# FlagOS Runtime 2.3.0 for Kunlunxin XRE 5.37.1.
ARG KUNLUNXIN_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-kunlunxin-xre5.37.1:2.3.0@sha256:cc0deb9c09cd49977e754892e85f4414835d142379949cc2d66261f63b47959d
FROM ${KUNLUNXIN_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG KUNLUNXIN_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-kunlunxin/simple
ARG KUNLUNXIN_VLLM_REQUIREMENT=vllm==0.32.0+flagos
ARG FLAGCX_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-kunlunxin/packages/flagcx/0.14.1+xre5.37.1/flagcx-0.14.1+xre5.37.1-cp310-cp310-linux_x86_64.whl
ARG FLAGCX_WHEEL_SHA256=b49203511f7906aaa3cebdbf304aff521a86f0540f6b57c6af6154092f8a658e
ARG FLAGGEMS_VLLM_COMMIT=8ae5c22f5af6d650216f881c329c4dff20c182e1
ARG FLAGGEMS_VLLM_SHA256=eab3544a5a6a2efaf7b3465a04ea0f867f77cfbe144ed7f2b8ad20c7b28bb36c
ARG VCS_REF=unknown
ARG PLUGIN_SOURCE_SHA=77f141a961d75e54c0cf209de80c99a7c7eec04b

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.source="https://github.com/flagos-ai/vllm-plugin-FL" \
      org.opencontainers.image.revision="${VCS_REF}" \
      io.flagos.accelerator.backend="kunlunxin" \
      io.flagos.vllm.version="0.32.0" \
      io.flagos.xre.version="5.37.1" \
      io.flagos.host.driver.version="5.0.21.47" \
      io.flagos.plugin.source.sha="${PLUGIN_SOURCE_SHA}"

# Use vendor ops during multimodal profiling on P800.
ENV GEMS_VENDOR=kunlunxin \
    VLLM_PLUGINS=fl \
    USE_FLAGGEMS=1 \
    VLLM_FL_PREFER=flagos

# Install the vendor-provided, Python 3.10-compatible vLLM wheel.
RUN PYTHONPATH=/opt/triton \
    PIP_EXTRA_INDEX_URL="${KUNLUNXIN_INDEX_URL}${EXTRA_INDEX_URL:+ $EXTRA_INDEX_URL}" \
    python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        "${KUNLUNXIN_VLLM_REQUIREMENT}" \
        "flagcx @ ${FLAGCX_WHEEL_URL}#sha256=${FLAGCX_WHEEL_SHA256}" \
        "torch==2.9.0+cu129" "torchvision==0.24.0+cu129" "torchaudio==2.9.0+cu129" \
        "numpy==1.26.4" "regex==2026.4.4"

# Runner V2 on P800 uses this provider's sampling and grammar operators.
RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" wheel \
        packaging pybind11 pyyaml sqlalchemy "cmake>=3.26" \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation \
        "flaggems-vllm @ https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/${FLAGGEMS_VLLM_COMMIT}#sha256=${FLAGGEMS_VLLM_SHA256}"

# Build the plugin, including vllm_fl._C, from the pinned checkout inside the
# runtime. The runtime ships CMake 3.22, while csrc/ requires 3.26; nvcc is only
# needed because Torch's CMake package config enables the CUDA language.
COPY . /opt/vllm-plugin-FL
RUN SETUPTOOLS_SCM_PRETEND_VERSION="0.5.0.dev0+g${PLUGIN_SOURCE_SHA}.xre5.37.1" \
       VLLM_VENDOR=cuda \
       PATH="/usr/local/cuda/bin:${PATH}" \
       CUDACXX=/usr/local/cuda/bin/nvcc \
       CMAKE_ARGS="-DTorch_DIR=$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)' 2>/dev/null | tail -1)/Torch" \
       python -m pip wheel --no-build-isolation --no-deps --no-cache-dir \
           -w /opt/kunlunxin-wheels /opt/vllm-plugin-FL

RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        "modelscope==1.35.3" \
    && python -m pip install --no-deps --force-reinstall /opt/kunlunxin-wheels/vllm_plugin_fl-*.whl \
    && python -m pip check

FROM base AS ci
RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        pytest pytest-cov pytest-timeout pytest-json-report pytest-xdist \
        requests openai pillow decorator pyyaml sqlalchemy pybind11 \
    && python -m pip check

FROM ci AS dev
RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        pre-commit ruff

FROM base AS release
