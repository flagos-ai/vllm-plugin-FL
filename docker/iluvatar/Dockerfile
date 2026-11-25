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

# FlagOS Runtime 2.3.0 for Iluvatar CoreX 5.0.1.
ARG ILUVATAR_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-iluvatar-corex5.0.1:2.3.0@sha256:fb0280be3f2b1c7feba3b1b933dce9e6ae85690c0b7edb615d7577c1c632777c

# ---------- base stage ----------
FROM ${ILUVATAR_BASE_IMAGE} AS base
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG ILUVATAR_BASE_IMAGE
ARG VLLM_VERSION=0.32.0
ARG VLLM_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-iluvatar/packages/vllm/0.32.0+flagos/vllm-0.32.0+flagos-py3-none-any.whl
ARG VLLM_WHEEL_SHA256=8863d2a86ae2928f3a5575e2eef829ef91654cc16469aad26cb63a7f009f6b2a
ARG FLAGGEMS_VERSION=5.5.0.dev20261118
ARG FLAGGEMS_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-daily/packages/flag-gems/5.5.0.dev20261118/flag_gems-5.5.0.dev20261118-py3-none-any.whl
ARG FLAGGEMS_WHEEL_SHA256=cae4b8409763c88e3e6e7ead44cc900f20de391a1000fb7ad6efe8c215b86316
ARG FLAGGEMS_VLLM_VERSION=0.3.0.dev20261119
ARG FLAGGEMS_VLLM_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-daily/packages/flaggems-vllm/0.3.0.dev20261119/flaggems_vllm-0.3.0.dev20261119-py3-none-any.whl
ARG FLAGGEMS_VLLM_WHEEL_SHA256=b9947e9da1c8ca02eec68f8a0a0fa72595507a79b24f3c2f77cd00af344d42d6
ARG PLUGIN_SOURCE_SHA=unknown
ARG PLUGIN_VERSION=0.5.0.dev0
ARG INDEX_URL=https://pypi.org/simple
ARG VENDOR_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-iluvatar/simple
ARG EXTRA_INDEX_URL=https://pypi.org/simple
ARG DAILY_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-daily/simple
ARG BUILD_DATE=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=dev
ARG SOURCE_URL=https://github.com/flagos-ai/vllm-plugin-FL

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.description="Iluvatar BI-V150 CoreX 5.0.1 CI image for vllm-plugin-FL (vLLM 0.32)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${PLUGIN_SOURCE_SHA}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${ILUVATAR_BASE_IMAGE}" \
      io.flagos.accelerator.backend="iluvatar" \
      io.flagos.image.stage="${IMAGE_STAGE}" \
      io.flagos.runtime.version="2.3.0" \
      io.flagos.corex.version="5.0.1" \
      io.flagos.vllm.version="${VLLM_VERSION}" \
      io.flagos.flaggems.version="${FLAGGEMS_VERSION}" \
      io.flagos.plugin.source.commit="${PLUGIN_SOURCE_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VLLM_PLUGINS=fl \
    GEMS_VENDOR=iluvatar \
    FLAGGEMS_VERSION=${FLAGGEMS_VERSION}

# Resolve the vendor vLLM wheel against the Runtime Torch ABI.
# Expose Triton to pip only for this installation.
RUN PYTHONPATH=/opt/triton python -m pip install --no-cache-dir \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${VENDOR_INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        --extra-index-url "${DAILY_INDEX_URL}" \
        "torch==2.13.0+corex.5.0.1" "torchvision==0.28.0+corex.5.0.1" \
        "torchaudio==2.11.0+corex.5.0.1" "flagcx==0.14.1+corex5.0.1" \
        "numpy==1.26.4" "opencv-python-headless==5.0.0.93+flagos" \
        "compressed-tensors==0.17.0+flagos" "xgrammar==0.2.8+flagos" \
        "vllm @ ${VLLM_WHEEL_URL}#sha256=${VLLM_WHEEL_SHA256}"

# Install the pinned provider wheels without re-resolving the Runtime stack.
RUN python -m pip install --no-cache-dir --no-deps \
        "flag-gems @ ${FLAGGEMS_WHEEL_URL}#sha256=${FLAGGEMS_WHEEL_SHA256}" \
        "flaggems-vllm @ ${FLAGGEMS_VLLM_WHEEL_URL}#sha256=${FLAGGEMS_VLLM_WHEEL_SHA256}" \
    && python -m pip install --no-cache-dir --no-deps \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        setuptools-scm==9.2.2 wheel==0.45.1

# Build and install the plugin wheel from the archived source revision.
RUN --mount=type=bind,target=/src,rw \
    cd /src \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="${PLUGIN_VERSION}" \
        python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/plugin-wheels . \
    && python -m pip install --no-deps --no-index /tmp/plugin-wheels/vllm_plugin_fl-*.whl \
    && rm -rf /tmp/plugin-wheels

# Install the plugin's model-loading dependency.
RUN python -m pip install --no-cache-dir \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        modelscope==1.35.3 \
    && python -m pip check

WORKDIR /workspace

# ---------- ci stage ----------
FROM base AS ci

ARG IMAGE_STAGE=ci
LABEL io.flagos.image.stage="${IMAGE_STAGE}"

# Install the test tools used by UT, FT, E2E and benchmark.
RUN python -m pip install --no-cache-dir \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        pytest==8.4.1 pytest-cov pytest-json-report pytest-timeout \
        "pytest-xdist>=3.0.0" pytest-metadata \
        requests openai pillow decorator pyyaml sqlalchemy \
    && python -m pip check

# ---------- dev stage ----------
FROM ci AS dev

ARG IMAGE_STAGE=dev
LABEL io.flagos.image.stage="${IMAGE_STAGE}"

# Install developer tools on top of the CI environment.
RUN python -m pip install --no-cache-dir \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        ruff pre-commit

# ---------- release stage ----------
FROM base AS release

ARG IMAGE_STAGE=release
LABEL io.flagos.image.stage="${IMAGE_STAGE}"
