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

# Unify the Iluvatar BI-V150 CI image on the published FlagOS Runtime 2.3.0
# for Iluvatar CoreX 5.0.1. The runtime already contains Python 3.12, Torch
# 2.13.0 / TorchVision 0.28.0 / TorchAudio 2.11.0 (+corex.5.0.1), FlagTree
# 0.7.1a1+iluvatar3.6 (with its Triton 3.6.0), FlagCX 0.14.1+corex5.0.1 and
# NumPy 1.26.4. Nothing below replaces those packages; the constraints file
# enforces that and the final assertion fails the build on any drift.
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

COPY docker/iluvatar/constraints.txt /opt/iluvatar/constraints.txt

# Download the validated wheels under their canonical names and check SHA256.
RUN mkdir -p /opt/iluvatar/wheels \
    && VLLM_WHL="/opt/iluvatar/wheels/${VLLM_WHEEL_URL##*/}" \
    && GEMS_WHL="/opt/iluvatar/wheels/${FLAGGEMS_WHEEL_URL##*/}" \
    && GEMS_VLLM_WHL="/opt/iluvatar/wheels/${FLAGGEMS_VLLM_WHEEL_URL##*/}" \
    && curl -fsSL --retry 3 -o "${VLLM_WHL}" "${VLLM_WHEEL_URL}" \
    && curl -fsSL --retry 3 -o "${GEMS_WHL}" "${FLAGGEMS_WHEEL_URL}" \
    && curl -fsSL --retry 3 -o "${GEMS_VLLM_WHL}" "${FLAGGEMS_VLLM_WHEEL_URL}" \
    && printf '%s  %s\n' \
        "${VLLM_WHEEL_SHA256}" "${VLLM_WHL}" \
        "${FLAGGEMS_WHEEL_SHA256}" "${GEMS_WHL}" \
        "${FLAGGEMS_VLLM_WHEEL_SHA256}" "${GEMS_VLLM_WHL}" \
        | sha256sum -c -

# Install vLLM with its Python dependencies under constraints. The runtime's
# Torch/TorchVision/TorchAudio, NumPy, FlagTree/Triton and FlagCX stay exactly
# as shipped; FlagGems and FlagGems-vllm go in from the pinned wheels without
# dependency resolution so no vendor extra (e.g. flag_gems[iluvatar-corex450])
# can leak in. The .pth keeps FlagTree's Triton importable even though CI setup
# unsets PYTHONPATH.
RUN pip install --no-cache-dir \
        --constraint /opt/iluvatar/constraints.txt \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${VENDOR_INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        --extra-index-url "${DAILY_INDEX_URL}" \
        /opt/iluvatar/wheels/vllm-*.whl \
    && pip install --no-cache-dir --no-deps \
        /opt/iluvatar/wheels/flag_gems-*.whl \
        /opt/iluvatar/wheels/flaggems_vllm-*.whl \
    && pip install --no-cache-dir --no-deps \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        setuptools-scm==9.2.2 wheel==0.45.1 \
    && echo /opt/flagtree > /flagos/lib/python3.12/site-packages/flagtree.pth \
    && rm -rf /opt/iluvatar/wheels

# The plugin is built from the locked source revision supplied by build.sh
# (`git archive` of PLUGIN_SOURCE_SHA), never from the working tree.
COPY pyproject.toml setup.py README.md LICENSE /opt/iluvatar/plugin-src/
COPY vllm_fl /opt/iluvatar/plugin-src/vllm_fl
RUN cd /opt/iluvatar/plugin-src \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="${PLUGIN_VERSION}" \
        pip wheel --no-deps --no-build-isolation --wheel-dir /opt/iluvatar/plugin-build . \
    && pip install --no-deps --no-index /opt/iluvatar/plugin-build/vllm_plugin_fl-*.whl \
    && pip install --no-cache-dir \
        --constraint /opt/iluvatar/constraints.txt \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        modelscope==1.35.3 \
    && rm -rf /opt/iluvatar/plugin-src /opt/iluvatar/plugin-build

# Fail the build on any stack drift: exact vendor versions, the validated
# empty vLLM wheel, FlagTree-provided Triton, and a clean `pip check`.
RUN cd /tmp \
    && python3 -c "import importlib.metadata as m, importlib.util as u, torch, vllm, flag_gems, flagcx, triton; \
assert m.version('torch') == '2.13.0+corex.5.0.1', m.version('torch'); \
assert m.version('vllm') == '0.32.0+flagos', m.version('vllm'); \
assert u.find_spec('vllm._C') is None, 'vllm wheel is not an empty build'; \
assert m.version('torchvision') == '0.28.0+corex.5.0.1'; \
assert m.version('torchaudio') == '2.11.0+corex.5.0.1'; \
assert m.version('numpy') == '1.26.4'; \
assert m.version('flagcx') == '0.14.1+corex5.0.1', m.version('flagcx'); \
assert flag_gems.__version__.startswith('5.5.0'), flag_gems.__version__; \
assert triton.__version__.startswith('3.6.0') and triton.__file__.startswith('/opt/flagtree/'), triton.__file__; \
assert m.version('vllm-plugin-fl').startswith('0.5.0.dev0'); \
print('stack contract ok:', m.version('torch'), m.version('vllm'), m.version('flaggems-vllm'), triton.__version__, flagcx.__file__)" \
    && pip check

WORKDIR /workspace

# ---------- dev stage ----------
FROM base AS dev

ARG IMAGE_STAGE=dev
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL=https://pypi.org/simple

LABEL io.flagos.image.stage="${IMAGE_STAGE}"

RUN pip install --no-cache-dir \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        pytest==8.4.1 \
        pytest-cov \
        pytest-json-report \
        pytest-timeout \
        ruff \
        pre-commit

WORKDIR /workspace

# ---------- ci stage ----------
FROM dev AS ci

ARG IMAGE_STAGE=ci
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL=https://pypi.org/simple

LABEL io.flagos.image.stage="${IMAGE_STAGE}"

RUN pip install --no-cache-dir \
        --index-url "${INDEX_URL}" \
        --extra-index-url "${EXTRA_INDEX_URL}" \
        "pytest-xdist>=3.0.0" \
        pytest-metadata \
        requests \
        openai \
        pillow \
        decorator \
        pyyaml \
        sqlalchemy

WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release

ARG IMAGE_STAGE=release

LABEL io.flagos.image.stage="${IMAGE_STAGE}"

WORKDIR /workspace
