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

# Published with the FlagRelease MiniCPM-OCR-1.0 MetaX deployment recipe:
# https://modelscope.cn/models/FlagRelease/MiniCPM-OCR-1.0-BF16-metax-FlagOS
ARG METAX_BASE_IMAGE=harbor.baai.ac.cn/flagrelease-public/minicpm-ocr-1.0-metax-c550-gems5.5.0-pluginfl-0601420-vllm0.24.0-maca3.8.1.3:20261021@sha256:7c810404a21cb60d0a6d1f532baca82d1dea8ee428231f7fb066d609aacbadcf

# ---------- base stage ----------
FROM ${METAX_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    GEMS_VENDOR=metax \
    VLLM_PLUGINS=fl \
    PATH="/flagos/bin:/opt/conda/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# docker/build.sh uses docker/metax as this platform's build context.
COPY runtime_constraints.py /opt/metax-stack/runtime_constraints.py
RUN python /opt/metax-stack/runtime_constraints.py capture /opt/metax-stack

# ---------- shared vLLM 0.32 runtime ----------
FROM base AS runtime

ARG VLLM_VERSION=0.32.0
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/refs/tags/v${VLLM_VERSION}
ARG VLLM_SOURCE_SHA256=bf592c132f27bbb5f65cff0bbd2ee873eae722b79c8286934fd5b5cf88b43d61
ARG UV_VERSION=0.13.4
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL

# Upstream Python dependencies can require a newer stock torch. Overrides
# retain the installed native MACA packages instead of selecting CUDA wheels
# or failing resolution as ordinary constraints would.
ENV UV_OVERRIDE=/opt/metax-stack/native-overrides.txt \
    FL_METAX_STACK_MANIFEST=/opt/metax-stack/native-stack.json

RUN python -m pip install --no-cache-dir --no-deps \
        --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        "uv==${UV_VERSION}" \
    && uv --config-file /opt/metax-stack/uv.toml pip install --python "$(command -v python)" --no-cache --no-build-isolation \
        --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        'setuptools>=78.1.1,<81.0.0' wheel 'setuptools-scm>=8' \
        'setuptools-rust>=1.9' scikit-build-core==0.11 pybind11 ninja cmake

# setup.py imports tools/build_rust.py even in empty mode, so retain the full
# upstream archive. The Rust frontend is optional for the empty target.
# Build an ordinary wheel: the runtime must not depend on an editable checkout.
RUN curl --fail --location --retry 5 "${VLLM_SOURCE_URL}" -o /tmp/vllm-source.tar.gz \
    && echo "${VLLM_SOURCE_SHA256}  /tmp/vllm-source.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/vllm-fl-empty-src \
    && tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /opt/vllm-fl-empty-src \
    && test -f /opt/vllm-fl-empty-src/tools/build_rust.py \
    && uv --config-file /opt/metax-stack/uv.toml pip install --python "$(command -v python)" --no-cache --no-build-isolation \
        --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        -r /opt/vllm-fl-empty-src/requirements/common.txt \
        transformers==5.19.0 tokenizers==0.23.3 depyf==0.20.0 \
    && python -m pip uninstall -y vllm-plugin-fl \
    && VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" python -m pip wheel -v \
        --no-build-isolation --no-deps --wheel-dir /opt/vllm-empty-wheels \
        /opt/vllm-fl-empty-src \
    && python -m pip install --no-build-isolation --no-deps \
        /opt/vllm-empty-wheels/vllm-*.whl \
    && python /opt/metax-stack/runtime_constraints.py verify \
        /opt/metax-stack --vllm-version "${VLLM_VERSION}"

# Bake the plugin's Python dependencies and the router ABI into every target.
ENV VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
COPY install_flaggems_vllm.py /opt/metax-stack/install_flaggems_vllm.py
RUN uv --config-file /opt/metax-stack/uv.toml pip install --python "$(command -v python)" --no-cache \
        --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        decorator pyyaml sqlalchemy modelscope==1.35.3 \
    && python /opt/metax-stack/install_flaggems_vllm.py \
    && python /opt/metax-stack/runtime_constraints.py verify \
        /opt/metax-stack --vllm-version "${VLLM_VERSION}"

WORKDIR /workspace

# ---------- dev stage ----------
FROM runtime AS dev

ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL

RUN uv --config-file /opt/metax-stack/uv.toml pip install --python "$(command -v python)" --no-cache \
        --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        pytest pytest-cov pytest-timeout pytest-json-report pre-commit ruff \
    && python /opt/metax-stack/runtime_constraints.py verify /opt/metax-stack

# ---------- ci stage ----------
FROM dev AS ci

# ---------- release stage ----------
FROM runtime AS release

# Replace the source and package labels inherited from the vendor base.
LABEL org.opencontainers.image.source="https://github.com/flagos-ai/vllm-plugin-FL" \
    org.opencontainers.image.version="${VLLM_VERSION}-metax-release" \
    org.opencontainers.image.revision="" \
    org.flagos.components.flaggems_vllm.commit="e00557103e284db5b57168386db60faa051989ef" \
    org.flagos.components.vllm_plugin_fl="not-installed" \
    flagos.plugin="not-installed"

# Install the desired FL plugin checkout with --no-build-isolation --no-deps.
WORKDIR /workspace
