# Copyright 2026 FlagOS Contributors
# Ascend vLLM 0.32.0 inference/CI image based on FlagOS Runtime 2.3.0 (CANN 9.2.1).
# Build context: repository root. Pin the verified FlagOS Runtime 2.3.0 manifest digest.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.2.1:2.3.0@sha256:92600f84feee2a2a738bbc77abdb02c8c7af925a6da379b23dde6ded6cba50db
FROM ${BASE_IMAGE} AS ci
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG FLAGOS_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-ascend/simple
ARG INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
ARG EXTRA_INDEX_URL
ENV UV_PYTHON=/flagos/bin/python \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_HTTP_TIMEOUT=120

COPY docker/ascend/constraints.txt docker/ascend/requirements-ci.txt /opt/fl-ci/
RUN uv pip install --python /flagos/bin/python \
        --index "${FLAGOS_INDEX_URL}" \
        ${EXTRA_INDEX_URL:+--index "${EXTRA_INDEX_URL}"} \
        --default-index "${INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}" \
        --index-strategy unsafe-first-match \
        --constraint /opt/fl-ci/constraints.txt \
        -r /opt/fl-ci/requirements-ci.txt 'vllm==0.32.0+flagos' \
    && uv cache clean

COPY .github/scripts/install_flaggems_vllm.py /opt/fl-ci/install_flaggems_vllm.py
# Explicit bash invocation ensures BASH_ENV=/etc/bash_env.sh is loaded even
# when built by Podman/Buildah in default OCI mode (which ignores SHELL).
RUN bash -euo pipefail -c "python /opt/fl-ci/install_flaggems_vllm.py"

COPY .github/scripts/ascend/stack.py /opt/fl-ci/stack.py
RUN bash -euo pipefail -c "python -B /opt/fl-ci/stack.py --build-only"

COPY . /workspace/vllm-plugin-FL
WORKDIR /workspace/vllm-plugin-FL
ARG PLUGIN_SOURCE_SHA
ARG PLUGIN_SOURCE_VERSION=0.5.0.dev0
LABEL org.opencontainers.image.revision="${PLUGIN_SOURCE_SHA}"
RUN bash -euo pipefail -c 'test -n "${PLUGIN_SOURCE_SHA}" || { echo "ERROR: PLUGIN_SOURCE_SHA must be set" >&2; exit 1; }; \
    env -u VLLM_VENDOR SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="${PLUGIN_SOURCE_VERSION}" \
        python -m pip install --no-deps --no-build-isolation -e . \
    && python -m pip list --format=json > /opt/fl-ci/packages.json'

ENV VLLM_PLUGINS=fl \
    VLLM_FL_PLATFORM=ascend \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    USE_FLAGGEMS=1 \
    GEMS_VENDOR=ascend \
    USE_FLAGTUNE=0 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="${PLUGIN_SOURCE_VERSION}" \
    HF_ENDPOINT=https://hf-mirror.com \
    FL_MODEL_BASE_PATH=/data/models
ENTRYPOINT []
CMD ["/bin/bash"]
