# Copyright 2026 FlagOS Contributors
# NVIDIA vLLM 0.32.0 inference/CI image based on FlagOS Runtime 2.3.0.
# Build context: repository root. Pin the previously validated Runtime artifact.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0@sha256:edf29b9a28badaf72fc75fd0e057325df9610694914cca3f571e587038bf17e8
FROM ${BASE_IMAGE} AS ci
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG FLAGOS_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-nvidia/simple
ARG INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG EXTRA_INDEX_URL
ARG UBUNTU_MIRROR=https://mirrors.aliyun.com/ubuntu
ENV UV_PYTHON=/flagos/bin/python \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_HTTP_TIMEOUT=120

# TorchCodec needs FFmpeg shared libraries; the Runtime supplies build tools.
RUN sed -i -e "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" \
        -e "s|http://security.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY docker/cuda/constraints.txt docker/cuda/requirements-ci.txt /opt/fl-ci/
# XGBoost cu12 NCCL shares library paths with cu13; reinstall Runtime cu13 last.
RUN uv pip install --python /flagos/bin/python \
        --index "${FLAGOS_INDEX_URL}" \
        ${EXTRA_INDEX_URL:+--index "${EXTRA_INDEX_URL}"} \
        --default-index "${INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}" \
        --index-strategy unsafe-first-match \
        --constraint /opt/fl-ci/constraints.txt \
        -r /opt/fl-ci/requirements-ci.txt 'vllm[audio]==0.32.0+flagos' \
    && uv pip install --python /flagos/bin/python \
        --reinstall-package nvidia-nccl-cu13 \
        --index "${FLAGOS_INDEX_URL}" \
        --default-index "${INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}" \
        --index-strategy unsafe-first-match 'nvidia-nccl-cu13==2.29.7' \
    && uv cache clean

# Runtime BASH_ENV selects FlagTree only in bash; sh and direct Python need ENV.
ENV PYTHONPATH=/opt/flagtree

COPY .github/scripts/install_flaggems_vllm.py /opt/fl-ci/install_flaggems_vllm.py
COPY .github/scripts/nvidia/stack.py /opt/fl-ci/stack.py
RUN python /opt/fl-ci/install_flaggems_vllm.py \
    && python -B /opt/fl-ci/stack.py --build-only

COPY . /workspace/vllm-plugin-FL
WORKDIR /workspace/vllm-plugin-FL
ARG PLUGIN_SOURCE_SHA
ARG PLUGIN_SOURCE_VERSION=0.5.0.dev0
LABEL org.opencontainers.image.revision="${PLUGIN_SOURCE_SHA}"
RUN { test -n "${PLUGIN_SOURCE_SHA}" || { echo "ERROR: PLUGIN_SOURCE_SHA must be set" >&2; exit 1; }; } \
    && env -u VLLM_VENDOR SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="${PLUGIN_SOURCE_VERSION}" \
        python -m pip install --no-deps --no-build-isolation -e . \
    && python -m pip list --format=json > /opt/fl-ci/packages.json

ENV VLLM_PLUGINS=fl \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    USE_FLAGGEMS=1 \
    GEMS_VENDOR=nvidia \
    USE_FLAGTUNE=0
# The wheel closure supplies nvcc as a Python package, outside /usr/local/cuda.
RUN toolkit="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13")')" \
    && test -x "${toolkit}/bin/nvcc" \
    && ln -s "${toolkit}" /opt/cuda-toolkit \
    && ln -sfn lib "${toolkit}/lib64" \
    && ln -sfn libcudart.so.13 "${toolkit}/lib/libcudart.so"
ENV CUDA_HOME=/opt/cuda-toolkit \
    CUDA_PATH=/opt/cuda-toolkit \
    PATH="/opt/cuda-toolkit/bin:${PATH}"
ENTRYPOINT []
CMD ["/bin/bash"]
