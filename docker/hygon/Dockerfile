ARG VLLM_VERSION=0.13.0

# ---------- base stage ----------
FROM harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm${VLLM_VERSION}-ubuntu22.04-dtk26.04-rc1-py3.10-20260309 AS base

ARG PYTHON_VERSION=3.10
ARG UV_VERSION=0.11.16
ENV DEBIAN_FRONTEND=noninteractive

# Install uv
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install pip via uv
RUN uv pip install --system pip setuptools wheel

# ---------- dev stage ----------
FROM base AS dev

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.13.0

# Install dev tools
RUN uv pip install --system \
        pytest \
        pytest-cov \
        pytest-json-report \
        ruff \
        pre-commit \
        ninja \
        cmake

# ---------- ci stage ----------
FROM base AS ci

ARG INDEX_URL
ARG EXTRA_INDEX_URL

# Install dev/test tools
RUN uv pip install --system \
        pytest \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        pycountry \
        numpy \
        requests \
        decorator \
        soundfile \
        librosa \
        "modelscope>=1.18.1" \
        ruff \
        pre-commit \
        ninja \
        cmake

ARG FLAGCX_VERSION=v0.9.0
ARG FLAGGEMS_VERSION=v5.0.0

# Install FlagGems
RUN uv pip install --system \
        scikit-build-core==0.11 \
        pybind11 \
        ninja \
        cmake \
    && git clone -b ${FLAGGEMS_VERSION} --depth 1 https://github.com/flagos-ai/FlagGems /workspace/FlagGems \
    && uv pip install --system --no-build-isolation /workspace/FlagGems

# Install FlagCX (DCU)
# Compilation terminated:
# "du_adaptor.h:7:10: fatal error: nccl.h: No such file or directory"
# RUN git clone https://github.com/flagos-ai/FlagCX.git /workspace/FlagCX \
#     && cd /workspace/FlagCX \
#     && git checkout ${FLAGCX_VERSION} \
#     && git submodule update --init --recursive \
#     && make USE_DU=1
#     && cd plugin/torch \
#     && FLAGCX_ADAPTOR=du uv pip install --system --no-build-isolation .

# ENV FLAGCX_PATH=/workspace/FlagCX

# Install DeepGEMM
# RUN git clone https://github.com/deepseek-ai/DeepGEMM.git /workspace/DeepGEMM \
#     && cd /workspace/DeepGEMM \
#     && git checkout 35c4bc8 \
#     && git submodule update --init --recursive \
#     && TORCH_DEVICE_BACKEND_AUTOLOAD=0 uv pip install --system --no-build-isolation .

# Set environment variables for vLLM
ENV VLLM_PLUGINS=fl \
    GEMS_VENDOR=hygon

WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.13.0

WORKDIR /workspace