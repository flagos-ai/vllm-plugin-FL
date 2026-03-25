ARG VLLM_VERSION=0.13.0

# ---------- base stage ----------
FROM quay.io/ascend/vllm-ascend:v${VLLM_VERSION}rc1-a3 AS base

RUN pip install --upgrade pip setuptools

# ---------- dev stage ----------
FROM base AS dev

# Install dev tools
RUN pip install \
        pytest \
        pytest-cov \
        pytest-json-report \
        ruff \
        pre-commit \
        ninja \
        cmake

# ---------- ci stage ----------
FROM base AS ci

# Install dev/test tools
RUN pip install --upgrade pip
RUN pip install \
        pytest \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        numpy \
        requests \
        decorator \
        "modelscope>=1.18.1" \
        ruff \
        pre-commit \
        ninja \
        cmake

# Install FlagGems (NPU backend)
RUN pip install -U scikit-build-core==0.11 pybind11 \
    && git clone https://github.com/flagos-ai/FlagGems /workspace/FlagGems \
    && pip install --no-build-isolation \
        --config-settings=cmake.define.FLAGGEMS_BACKEND=NPU \
        /workspace/FlagGems

# Install FlagTree
RUN pip install flagtree==0.4.0+ascend3.2 \
        --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
        --trusted-host=resource.flagos.net


ENV VLLM_PLUGINS=fl
ENV TRITON_ALL_BLOCKS_PARALLEL=1

WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.13.0

# Install vLLM
# Todo

WORKDIR /workspace
