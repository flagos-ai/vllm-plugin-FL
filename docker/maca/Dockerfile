ARG VLLM_VERSION=0.13.0
ARG MACA_VERSION=3.3.0.303
ARG TORCH_VERSION=2.8
ARG PYTHON_VERSION=312
ARG UBUNTU_VERSION=22.04

# ---------- base stage ----------
FROM cr.metax-tech.com/public-ai-release/maca/vllm-metax:${VLLM_VERSION}-maca.ai${MACA_VERSION}-torch${TORCH_VERSION}-py${PYTHON_VERSION}-ubuntu${UBUNTU_VERSION}-amd64 AS base

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.7.12
ARG VLLM_VERSION=0.13.0

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- ci stage ----------
FROM base AS ci

ENV CONDA_PREFIX=/opt/conda \
    CONDA_EXE=/opt/conda/bin/conda \
    CONDA_PYTHON_EXE=/opt/conda/bin/python \
    CONDA_DEFAULT_ENV=base \
    PATH="/opt/conda/bin:/opt/conda/condabin:${PATH}"

ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.13.0

# Install FlagGems
RUN pip install \
        scikit-build-core==0.11 \
        sqlalchemy==2.0.48 \
        pybind11 \
        ninja \
        cmake \
        -i ${INDEX_URL} \
    && git clone -b v4.2.0 --depth 1 https://github.com/flagos-ai/FlagGems /workspace/FlagGems \
    && pip install --no-build-isolation /workspace/FlagGems

# Install Testing dependencies to save time for CI
RUN pip install \
        pytest \
        pytest-cov \
        pytest-xdist \
        pytest-timeout \
        pytest-json-report \
        numpy \
        requests \
        openai \
        decorator \
        soundfile \
        librosa \
        modelscope>=1.18.1 \
        -i ${INDEX_URL}

# Pre-download vLLM test audio assets
RUN mkdir -p /root/.cache/vllm/assets/vllm_public_assets \
    && curl -fSL -o /root/.cache/vllm/assets/vllm_public_assets/mary_had_lamb.ogg \
        https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg \
    && curl -fSL -o /root/.cache/vllm/assets/vllm_public_assets/winning_call.ogg \
        https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/winning_call.ogg

# Set environment variables for vLLM
ENV VLLM_PLUGINS=fl \
    GEMS_VENDOR=metax

WORKDIR /workspace

# ---------- release stage ----------
