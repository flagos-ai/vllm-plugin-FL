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
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# ---------- ci stage ----------
FROM base AS ci

SHELL ["/bin/bash", "-c"]

ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.13.0

# Install FlagGems
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base && \
    pip install \
        scikit-build-core==0.11 \
        sqlalchemy==2.0.48 \
        pybind11 \
        ninja \
        cmake \
        -i ${INDEX_URL} \
    && git clone -b v4.2.0 --depth 1 https://github.com/flagos-ai/FlagGems /workspace/FlagGems \
    && pip install --no-build-isolation /workspace/FlagGems

# Set environment variables for vLLM and Triton
ENV VLLM_PLUGINS=fl
ENV GEMS_VENDOR=metax

WORKDIR /workspace

# ---------- release stage ----------
