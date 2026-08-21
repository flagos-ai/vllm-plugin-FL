ARG MUSA_BASE_IMAGE=registry.mthreads.com/mcconline/inference/vllm:v0.20.2-ph1-4.3.5-torch2.7.1-v1.1.0

# ---------- base stage ----------
FROM ${MUSA_BASE_IMAGE} AS base

ARG PYTHON_VERSION=3.10
ARG VLLM_VERSION=0.20.2
ARG FLAGGEMS_VERSION=5.0.0

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl

# Keep the vendor runtime from the base image and add only the tools required
# to build the checked-out plugin in CI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        "setuptools>=77.0.3,<81.0.0" \
        wheel \
        setuptools-scm \
        scikit-build-core==0.11 \
        pybind11 \
        ninja \
        cmake

# Fail early when a different vendor base image is supplied accidentally.
RUN python -c 'from importlib.metadata import version; \
vllm = version("vllm"); \
flag_gems = version("flag-gems"); \
assert "empty" in vllm, vllm; \
assert flag_gems == "5.0.0", flag_gems; \
print("vLLM:", vllm); \
print("FlagGems:", flag_gems)'

WORKDIR /workspace

# ---------- dev stage ----------
FROM base AS dev

RUN python -m pip install --no-cache-dir \
        pytest==8.4.1 \
        pytest-cov \
        pytest-json-report \
        ruff \
        pre-commit

WORKDIR /workspace

# ---------- ci stage ----------
FROM dev AS ci

RUN python -m pip install --no-cache-dir \
        pytest-timeout \
        pytest-metadata

WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release

WORKDIR /workspace
