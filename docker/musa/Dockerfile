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

# vLLM 0.24 image, layered on the FlagOS-packaged MUSA stack
# (MUSA 4.3.5 / torch 2.9.0 / torch_musa / vLLM 0.24.0 / plugin 0.3.0).
# The v0.20.2 image that wrapped Moore Threads' own registry image directly
# is kept as Dockerfile.v0.20.2.
ARG MUSA_BASE_IMAGE=harbor.baai.ac.cn/plugin/musa-ph1.4.3.5-tree0.6.1a2-triton3.6.0-cxnone-plugin0.3.0-vllm0.24.0-cp310-pt290-x64:20260804

# ---------- base stage ----------
FROM ${MUSA_BASE_IMAGE} AS base

ARG MUSA_BASE_IMAGE
ARG VLLM_VERSION=0.24.0
ARG TORCH_VERSION=2.9.0
ARG FLAGGEMS_VERSION=5.0.0
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=dev
ARG SOURCE_URL=https://github.com/flagos-ai/vllm-plugin-FL

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.description="Moore Threads MUSA image for vllm-plugin-FL (vLLM 0.24)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${MUSA_BASE_IMAGE}" \
      io.flagos.accelerator.backend="musa" \
      io.flagos.image.stage="${IMAGE_STAGE}" \
      io.flagos.vllm.version="${VLLM_VERSION}" \
      io.flagos.torch.version="${TORCH_VERSION}" \
      io.flagos.flaggems.version="${FLAGGEMS_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl

# Fail early when a wrong vendor base image is supplied: the stack must
# provide torch + torch_musa + vLLM 0.24 + FlagGems (setup.sh relies on all
# four at job time). Metadata-only on purpose — importing flag_gems or
# torch_musa initializes the MUSA driver, which only succeeds on a host with
# Mthreads GPUs.
RUN python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

def dist(*names):
    for name in names:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    raise SystemExit(f"none of {names} installed")

vllm = dist("vllm")
torch = dist("torch")
torch_musa = dist("torch-musa", "torch_musa")
flag_gems = dist("flag-gems", "flag_gems")

assert vllm.startswith("0.24"), (
    f"base image carries vLLM {vllm}, expected 0.24.x — wrong MUSA_BASE_IMAGE?"
)
assert "empty" in vllm, (
    f"base image carries vLLM {vllm}, expected the empty-device build "
    "(the plugin registers the MUSA platform itself)"
)
assert torch.startswith("2.9"), (
    f"base image carries torch {torch}, expected 2.9.x"
)
print(f"vLLM {vllm} / torch {torch} / torch_musa {torch_musa} / "
      f"FlagGems {flag_gems} — base stack OK")
PY

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
