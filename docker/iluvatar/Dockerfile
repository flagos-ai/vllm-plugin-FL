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

# vLLM 0.24 image for Iluvatar BI-V150, layered on the validated CoreX 4.5
# stack supplied for vLLM 0.24. Keep the packaged vendor software matrix.
ARG ILUVATAR_BASE_IMAGE=harbor.baai.ac.cn/plugin/iluvatar-corex4.5.0-flagtree0.6.0-triton3.6.0-cxnone-vllm_fl0.24.0:20260909

# ---------- base stage ----------
FROM ${ILUVATAR_BASE_IMAGE} AS base

ARG ILUVATAR_BASE_IMAGE
ARG VLLM_VERSION=0.24.0
ARG TORCH_VERSION=2.10.0
ARG DRIVER_VERSION=4.5.0
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=dev
ARG SOURCE_URL=https://github.com/flagos-ai/vllm-plugin-FL

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.description="Iluvatar BI-V150 image for vllm-plugin-FL (vLLM 0.24)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${ILUVATAR_BASE_IMAGE}" \
      io.flagos.accelerator.backend="iluvatar" \
      io.flagos.image.stage="${IMAGE_STAGE}" \
      io.flagos.vllm.version="${VLLM_VERSION}" \
      io.flagos.torch.version="${TORCH_VERSION}" \
      io.flagos.driver.version="${DRIVER_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl \
    GEMS_VENDOR=iluvatar

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf "$(command -v python3.12)" /usr/local/bin/python \
    && python -m pip --version

RUN python -m pip install --no-cache-dir \
        "setuptools>=77.0.3,<81.0.0" \
        wheel \
        setuptools-scm \
        setuptools-rust \
        "cloudpickle>=3.1.0" \
        scikit-build-core==0.11 \
        pybind11 \
        ninja \
        cmake

# Metadata-only validation avoids initializing the CoreX driver during build.
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
flag_gems = dist("flag-gems", "flag_gems")

assert vllm.startswith("0.24"), vllm
assert "empty" in vllm, vllm
assert torch.startswith("2.10"), torch
assert flag_gems.startswith("5.3"), flag_gems
print(f"vLLM {vllm} / torch {torch} / FlagGems {flag_gems} - base stack OK")
PY

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

WORKDIR /workspace
