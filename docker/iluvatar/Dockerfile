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

# Extend the published 0.32 CoreX stack without replacing its runtime packages.
# This Dockerfile is not the recipe that originally produced the published image.
ARG ILUVATAR_BASE_IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-iluvatar-ci

# ---------- base stage ----------
FROM ${ILUVATAR_BASE_IMAGE} AS base

ARG ILUVATAR_BASE_IMAGE
ARG VLLM_VERSION=0.32.0
ARG TORCH_VERSION=2.10.0
ARG DRIVER_VERSION=4.5.0
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=dev
ARG SOURCE_URL=https://github.com/flagos-ai/vllm-plugin-FL

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.description="Iluvatar BI-V150 image for vllm-plugin-FL (vLLM 0.32)" \
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

RUN env -u PYTHONPATH python - <<'PY'
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

assert vllm.startswith("0.32.0"), vllm
assert "empty" in vllm, vllm
assert torch.startswith("2.10"), torch
assert flag_gems.startswith("5.5.0"), flag_gems
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
        "pytest-xdist>=3.0.0" \
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
