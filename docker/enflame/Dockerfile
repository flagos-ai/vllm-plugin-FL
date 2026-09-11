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

# vLLM 0.24 image for Enflame S60. The base is the vendor-delivered
# torch_gcu 2.11.0 stack (torch 2.11 / triton_gcu 3.6 / FlagGems 5.4.0.dev0,
# validated against host driver 1.9.10); this Dockerfile replaces its vLLM
# with the official 0.24.0 empty-device build so the FL plugin registers the
# GCU platform itself. The previous thin wrapper around the 0.20.2 vendor
# image is kept as Dockerfile.v0.20.2.
ARG ENFLAME_BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-enflame-gems_5.4.0.dev0-sglang_0.5.11-sglang_plugin_0.1.0-cx_0.13.0-python_3.12.8-torch_gcu_2.11.0_3.8.20260713-pcp_tops3.8.20260714-gpu_s60-arc_amd64-driver_1.9.10:202608141853
# Override with a mirror (e.g. https://gh-proxy.com/https://github.com/...) on
# hosts with unreliable direct GitHub access.
ARG VLLM_SOURCE_URL=https://github.com/vllm-project/vllm.git

# ---------- base stage ----------
FROM ${ENFLAME_BASE_IMAGE} AS base

ARG ENFLAME_BASE_IMAGE
ARG VLLM_VERSION=0.24.0
ARG VLLM_SOURCE_URL
ARG DRIVER_VERSION=1.9.10
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=dev
ARG SOURCE_URL=https://github.com/flagos-ai/vllm-plugin-FL

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.description="Enflame S60 image for vllm-plugin-FL (vLLM 0.24)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${ENFLAME_BASE_IMAGE}" \
      io.flagos.accelerator.backend="enflame" \
      io.flagos.image.stage="${IMAGE_STAGE}" \
      io.flagos.vllm.version="${VLLM_VERSION}" \
      io.flagos.driver.version="${DRIVER_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=99999 \
    TORCH_GCU_ENABLE_INT64_AND_UINT64=1 \
    ENABLE_I64_CHECK=0 \
    TORCHGCU_INDUCTOR_ENABLE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Build tooling for the vLLM empty-mode install below. vLLM 0.24 needs
# setuptools >= 77 (PEP 639 SPDX license) and setuptools-rust (new Rust
# extensions); both were missing from the vendor bundle. cloudpickle is
# upgraded because the bundle ships 1.6.0, which cannot pickle Python 3.12
# code objects (no qualname field) and breaks vLLM's model-registry
# subprocess.
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

# Replace the vendor vLLM with the official empty-device build; the plugin
# registers the GCU platform via the vLLM platform_plugins entry point.
RUN python -m pip uninstall -y vllm || true \
    && python -m pip cache remove vllm || true \
    && SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && find "${SITE_PACKAGES}" -maxdepth 1 \
        \( -name "vllm" -o -name "vllm-*.dist-info" -o -name "vllm-*.egg-info" \) \
        -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /tmp/vllm \
    && git clone --depth 1 --branch "v${VLLM_VERSION}" \
        "${VLLM_SOURCE_URL}" /tmp/vllm \
    && cd /tmp/vllm \
    && VLLM_TARGET_DEVICE=empty python -m pip install --no-cache-dir \
        --no-build-isolation --no-deps . \
    && cd / \
    && rm -rf /tmp/vllm

# Fail early when a wrong vendor base image is supplied. Metadata-only on
# purpose — importing flag_gems/torch_gcu initializes the GCU driver, which
# only succeeds on a host with Enflame GPUs.
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
torch_gcu = dist("torch-gcu", "torch_gcu")
flag_gems = dist("flag-gems", "flag_gems")

assert vllm.startswith("0.24"), (
    f"base image carries vLLM {vllm}, expected 0.24.x — wrong ENFLAME_BASE_IMAGE?"
)
assert "empty" in vllm, (
    f"base image carries vLLM {vllm}, expected the empty-device build"
)
assert torch.startswith("2.11"), (
    f"base image carries torch {torch}, expected 2.11.x"
)
print(f"vLLM {vllm} / torch {torch} / torch_gcu {torch_gcu} / "
      f"FlagGems {flag_gems} — base stack OK")
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
