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

# vLLM 0.28 empty-device recipe for Enflame S60. Keep the vendor torch_gcu
# 2.11 stack and install official Enflame compiler/operator wheels.
# The runtime is validated on driver 1.10.6. The old vendor wrapper
# remains in Dockerfile.v0.20.2.
ARG ENFLAME_BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-enflame-gems_5.4.0.dev0-sglang_0.5.11-sglang_plugin_0.1.0-cx_0.13.0-python_3.12.8-torch_gcu_2.11.0_3.8.20260713-pcp_tops3.8.20260714-gpu_s60-arc_amd64-driver_1.9.10:202608141853
# Override with a mirror (e.g. https://gh-proxy.com/https://github.com/...) on
# hosts with unreliable direct GitHub access.
ARG VLLM_SOURCE_URL=https://github.com/vllm-project/vllm.git

# ---------- base stage ----------
FROM ${ENFLAME_BASE_IMAGE} AS base

ARG ENFLAME_BASE_IMAGE
ARG VLLM_VERSION=0.28.0
ARG VLLM_SOURCE_URL
ARG VLLM_SOURCE_COMMIT=2cf0a6915ce544dc493a0990f2ea38d81601128a
ARG DRIVER_VERSION=1.10.6
ARG FLAGOS_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-enflame/simple
ARG FLAGGEMS_VERSION=5.4.0
ARG FLAGTREE_VERSION=0.7.0+enflame3.6
ARG TRITON_GCU_VERSION=3.6.0+1.0.20260722
ARG OPENAI_VERSION=3.28.0
ARG PANDAS_VERSION=3.0.6
ARG FLAGGEMS_SHA256=09cb175e29f4fdfc78b87df46c4500d5a9b451e12cd64dda9a7d669f522bf5e5
ARG FLAGTREE_SHA256=94b3aaf9f9aefa02f535ee5339f16c5304c52a1de5e7e749f742f7b4bf01332d
ARG TRITON_GCU_SHA256=59d7a450a48f8de2a965ea6e456ea7c13bf53cb83abd5df929e4e8558da5f8f9
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=dev
ARG SOURCE_URL=https://github.com/flagos-ai/vllm-plugin-FL

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.description="Enflame S60 image for vllm-plugin-FL (vLLM ${VLLM_VERSION})" \
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

# Build tooling for vLLM 0.28 empty mode: setuptools >= 77 (PEP 639 SPDX
# license) and setuptools-rust are missing from the vendor bundle. cloudpickle is
# upgraded because the bundle ships 1.6.0, which cannot pickle Python 3.12
# code objects (no qualname field) and breaks vLLM's model-registry
# subprocess.
RUN python -m pip install --no-cache-dir \
        "setuptools>=77.0.3,<81.0.0" \
        wheel \
        "setuptools-scm>=8" \
        "setuptools-rust>=1.9" \
        "packaging>=26.0" \
        "cloudpickle>=3.1.0" \
        "PyYAML==6.0.1" \
        "sqlalchemy>=1.4.31,<2.1" \
        scikit-build-core==0.11 \
        pybind11 \
        ninja \
        "cmake>=3.26.1"

# FlagTree supplies the triton Python namespace. Preserve the vendor SDK
# and install the matching triton-gcu shim before FlagTree. Changing wheel
# versions also requires matching hashes from the official backend index.
RUN printf '%s --hash=sha256:%s\n' \
        "flag-gems==${FLAGGEMS_VERSION}" "${FLAGGEMS_SHA256}" \
        "flagtree==${FLAGTREE_VERSION}" "${FLAGTREE_SHA256}" \
        "triton-gcu==${TRITON_GCU_VERSION}" "${TRITON_GCU_SHA256}" \
        > /tmp/enflame-wheel-requirements.txt \
    && python -m pip download --no-cache-dir --no-deps --only-binary=:all: \
        --require-hashes --index-url "${FLAGOS_INDEX_URL}" \
        --dest /tmp/enflame-wheels -r /tmp/enflame-wheel-requirements.txt \
    && python -m pip uninstall -y triton \
    && python -m pip install --no-deps --no-index --find-links /tmp/enflame-wheels \
        "triton-gcu==${TRITON_GCU_VERSION}" \
    && python -m pip install --no-deps --no-index --find-links /tmp/enflame-wheels \
        "flagtree==${FLAGTREE_VERSION}" \
    && python -m pip install --no-deps --no-index --find-links /tmp/enflame-wheels \
        "flag-gems==${FLAGGEMS_VERSION}" \
    && rm -rf /tmp/enflame-wheels /tmp/enflame-wheel-requirements.txt

# Replace the vendor vLLM with the official empty-device build; the plugin
# registers the GCU platform via the vLLM platform_plugins entry point.
RUN (python -m pip uninstall -y vllm || true) \
    && (python -m pip cache remove vllm || true) \
    && SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && find "${SITE_PACKAGES}" -maxdepth 1 \
        \( -name "vllm" -o -name "vllm-*.dist-info" -o -name "vllm-*.egg-info" \) \
        -exec rm -rf {} + 2>/dev/null \
    && rm -rf /tmp/vllm \
    && git clone --depth 1 --branch "v${VLLM_VERSION}" \
        "${VLLM_SOURCE_URL}" /tmp/vllm \
    && cd /tmp/vllm \
    && test "$(git rev-parse HEAD)" = "${VLLM_SOURCE_COMMIT}" \
    && printf '%s\n' \
        'torch==2.11.0+cpu' \
        'torch-gcu==2.11.0+3.8.20260713' \
        'triton==3.6.0' \
        "triton-gcu==${TRITON_GCU_VERSION}" \
        "flagtree==${FLAGTREE_VERSION}" \
        "flag-gems==${FLAGGEMS_VERSION}" \
        > /tmp/enflame-vendor-constraints.txt \
    && python -m pip install --no-cache-dir -r requirements/common.txt \
        -c /tmp/enflame-vendor-constraints.txt \
    && python -m pip install --no-cache-dir --upgrade \
        "openai==${OPENAI_VERSION}" "pandas==${PANDAS_VERSION}" \
        -c /tmp/enflame-vendor-constraints.txt \
    && VLLM_TARGET_DEVICE=empty python -m pip install --no-cache-dir \
        --no-build-isolation --no-deps . \
    && cd / \
    && rm -rf /tmp/vllm /tmp/enflame-vendor-constraints.txt

# vLLM common dependencies may install upstream Triton into the same Python
# namespace as FlagTree. Restore the pinned Enflame compiler after resolving
# those dependencies; package version metadata alone cannot detect an overlay.
RUN printf '%s --hash=sha256:%s\n' \
        "flagtree==${FLAGTREE_VERSION}" "${FLAGTREE_SHA256}" \
        > /tmp/enflame-flagtree-requirements.txt \
    && python -m pip download --no-cache-dir --no-deps --only-binary=:all: \
        --require-hashes --index-url "${FLAGOS_INDEX_URL}" \
        --dest /tmp/enflame-flagtree-wheel -r /tmp/enflame-flagtree-requirements.txt \
    && python -m pip uninstall -y triton \
    && python -m pip install --force-reinstall --no-deps --no-index \
        --find-links /tmp/enflame-flagtree-wheel "flagtree==${FLAGTREE_VERSION}" \
    && rm -rf /tmp/enflame-flagtree-wheel /tmp/enflame-flagtree-requirements.txt

COPY verify_compiler.py /usr/local/share/vllm-plugin-fl/enflame-verify-compiler.py
RUN python /usr/local/share/vllm-plugin-fl/enflame-verify-compiler.py \
    --expected-version "${FLAGTREE_VERSION}"

# Fail early when a wrong vendor base image is supplied. Metadata-only on
# purpose — importing flag_gems/torch_gcu initializes the GCU driver, which
# only succeeds on a host with Enflame GPUs.
RUN VLLM_EXPECTED_VERSION="${VLLM_VERSION}" \
    FLAGGEMS_EXPECTED_VERSION="${FLAGGEMS_VERSION}" \
    FLAGTREE_EXPECTED_VERSION="${FLAGTREE_VERSION}" \
    OPENAI_EXPECTED_VERSION="${OPENAI_VERSION}" \
    PANDAS_EXPECTED_VERSION="${PANDAS_VERSION}" \
    TRITON_GCU_EXPECTED_VERSION="${TRITON_GCU_VERSION}" python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import os

from packaging.version import Version

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
flagtree = dist("flagtree")
triton_gcu = dist("triton-gcu", "triton_gcu")
openai = dist("openai")
pandas = dist("pandas")

assert Version(vllm).base_version == os.environ["VLLM_EXPECTED_VERSION"], (
    f"installed vLLM {vllm}, expected {os.environ['VLLM_EXPECTED_VERSION']}"
)
assert "empty" in vllm, (
    f"base image carries vLLM {vllm}, expected the empty-device build"
)
assert torch == "2.11.0+cpu", f"vendor torch was replaced: {torch}"
assert torch_gcu == "2.11.0+3.8.20260713", f"wrong torch_gcu: {torch_gcu}"
assert flag_gems == os.environ["FLAGGEMS_EXPECTED_VERSION"], flag_gems
assert flagtree == os.environ["FLAGTREE_EXPECTED_VERSION"], flagtree
assert triton_gcu == os.environ["TRITON_GCU_EXPECTED_VERSION"], triton_gcu
assert openai == os.environ["OPENAI_EXPECTED_VERSION"], openai
assert pandas == os.environ["PANDAS_EXPECTED_VERSION"], pandas
print(f"vLLM {vllm} / torch {torch} / torch_gcu {torch_gcu} / "
      f"FlagGems {flag_gems} / FlagTree {flagtree} / triton_gcu {triton_gcu} "
      f"/ OpenAI SDK {openai} / pandas {pandas} "
      "— package metadata OK; device/Gate validation remains separate")
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
