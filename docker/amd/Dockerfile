# syntax=docker/dockerfile:1
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
#
# AMD ROCm image for gfx1100 (RDNA3), validated on Radeon PRO W7900D.
#
# Two source images are needed because neither one alone is sufficient:
#   * The ROCm base image has a Torch built for gfx1100, so the GPUs are
#     visible, but it ships upstream Triton 3.8 and no FlagGems.
#   * The FlagOS runtime image carries FlagTree 0.7.0+amd3.6, but its Torch
#     reports an empty torch.cuda.get_arch_list(), so no GPU is visible.
# FlagTree is therefore copied out of the FlagOS image into the ROCm base.
# FlagTree is not on PyPI; this COPY --from is how the AMD build is obtained.

ARG AMD_BASE_IMAGE=docker.io/vllm/vllm-openai-rocm:latest
ARG FLAGOS_RUNTIME_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-amd-rocm7.2.4:2.3.0

FROM ${FLAGOS_RUNTIME_IMAGE} AS flagos

FROM ${AMD_BASE_IMAGE} AS base
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG PYTHON_VERSION=3.12
ARG VLLM_VERSION=0.32.0
ARG FLAGTREE_VERSION=0.7.0+amd3.6
ARG FLAGGEMS_COMMIT=a8fdfa0
ARG FLAGCX_COMMIT=591cfbf
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL

# GEMS_VENDOR/USE_FLAGGEMS mirror the other FlagOS backends. Graph mode is not
# supported on this backend yet, so the service runs eager; see docs/usage/amd.md.
ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl \
    USE_FLAGGEMS=1 \
    GEMS_VENDOR=amd \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Refuse a base image whose Torch cannot see gfx1100: that failure is otherwise
# only discovered at serve time, as an empty device list.
RUN python3 -c 'import torch; \
archs = torch.cuda.get_arch_list(); \
print("torch:", torch.__version__, "| hip:", torch.version.hip); \
print("arch_list:", archs); \
assert any(a.startswith("gfx1100") for a in archs), \
    "base image Torch is not built for gfx1100: %r" % (archs,)'

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- FlagTree supplies Triton ----------
# FlagGems requires Triton <= 3.6; the ROCm base image ships 3.8. Move the
# image's Triton aside and let FlagTree own the `triton` module via a .pth.
COPY --from=flagos /opt/flagtree /opt/flagtree
RUN SP="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && for d in triton pytorch_triton; do \
        [ -d "${SP}/${d}" ] && mv "${SP}/${d}" "${SP}/.${d}-image-backup" || true; \
    done \
    && printf '/opt/flagtree\n' > "${SP}/zz-flagtree.pth" \
    && EXPECT_FLAGTREE="${FLAGTREE_VERSION}" python3 -c 'import triton, importlib.metadata as md; \
import os; \
print("triton:", triton.__version__, triton.__file__); \
assert "/opt/flagtree" in os.path.realpath(triton.__file__), triton.__file__; \
ft = md.version("flagtree"); \
print("flagtree:", ft); \
assert ft == os.environ["EXPECT_FLAGTREE"], (ft, os.environ["EXPECT_FLAGTREE"])'

# ---------- vLLM, rebuilt as an empty distribution ----------
# Non-NVIDIA backends use empty builds uniformly: no vllm._C / _rocm_C.
RUN python3 -m pip install --no-cache-dir \
        ${INDEX_URL:+--index-url "${INDEX_URL}"} \
        ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
        "setuptools>=77,<81" setuptools-scm wheel pybind11 ninja cmake \
    && git clone --depth 1 --branch "v${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm.git /opt/vllm-src \
    && cd /opt/vllm-src \
    && VLLM_TARGET_DEVICE=empty VLLM_USE_PRECOMPILED=0 \
       SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" \
       python3 -m pip wheel --no-deps --no-build-isolation \
           --wheel-dir /opt/wheels /opt/vllm-src \
    && python3 -m pip install --no-cache-dir "/opt/wheels/vllm-${VLLM_VERSION}+empty-"*.whl \
    && rm -rf /opt/vllm-src \
    && python3 -c 'from importlib.metadata import version; \
v = version("vllm"); print("vllm:", v); assert v.endswith("+empty"), v'

# ---------- FlagGems, FlagGems-vllm and FlagCX ----------
# FlagGems must come from main: the gfx1100 fix that sets
# triton_extra_name="hip" for the _amd backend is not in the 5.4.0 release.
RUN git clone https://github.com/flagos-ai/FlagGems.git /opt/FlagGems \
    && git -C /opt/FlagGems checkout "${FLAGGEMS_COMMIT}" \
    && python3 -m pip install --no-cache-dir --no-build-isolation \
        ${INDEX_URL:+--index-url "${INDEX_URL}"} \
        /opt/FlagGems \
    && rm -rf /opt/FlagGems

COPY .github/scripts/install_flaggems_vllm.py /opt/amd/install_flaggems_vllm.py
RUN PIP_INDEX_URL="${INDEX_URL}" PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-}" \
        python3 /opt/amd/install_flaggems_vllm.py

# FlagCX's AMD adaptor uses RCCL. libflagcx.so must be loadable at runtime, and
# the plugin selects FlagCX purely from FLAGCX_PATH being set.
RUN git clone https://github.com/flagos-ai/FlagCX.git /opt/FlagCX \
    && git -C /opt/FlagCX checkout "${FLAGCX_COMMIT}" \
    && cd /opt/FlagCX \
    && make USE_AMD=1 -j"$(nproc)" \
    && FLAGCX_ADAPTOR=amd USE_AMD=1 FLAGCX_HOME=/opt/FlagCX \
        python3 -m pip install --no-cache-dir --no-deps --no-build-isolation .
ENV FLAGCX_PATH=/opt/FlagCX \
    LD_LIBRARY_PATH=/opt/FlagCX/build/lib:${LD_LIBRARY_PATH}

# ---------- the plugin ----------
COPY pyproject.toml setup.py README.md LICENSE /opt/vllm-src/vllm-plugin-fl/
COPY vllm_fl /opt/vllm-src/vllm-plugin-fl/vllm_fl
RUN python3 -m pip wheel --no-deps --no-build-isolation \
        --wheel-dir /opt/wheels /opt/vllm-src/vllm-plugin-fl \
    && python3 -m pip install --no-cache-dir --no-deps \
        /opt/wheels/vllm_plugin_fl-*.whl

# Fail the build if the stack is not what this backend was validated with.
RUN VLLM_PLUGINS=fl python3 -c 'import importlib.metadata as md; \
import triton, os; \
import vllm_fl; \
from vllm_fl.platform import PlatformFL as P; \
assert "/opt/flagtree" in os.path.realpath(triton.__file__), triton.__file__; \
assert P.vendor_name == "amd", P.vendor_name; \
assert P().is_rocm(), "is_rocm() must be True on AMD"; \
assert P.dist_backend == "flagcx", P.dist_backend; \
print("vllm      :", md.version("vllm")); \
print("flagtree  :", md.version("flagtree")); \
print("flag_gems :", md.version("flag-gems")); \
print("vendor    :", P.vendor_name, "| dist_backend:", P.dist_backend)'

WORKDIR /workspace

# The ROCm base image's entrypoint is `vllm serve`, which swallows any command
# passed to `docker run`. Pass `--entrypoint python3` (or bash) to inspect the
# image; `docker run <image> <model>` serves directly, as upstream intends.

# ---------- ci stage ----------
FROM base AS ci
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
RUN python3 -m pip install --no-cache-dir \
        ${INDEX_URL:+--index-url "${INDEX_URL}"} \
        ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
        pytest pytest-cov pytest-timeout pytest-json-report ruff pre-commit
WORKDIR /workspace

# ---------- dev stage ----------
FROM ci AS dev
WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release
WORKDIR /workspace
