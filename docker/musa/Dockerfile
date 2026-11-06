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

# FlagOS 2.2 release runtime for Moore Threads: MUSA SDK 4.3.6, Python 3.10,
# torch 2.9.0 + torch_musa 2.9.0, FlagTree 0.7.0, FlagGems 5.4.0.
# https://flagos.io/resourcedownload -> release-info/runtime/mthreads-musa4.3.6
ARG MUSA_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-mthreads-musa4.3.6:2.2.0

# ---------- base stage ----------
FROM ${MUSA_BASE_IMAGE} AS base

ARG PYTHON_VERSION=3.10
ARG VLLM_VERSION=0.32.0
ARG FLAGGEMS_VERSION=5.4.0
ARG INDEX_URL=""
ARG EXTRA_INDEX_URL=""
# Source archive for the vLLM tree; override with an internal mirror when the
# build host cannot reach codeload.github.com.
ARG VLLM_SOURCE_URL=""

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl

# Keep the FlagOS runtime stack and add only the tools needed to build the
# checked-out plugin in CI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -n "${INDEX_URL}" ]; then export PIP_INDEX_URL="${INDEX_URL}"; fi; \
    if [ -n "${EXTRA_INDEX_URL}" ]; then export PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL}"; fi; \
    python -m pip install --no-cache-dir \
        "setuptools>=77.0.3,<81.0.0" \
        wheel \
        "setuptools-scm>=8" \
        "setuptools-rust>=1.9" \
        scikit-build-core==0.11 \
        pybind11 \
        ninja \
        cmake

# vLLM 0.32 requires Python >= 3.11 while torch_musa only ships cp310 wheels, so
# the empty wheel is installed with --ignore-requires-python and the 3.11-only
# names it imports at module load are back-filled by the .pth shim installed
# below.
RUN set -eu; \
    url="${VLLM_SOURCE_URL:-https://codeload.github.com/vllm-project/vllm/tar.gz/refs/tags/v${VLLM_VERSION}}"; \
    mkdir -p /workspace/vllm; \
    curl -fsSL --retry 5 --retry-delay 5 --max-time 900 "$url" -o /tmp/vllm-src.tar.gz; \
    tar -xzf /tmp/vllm-src.tar.gz -C /workspace/vllm --strip-components=1; \
    rm -f /tmp/vllm-src.tar.gz; \
    VLLM_TARGET_DEVICE=empty \
        VLLM_USE_PRECOMPILED=0 \
        VLLM_USE_PRECOMPILED_RUST=0 \
        VLLM_VENDOR= \
        SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}+empty" \
        python -m pip install --no-cache-dir --no-build-isolation --no-deps \
            --ignore-requires-python /workspace/vllm

# vLLM's own dependencies are installed with --no-deps so that pip cannot
# replace the vendor stack (compressed-tensors==0.17.0 asks for torch>=2.10 and
# would install an upstream CUDA torch over the MUSA build).
COPY req032.txt /workspace/req032.txt

RUN if [ -n "${INDEX_URL}" ]; then export PIP_INDEX_URL="${INDEX_URL}"; fi; \
    if [ -n "${EXTRA_INDEX_URL}" ]; then export PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL}"; fi; \
    python -m pip install --no-cache-dir --ignore-requires-python --no-deps \
        -r /workspace/req032.txt

# Extras and transitive dependencies skipped by --no-deps above.
RUN if [ -n "${INDEX_URL}" ]; then export PIP_INDEX_URL="${INDEX_URL}"; fi; \
    if [ -n "${EXTRA_INDEX_URL}" ]; then export PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL}"; fi; \
    python -m pip install --no-cache-dir --ignore-requires-python \
        "fastapi[standard]>=0.133.0,<0.137.0"

# Transitive dependencies of the list above (skipped by --no-deps), plus extras
# such as fastapi[standard] -> uvloop. rpds-py is pinned to the last release that
# ships a cp310 wheel; newer releases are Rust sdists and would bootstrap a
# toolchain from static.rust-lang.org during the build. websockets is pinned to
# the last release that still supports Python 3.10: 17.x drops the 3.10 branch in
# its typing module (``logging.LoggerAdapter[Any]``), which is a TypeError at
# import time, and --ignore-requires-python above means pip no longer filters
# such wheels out by Requires-Python.
RUN if [ -n "${INDEX_URL}" ]; then export PIP_INDEX_URL="${INDEX_URL}"; fi; \
    if [ -n "${EXTRA_INDEX_URL}" ]; then export PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL}"; fi; \
    python -m pip install --no-cache-dir --ignore-requires-python \
        aiohappyeyeballs \
        aiosignal \
        annotated-doc \
        anyio \
        apache-tvm-ffi \
        astor \
        async-timeout \
        attrs \
        certifi \
        charset-normalizer \
        click \
        dill \
        dnspython \
        docstring-parser \
        email-validator \
        frozenlist \
        h11 \
        hf-xet \
        httpcore \
        httpx2 \
        idna \
        interegular \
        jiter \
        jmespath \
        jsonschema-specifications \
        loguru \
        mcp-types \
        multidict \
        opentelemetry-exporter-otlp-proto-grpc \
        opentelemetry-exporter-otlp-proto-http \
        opentelemetry-semantic-conventions \
        propcache \
        pycountry \
        pydantic-core \
        pydantic-extra-types \
        pyjwt \
        referencing \
        rpds-py==0.30.0 \
        sniffio \
        sse-starlette \
        supervisor \
        typing-inspection \
        typer \
        urllib3 \
        websockets==16.0 \
        yarl \
    && python -m pip install --no-cache-dir --no-deps \
        "python-multipart==0.0.20"

# Must run before any vLLM import: py3.10 back-fills for the 0.32 stack.
COPY vllm_fl_py310_compat.py zz_vllm_fl_py310_compat.pth /tmp/compat/

RUN python -c "import shutil, site; sp = site.getsitepackages()[0]; \
[shutil.copy('/tmp/compat/' + f, sp) for f in ('vllm_fl_py310_compat.py', \
'zz_vllm_fl_py310_compat.pth')]" \
    && rm -rf /tmp/compat

# Fail early when a different vendor base image is supplied accidentally.
RUN python -c 'from importlib.metadata import version; \
import vllm; \
vllm_version = version("vllm"); \
flag_gems = version("flag-gems"); \
websockets_version = version("websockets"); \
assert vllm_version.startswith("0.32.0"), vllm_version; \
assert flag_gems == "5.4.0", flag_gems; \
assert websockets_version.startswith("16."), websockets_version; \
print("vLLM:", vllm_version); \
print("FlagGems:", flag_gems); \
print("websockets:", websockets_version)' \
    && rm -rf /workspace/vllm

WORKDIR /workspace
ENTRYPOINT []
CMD ["python", "-I", "-B", "-m", "vllm.entrypoints.openai.api_server"]

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
