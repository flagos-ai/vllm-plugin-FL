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

# FlagOS Runtime 2.3.0 for Kunlunxin XRE 5.37.1.
ARG KUNLUNXIN_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-kunlunxin-xre5.37.1:2.3.0@sha256:cc0deb9c09cd49977e754892e85f4414835d142379949cc2d66261f63b47959d

# ---------- base stage ----------
FROM ${KUNLUNXIN_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG KUNLUNXIN_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-kunlunxin/simple
ARG KUNLUNXIN_VLLM_REQUIREMENT=vllm==0.32.0+flagos
ARG FLAGGEMS_REQUIREMENT=flag-gems==5.5.0.dev20261124
ARG DAILY_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-daily/simple
ARG FLAGCX_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-kunlunxin/packages/flagcx/0.14.1+xre5.37.1/flagcx-0.14.1+xre5.37.1-cp310-cp310-linux_x86_64.whl
ARG FLAGCX_WHEEL_SHA256=b49203511f7906aaa3cebdbf304aff521a86f0540f6b57c6af6154092f8a658e
ARG FLAGGEMS_VLLM_COMMIT=8ae5c22f5af6d650216f881c329c4dff20c182e1
ARG FLAGGEMS_VLLM_SHA256=eab3544a5a6a2efaf7b3465a04ea0f867f77cfbe144ed7f2b8ad20c7b28bb36c
ARG PLUGIN_SOURCE_SHA=unknown

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.source="https://github.com/flagos-ai/vllm-plugin-FL" \
      org.opencontainers.image.revision="${PLUGIN_SOURCE_SHA}" \
      io.flagos.accelerator.backend="kunlunxin" \
      io.flagos.vllm.version="0.32.0" \
      io.flagos.xre.version="5.37.1" \
      io.flagos.host.driver.version="5.0.21.47" \
      io.flagos.plugin.source.sha="${PLUGIN_SOURCE_SHA}"

# Use vendor ops during multimodal profiling on P800.
ENV GEMS_VENDOR=kunlunxin \
    VLLM_PLUGINS=fl \
    USE_FLAGGEMS=1 \
    VLLM_FL_PREFER=flagos

# Resolve the vendor vLLM wheel against the Runtime Torch ABI.
# Expose Triton to pip only for this installation.
RUN PYTHONPATH=/opt/triton \
    PIP_EXTRA_INDEX_URL="${KUNLUNXIN_INDEX_URL}${EXTRA_INDEX_URL:+ $EXTRA_INDEX_URL}" \
    python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        "${KUNLUNXIN_VLLM_REQUIREMENT}" \
        "flagcx @ ${FLAGCX_WHEEL_URL}#sha256=${FLAGCX_WHEEL_SHA256}" \
        "torch==2.9.0+cu129" "torchvision==0.24.0+cu129" "torchaudio==2.9.0+cu129" \
        "numpy==1.26.4" "regex==2026.4.4"

# Install build tools and the FlagGems stack.
# P800 Runner V2 needs the pinned provider for sampling and grammar masking.
RUN PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-}" \
    python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" wheel \
        packaging pybind11 pyyaml sqlalchemy "cmake>=3.26" \
    && python -m pip install --no-cache-dir --no-deps --index-url "${DAILY_INDEX_URL}" \
        "${FLAGGEMS_REQUIREMENT}" \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation \
        "flaggems-vllm @ https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/${FLAGGEMS_VLLM_COMMIT}#sha256=${FLAGGEMS_VLLM_SHA256}"

# Install the plugin's model-loading dependency.
RUN PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-}" \
    python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        "modelscope==1.35.3"

WORKDIR /workspace

# ---------- plugin wheel stage ----------
FROM base AS plugin-wheel

# Build the native extension against the Runtime Torch using the archived checkout.
WORKDIR /src
COPY pyproject.toml setup.py README.md LICENSE ./
COPY csrc ./csrc
COPY vllm_fl ./vllm_fl
RUN torch_package="$(python -c 'from importlib.util import find_spec; print(find_spec("torch").submodule_search_locations[0])')" \
    && torch_dir="${torch_package}/share/cmake/Torch" \
    && test -f "${torch_dir}/TorchConfig.cmake" \
    && cuda_home="${CUDA_HOME:-/usr/local/cuda}" \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="0.5.0.dev0+g${PLUGIN_SOURCE_SHA}.xre5.37.1" \
        VLLM_VENDOR=cuda PATH="${cuda_home}/bin:${PATH}" CUDACXX="${cuda_home}/bin/nvcc" \
        CMAKE_ARGS="-DTorch_DIR=${torch_dir}" \
        python -m pip wheel --no-build-isolation --no-deps --no-cache-dir -w /wheels .

# ---------- runtime stage ----------
FROM base AS runtime

# Install the wheel without copying the build checkout into the runtime.
RUN --mount=type=bind,from=plugin-wheel,source=/wheels,target=/wheels \
    python -m pip install --no-deps --no-index --force-reinstall /wheels/vllm_plugin_fl-*.whl \
    && python -m pip check

# ---------- ci stage ----------
FROM runtime AS ci

# Install the test tools used by UT, FT, E2E and benchmark.
RUN PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-}" \
    python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        pytest pytest-cov pytest-timeout pytest-json-report pytest-xdist \
        requests openai pillow decorator pyyaml sqlalchemy pybind11 \
    && python -m pip check

# ---------- dev stage ----------
FROM ci AS dev

# Install developer tools on top of the CI environment.
RUN PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-}" \
    python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        pre-commit ruff

# ---------- release stage ----------
FROM runtime AS release
