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


# Official FlagOS Runtime 2.3.0 for Kunlunxin XRE 5.37.1.
# https://flagos-ai.github.io/release-info/zh-cn/runtime/kunlunxin-xre5.37.1/
# The digest is the published manifest digest; the config digest is
# sha256:4961327d52a6feb2a71759e7e0055ce41c373ecbf42a802132ee8eb3ed52a663.
ARG KUNLUNXIN_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-kunlunxin-xre5.37.1:2.3.0@sha256:cc0deb9c09cd49977e754892e85f4414835d142379949cc2d66261f63b47959d
FROM ${KUNLUNXIN_BASE_IMAGE} AS dependencies

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VCS_REF=unknown
ARG PLUGIN_SOURCE_SHA=77f141a961d75e54c0cf209de80c99a7c7eec04b

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.source="https://github.com/flagos-ai/vllm-plugin-FL" \
      org.opencontainers.image.revision="${VCS_REF}" \
      io.flagos.accelerator.backend="kunlunxin" \
      io.flagos.vllm.version="0.32.0" \
      io.flagos.xre.version="5.37.1" \
      io.flagos.host.driver.version="5.0.21.47" \
      io.flagos.plugin.source.sha="${PLUGIN_SOURCE_SHA}"

# The application image this replaces exported VLLM_FL_PREFER=flagos plus an
# operator whitelist; the Runtime 2.3.0 base exports neither. PREFER=flagos is
# required: without it the multimodal encoder profiling forward falls back to
# kernels that exceed the XPU launch watchdog on P800. The whitelist is
# deliberately not set for the same reason, since it forces native fallbacks.
ENV PATH="/flagos/bin:${PATH}" \
    GEMS_VENDOR=kunlunxin \
    VLLM_PLUGINS=fl \
    USE_FLAGGEMS=1 \
    VLLM_FL_PREFER=flagos

# Pinned FlagCX 0.14.1 wheel, the Python 3.10 patched vLLM 0.32.0+flagos wheel,
# and the runtime constraints that keep Torch/FlagTree/Triton/FlagGems/NumPy.
COPY docker/kunlunxin/install-vllm.sh docker/kunlunxin/python310_compat.py docker/kunlunxin/constraints.txt /opt/kunlunxin-build/
RUN bash /opt/kunlunxin-build/install-vllm.sh

# Register the vLLM FlagGems providers in every derived image.
COPY .github/scripts/install_flaggems_vllm.py /opt/kunlunxin-build/install_flaggems_vllm.py
RUN PIP_INDEX_URL="${INDEX_URL:-https://pypi.org/simple}" CI_FLAGGEMS_VLLM_VENDOR=kunlunxin \
    python /opt/kunlunxin-build/install_flaggems_vllm.py --source-archive

# Build the plugin, including vllm_fl._C, from the pinned checkout inside the
# runtime. The runtime ships CMake 3.22, while csrc/ requires 3.26; nvcc is only
# needed because Torch's CMake package config enables the CUDA language.
COPY . /opt/vllm-plugin-FL
RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        --constraint /opt/kunlunxin-build/constraints.txt "cmake>=3.26" \
    && SETUPTOOLS_SCM_PRETEND_VERSION="0.5.0.dev0+g${PLUGIN_SOURCE_SHA}.xre5.37.1" \
       VLLM_VENDOR=cuda \
       PATH="/usr/local/cuda/bin:${PATH}" \
       CUDACXX=/usr/local/cuda/bin/nvcc \
       CMAKE_ARGS="-DTorch_DIR=$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)' 2>/dev/null | tail -1)/Torch" \
       python -m pip wheel --no-build-isolation --no-deps --no-cache-dir \
           -w /opt/kunlunxin-wheels /opt/vllm-plugin-FL

FROM dependencies AS runtime
RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        --constraint /opt/kunlunxin-build/constraints.txt \
        "modelscope==1.35.3" \
    && python -m pip install --no-deps --force-reinstall /opt/kunlunxin-wheels/vllm_plugin_fl-*.whl

FROM runtime AS base

FROM base AS ci
RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        --constraint /opt/kunlunxin-build/constraints.txt \
        pytest pytest-cov pytest-timeout pytest-json-report pytest-xdist \
        requests openai pillow decorator pyyaml sqlalchemy pybind11

FROM ci AS dev
RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        --constraint /opt/kunlunxin-build/constraints.txt \
        pre-commit ruff

FROM base AS release
