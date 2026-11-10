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


# Latest Kunlunxin vLLM application image on the FlagOS release page.
# https://flagos-ai.github.io/release-info/application/vllm0.24.0-kunlunxin-xre5.37.1/
ARG KUNLUNXIN_BASE_IMAGE=harbor.baai.ac.cn/flagos-app/vllm0.24.0-kunlunxin-xre5.37.1:2.3.0-0.3.0@sha256:db1cceff13d438142abfd695500ac0820ebfe6ae13d909017a2ce073ac607b17
FROM ${KUNLUNXIN_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="vllm-plugin-fl" \
      org.opencontainers.image.source="https://github.com/flagos-ai/vllm-plugin-FL" \
      org.opencontainers.image.revision="${VCS_REF}" \
      io.flagos.accelerator.backend="kunlunxin" \
      io.flagos.vllm.version="0.32.0"

ENV PATH="/flagos/bin:${PATH}" \
    GEMS_VENDOR=kunlunxin \
    VLLM_PLUGINS=fl \
    USE_FLAGGEMS=1

COPY docker/kunlunxin/install-vllm.sh docker/kunlunxin/python310_compat.py /opt/kunlunxin-build/
RUN bash /opt/kunlunxin-build/install-vllm.sh

# Register the vLLM FlagGems providers in every derived image.
COPY .github/scripts/install_flaggems_vllm.py /opt/kunlunxin-build/install_flaggems_vllm.py
RUN PIP_INDEX_URL="${INDEX_URL:-https://pypi.org/simple}" \
    python /opt/kunlunxin-build/install_flaggems_vllm.py --source-archive

ENV VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding

COPY . /opt/vllm-plugin-FL
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.5.0.dev0 \
        python -m pip install --no-build-isolation --no-deps /opt/vllm-plugin-FL \
    && python -c 'from importlib.metadata import version; assert version("vllm") == "0.32.0+flagos"; import vllm.utils.import_utils'

FROM base AS ci
RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        pytest pytest-cov pytest-timeout pytest-json-report pytest-xdist \
        requests openai pillow decorator pyyaml sqlalchemy pybind11

FROM ci AS dev
RUN python -m pip install --no-cache-dir --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        pre-commit ruff

FROM base AS release
