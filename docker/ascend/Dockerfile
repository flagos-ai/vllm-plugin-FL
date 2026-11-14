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

# Latest published 910C application image from the FlagOS release catalog.
# Keep its native CANN/PyTorch/FlagTree/FlagGems stack; replace vLLM and FL.
ARG ASCEND_BASE_IMAGE=harbor.baai.ac.cn/flagos-app/vllm0.24.0-ascend-cann9.0.0-910c:2.2.0-0.3.0@sha256:fc83d07a4fbecd090dc471e9caa5d978dc34715fcfcca537b7bbf720a298dd51

FROM ${ASCEND_BASE_IMAGE} AS base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG VLLM_VERSION=0.32.0
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/refs/tags/v${VLLM_VERSION}
ARG VLLM_SOURCE_SHA256=bf592c132f27bbb5f65cff0bbd2ee873eae722b79c8286934fd5b5cf88b43d61
ARG UV_VERSION=0.13.4
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_PLUGINS=fl \
    VLLM_FL_PLATFORM=ascend \
    USE_C_EXTENSION=0 \
    HF_ENDPOINT=https://hf-mirror.com \
    FL_MODEL_BASE_PATH=/data/models \
    UV_OVERRIDE=/opt/ascend-stack/native-overrides.txt

# Job-container steps may use sh or execute Python directly, bypassing BASH_ENV.
# Keep the image-selected compiler and runtime libraries available in those paths.
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    PYTHONPATH=/opt/flagtree:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe \
    LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}

COPY runtime_constraints.py /opt/ascend-stack/runtime_constraints.py
# compiler selects the image's FlagTree environment; capture the exact
# installed versions rather than reinstalling the old 0.28 recipe's stack.
RUN compiler flagtree \
    && python /opt/ascend-stack/runtime_constraints.py capture /opt/ascend-stack \
    && python -m pip install --no-cache-dir --no-deps \
        --index-url "${INDEX_URL}" "uv==${UV_VERSION}" \
    && uv --config-file /opt/ascend-stack/uv.toml pip install \
        --python "$(command -v python)" --no-cache --no-build-isolation \
        --index-url "${INDEX_URL}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        'setuptools>=78.1.1,<81.0.0' 'setuptools-scm>=8' \
        'setuptools-rust>=1.9' wheel scikit-build-core==0.11 pybind11 ninja cmake

# Keep the complete source archive: setup.py imports tools/build_rust.py
# even for empty mode. Build an ordinary wheel, without CUDA extensions.
RUN curl --fail --location --retry 5 "${VLLM_SOURCE_URL}" -o /tmp/vllm-source.tar.gz \
    && echo "${VLLM_SOURCE_SHA256}  /tmp/vllm-source.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/vllm-fl-empty-src \
    && tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /opt/vllm-fl-empty-src \
    && test -f /opt/vllm-fl-empty-src/tools/build_rust.py \
    && uv --config-file /opt/ascend-stack/uv.toml pip install \
        --python "$(command -v python)" --no-cache --no-build-isolation \
        --index-url "${INDEX_URL}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        -r /opt/vllm-fl-empty-src/requirements/common.txt \
    && python -m pip uninstall -y vllm vllm-ascend vllm-plugin-fl \
    && VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" \
        python -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir /opt/vllm-empty-wheels /opt/vllm-fl-empty-src \
    && python -m pip install --no-deps /opt/vllm-empty-wheels/vllm-*.whl \
    && python /opt/ascend-stack/runtime_constraints.py verify \
        /opt/ascend-stack --vllm-version "${VLLM_VERSION}"

WORKDIR /workspace
# Do not inherit the 0.24 application's vllm-serve launcher.
CMD ["bash"]

FROM base AS dev
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
RUN uv --config-file /opt/ascend-stack/uv.toml pip install \
        --python "$(command -v python)" --no-cache \
        --index-url "${INDEX_URL}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        pytest pytest-cov pytest-xdist pytest-timeout pytest-json-report \
        pytest-metadata ruff pre-commit \
    && python /opt/ascend-stack/runtime_constraints.py verify /opt/ascend-stack

FROM dev AS ci
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
RUN uv --config-file /opt/ascend-stack/uv.toml pip install \
        --python "$(command -v python)" --no-cache \
        --index-url "${INDEX_URL}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        requests openai decorator pyyaml sqlalchemy pybind11 \
        modelscope==1.35.3 'huggingface_hub>=1.31.0' \
    && python /opt/ascend-stack/runtime_constraints.py verify /opt/ascend-stack

FROM base AS release
# Install the desired FL plugin wheel with --no-build-isolation --no-deps.
WORKDIR /workspace
