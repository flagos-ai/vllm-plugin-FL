# Copyright (c) 2026 BAAI. All rights reserved.
# Latest Sunrise application image on the FlagOS release page, 2026-11-04:
# https://flagos-ai.github.io/release-info/application/vllm0.24.0-sunrise-tangrt1.2.0/
ARG SUNRISE_BASE_IMAGE=harbor.baai.ac.cn/flagos-app/vllm0.24.0-sunrise-tangrt1.2.0:2.3.0-0.3.0
# An optional named build context supplies only the public pinned Git bundle.
FROM scratch AS flaggems_provider

FROM ${SUNRISE_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG SUNRISE_PYTHON=/flagos/bin/python
ENV DEBIAN_FRONTEND=noninteractive \
    GEMS_VENDOR=sunrise \
    VLLM_PLUGINS=fl \
    USE_FLAGGEMS=1 \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_VENDOR="" \
    SUNRISE_PYTHON=${SUNRISE_PYTHON} \
    UV_OVERRIDE=/opt/sunrise-stack/native-overrides.txt \
    FL_SUNRISE_STACK_MANIFEST=/opt/sunrise-stack/native-stack.json

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git tmux \
    && rm -rf /var/lib/apt/lists/*

# Repository-root build context. Compiler selection is inherited from the
# published image and must expose its matched FlagTree/Triton and FlagGems.
COPY docker/sunrise/environment.sh docker/sunrise/entrypoint.sh \
    docker/sunrise/runtime_constraints.py docker/sunrise/python310_compat.py \
    docker/sunrise/verify_runtime.py /opt/sunrise-build/
RUN source /opt/sunrise-build/environment.sh \
    && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    && "${SUNRISE_PYTHON}" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 14), sys.version' \
    && "${SUNRISE_PYTHON}" /opt/sunrise-build/runtime_constraints.py capture /opt/sunrise-stack

FROM base AS runtime
ARG VLLM_VERSION=0.32.0
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/refs/tags/v${VLLM_VERSION}
ARG VLLM_SOURCE_SHA256=bf592c132f27bbb5f65cff0bbd2ee873eae722b79c8286934fd5b5cf88b43d61
ARG UV_VERSION=0.13.4
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
ARG PLUGIN_VERSION=0.5.0.dev0

RUN source /opt/sunrise-build/environment.sh \
    && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    && python -m pip install --no-cache-dir --no-deps \
        --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} "uv==${UV_VERSION}" \
    && uv --config-file /opt/sunrise-stack/uv.toml pip install --python "$(command -v python)" \
        --no-cache --no-build-isolation --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        'setuptools>=78.1.1,<81.0.0' wheel 'setuptools-scm>=8' \
        'setuptools-rust>=1.9' packaging jinja2 scikit-build-core==0.11 pybind11 ninja cmake sqlalchemy

# Keep tools/build_rust.py: setup.py imports it even for the empty target.
# On the vendor cp310 stack, apply the explicit third-party compatibility
# build before dependency resolution. The JSON receipt records every patch.
RUN source /opt/sunrise-build/environment.sh \
    && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    && curl --fail --location --retry 5 "${VLLM_SOURCE_URL}" -o /tmp/vllm-source.tar.gz \
    && echo "${VLLM_SOURCE_SHA256}  /tmp/vllm-source.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/vllm-fl-empty-src \
    && tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /opt/vllm-fl-empty-src \
    && test -f /opt/vllm-fl-empty-src/tools/build_rust.py \
    && if python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then \
        python /opt/sunrise-build/python310_compat.py /opt/vllm-fl-empty-src \
            --manifest /opt/sunrise-stack/python310-compat.json; \
    fi \
    && uv --config-file /opt/sunrise-stack/uv.toml pip install --python "$(command -v python)" \
        --no-cache --no-build-isolation --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        -r /opt/vllm-fl-empty-src/requirements/common.txt \
    && VLLM_TARGET_DEVICE=empty VLLM_USE_PRECOMPILED=0 \
        SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" python -m pip wheel -v \
        --no-build-isolation --no-deps --wheel-dir /opt/vllm-empty-wheels /opt/vllm-fl-empty-src \
    && python -m pip install --no-build-isolation --no-deps /opt/vllm-empty-wheels/vllm-*.whl \
    && python /opt/sunrise-build/runtime_constraints.py verify \
        /opt/sunrise-stack --vllm-version "${VLLM_VERSION}"

# The provider is independent of plugin edits. Keep its immutable source install
# in a cached layer and avoid HTTP/2 partial-clone failures through the gateway.
COPY .github/scripts/install_flaggems_vllm.py /opt/sunrise-build/install_flaggems_vllm.py
RUN --mount=type=secret,id=git_config,required=false \
    --mount=type=bind,from=flaggems_provider,target=/opt/flaggems-provider \
    source /opt/sunrise-build/environment.sh \
    && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    && if [ -f /run/secrets/git_config ]; then \
        export GIT_CONFIG_GLOBAL=/run/secrets/git_config; \
    fi \
    && export GIT_CONFIG_COUNT=3 \
        GIT_CONFIG_KEY_0=http.version GIT_CONFIG_VALUE_0=HTTP/1.1 \
        GIT_CONFIG_KEY_1=http.lowSpeedLimit GIT_CONFIG_VALUE_1=1 \
        GIT_CONFIG_KEY_2=http.lowSpeedTime GIT_CONFIG_VALUE_2=60 \
    && if [ -f /opt/flaggems-provider/flaggems-vllm.bundle ]; then \
        provider_commit=$(python /opt/sunrise-build/install_flaggems_vllm.py --sunrise --print-commit) \
        && \
        git bundle list-heads /opt/flaggems-provider/flaggems-vllm.bundle HEAD \
            | grep -Fqx "${provider_commit} HEAD" \
        && export GIT_CONFIG_COUNT=4 \
            GIT_CONFIG_KEY_3=url./opt/flaggems-provider/flaggems-vllm.bundle.insteadOf \
            GIT_CONFIG_VALUE_3=https://github.com/flagos-ai/FlagGems-vllm.git; \
    fi \
    && python /opt/sunrise-build/install_flaggems_vllm.py --sunrise \
    && python /opt/sunrise-build/runtime_constraints.py verify \
        /opt/sunrise-stack --vllm-version "${VLLM_VERSION}"

# Install the requested checkout without resolving a replacement vendor stack.
# A stable build version also works when Docker excludes worktree .git files.
COPY . /opt/vllm-plugin-fl/
RUN source /opt/sunrise-build/environment.sh \
    && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    && SETUPTOOLS_SCM_PRETEND_VERSION="${PLUGIN_VERSION}" \
        python -m pip install --no-build-isolation --no-deps /opt/vllm-plugin-fl \
    && python /opt/sunrise-build/runtime_constraints.py verify \
        /opt/sunrise-stack --vllm-version "${VLLM_VERSION}"

WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "/opt/sunrise-build/entrypoint.sh"]
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]

FROM runtime AS dev
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
RUN source /opt/sunrise-build/environment.sh \
    && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    && uv --config-file /opt/sunrise-stack/uv.toml pip install --python "$(command -v python)" \
        --no-cache --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        pytest pytest-cov pytest-timeout pytest-json-report pre-commit ruff \
    && python /opt/sunrise-build/runtime_constraints.py verify /opt/sunrise-stack

FROM dev AS ci
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
RUN source /opt/sunrise-build/environment.sh \
    && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    && uv --config-file /opt/sunrise-stack/uv.toml pip install --python "$(command -v python)" \
        --no-cache --index-url "${INDEX_URL:-https://pypi.org/simple}" \
        ${EXTRA_INDEX_URL:+--extra-index-url ${EXTRA_INDEX_URL}} \
        decorator pyyaml sqlalchemy 'modelscope>=1.18.1' \
    && python /opt/sunrise-build/runtime_constraints.py verify /opt/sunrise-stack

FROM runtime AS release
