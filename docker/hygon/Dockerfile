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

ARG HYGON_BASE_IMAGE=harbor.baai.ac.cn/plugin/hygon-dtk26.04-tree0.6.0hcu3.6-triton3.6.0-cxnone-plugin0.3.0-vllm0.24.0-cp310-pt2100-x64:202609041747@sha256:d2530485457ede1d17dbb26704a3d4408d760ff3e363402599e1738dc5828807

# An optional named context overrides this empty stage with a wheel-only directory.
FROM scratch AS vllm_wheels

FROM ${HYGON_BASE_IMAGE} AS base
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG VLLM_VERSION=0.28.0
ARG VLLM_SOURCE_SHA=2cf0a6915ce544dc493a0990f2ea38d81601128a
ARG PLUGIN_SOURCE_SHA
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL

# Keep vendor packages and SDK files; explicitly provide their runtime paths.
ENV DEBIAN_FRONTEND=noninteractive \
    ROCM_PATH=/opt/dtk \
    HIP_PATH=/opt/dtk/hip \
    DTKROOT=/opt/dtk \
    HYHAL_PATH=/opt/hyhal \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/opt/mpi/lib:/opt/hwloc/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib \
    GEMS_VENDOR=hygon \
    USE_FLAGGEMS=1 \
    VLLM_PLUGINS=fl \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_VENDOR="" \
    PYTHONPATH=""

ENV VIRTUAL_ENV=/opt/vllm-fl/venv \
    PATH=/opt/vllm-fl/venv/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/hyhal/bin:${PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    HYGON_FLAGGEMS_VERSION=0.1.0 \
    HYGON_FLAGTREE_VERSION=0.6.0+hcu3.6

# Repository-root context; copy only build inputs, never local credentials or .git.
COPY .github/scripts/hygon/verify_install.py /opt/hygon/verify_install.py
RUN /usr/bin/python3.10 -I -B /opt/hygon/verify_install.py \
        --static-only --write-vendor-snapshot /opt/hygon/vendor-base.json \
    && /usr/bin/python3.10 -I -B -m pip uninstall -y vllm vllm-plugin-fl \
    && /usr/bin/python3.10 -I -B -m virtualenv \
        --python /usr/bin/python3.10 --symlinks --system-site-packages \
        --no-download --no-periodic-update --seeder app-data \
        --no-setuptools --pip 26.0.1 --app-data /opt/hygon/app-data "$VIRTUAL_ENV"

# Keep the image's ordinary Torch/Gems/Tree/Triton and existing build tools.
# Explicit Python-only tooling and runtime wheels: dependency resolution cannot
# replace the vendor Torch, FlagTree/Triton, FlagGems or SDK.
RUN python -I -B -m pip install --no-deps \
        ${INDEX_URL:+--index-url "${INDEX_URL}"} \
        ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
        setuptools-rust \
    && python -I -B -m pip install --no-deps \
        ${INDEX_URL:+--index-url "${INDEX_URL}"} \
        ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
        openai==2.25.0 huggingface-hub==1.27.0 fastapi==0.136.3 \
        starlette==1.7.0 llguidance==1.7.6 mistral-common==1.12.0 \
        model-hosting-container-standards==0.1.16 outlines-core==0.2.14 \
        prometheus-fastapi-instrumentator==8.1.0 xgrammar==0.2.8 \
        apache-tvm-ffi==0.1.11 click==8.4.2 hf-xet==1.5.2

ARG VLLM_WHEEL_SHA256=""

# Optional BuildKit secret git_config supplies a git-only proxy; it is never
# copied to the image or passed in a build ARG/ENV. Direct Git needs no secret.
# Default: build upstream from the released Git revision. An explicitly hashed
# ordinary empty wheel can instead be provided by the vllm_wheels named context.
# Empty mode has no CUDA/HIP _C;
# optional Rust failures remain visible and precompiled downloads are disabled.
RUN --mount=type=secret,id=git_config,target=/run/secrets/git_config,required=false \
    --mount=type=bind,from=vllm_wheels,target=/run/vllm_wheels,ro \
    test "$VLLM_VERSION" = 0.28.0 \
    && mkdir -p /opt/hygon/wheels /opt/hygon/cargo /opt/hygon/rustup \
    && shopt -s nullglob \
    && wheels=(/run/vllm_wheels/*.whl) \
    && if (( ${#wheels[@]} )); then \
        [[ ${#wheels[@]} = 1 && "$VLLM_WHEEL_SHA256" =~ ^[0-9a-f]{64}$ ]] \
        && test "${wheels[0]##*/}" = 'vllm-0.28.0+empty-cp310-cp310-linux_x86_64.whl' \
        && printf '%s  %s\n' "$VLLM_WHEEL_SHA256" "${wheels[0]}" | sha256sum -c - \
        && python -I -S -B -c 'import email,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); names=z.namelist(); entries=[n for n in names if n.endswith(".dist-info/METADATA")]; assert len(entries)==1; m=email.message_from_bytes(z.read(entries[0])); assert m["Name"]=="vllm" and m["Version"]=="0.28.0+empty"; assert not any(n.startswith("vllm/_C") and n.endswith((".so",".pyd")) for n in names)' "${wheels[0]}" \
        && cp "${wheels[0]}" /opt/hygon/wheels/; \
    else \
        test -z "$VLLM_WHEEL_SHA256" \
        && GIT_CONFIG_GLOBAL=/run/secrets/git_config git clone --depth 1 --branch "v${VLLM_VERSION}" \
            https://github.com/vllm-project/vllm.git /opt/vllm-src/vllm \
        && test "$(git -C /opt/vllm-src/vllm rev-parse HEAD)" = "$VLLM_SOURCE_SHA" \
        && env -u VLLM_VENDOR -u VLLM_VERSION_OVERRIDE \
            -u SETUPTOOLS_SCM_PRETEND_VERSION -u SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM \
            -u VLLM_PRECOMPILED_WHEEL_LOCATION \
            VLLM_TARGET_DEVICE=empty VLLM_USE_PRECOMPILED=0 \
            VLLM_USE_PRECOMPILED_RUST=0 VLLM_DOCKER_BUILD_CONTEXT=0 \
            VLLM_REQUIRE_RUST_FRONTEND=0 CARGO_NET_OFFLINE=true \
            RUSTUP_AUTO_INSTALL=0 CARGO_HOME=/opt/hygon/cargo RUSTUP_HOME=/opt/hygon/rustup \
            python -I -B -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
            --wheel-dir /opt/hygon/wheels /opt/vllm-src/vllm; \
    fi \
    && python -I -B -m pip install --no-deps --no-index --no-compile \
        /opt/hygon/wheels/vllm-*.whl

COPY pyproject.toml setup.py README.md LICENSE /opt/vllm-src/vllm-plugin-fl/
COPY vllm_fl /opt/vllm-src/vllm-plugin-fl/vllm_fl
# The version is explicitly bound to the build wrapper's actual Git HEAD;
# no source tree or editable finder is used by the installed package.
RUN [[ "$PLUGIN_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] \
    && env -u VLLM_VENDOR -u SETUPTOOLS_SCM_PRETEND_VERSION \
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="0.0.0+g${PLUGIN_SOURCE_SHA}" \
        python -I -B -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
        --wheel-dir /opt/hygon/wheels /opt/vllm-src/vllm-plugin-fl \
    && python -I -B -m pip install --no-deps --no-index --no-compile \
        /opt/hygon/wheels/vllm_plugin_fl-*.whl \
    && python -I -B /opt/hygon/verify_install.py --static-only \
        --compare-vendor-snapshot /opt/hygon/vendor-base.json
LABEL org.opencontainers.image.revision=${PLUGIN_SOURCE_SHA}
WORKDIR /workspace

FROM base AS ci
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
RUN python -I -B -m pip install \
        ${INDEX_URL:+--index-url "${INDEX_URL}"} \
        ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
        pytest pytest-cov pytest-timeout pytest-json-report ruff pre-commit \
    && python -I -B /opt/hygon/verify_install.py --static-only \
        --compare-vendor-snapshot /opt/hygon/vendor-base.json
WORKDIR /workspace

FROM ci AS dev
WORKDIR /workspace

FROM base AS release
WORKDIR /workspace
