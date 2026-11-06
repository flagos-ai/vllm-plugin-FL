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

# FlagOS 2.2 Hygon release image: DTK 26.04, Python 3.10, vendor Torch 2.9,
# FlagGems 5.4.0 and FlagTree 0.7.0. Upstream vLLM 0.32 requires Python >= 3.11,
# so .github/scripts/hygon/patch_vllm_py310.py adapts the source before the
# empty wheel is built. See docs/usage/hygon.md#vllm-032-experimental-path.
ARG HYGON_BASE_IMAGE=harbor.baai.ac.cn/flagos-app/vllm0.24.0-hygon-dtk26.04:2.2.0-0.3.0@sha256:40fcd205387f48339d6b6e143ccc9e9c1566e37ba26e73f07f0d9b13465c70f5

# An optional named context overrides this empty stage with a wheel-only directory.
FROM scratch AS vllm_wheels

FROM ${HYGON_BASE_IMAGE} AS base
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG VLLM_VERSION=0.32.0
ARG VLLM_SOURCE_SHA=5df63a7203cdb9328d824a27b682e77c7b9928be
ARG PLUGIN_SOURCE_SHA
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL

# Keep the vendor Python environment (/flagos), Torch, FlagGems, FlagTree and SDK
# paths; the base image already exports LD_LIBRARY_PATH for DTK and HyHAL, and
# PYTHONPATH=/opt/flagtree, which provides the FlagTree (Triton) distribution.
# LD_LIBRARY_PATH is mirrored from /etc/profile.d/vendor.sh: GitHub Actions runs
# container steps with `sh`, which ignores BASH_ENV, so a profile-only export
# leaves `import torch` without libgalaxyhip.so.5. Keep both in sync.
ENV DEBIAN_FRONTEND=noninteractive \
    ROCM_PATH=/opt/dtk \
    HIP_PATH=/opt/dtk/hip \
    DTKROOT=/opt/dtk \
    HYHAL_PATH=/opt/hyhal \
    LD_LIBRARY_PATH="/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/lib:/opt/dtk/.hyhal/lib64:" \
    GEMS_VENDOR=hygon \
    USE_FLAGGEMS=1 \
    VLLM_PLUGINS=fl \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_VENDOR="" \
    VIRTUAL_ENV=/flagos \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    HYGON_FLAGGEMS_VERSION=5.4.0 \
    HYGON_FLAGTREE_VERSION=0.7.0+hcu3.6

# Repository-root context; copy only build inputs, never local credentials or .git.
COPY .github/scripts/hygon/verify_install.py /opt/hygon/verify_install.py
COPY .github/scripts/hygon/patch_vllm_py310.py /opt/hygon/patch_vllm_py310.py
COPY .github/scripts/install_flaggems_vllm.py /opt/hygon/install_flaggems_vllm.py
# The base image supplies FlagTree through PYTHONPATH; also expose it through a
# site-packages path file so isolated (python -I) verification can find it.
RUN printf '%s\n' /opt/flagtree \
        > /flagos/lib/python3.10/site-packages/flagtree-opt.pth \
    && python -I -B /opt/hygon/verify_install.py \
        --static-only --write-vendor-snapshot /opt/hygon/vendor-base.json \
    && python -I -B -m pip uninstall -y vllm vllm-plugin-fl

ARG VLLM_WHEEL_SHA256=""

# Optional BuildKit secret git_config supplies a git-only proxy; it is never
# copied to the image or passed in a build ARG/ENV. Direct Git needs no secret.
# Default: build upstream from the released Git revision. An explicitly hashed
# ordinary empty wheel can instead be provided by the vllm_wheels named context.
# Empty mode has no CUDA/HIP _C;
# optional Rust failures remain visible and precompiled downloads are disabled.
RUN --mount=type=secret,id=git_config,target=/run/secrets/git_config,required=false \
    --mount=type=bind,from=vllm_wheels,target=/run/vllm_wheels,ro \
    test "$VLLM_VERSION" = 0.32.0 \
    && mkdir -p /opt/hygon/wheels /opt/hygon/cargo /opt/hygon/rustup \
    && shopt -s nullglob \
    && wheels=(/run/vllm_wheels/*.whl) \
    && if (( ${#wheels[@]} )); then \
        [[ ${#wheels[@]} = 1 && "$VLLM_WHEEL_SHA256" =~ ^[0-9a-f]{64}$ ]] \
        && test "${wheels[0]##*/}" = 'vllm-0.32.0+empty-cp310-cp310-linux_x86_64.whl' \
        && printf '%s  %s\n' "$VLLM_WHEEL_SHA256" "${wheels[0]}" | sha256sum -c - \
        && python -I -S -B -c 'import email,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); names=z.namelist(); entries=[n for n in names if n.endswith(".dist-info/METADATA")]; assert len(entries)==1; m=email.message_from_bytes(z.read(entries[0])); assert m["Name"]=="vllm" and m["Version"]=="0.32.0+empty"; assert not any(n.startswith("vllm/_C") and n.endswith((".so",".pyd")) for n in names)' "${wheels[0]}" \
        && cp "${wheels[0]}" /opt/hygon/wheels/; \
    else \
        test -z "$VLLM_WHEEL_SHA256" \
        && GIT_CONFIG_GLOBAL=/run/secrets/git_config git clone --depth 1 --branch "v${VLLM_VERSION}" \
            https://github.com/vllm-project/vllm.git /opt/vllm-src/vllm \
        && test "$(git -C /opt/vllm-src/vllm rev-parse HEAD)" = "$VLLM_SOURCE_SHA" \
        && python -I -B /opt/hygon/patch_vllm_py310.py /opt/vllm-src/vllm \
        && env -u VLLM_VENDOR \
            -u SETUPTOOLS_SCM_PRETEND_VERSION -u SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM \
            -u VLLM_PRECOMPILED_WHEEL_LOCATION \
            VLLM_TARGET_DEVICE=empty VLLM_USE_PRECOMPILED=0 \
            VLLM_USE_PRECOMPILED_RUST=0 VLLM_DOCKER_BUILD_CONTEXT=0 \
            VLLM_REQUIRE_RUST_FRONTEND=0 VLLM_VERSION_OVERRIDE=0.32.0+empty \
            CARGO_NET_OFFLINE=true \
            RUSTUP_AUTO_INSTALL=0 CARGO_HOME=/opt/hygon/cargo RUSTUP_HOME=/opt/hygon/rustup \
            python -I -B -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
            --wheel-dir /opt/hygon/wheels /opt/vllm-src/vllm; \
    fi \
    && python -I -B -m pip install --no-deps --no-index --no-compile \
        /opt/hygon/wheels/vllm-*.whl

COPY pyproject.toml setup.py README.md LICENSE /opt/vllm-src/vllm-plugin-fl/
COPY vllm_fl /opt/vllm-src/vllm-plugin-fl/vllm_fl
# docker/build.sh supplies an archive of PLUGIN_SOURCE_SHA for these COPY inputs.
# The installed package uses ordinary wheels, without an editable finder.
RUN --mount=type=secret,id=git_config,target=/run/secrets/git_config,required=false \
    [[ "$PLUGIN_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] \
    && env -u VLLM_VENDOR -u SETUPTOOLS_SCM_PRETEND_VERSION \
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="0.0.0+g${PLUGIN_SOURCE_SHA}" \
        python -I -B -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
        --wheel-dir /opt/hygon/wheels /opt/vllm-src/vllm-plugin-fl \
    && python -I -B -m pip install --no-deps --no-index --no-compile \
        /opt/hygon/wheels/vllm_plugin_fl-*.whl \
    && GIT_CONFIG_GLOBAL=/run/secrets/git_config \
        PIP_INDEX_URL="${INDEX_URL}" PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-}" \
        python -I -B /opt/hygon/install_flaggems_vllm.py \
    && python -I -B /opt/hygon/verify_install.py --static-only \
        --compare-vendor-snapshot /opt/hygon/vendor-base.json
LABEL org.opencontainers.image.revision=${PLUGIN_SOURCE_SHA}
WORKDIR /workspace

FROM base AS ci
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
# pytest-xdist is required: tests/run.py drives unit tests with ``pytest -n``.
# Baking it in keeps setup.sh's fallback install from needing PyPI on the runner.
RUN python -I -B -m pip install \
        ${INDEX_URL:+--index-url "${INDEX_URL}"} \
        ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
        pytest pytest-cov pytest-timeout pytest-json-report pytest-xdist \
        ruff pre-commit \
    && python -I -B /opt/hygon/verify_install.py --static-only \
        --compare-vendor-snapshot /opt/hygon/vendor-base.json
WORKDIR /workspace

FROM ci AS dev
WORKDIR /workspace

FROM base AS release
WORKDIR /workspace
