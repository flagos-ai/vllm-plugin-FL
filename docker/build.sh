#!/bin/bash


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

set -euo pipefail

# ==============================================================================
# Docker Image Build Script
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Version defaults (override via environment variables) ----
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_VERSION="${UV_VERSION:-0.7.12}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"
CANN_VERSION="${CANN_VERSION:-8.5.1}"
CANN_CHIP="${CANN_CHIP:-910b}"
HYGON_BASE_IMAGE="${HYGON_BASE_IMAGE:-harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.20.0-ubuntu22.04-dtk26.04-py3.10-MiniCPM-V-4.6}"
HYGON_VLLM_VERSION="${HYGON_VLLM_VERSION:-0.20.2}"
HYGON_DTK_VERSION="${HYGON_DTK_VERSION:-26.04}"
HYGON_PYTHON_VERSION="${HYGON_PYTHON_VERSION:-3.10}"
HYGON_RUNTIME_LIB_DIR="${HYGON_RUNTIME_LIB_DIR:-/opt/hyhal/lib}"
HYGON_RUNTIME_ROOTS="${HYGON_RUNTIME_ROOTS:-/opt/dtk:/opt/hyhal:/usr/local/hyhal}"
FLAGGEMS_VERSION="${FLAGGEMS_VERSION:-62d70b9e858ec407572153ee8cdf65cc24a637d5}"
VLLM_PLUGIN_FL_VERSION="${VLLM_PLUGIN_FL_VERSION:-ffa2ee3eb3831f3873dd0966d12fc8e0b4e6e3d4}"

# ---- Build options ----
PLATFORM="${PLATFORM:-cuda}"
TARGET="dev"
IMAGE_NAME="harbor.baai.ac.cn/flagscale/vllm-plugin-fl"
IMAGE_TAG=""
INDEX_URL="${INDEX_URL:-}"
EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-}"
NO_CACHE=""
EXTRA_BUILD_ARGS=()

# ==============================================================================
# Helper functions
# ==============================================================================

err() {
    printf "ERROR: %s\n" "$1" >&2
    exit 1
}

msg() {
    printf ">>> %s\n" "$1"
}

cleanup_hygon_runtime_overlay() {
    if [[ -n "${HYGON_RUNTIME_OVERLAY:-}" ]]; then
        rm -rf "${HYGON_RUNTIME_OVERLAY}"
    fi
}

stage_hygon_runtime_files() {
    local src rel
    for src in "$@"; do
        [[ -e "${src}" ]] || continue
        rel="${src#/}"
        mkdir -p "${HYGON_RUNTIME_OVERLAY}/$(dirname "${rel}")"
        cp -aL "${src}" "${HYGON_RUNTIME_OVERLAY}/${rel}"
    done
}

prepare_hygon_runtime_overlay() {
    HYGON_RUNTIME_OVERLAY="${SCRIPT_DIR}/hygon/.hygon-runtime"
    rm -rf "${HYGON_RUNTIME_OVERLAY}"
    mkdir -p "${HYGON_RUNTIME_OVERLAY}/opt/hyhal/lib"

    mapfile -t HYGON_RUNTIME_LIBS < <(
        find "${HYGON_RUNTIME_LIB_DIR}" -maxdepth 1 -name 'librocm_smi64.so*' -print 2>/dev/null | sort
    )
    if [[ "${#HYGON_RUNTIME_LIBS[@]}" -eq 0 ]]; then
        err "Hygon runtime libraries not found: ${HYGON_RUNTIME_LIB_DIR}/librocm_smi64.so*"
    fi

    stage_hygon_runtime_files "${HYGON_RUNTIME_LIBS[@]}"

    local runtime_roots=()
    local existing_runtime_roots=()
    local root
    IFS=: read -r -a runtime_roots <<< "${HYGON_RUNTIME_ROOTS}"
    for root in "${runtime_roots[@]}"; do
        if [[ -d "${root}" ]]; then
            existing_runtime_roots+=("${root}")
        fi
    done
    if [[ "${#existing_runtime_roots[@]}" -eq 0 ]]; then
        err "Hygon runtime roots not found: ${HYGON_RUNTIME_ROOTS}"
    fi

    mapfile -t HYGON_HSA_RUNTIME_FILES < <(
        find "${existing_runtime_roots[@]}" \( -type f -o -type l \) \
            \( -name 'libhsa-runtime64.so*' \
            -o -name 'hsa-runtime64*cmake' \
            -o -path '*/hsa-runtime64/*.cmake' \) \
            -print 2>/dev/null | sort -u
    )
    if [[ "${#HYGON_HSA_RUNTIME_FILES[@]}" -eq 0 ]]; then
        err "Hygon HSA runtime files not found under: ${HYGON_RUNTIME_ROOTS}"
    fi
    stage_hygon_runtime_files "${HYGON_HSA_RUNTIME_FILES[@]}"

    mapfile -t HYGON_ROCM_SMI_CMAKE_FILES < <(
        find "${existing_runtime_roots[@]}" \( -type f -o -type l \) \
            \( -name 'rocm_smi*cmake' -o -path '*/rocm_smi/*.cmake' \) \
            -print 2>/dev/null | sort -u
    )
    if [[ "${#HYGON_ROCM_SMI_CMAKE_FILES[@]}" -gt 0 ]]; then
        stage_hygon_runtime_files "${HYGON_ROCM_SMI_CMAKE_FILES[@]}"
    fi

    trap cleanup_hygon_runtime_overlay EXIT
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build the vllm-plugin-FL Docker image.

OPTIONS:
    --platform PLATFORM    Platform to build: cuda, ascend, hygon (default: ${PLATFORM})
    --target TARGET        Build target: dev, ci, release (default: ${TARGET})
    --image-name NAME      Image name (default: ${IMAGE_NAME})
    --image-tag TAG        Image tag (default: auto-generated)
    --index-url URL        PyPI index URL (for custom mirrors)
    --extra-index-url URL  Extra PyPI index URL
    --build-arg K=V        Pass build-arg to docker (can be repeated)
    --no-cache             Build without cache
    --help                 Show this help message

VERSIONS (override via environment variables):
    PYTHON_VERSION       Python version (default: ${PYTHON_VERSION})
    UV_VERSION           uv version (default: ${UV_VERSION})
    VLLM_VERSION         vLLM version (default: ${VLLM_VERSION})
    UBUNTU_VERSION       Ubuntu version (default: ${UBUNTU_VERSION})
  CUDA:
    CUDA_VERSION         CUDA version (default: ${CUDA_VERSION})
  Ascend:
    CANN_VERSION         CANN version (default: ${CANN_VERSION})
    CANN_CHIP            CANN chip: 910b, a3 (default: ${CANN_CHIP})
  Hygon:
    HYGON_BASE_IMAGE     Base image (default: ${HYGON_BASE_IMAGE})
    HYGON_VLLM_VERSION   vLLM version installed in empty mode (default: ${HYGON_VLLM_VERSION})
    HYGON_DTK_VERSION    DTK version used in generated image tag (default: ${HYGON_DTK_VERSION})
    HYGON_PYTHON_VERSION Python version in Hygon base image tag (default: ${HYGON_PYTHON_VERSION})
    HYGON_RUNTIME_LIB_DIR Hygon runtime library source dir (default: ${HYGON_RUNTIME_LIB_DIR})
    HYGON_RUNTIME_ROOTS  Colon-separated roots for Hygon runtime overlay files (default: ${HYGON_RUNTIME_ROOTS})
    FLAGGEMS_VERSION     FlagGems git ref (default: ${FLAGGEMS_VERSION})
    VLLM_PLUGIN_FL_VERSION vllm-plugin-FL git ref (default: ${VLLM_PLUGIN_FL_VERSION})

EXAMPLES:
    # Build CUDA dev image
    ./build.sh --target dev

    # Build Ascend CI image for 910b
    ./build.sh --platform ascend --target ci

    # Build Ascend CI image for A3
    CANN_CHIP=a3 ./build.sh --platform ascend --target ci --build-arg SOC_VERSION=ascend910_9391

    # Build Hygon CI image
    ./build.sh --platform hygon --target ci

    # Build with custom PyPI mirror
    ./build.sh --target dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple

    # Build with extra docker build args
    ./build.sh --target dev --build-arg HTTP_PROXY=http://proxy:8080
EOF
    exit 0
}

# ==============================================================================
# Parse arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            PLATFORM="$2"; shift 2 ;;
        --target)
            TARGET="$2"; shift 2 ;;
        --image-name)
            IMAGE_NAME="$2"; shift 2 ;;
        --image-tag)
            IMAGE_TAG="$2"; shift 2 ;;
        --index-url)
            INDEX_URL="$2"; shift 2 ;;
        --extra-index-url)
            EXTRA_INDEX_URL="$2"; shift 2 ;;
        --build-arg)
            EXTRA_BUILD_ARGS+=("--build-arg" "$2"); shift 2 ;;
        --no-cache)
            NO_CACHE="--no-cache"; shift ;;
        --help|-h)
            usage ;;
        *)
            err "Unknown argument: $1. Use --help for usage." ;;
    esac
done

# ==============================================================================
# Validate
# ==============================================================================

if [[ "${TARGET}" != "dev" && "${TARGET}" != "ci" && "${TARGET}" != "release" ]]; then
    err "Invalid target '${TARGET}'. Must be 'dev', 'ci', or 'release'."
fi

if ! command -v docker &>/dev/null; then
    err "docker is not installed or not in PATH."
fi

DOCKERFILE="${SCRIPT_DIR}/${PLATFORM}/Dockerfile"
if [[ ! -f "${DOCKERFILE}" ]]; then
    err "Dockerfile not found: ${DOCKERFILE}"
fi

# ==============================================================================
# Build
# ==============================================================================

# Build context is the platform-specific directory (e.g. docker/ascend/)
BUILD_CONTEXT="${SCRIPT_DIR}/${PLATFORM}"

# Platform-specific build args and auto-tag
BUILD_ARGS=(
    --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}"
)

if [[ "${PLATFORM}" == "cuda" ]]; then
    BUILD_ARGS+=(
        --build-arg "CUDA_VERSION=${CUDA_VERSION}"
        --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
        --build-arg "VLLM_VERSION=${VLLM_VERSION}"
        --build-arg "UV_VERSION=${UV_VERSION}"
        --build-arg "INDEX_URL=${INDEX_URL}"
        --build-arg "EXTRA_INDEX_URL=${EXTRA_INDEX_URL}"
    )
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="cuda${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}-py${PYTHON_VERSION}-${TARGET}"
    fi
elif [[ "${PLATFORM}" == "ascend" ]]; then
    BUILD_ARGS+=(
        --build-arg "CANN_VERSION=${CANN_VERSION}"
        --build-arg "CANN_CHIP=${CANN_CHIP}"
        --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
        --build-arg "VLLM_VERSION=${VLLM_VERSION}"
    )
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="cann${CANN_VERSION}-${CANN_CHIP}-ubuntu${UBUNTU_VERSION}-py${PYTHON_VERSION}-${TARGET}"
    fi
elif [[ "${PLATFORM}" == "hygon" ]]; then
    PYTHON_VERSION="${HYGON_PYTHON_VERSION}"
    VLLM_VERSION="${HYGON_VLLM_VERSION}"
    BUILD_ARGS+=(
        --build-arg "HYGON_BASE_IMAGE=${HYGON_BASE_IMAGE}"
        --build-arg "PYTHON_VERSION=${HYGON_PYTHON_VERSION}"
        --build-arg "VLLM_VERSION=${HYGON_VLLM_VERSION}"
        --build-arg "INDEX_URL=${INDEX_URL}"
        --build-arg "EXTRA_INDEX_URL=${EXTRA_INDEX_URL}"
        --build-arg "FLAGGEMS_VERSION=${FLAGGEMS_VERSION}"
        --build-arg "VLLM_PLUGIN_FL_VERSION=${VLLM_PLUGIN_FL_VERSION}"
    )
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="hygon-vllm${VLLM_VERSION}-dtk${HYGON_DTK_VERSION}-py${HYGON_PYTHON_VERSION}-${TARGET}"
    fi
else
    err "Unknown platform '${PLATFORM}'. Must be 'cuda', 'ascend', or 'hygon'."
fi

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

msg "Building image: ${FULL_IMAGE}"
msg "  Platform:       ${PLATFORM}"
msg "  Target:         ${TARGET}"
if [[ "${PLATFORM}" == "cuda" ]]; then
    msg "  CUDA:           ${CUDA_VERSION}"
elif [[ "${PLATFORM}" == "ascend" ]]; then
    msg "  CANN:           ${CANN_VERSION}"
    msg "  Chip:           ${CANN_CHIP}"
elif [[ "${PLATFORM}" == "hygon" ]]; then
    msg "  DTK:            ${HYGON_DTK_VERSION}"
    msg "  Hygon Python:   ${HYGON_PYTHON_VERSION}"
    msg "  Base image:     ${HYGON_BASE_IMAGE}"
    msg "  Runtime libs:   ${HYGON_RUNTIME_LIB_DIR}"
    msg "  FlagGems:       ${FLAGGEMS_VERSION}"
    msg "  Plugin:         ${VLLM_PLUGIN_FL_VERSION}"
fi
msg "  Ubuntu:         ${UBUNTU_VERSION}"
msg "  Python:         ${PYTHON_VERSION}"
msg "  vLLM:           ${VLLM_VERSION}"
msg ""

if [[ "${PLATFORM}" == "hygon" ]]; then
    prepare_hygon_runtime_overlay
fi

docker build \
    -f "${DOCKERFILE}" \
    --target "${TARGET}" \
    "${BUILD_ARGS[@]}" \
    ${NO_CACHE} \
    "${EXTRA_BUILD_ARGS[@]+"${EXTRA_BUILD_ARGS[@]}"}" \
    -t "${FULL_IMAGE}" \
    "${BUILD_CONTEXT}"

msg "Build complete: ${FULL_IMAGE}"
