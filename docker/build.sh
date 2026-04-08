#!/bin/bash

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

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build the vllm-plugin-FL Docker image.

OPTIONS:
    --platform PLATFORM    Platform to build: cuda, ascend (default: ${PLATFORM})
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

EXAMPLES:
    # Build CUDA dev image
    ./build.sh --target dev

    # Build Ascend CI image for 910b
    ./build.sh --platform ascend --target ci

    # Build Ascend CI image for A3
    CANN_CHIP=a3 ./build.sh --platform ascend --target ci --build-arg SOC_VERSION=ascend910_9391

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
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
    --build-arg "VLLM_VERSION=${VLLM_VERSION}"
)

if [[ "${PLATFORM}" == "cuda" ]]; then
    BUILD_ARGS+=(
        --build-arg "CUDA_VERSION=${CUDA_VERSION}"
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
    )
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="cann${CANN_VERSION}-${CANN_CHIP}-ubuntu${UBUNTU_VERSION}-py${PYTHON_VERSION}-${TARGET}"
    fi
else
    err "Unknown platform '${PLATFORM}'. Must be 'cuda' or 'ascend'."
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
fi
msg "  Ubuntu:         ${UBUNTU_VERSION}"
msg "  Python:         ${PYTHON_VERSION}"
msg "  vLLM:           ${VLLM_VERSION}"
msg ""

docker build \
    -f "${DOCKERFILE}" \
    --target "${TARGET}" \
    "${BUILD_ARGS[@]}" \
    ${NO_CACHE} \
    "${EXTRA_BUILD_ARGS[@]+"${EXTRA_BUILD_ARGS[@]}"}" \
    -t "${FULL_IMAGE}" \
    "${BUILD_CONTEXT}"

msg "Build complete: ${FULL_IMAGE}"
