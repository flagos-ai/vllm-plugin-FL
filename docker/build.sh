#!/bin/bash

set -euo pipefail

# ==============================================================================
# Docker Image Build Script
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Version defaults (override via environment variables) ----
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_VERSION="${UV_VERSION:-0.7.12}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"
VLLM_VERSION="${VLLM_VERSION:-0.13.0}"

# ---- Build options ----
PLATFORM="${PLATFORM:-cuda}"
TARGET="dev"
IMAGE_NAME="localhost:5000/vllm-plugin-fl"
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
    --platform PLATFORM    Platform to build (default: ${PLATFORM})
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
    CUDA_VERSION         CUDA version (default: ${CUDA_VERSION})
    UBUNTU_VERSION       Ubuntu version (default: ${UBUNTU_VERSION})
    VLLM_VERSION         vLLM version (default: ${VLLM_VERSION})

EXAMPLES:
    # Build dev image with defaults
    ./build.sh --target dev

    # Build release image with custom CUDA version
    CUDA_VERSION=12.8.1 ./build.sh --target release

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

# ==============================================================================
# Build
# ==============================================================================

# Auto-generate tag if not specified
if [[ -z "${IMAGE_TAG}" ]]; then
    IMAGE_TAG="cuda${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}-py${PYTHON_VERSION}-${TARGET}"
fi

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

msg "Building image: ${FULL_IMAGE}"
msg "  Platform:       ${PLATFORM}"
msg "  Target:         ${TARGET}"
msg "  CUDA:           ${CUDA_VERSION}"
msg "  Ubuntu:         ${UBUNTU_VERSION}"
msg "  Python:         ${PYTHON_VERSION}"
msg "  uv:             ${UV_VERSION}"
msg "  vLLM:           ${VLLM_VERSION}"
msg ""

docker build \
    -f "${SCRIPT_DIR}/${PLATFORM}/Dockerfile" \
    --target "${TARGET}" \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    --build-arg "UV_VERSION=${UV_VERSION}" \
    --build-arg "VLLM_VERSION=${VLLM_VERSION}" \
    --build-arg "INDEX_URL=${INDEX_URL}" \
    --build-arg "EXTRA_INDEX_URL=${EXTRA_INDEX_URL}" \
    ${NO_CACHE} \
    "${EXTRA_BUILD_ARGS[@]+"${EXTRA_BUILD_ARGS[@]}"}" \
    -t "${FULL_IMAGE}" \
    "${PROJECT_ROOT}"

msg "Build complete: ${FULL_IMAGE}"
