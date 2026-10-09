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
METAX_BASE_IMAGE="${METAX_BASE_IMAGE:-harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:vllm-metax-0.20.0-maca.ai3.7.0.107-torch2.8-py312-ubuntu22.04-amd64}"
METAX_PYTHON_VERSION="${METAX_PYTHON_VERSION:-3.12}"
METAX_PYTHON_TAG="${METAX_PYTHON_TAG:-py312}"
METAX_MACA_VERSION="${METAX_MACA_VERSION:-3.7.0.107}"
METAX_VLLM_VERSION="${METAX_VLLM_VERSION:-0.20.2}"
MUSA_BASE_IMAGE="${MUSA_BASE_IMAGE:-registry.mthreads.com/mcconline/inference/vllm:v0.20.2-ph1-4.3.5-torch2.7.1-v1.1.0}"
MUSA_VERSION="${MUSA_VERSION:-4.3.5}"
MUSA_VLLM_VERSION="${MUSA_VLLM_VERSION:-0.20.2}"
MUSA_PYTHON_VERSION="${MUSA_PYTHON_VERSION:-3.10}"
MUSA_TORCH_VERSION="${MUSA_TORCH_VERSION:-2.7.1}"
MUSA_FLAGGEMS_VERSION="${MUSA_FLAGGEMS_VERSION:-5.0.0}"
ASCEND_VLLM_VERSION="${ASCEND_VLLM_VERSION:-0.20.2}"
ASCEND_BASE_IMAGE="${ASCEND_BASE_IMAGE:-quay.io/ascend/vllm-ascend:v0.20.2rc1-a3}"
ASCEND_FLAGGEMS_VERSION="${ASCEND_FLAGGEMS_VERSION:-3e6528cf04f5f964a7b0fa6628de6f0410dbfd02}"
ENFLAME_BASE_IMAGE="${ENFLAME_BASE_IMAGE:-harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-enflame-gems_5.4.0.dev0-sglang_0.5.11-sglang_plugin_0.1.0-cx_0.13.0-python_3.12.8-torch_gcu_2.11.0_3.8.20260713-pcp_tops3.8.20260714-gpu_s60-arc_amd64-driver_1.9.10:202608141853}"
ENFLAME_DRIVER_VERSION="${ENFLAME_DRIVER_VERSION:-1.9.10}"
ENFLAME_PYTHON_VERSION="${ENFLAME_PYTHON_VERSION:-3.12}"
ENFLAME_VLLM_VERSION="${ENFLAME_VLLM_VERSION:-0.24.0}"
KUNLUNXIN_BASE_IMAGE="${KUNLUNXIN_BASE_IMAGE:-harbor.baai.ac.cn/plugin/xvllm-ubuntu2204-py310-torch29-0200:v20.0.10.0}"
KUNLUNXIN_DRIVER_VERSION="${KUNLUNXIN_DRIVER_VERSION:-5.0.21.43}"
KUNLUNXIN_PYTHON_VERSION="${KUNLUNXIN_PYTHON_VERSION:-3.10}"
KUNLUNXIN_TORCH_VERSION="${KUNLUNXIN_TORCH_VERSION:-2.9.0}"
KUNLUNXIN_VLLM_VERSION="${KUNLUNXIN_VLLM_VERSION:-0.20.2}"
KUNLUNXIN_FLAGGEMS_REF="${KUNLUNXIN_FLAGGEMS_REF:-v5.0.0}"
KUNLUNXIN_FLAGCX_REF="${KUNLUNXIN_FLAGCX_REF:-v0.13.0}"
KUNLUNXIN_PLUGIN_FL_REF="${KUNLUNXIN_PLUGIN_FL_REF:-38e7dbc20197e2db742c4e4c9687d36ea4df9900}"
HYGON_BASE_IMAGE="${HYGON_BASE_IMAGE:-harbor.baai.ac.cn/plugin/hygon-dtk26.04-tree0.6.0hcu3.6-triton3.6.0-cxnone-plugin0.3.0-vllm0.24.0-cp310-pt2100-x64:202609041747@sha256:d2530485457ede1d17dbb26704a3d4408d760ff3e363402599e1738dc5828807}"
HYGON_VLLM_VERSION="${HYGON_VLLM_VERSION:-0.28.0}"
HYGON_DTK_VERSION="${HYGON_DTK_VERSION:-26.04}"
HYGON_PYTHON_VERSION="${HYGON_PYTHON_VERSION:-3.10}"
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
BUILD_SECRET_ARGS=()

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
    --platform PLATFORM    Platform to build: cuda, ascend, hygon, metax, musa, enflame, kunlunxin (default: ${PLATFORM})
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
    ASCEND_VLLM_VERSION  vLLM version in the validated image (default: ${ASCEND_VLLM_VERSION})
    ASCEND_BASE_IMAGE    Validated Ascend vLLM base image (default: ${ASCEND_BASE_IMAGE})
    ASCEND_FLAGGEMS_VERSION FlagGems git ref for Ascend (default: ${ASCEND_FLAGGEMS_VERSION})
  MetaX:
    METAX_BASE_IMAGE     Base image (default: ${METAX_BASE_IMAGE})
    METAX_MACA_VERSION   MACA version used in generated image tag (default: ${METAX_MACA_VERSION})
    METAX_PYTHON_VERSION Python version used in generated image tag (default: ${METAX_PYTHON_VERSION})
    METAX_PYTHON_TAG     Python tag fragment used in generated image tag (default: ${METAX_PYTHON_TAG})
    METAX_VLLM_VERSION   vLLM version installed in empty mode (default: ${METAX_VLLM_VERSION})
  MUSA:
    MUSA_BASE_IMAGE      Moore Threads base image (default: ${MUSA_BASE_IMAGE})
    MUSA_VERSION         MUSA version used in image tag (default: ${MUSA_VERSION})
    MUSA_VLLM_VERSION    vLLM empty-mode version (default: ${MUSA_VLLM_VERSION})
    MUSA_PYTHON_VERSION  Python version in base image (default: ${MUSA_PYTHON_VERSION})
    MUSA_TORCH_VERSION   PyTorch version in base image (default: ${MUSA_TORCH_VERSION})
    MUSA_FLAGGEMS_VERSION FlagGems version in base image (default: ${MUSA_FLAGGEMS_VERSION})
  Enflame:
    ENFLAME_BASE_IMAGE     Base image (default: ${ENFLAME_BASE_IMAGE})
    ENFLAME_DRIVER_VERSION Driver version used in generated image tag (default: ${ENFLAME_DRIVER_VERSION})
    ENFLAME_PYTHON_VERSION Python version in the base image (default: ${ENFLAME_PYTHON_VERSION})
    ENFLAME_VLLM_VERSION   vLLM version in the base image (default: ${ENFLAME_VLLM_VERSION})
  Kunlunxin:
    KUNLUNXIN_BASE_IMAGE     Base image (default: ${KUNLUNXIN_BASE_IMAGE})
    KUNLUNXIN_DRIVER_VERSION Driver version used in generated image tag (default: ${KUNLUNXIN_DRIVER_VERSION})
    KUNLUNXIN_PYTHON_VERSION Python version in the base image (default: ${KUNLUNXIN_PYTHON_VERSION})
    KUNLUNXIN_TORCH_VERSION  PyTorch version in the base image (default: ${KUNLUNXIN_TORCH_VERSION})
    KUNLUNXIN_VLLM_VERSION   vLLM version installed in empty mode (default: ${KUNLUNXIN_VLLM_VERSION})
    KUNLUNXIN_FLAGGEMS_REF   FlagGems git ref (default: ${KUNLUNXIN_FLAGGEMS_REF})
    KUNLUNXIN_FLAGCX_REF     FlagCX git ref (default: ${KUNLUNXIN_FLAGCX_REF})
    KUNLUNXIN_PLUGIN_FL_REF  vllm-plugin-FL git ref (default: ${KUNLUNXIN_PLUGIN_FL_REF})
  Hygon:
    HYGON_BASE_IMAGE     Base image (default: ${HYGON_BASE_IMAGE})
    HYGON_VLLM_VERSION   vLLM version installed in empty mode (default: ${HYGON_VLLM_VERSION})
    HYGON_DTK_VERSION    DTK version used in generated image tag (default: ${HYGON_DTK_VERSION})
    HYGON_PYTHON_VERSION Python version in Hygon base image tag (default: ${HYGON_PYTHON_VERSION})
    PLUGIN_SOURCE_SHA   Plugin Git revision (default: current repository HEAD)
    HYGON_GIT_CONFIG    Optional Git-only proxy config mounted as a BuildKit secret

EXAMPLES:
    # Build CUDA dev image
    ./build.sh --target dev

    # Build the validated Ascend CI image
    ./build.sh --platform ascend --target ci

    # Override the Ascend base image when validating a new stack
    ASCEND_BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.20.2rc1-a3 \
        ./build.sh --platform ascend --target ci

    # Build Hygon CI image
    ./build.sh --platform hygon --target ci

    # Build MetaX CI image
    ./build.sh --platform metax --target ci --image-name harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl

    # Build Moore Threads MUSA dev image
    ./build.sh --platform musa --target dev

    # Build Enflame CI image
    ./build.sh --platform enflame --target ci --image-name harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl

    # Build Kunlunxin CI image
    ./build.sh --platform kunlunxin --target ci --image-name harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl

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
BUILD_ARGS=()

if [[ "${PLATFORM}" == "cuda" ]]; then
    BUILD_ARGS+=(
        --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}"
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
    VLLM_VERSION="${ASCEND_VLLM_VERSION}"
    BUILD_ARGS+=(
        --build-arg "ASCEND_BASE_IMAGE=${ASCEND_BASE_IMAGE}"
        --build-arg "FLAGGEMS_VERSION=${ASCEND_FLAGGEMS_VERSION}"
    )
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="ascend-vllm${VLLM_VERSION}-a3-${TARGET}"
    fi
elif [[ "${PLATFORM}" == "hygon" ]]; then
    PYTHON_VERSION="${HYGON_PYTHON_VERSION}"
    VLLM_VERSION="${HYGON_VLLM_VERSION}"
    BUILD_CONTEXT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    PLUGIN_SOURCE_SHA="${PLUGIN_SOURCE_SHA:-$(git -C "${BUILD_CONTEXT}" rev-parse HEAD)}"
    [[ "${PLUGIN_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]] || err "PLUGIN_SOURCE_SHA must be a full Git revision."
    if [[ "${IMAGE_NAME}" == "harbor.baai.ac.cn/flagscale/vllm-plugin-fl" ]]; then
        IMAGE_NAME="harbor.baai.ac.cn/plugin/vllm-plugin-fl"
    fi
    BUILD_ARGS+=(
        --build-arg "HYGON_BASE_IMAGE=${HYGON_BASE_IMAGE}"
        --build-arg "VLLM_VERSION=${HYGON_VLLM_VERSION}"
        --build-arg "PLUGIN_SOURCE_SHA=${PLUGIN_SOURCE_SHA}"
    )
    [[ -z "${INDEX_URL}" ]] || BUILD_ARGS+=(--build-arg "INDEX_URL=${INDEX_URL}")
    [[ -z "${EXTRA_INDEX_URL}" ]] || BUILD_ARGS+=(--build-arg "EXTRA_INDEX_URL=${EXTRA_INDEX_URL}")
    if [[ -n "${HYGON_GIT_CONFIG:-}" ]]; then
        [[ -f "${HYGON_GIT_CONFIG}" ]] || err "HYGON_GIT_CONFIG file does not exist."
        BUILD_SECRET_ARGS+=(--secret "id=git_config,src=${HYGON_GIT_CONFIG}")
    fi
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="v${HYGON_VLLM_VERSION}-hygon-${TARGET}"
    fi
elif [[ "${PLATFORM}" == "metax" ]]; then
    PYTHON_VERSION="${METAX_PYTHON_VERSION}"
    VLLM_VERSION="${METAX_VLLM_VERSION}"
    if [[ "${IMAGE_NAME}" == "harbor.baai.ac.cn/flagscale/vllm-plugin-fl" ]]; then
        IMAGE_NAME="harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl"
    fi
    BUILD_ARGS+=(
        --build-arg "METAX_BASE_IMAGE=${METAX_BASE_IMAGE}"
        --build-arg "VLLM_VERSION=${METAX_VLLM_VERSION}"
    )
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="vllm-metax-${METAX_VLLM_VERSION}-maca.ai${METAX_MACA_VERSION}-torch2.8-${METAX_PYTHON_TAG}-ubuntu22.04-amd64-ci-git"
    fi
elif [[ "${PLATFORM}" == "musa" ]]; then
    PYTHON_VERSION="${MUSA_PYTHON_VERSION}"
    VLLM_VERSION="${MUSA_VLLM_VERSION}"
    BUILD_ARGS+=(
        --build-arg "MUSA_BASE_IMAGE=${MUSA_BASE_IMAGE}"
        --build-arg "FLAGGEMS_VERSION=${MUSA_FLAGGEMS_VERSION}"
    )
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="musa${MUSA_VERSION}-vllm${VLLM_VERSION}-torch${MUSA_TORCH_VERSION}-py${MUSA_PYTHON_VERSION}-${TARGET}"
    fi
elif [[ "${PLATFORM}" == "enflame" ]]; then
    PYTHON_VERSION="${ENFLAME_PYTHON_VERSION}"
    VLLM_VERSION="${ENFLAME_VLLM_VERSION}"
    if [[ "${IMAGE_NAME}" == "harbor.baai.ac.cn/flagscale/vllm-plugin-fl" ]]; then
        IMAGE_NAME="harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl"
    fi
    BUILD_ARGS+=(
        --build-arg "ENFLAME_BASE_IMAGE=${ENFLAME_BASE_IMAGE}"
        --build-arg "VLLM_VERSION=${ENFLAME_VLLM_VERSION}"
        --build-arg "DRIVER_VERSION=${ENFLAME_DRIVER_VERSION}"
    )
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="v${ENFLAME_VLLM_VERSION}-enflame-ci"
    fi
elif [[ "${PLATFORM}" == "kunlunxin" ]]; then
    PYTHON_VERSION="${KUNLUNXIN_PYTHON_VERSION}"
    VLLM_VERSION="${KUNLUNXIN_VLLM_VERSION}"
    if [[ "${IMAGE_NAME}" == "harbor.baai.ac.cn/flagscale/vllm-plugin-fl" ]]; then
        IMAGE_NAME="harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl"
    fi
    BUILD_ARGS+=(
        --build-arg "KUNLUNXIN_BASE_IMAGE=${KUNLUNXIN_BASE_IMAGE}"
        --build-arg "VLLM_VERSION=${KUNLUNXIN_VLLM_VERSION}"
        --build-arg "FLAGGEMS_REF=${KUNLUNXIN_FLAGGEMS_REF}"
        --build-arg "FLAGCX_REF=${KUNLUNXIN_FLAGCX_REF}"
        --build-arg "VLLM_PLUGIN_FL_REF=${KUNLUNXIN_PLUGIN_FL_REF}"
        --build-arg "DRIVER_VERSION=${KUNLUNXIN_DRIVER_VERSION}"
        --build-arg "INDEX_URL=${INDEX_URL}"
        --build-arg "EXTRA_INDEX_URL=${EXTRA_INDEX_URL}"
    )
    if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="v${KUNLUNXIN_VLLM_VERSION}-kunlunxin-ci"
    fi
else
    err "Unknown platform '${PLATFORM}'. Must be 'cuda', 'ascend', 'hygon', 'metax', 'musa', 'enflame', or 'kunlunxin'."
fi

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

msg "Building image: ${FULL_IMAGE}"
msg "  Platform:       ${PLATFORM}"
msg "  Target:         ${TARGET}"
if [[ "${PLATFORM}" == "cuda" ]]; then
    msg "  CUDA:           ${CUDA_VERSION}"
elif [[ "${PLATFORM}" == "ascend" ]]; then
    msg "  Base image:     ${ASCEND_BASE_IMAGE}"
    msg "  FlagGems:       ${ASCEND_FLAGGEMS_VERSION}"
elif [[ "${PLATFORM}" == "hygon" ]]; then
    msg "  DTK:            ${HYGON_DTK_VERSION}"
    msg "  Hygon Python:   ${HYGON_PYTHON_VERSION}"
    msg "  Base image:     ${HYGON_BASE_IMAGE}"
    msg "  Vendor stack:   preserved from the pinned base image"
    msg "  Plugin source:  ${PLUGIN_SOURCE_SHA}"
elif [[ "${PLATFORM}" == "metax" ]]; then
    msg "  MACA:           ${METAX_MACA_VERSION}"
    msg "  MetaX Python:   ${METAX_PYTHON_VERSION}"
    msg "  Base image:     ${METAX_BASE_IMAGE}"
elif [[ "${PLATFORM}" == "musa" ]]; then
    msg "  MUSA:           ${MUSA_VERSION}"
    msg "  MUSA base:      ${MUSA_BASE_IMAGE}"
    msg "  MUSA PyTorch:   ${MUSA_TORCH_VERSION}"
    msg "  FlagGems:       ${MUSA_FLAGGEMS_VERSION}"
elif [[ "${PLATFORM}" == "enflame" ]]; then
    msg "  Driver:         ${ENFLAME_DRIVER_VERSION}"
    msg "  Enflame Python: ${ENFLAME_PYTHON_VERSION}"
    msg "  Base image:     ${ENFLAME_BASE_IMAGE}"
elif [[ "${PLATFORM}" == "kunlunxin" ]]; then
    msg "  Driver:         ${KUNLUNXIN_DRIVER_VERSION}"
    msg "  Kunlunxin base: ${KUNLUNXIN_BASE_IMAGE}"
    msg "  Kunlunxin PyTorch: ${KUNLUNXIN_TORCH_VERSION}"
    msg "  FlagGems:       ${KUNLUNXIN_FLAGGEMS_REF}"
    msg "  FlagCX:         ${KUNLUNXIN_FLAGCX_REF}"
    msg "  Plugin:         ${KUNLUNXIN_PLUGIN_FL_REF}"
fi
if [[ "${PLATFORM}" != "ascend" ]]; then
    msg "  Ubuntu:         ${UBUNTU_VERSION}"
    msg "  Python:         ${PYTHON_VERSION}"
fi
msg "  vLLM:           ${VLLM_VERSION}"
msg ""

docker build \
    -f "${DOCKERFILE}" \
    --target "${TARGET}" \
    "${BUILD_ARGS[@]}" \
    "${BUILD_SECRET_ARGS[@]}" \
    ${NO_CACHE} \
    "${EXTRA_BUILD_ARGS[@]+"${EXTRA_BUILD_ARGS[@]}"}" \
    -t "${FULL_IMAGE}" \
    "${BUILD_CONTEXT}"

msg "Build complete: ${FULL_IMAGE}"
