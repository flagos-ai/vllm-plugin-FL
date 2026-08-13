#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot build and test runner for vllm-plugin-FL Ascend custom ops.
#
# This script assumes the caller has already activated the CANN environment
# (e.g. source /usr/local/Ascend/ascend-toolkit/set_env.sh).  It will verify
# that CANN >= 9.0.0 is available, then build the torch extension, optionally
# build/install the CANN framework operator package, and finally run all tests
# under tests/custom_ops_tests/.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $*"; }

usage() {
    cat <<EOF
Usage: bash tests/custom_ops_tests/build_and_run.sh [OPTIONS]

One-shot build and test runner for vllm-plugin-FL Ascend custom ops.

Options:
  --build-ops            Build and install CANN framework operators from source
                         by running csrc/ascend/build_aclnn.sh.
  --soc <version>        SOC version passed to build_aclnn.sh, e.g. ascend910b
                         or ascend910_93. Defaults to \$SOC_VERSION, or
                         ascend910b if neither is set.
  --editable, -e         Install the package in editable mode
                         (VLLM_VENDOR=ascend pip install -e .) instead of
                         running VLLM_VENDOR=ascend python setup.py build_ext
                         --inplace.
  -h, --help             Show this help message.

Examples:
  # Default: build extension, check installed ops, run tests
  bash tests/custom_ops_tests/build_and_run.sh

  # Also compile and install CANN framework operators
  bash tests/custom_ops_tests/build_and_run.sh --build-ops

  # Compile operators for ascend910_93 and install editable
  bash tests/custom_ops_tests/build_and_run.sh --build-ops --soc ascend910_93 --editable
EOF
}

BUILD_OPS=0
SOC_VERSION_ARG=""
EDITABLE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-ops)
            BUILD_OPS=1
            ;;
        --soc)
            if [[ $# -lt 2 ]]; then
                log_error "--soc requires a value."
                usage
                exit 1
            fi
            SOC_VERSION_ARG="$2"
            shift
            ;;
        --editable|-e)
            EDITABLE=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# -----------------------------------------------------------------------------
# 1. CANN version check
# -----------------------------------------------------------------------------
check_cann() {
    log_step "Checking CANN environment ..."

    local ascend_home="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}"
    if [[ -z "$ascend_home" ]]; then
        log_error "ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME is not set."
        log_error "Please activate the CANN environment first, for example:"
        log_error "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        exit 1
    fi

    # Follow symlinks such as .../ascend-toolkit/latest
    ascend_home=$(readlink -f "$ascend_home" 2>/dev/null || echo "$ascend_home")

    if [[ ! -d "$ascend_home" ]]; then
        log_error "CANN path does not exist: $ascend_home"
        exit 1
    fi

    log_info "CANN home: $ascend_home"

    local version=""

    # Try to extract version from the directory name, e.g. /usr/local/Ascend/cann-9.0.0
    version=$(echo "$ascend_home" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)

    # Fallback: look for version.info or ascend_toolkit_install.info
    if [[ -z "$version" && -f "$ascend_home/version.info" ]]; then
        version=$(grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' "$ascend_home/version.info" | head -n1 || true)
    fi
    if [[ -z "$version" && -f "$ascend_home/ascend_toolkit_install.info" ]]; then
        version=$(grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' "$ascend_home/ascend_toolkit_install.info" | head -n1 || true)
    fi

    if [[ -z "$version" ]]; then
        log_error "Cannot detect CANN version from $ascend_home."
        log_error "Make sure you have sourced the correct CANN set_env.sh."
        exit 1
    fi

    log_info "Detected CANN version: $version"

    # Compare major.minor against 9.0
    local major minor
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)

    if [[ "$major" -lt 9 ]] || { [[ "$major" -eq 9 && "$minor" -lt 0 ]]; }; then
        log_error "CANN version $version is too old. This project requires CANN 9.0.0 or higher."
        log_error "Please install a compatible CANN toolkit and try again."
        exit 1
    fi

    log_info "CANN version check passed."
}

# -----------------------------------------------------------------------------
# 2. Optional environment configuration
# -----------------------------------------------------------------------------
ask_yes_no() {
    local prompt="$1"
    local response
    read -rp "$prompt [y/N]: " response
    case "$response" in
        [Yy]*) return 0 ;;
        *) return 1 ;;
    esac
}

configure_git_mirror() {
    log_step "Configuring git mirror ..."
    git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"
    log_info "Git mirror configured."
}

configure_pip_mirror() {
    log_step "Configuring pip mirror ..."
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    log_info "Pip mirror configured."
}

# -----------------------------------------------------------------------------
# 3. Submodule check
# -----------------------------------------------------------------------------
check_submodules() {
    log_step "Checking source submodules ..."

    local missing=0
    for sub in csrc/ascend/third_party/catlass csrc/ascend/third_party/pto-isa; do
        if [[ ! -e "$sub/.git" ]]; then
            log_warn "Submodule $sub is not initialized."
            missing=1
        fi
    done

    if [[ "$missing" -eq 1 ]]; then
        log_info "Initializing catlass and pto-isa submodules ..."
        git submodule update --init --recursive csrc/ascend/third_party/catlass
        git submodule update --init --recursive csrc/ascend/third_party/pto-isa
    else
        log_info "Submodules are already initialized."
    fi
}

# -----------------------------------------------------------------------------
# 4. Build/install the Python package
# -----------------------------------------------------------------------------
build_extension() {
    log_step "Building vllm_fl._C_ascend torch extension ..."

    if [[ ! -f "setup.py" ]]; then
        log_error "setup.py not found in $ROOT_DIR. Are you running this script from the project root?"
        exit 1
    fi

    VLLM_VENDOR=ascend python setup.py build_ext --inplace

    log_info "Torch extension built successfully."
}

install_editable() {
    log_step "Installing vllm-plugin-FL in editable mode ..."

    if [[ ! -f "setup.py" ]]; then
        log_error "setup.py not found in $ROOT_DIR."
        exit 1
    fi

    if ! command -v pip >/dev/null 2>&1; then
        log_error "pip is not available in the current environment."
        exit 1
    fi

    # This project imports torch_npu at build time, which is not available in
    # PEP 517 isolated build environments.  Use --no-build-isolation so the
    # current Python environment (where torch_npu is installed) is used.
    VLLM_VENDOR=ascend pip install --no-build-isolation -e .

    log_info "Editable install complete."
}

# -----------------------------------------------------------------------------
# 5. Build/install CANN framework operators (optional)
# -----------------------------------------------------------------------------
build_ops() {
    log_step "Building and installing CANN framework operators ..."

    local soc_args=()
    if [[ -n "$SOC_VERSION_ARG" ]]; then
        soc_args=("$SOC_VERSION_ARG")
    elif [[ -n "${SOC_VERSION:-}" ]]; then
        soc_args=("$SOC_VERSION")
    fi

    if [[ ${#soc_args[@]} -gt 0 ]]; then
        log_info "SOC version: ${soc_args[0]}"
    else
        log_info "SOC version: (default from build_aclnn.sh)"
    fi

    bash csrc/ascend/build_aclnn.sh "${soc_args[@]}"

    log_info "CANN framework operators built and installed."
}

# -----------------------------------------------------------------------------
# 6. Check CANN framework operator package
# -----------------------------------------------------------------------------
check_cann_framework_ops() {
    log_step "Checking CANN framework operator package ..."

    local vendor_dir="vllm_fl/_cann_ops_custom/vendors/custom_transformer"
    if [[ ! -d "$vendor_dir" ]]; then
        log_error "CANN framework operators are not installed at vllm_fl/_cann_ops_custom/"
        log_error ""
        log_error "If you already have a built .run package, install it with:"
        log_error "  bash csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run \\"
        log_error "      --install-path=\$(pwd)/vllm_fl/_cann_ops_custom"
        log_error ""
        log_error "Or build from source by adding --build-ops to this script:"
        log_error "  bash tests/custom_ops_tests/build_and_run.sh --build-ops [--soc <version>]"
        exit 1
    fi

    log_info "CANN framework operators found."
}

# -----------------------------------------------------------------------------
# 7. Set up CANN custom-op runtime environment
# -----------------------------------------------------------------------------
setup_cann_op_env() {
    log_step "Setting up CANN custom-op runtime environment ..."

    local vendor_dir="${ROOT_DIR}/vllm_fl/_cann_ops_custom/vendors/custom_transformer"
    local set_env_script="${vendor_dir}/bin/set_env.bash"

    # The package-provided set_env.bash appends to these variables, so make
    # sure they are defined before sourcing it.  This avoids an unbound
    # variable failure when this script is run with `set -u`.
    export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

    if [[ -f "$set_env_script" ]]; then
        log_info "Sourcing ${set_env_script} ..."
        # shellcheck source=/dev/null
        source "$set_env_script"
    fi

    # Always override the two path variables with the actual install location,
    # in case the package has been relocated since installation.
    export ASCEND_CUSTOM_OPP_PATH="$vendor_dir"
    export LD_LIBRARY_PATH="${vendor_dir}/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    log_info "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
}

# -----------------------------------------------------------------------------
# 8. Check / install FlagGems
# -----------------------------------------------------------------------------
check_flag_gems() {
    log_step "Checking FlagGems ..."

    if python -c "import flag_gems" >/dev/null 2>&1; then
        log_info "FlagGems is already installed."
        return 0
    fi

    log_warn "FlagGems not found. Installing from source ..."

    local parent_dir
    parent_dir=$(dirname "$ROOT_DIR")
    cd "$parent_dir"

    if [[ ! -d "FlagGems" ]]; then
        log_info "Cloning FlagGems ..."
        git clone https://github.com/flagos-ai/FlagGems
    else
        log_info "FlagGems directory already exists, skipping clone."
    fi

    cd FlagGems
    pip install --no-build-isolation -e .
    log_info "FlagGems installed."

    cd "$ROOT_DIR"
}

# -----------------------------------------------------------------------------
# 9. Run tests
# -----------------------------------------------------------------------------
run_tests() {
    log_step "Running custom ops tests ..."

    local test_dir="tests/custom_ops_tests"
    local failed=0

    if [[ ! -d "$test_dir" ]]; then
        log_error "Test directory not found: $test_dir"
        exit 1
    fi

    for test in "$test_dir"/test_*.py; do
        if [[ ! -f "$test" ]]; then
            continue
        fi
        log_info "Running $(basename "$test") ..."
        if ! python "$test"; then
            log_error "$(basename "$test") FAILED"
            failed=1
        fi
    done

    if [[ "$failed" -ne 0 ]]; then
        log_error "Some tests failed."
        exit 1
    fi

    log_info "All tests passed."
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log_info "Working directory: $ROOT_DIR"
    check_cann

    if ask_yes_no "Configure git mirror for GitHub (ghfast.top)?"; then
        configure_git_mirror
    fi
    if ask_yes_no "Configure pip mirror (Tsinghua)?"; then
        configure_pip_mirror
    fi

    check_submodules
    if [[ "$EDITABLE" -eq 1 ]]; then
        install_editable
    else
        build_extension
    fi
    if [[ "$BUILD_OPS" -eq 1 ]]; then
        build_ops
    fi
    check_cann_framework_ops
    setup_cann_op_env
    check_flag_gems
    run_tests
    log_info "Done."
}

main
