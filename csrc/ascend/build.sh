#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
#
# Low-level CANN custom-op build script for vllm-plugin-FL framework operators.
# This is the equivalent of vllm-ascend/csrc/build.sh, adapted to the
# vllm-plugin-FL directory layout where the CANN CMakeLists.txt lives under
# csrc/ascend/.

set -e

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/output
USER_ID=$(id -u)
PARENT_JOB="false"
CHECK_COMPATIBLE="true"
VERBOSE="false"

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi

ENABLE_BUILD_PKG="OFF"
ENABLE_BUILT_IN="OFF"

BASE_CUSTOM_OPTION="-DBUILD_OPEN_PROJECT=ON -DBUILD_TYPE=Release -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON"
CUSTOM_OPTION="${BASE_CUSTOM_OPTION}"

function help_info() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo
    echo "-h|--help            Displays help message."
    echo
    echo "-n|--op-name         Specifies the compiled operator. If there are multiple values, separate them with semicolons and use quotation marks. The default is all."
    echo "                     For example: -n \"flash_attention_score\" or -n \"flash_attention_score;flash_attention_score_grad\""
    echo
    echo "-c|--compute-unit    Specifies the chip type. If there are multiple values, separate them with semicolons and use quotation marks. The default is ascend910b."
    echo "                     For example: -c \"ascend910b\" or -c \"ascend910b;ascend310p\""
    echo
    echo "--pkg                Build a self-extracting .run package."
    echo
    echo "--ops=OPS            Same as -n (upstream style)."
    echo
    echo "--soc=SOC            Same as -c (upstream style)."
    echo
    echo "--verbose            Displays more compilation information."
    echo
}

function log() {
    local current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${current_time}] $1"
}

function set_env()
{
    source ${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash || echo "0"

    export BISHENG_REAL_PATH=$(which bisheng || true)

    if [ -z "${BISHENG_REAL_PATH}" ];then
        log "Error: bisheng compilation tool not found, Please check whether the cann package or environment variables are set."
        exit 1
    fi
}

function clean()
{
    if [ -n "${BUILD_DIR}" ];then
        rm -rf ${BUILD_DIR}
    fi
    if [ -n "${OUTPUT_DIR}" ];then
        rm -rf ${OUTPUT_DIR}
    fi
    mkdir -p ${BUILD_DIR} ${OUTPUT_DIR}
}

function cmake_config()
{
    local extra_option="$1"
    log "Info: cmake config ${CUSTOM_OPTION} ${extra_option} ."
    cmake -S ${CURRENT_DIR} -B ${BUILD_DIR} ${CUSTOM_OPTION} ${extra_option}
}

function build()
{
    local target="$1"
    if [ "${VERBOSE}" == "true" ];then
        local option="--verbose"
    fi
    cmake --build ${BUILD_DIR} --target ${target} ${JOB_NUM} ${option}
}

function gen_bisheng(){
    local ccache_program=$1
    local gen_bisheng_dir=${BUILD_DIR}/gen_bisheng_dir

    if [ ! -d "${gen_bisheng_dir}" ];then
        mkdir -p ${gen_bisheng_dir}
    fi

    pushd ${gen_bisheng_dir} > /dev/null
    cat > bisheng <<EOF
#!/bin/bash
ccache_args="${ccache_program} ${BISHENG_REAL_PATH}"
args="\$@"
EOF
    if [ "${VERBOSE}" == "true" ];then
        echo 'echo "${ccache_args} ${args}"' >> bisheng
    fi
    cat >> bisheng <<'EOF'
eval "${ccache_args} ${args}"
EOF
    chmod +x bisheng

    export PATH=${gen_bisheng_dir}:$PATH
    popd > /dev/null
}

function build_package(){
    build package
}

while [[ $# -gt 0 ]]; do
    case $1 in
    -h|--help)
        help_info
        exit
        ;;
    -n|--op-name)
        ascend_op_name="$2"
        shift 2
        ;;
    -c|--compute-unit)
        ascend_compute_unit="$2"
        shift 2
        ;;
    --pkg)
        ENABLE_BUILD_PKG="ON"
        ENABLE_BUILT_IN="OFF"
        shift
        ;;
    --ops)
        ascend_op_name="$2"
        shift 2
        ;;
    --soc)
        ascend_compute_unit="$2"
        shift 2
        ;;
    --ops=*)
        ascend_op_name="${1#*=}"
        shift
        ;;
    --soc=*)
        ascend_compute_unit="${1#*=}"
        shift
        ;;
    --verbose)
        VERBOSE="true"
        shift
        ;;
    *)
        help_info
        exit 1
        ;;
    esac
done

CUSTOM_OPTION="${BASE_CUSTOM_OPTION} -DENABLE_BUILD_PKG=${ENABLE_BUILD_PKG} -DENABLE_BUILT_IN=${ENABLE_BUILT_IN}"

if [ -n "${ascend_compute_unit}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_COMPUTE_UNIT=${ascend_compute_unit}"
fi

if [ -n "${ascend_op_name}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_OP_NAME=${ascend_op_name}"
fi

if [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
else
    log "Error: Please set the toolkit package installation directory through ASCEND_HOME_PATH or ASCEND_OPP_PATH."
    exit 1
fi

if [ "${PARENT_JOB}" == "false" ];then
    CPU_NUM=$(($(grep -c "^processor" /proc/cpuinfo)*2))
    JOB_NUM="-j${CPU_NUM}"
fi

CUSTOM_OPTION="${CUSTOM_OPTION} -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} -DCHECK_COMPATIBLE=${CHECK_COMPATIBLE} -DCANN_3RD_LIB_PATH=${CURRENT_DIR}/third_party"

set_env
clean

ccache_system=$(which ccache || true)
if [ -n "${ccache_system}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=ON -DCUSTOM_CCACHE=${ccache_system}"
    gen_bisheng ${ccache_system}
fi

cmake_config
build_package

log "Info: CANN framework operator package built at ${BUILD_DIR}."
log "Info: Install with: bash ${BUILD_DIR}/cann-ops-transformer-*.run --install-path=\$YOUR_INSTALL_DIR"
