#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Top-level CANN custom-op build entry for vllm-plugin-FL framework operators.
# Equivalent to vllm-ascend/csrc/build_aclnn.sh: handles SOC detection, catlass
# setup, operator selection and final installation.

set -e

ROOT_DIR=$(cd "$(dirname $(readlink -f ${BASH_SOURCE[0]}))/../../" && pwd)
SOC_VERSION="ascend910b"
CLEAN_THIRD_PARTY=0

# Parse arguments. The SOC_VERSION positional argument can appear anywhere;
# --clean-third-party is the only supported flag.
for arg in "$@"; do
    case "$arg" in
        --clean-third-party)
            CLEAN_THIRD_PARTY=1
            ;;
        --*)
            echo "Unknown option: $arg"
            exit 1
            ;;
        *)
            SOC_VERSION="$arg"
            ;;
    esac
done

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    echo "No custom aclnn ops for ASCEND310 series."
    exit 0
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    # ASCEND910B (A2) series
    git config --global --add safe.directory "$ROOT_DIR" || true
    CATLASS_PATH=${ROOT_DIR}/csrc/ascend/third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}

    CUSTOM_OPS="moe_grouped_matmul;grouped_matmul_swiglu_quant;grouped_matmul_swiglu_quant_v2;grouped_matmul_swiglu_quant_weight_nz_tensor_list;lightning_indexer;lightning_indexer_quant;sparse_flash_attention;kv_quant_sparse_flash_attention;sparse_attn_sharedkv;vllm_quant_lightning_indexer;compressor;compressor_metadata;fused_gdn_gating;inplace_partial_rotary_mul;recurrent_gated_delta_rule;rms_norm_dynamic_quant;store_kv_block;reshape_and_cache_bnsd;ngram_spec_decode;matmul_allreduce_add_rmsnorm;moe_init_routing_custom;moe_gating_top_k;moe_gating_top_k_hash;add_rms_norm_bias;apply_top_k_top_p_custom;causal_conv1d;chunk_fwd_o;chunk_gated_delta_rule_fwd_h;chunk_gated_delta_rule;copy_and_expand_eagle_inputs;dequant_swiglu_quant;hamming_dist_top_k;hc_post;hc_pre;hc_pre_inv_rms;hc_pre_sinkhorn;scatter_nd_update_v2;transpose_kv_cache_by_block;"
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    # ASCEND910C (A3) series
    git config --global --add safe.directory "$ROOT_DIR" || true
    CATLASS_PATH=${ROOT_DIR}/csrc/ascend/third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}

    if [[ -n "${HCCL_STRUCT_FILE_PATH}" ]]; then
        yes | cp "${HCCL_STRUCT_FILE_PATH}" "${ROOT_DIR}/csrc/ascend/utils/inc/kernel"
    fi

    CUSTOM_OPS_ARRAY=(
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer_vllm"
        "sparse_flash_attention"
        "dispatch_ffn_combine"
        "dispatch_ffn_combine_bf16"
        "dispatch_gmm_combine_decode"
        "moe_combine_normal"
        "moe_dispatch_normal"
        "dispatch_layout"
        "notify_dispatch"
        "moe_init_routing_custom"
        "moe_gating_top_k"
        "add_rms_norm_bias"
        "apply_top_k_top_p_custom"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "moe_grouped_matmul"
        "lightning_indexer_quant"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
else
    echo "Unsupported SOC version: $SOC_VERSION"
    exit 1
fi

# Build custom ops
cd ${ROOT_DIR}/csrc/ascend
rm -rf build output build_out

echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
bash build.sh --pkg --ops="$CUSTOM_OPS" --soc="$SOC_ARG"

# Install custom ops to vllm_fl/_cann_ops_custom (isolated from system CANN).
INSTALL_DIR=${ROOT_DIR}/vllm_fl/_cann_ops_custom
RUN_PACKAGE=$(ls build/cann-ops-transformer*.run 2> /dev/null | head -n1)
if [[ -z "${RUN_PACKAGE}" ]]; then
    RUN_PACKAGE=$(ls build_out/cann-ops-transformer*.run 2> /dev/null | head -n1)
fi
if [[ -z "${RUN_PACKAGE}" ]]; then
    echo "Error: no .run package found under build/ or build_out/"
    exit 1
fi

echo "installing ${RUN_PACKAGE} to ${INSTALL_DIR}"
bash "${RUN_PACKAGE}" --install-path="${INSTALL_DIR}"

# Clean downloaded third-party build artifacts only when explicitly requested.
# catlass and pto-isa are source submodules and must be kept.
clean_third_party_artifacts() {
    local third_party_dir="${ROOT_DIR}/csrc/ascend/third_party"
    echo "[build_aclnn] cleaning downloaded third-party build artifacts ..."
    rm -rf "${third_party_dir}/abseil-cpp"
    rm -rf "${third_party_dir}/ascend_protobuf"
    rm -rf "${third_party_dir}/json"
    rm -rf "${third_party_dir}/pkg"
}
if [[ "${CLEAN_THIRD_PARTY}" == "1" ]]; then
    clean_third_party_artifacts
fi

echo "CANN framework operators built and installed for $SOC_VERSION."
