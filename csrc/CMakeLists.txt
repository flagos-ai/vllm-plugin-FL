# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project
#
# vLLM-FL extension dispatcher.
# For Ascend, this file directly builds the torch extension (_C_ascend) and
# the kernels shared library. The CANN framework operators are built
# independently from csrc/ascend/ via build_aclnn.sh.

cmake_minimum_required(VERSION 3.26)
project(vllm_fl_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED VLLM_VENDOR)
    if(DEFINED ENV{VLLM_VENDOR})
        set(VLLM_VENDOR $ENV{VLLM_VENDOR})
    endif()
endif()

if(NOT VLLM_VENDOR)
    message(FATAL_ERROR
        "VLLM_VENDOR is required but not specified.\n"
        "Set VLLM_VENDOR=cuda or VLLM_VENDOR=ascend."
    )
endif()

set(SUPPORTED_VENDORS cuda ascend)
if(NOT VLLM_VENDOR IN_LIST SUPPORTED_VENDORS)
    message(FATAL_ERROR
        "Unsupported vendor: ${VLLM_VENDOR}\n"
        "Supported vendors: ${SUPPORTED_VENDORS}"
    )
endif()

message(STATUS "vLLM-FL Extensions vendor: ${VLLM_VENDOR}")

if(VLLM_PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${VLLM_PYTHON_EXECUTABLE})
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python: ${Python_EXECUTABLE} (${Python_VERSION})")

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})

find_package(Torch REQUIRED)
message(STATUS "PyTorch: ${Torch_VERSION}")

if(VLLM_VENDOR STREQUAL "ascend")
    set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

    if(NOT DEFINED TORCH_NPU_PATH)
        message(FATAL_ERROR
            "TORCH_NPU_PATH is required for Ascend builds. "
            "It is normally supplied by setup.py."
        )
    endif()
    message(STATUS "torch_npu package path: ${TORCH_NPU_PATH}")
    include_directories(${TORCH_NPU_PATH}/include)
    link_directories(${TORCH_NPU_PATH}/lib)

    # Locate the AscendC kernel cmake helpers shipped with the CANN toolkit.
    if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
        set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
        set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
        set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    else()
        message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
    endif()
    include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

    set(VLLM_FL_ASCEND_SRCS
        ascend/camem_allocator.cpp
        ascend/torch_binding.cpp
        ascend/torch_binding_meta.cpp
        ascend/aclnn_torch_adapter/op_api_common.h
        ascend/aclnn_torch_adapter/NPUBridge.cpp
        ascend/aclnn_torch_adapter/NPUBridge.h
        ascend/aclnn_torch_adapter/NPUStorageImpl.cpp
        ascend/aclnn_torch_adapter/NPUStorageImpl.h
        # Framework operator torch adapters (op_host/op_kernel/ are built by the
        # CANN toolchain and are present here for reference / vendor packaging).
        ascend/attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h
        ascend/attention/kv_quant_sparse_flash_attention/kv_quant_sparse_flash_attention_torch_adpt.h
        ascend/attention/lightning_indexer/lightning_indexer_torch_adpt.h
        ascend/attention/lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h
        ascend/attention/lightning_indexer_vllm/lightning_indexer_vllm_torch_adpt.h
        ascend/attention/ngram_spec_decode/ngram_spec_decode_torch_adpt.h
        ascend/attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h
        ascend/attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h
        ascend/attention/store_kv_block/store_kv_block_torch_adpt.h
        ascend/gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h
        ascend/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h
        ascend/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h
        ascend/mc2/matmul_allreduce_add_rmsnorm/matmul_allreduce_add_rmsnorm_torch_adpt.h
        ascend/moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h
        ascend/moe/apply_top_k_top_p_custom/apply_top_k_top_p_custom_torch_adpt.h
        ascend/moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h
        ascend/moe/moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h
    )

    file(GLOB VLLM_FL_ASCEND_KERNELS ascend/kernels/*.cpp)
    ascendc_library(vllm_fl_kernels SHARED ${VLLM_FL_ASCEND_KERNELS})

    Python_add_library(_C_ascend MODULE WITH_SOABI ${VLLM_FL_ASCEND_SRCS})
    target_compile_definitions(_C_ascend PRIVATE TORCH_EXTENSION_NAME=_C_ascend)

    target_include_directories(_C_ascend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend/aclnn_torch_adapter
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
    )

    target_link_libraries(_C_ascend PRIVATE
        ${TORCH_LIBRARIES}
        torch_npu
        vllm_fl_kernels
        ascendcl
        opapi
        tiling_api
        register
        platform
        ascendalog
        dl
    )

    target_compile_options(_C_ascend PRIVATE -O3)
    target_link_options(_C_ascend PRIVATE "-Wl,-rpath,$ORIGIN")

    install(TARGETS _C_ascend vllm_fl_kernels LIBRARY DESTINATION vllm_fl COMPONENT _C_ascend)

elseif(VLLM_VENDOR STREQUAL "cuda")
    add_subdirectory(cuda)
else()
    message(FATAL_ERROR "Unsupported vendor: ${VLLM_VENDOR}")
endif()

# Optional PTO chunk-GDN megakernel pre-compilation (Ascend only).
option(BUILD_PTO_CHUNK_GDN "Pre-compile PTO GDN megakernel via Bisheng" OFF)
if(VLLM_VENDOR STREQUAL "ascend" AND BUILD_PTO_CHUNK_GDN)
    add_subdirectory(ascend/pto_chunk_gdn EXCLUDE_FROM_ALL)
endif()
