# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project
#
# This weak-ref operator is a host-side Tensor wrapper. It registers on the
# CUDA dispatch key, but does not launch a CUDA kernel, so it must also build
# in HIP/ROCm CUDA-like PyTorch environments where nvcc is unavailable.

set(VLLM_FL_CUDA_SRCS
    weak_ref_tensor.cpp
    torch_bindings.cpp
)

Python_add_library(_C MODULE WITH_SOABI ${VLLM_FL_CUDA_SRCS})
target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)

target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})

target_compile_options(_C PRIVATE -O3)

install(TARGETS _C LIBRARY DESTINATION vllm_fl COMPONENT _C)
