# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project
#
# CUDA backend for vLLM-FL

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

# =============================================================================
# Source files
# =============================================================================

set(VLLM_FL_CUDA_SRCS
    weak_ref_tensor.cu
    torch_bindings.cpp
)

# =============================================================================
# Define extension target
# =============================================================================

# Create Python extension module named _C
# This will be importable as: import vllm_fl._C
Python_add_library(_C MODULE WITH_SOABI ${VLLM_FL_CUDA_SRCS})

# Set TORCH_EXTENSION_NAME so TORCH_LIBRARY_EXPAND works
target_compile_definitions(_C PRIVATE "-DTORCH_EXTENSION_NAME=_C")

# Include directories
target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(_C PRIVATE
    torch
    CUDA::cudart
    CUDA::cuda_driver
)

# CUDA settings
set_target_properties(_C PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

target_compile_options(_C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# =============================================================================
# Install to vllm_fl package directory
# =============================================================================

install(TARGETS _C LIBRARY DESTINATION vllm_fl COMPONENT _C)
