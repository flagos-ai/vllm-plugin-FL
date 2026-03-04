# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project
#
# vLLM-FL C++ extensions - Root CMakeLists.txt

cmake_minimum_required(VERSION 3.26)
project(vllm_fl_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Vendor Selection (REQUIRED - no auto-detection)
# =============================================================================

if(NOT DEFINED VLLM_VENDOR)
    if(DEFINED ENV{VLLM_VENDOR})
        set(VLLM_VENDOR $ENV{VLLM_VENDOR})
    endif()
endif()

if(NOT VLLM_VENDOR)
    message(FATAL_ERROR
        "VLLM_VENDOR is required but not specified.\n"
        "Please set VLLM_VENDOR environment variable or cmake option:\n"
        "  export VLLM_VENDOR=cuda    # For NVIDIA CUDA\n"
        "  export VLLM_VENDOR=ascend  # For Huawei Ascend\n"
        "\n"
        "Or pass to cmake:\n"
        "  cmake -DVLLM_VENDOR=cuda .."
    )
endif()

set(SUPPORTED_VENDORS cuda ascend)
if(NOT VLLM_VENDOR IN_LIST SUPPORTED_VENDORS)
    message(FATAL_ERROR
        "Unsupported vendor: ${VLLM_VENDOR}\n"
        "Supported vendors: ${SUPPORTED_VENDORS}"
    )
endif()

message(STATUS "==============================================")
message(STATUS "vLLM-FL Extensions: ${VLLM_VENDOR}")
message(STATUS "==============================================")

# =============================================================================
# Find Python
# =============================================================================

if(VLLM_PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${VLLM_PYTHON_EXECUTABLE})
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python: ${Python_EXECUTABLE} (${Python_VERSION})")

# =============================================================================
# Find PyTorch
# =============================================================================

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})

find_package(Torch REQUIRED)
message(STATUS "PyTorch: ${Torch_VERSION}")

# =============================================================================
# Include directories
# =============================================================================

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# =============================================================================
# Build Vendor Backend
# =============================================================================

add_subdirectory(${VLLM_VENDOR})
