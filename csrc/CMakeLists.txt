# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

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
        "Set VLLM_VENDOR=cuda."
    )
endif()

set(SUPPORTED_VENDORS cuda)
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

# The extension wraps an existing Tensor; it does not compile CUDA kernels.
# TorchConfig.cmake probes nvcc/NVRTC even for this C++-only target, which
# fails in runtime images and in vendor CUDA-like builds without a toolkit.
# Read the same CPU-side include/library paths used by CppExtension instead.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
        "import json, torch; from torch.utils.cpp_extension import include_paths, library_paths; print(json.dumps({'version': torch.__version__, 'abi': int(torch._C._GLIBCXX_USE_CXX11_ABI), 'include_paths': include_paths(), 'library_paths': library_paths()}))"
    OUTPUT_VARIABLE TORCH_CONFIG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE TORCH_CONFIG_ERROR
    RESULT_VARIABLE TORCH_CONFIG_RESULT
)
if(NOT TORCH_CONFIG_RESULT EQUAL 0)
    message(FATAL_ERROR "Cannot query the active PyTorch: ${TORCH_CONFIG_ERROR}")
endif()
string(JSON TORCH_VERSION GET "${TORCH_CONFIG}" version)
string(JSON TORCH_CXX11_ABI GET "${TORCH_CONFIG}" abi)
foreach(kind IN ITEMS INCLUDE LIBRARY)
    string(TOLOWER "${kind}" key)
    string(JSON path_count LENGTH "${TORCH_CONFIG}" "${key}_paths")
    if(path_count EQUAL 0)
        message(FATAL_ERROR "PyTorch returned no ${key} paths")
    endif()
    math(EXPR path_last "${path_count} - 1")
    foreach(index RANGE ${path_last})
        string(JSON torch_path GET "${TORCH_CONFIG}" "${key}_paths" ${index})
        list(APPEND TORCH_${kind}_DIRS "${torch_path}")
    endforeach()
endforeach()
foreach(name IN ITEMS torch torch_cpu c10)
    find_library(TORCH_${name}_LIBRARY NAMES ${name}
        PATHS ${TORCH_LIBRARY_DIRS} NO_DEFAULT_PATH REQUIRED)
    list(APPEND TORCH_LIBRARIES "${TORCH_${name}_LIBRARY}")
endforeach()
message(STATUS "PyTorch: ${TORCH_VERSION} (C++ ABI ${TORCH_CXX11_ABI})")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${VLLM_VENDOR})
