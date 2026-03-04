# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project
#
# Ascend backend for vLLM-FL

# Ascend CANN toolkit
set(ASCEND_TOOLKIT_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
if(NOT ASCEND_TOOLKIT_PATH)
    set(ASCEND_TOOLKIT_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

if(NOT EXISTS "${ASCEND_TOOLKIT_PATH}/include/acl/acl.h")
    message(WARNING "Ascend CANN not found at ${ASCEND_TOOLKIT_PATH}. Skipping.")
    return()
endif()

message(STATUS "Ascend CANN: ${ASCEND_TOOLKIT_PATH}")

# =============================================================================
# Source files
# =============================================================================

set(VLLM_FL_ASCEND_SRCS
    weak_ref_tensor.cpp
    torch_bindings.cpp
)

# =============================================================================
# Define extension target
# =============================================================================

# Create Python extension module named _C
# This will be importable as: import vllm_fl._C
Python_add_library(_C MODULE WITH_SOABI ${VLLM_FL_ASCEND_SRCS})

# Set TORCH_EXTENSION_NAME so TORCH_LIBRARY_EXPAND works
target_compile_definitions(_C PRIVATE "-DTORCH_EXTENSION_NAME=_C")

# Include directories
target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${ASCEND_TOOLKIT_PATH}/include
    ${TORCH_INCLUDE_DIRS}
)

# Link libraries
get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
target_link_directories(_C PRIVATE
    ${TORCH_LIB_DIR}
    ${ASCEND_TOOLKIT_PATH}/lib64
)

target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})

# C++ settings
set_target_properties(_C PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# =============================================================================
# Install to vllm_fl package directory
# =============================================================================

install(TARGETS _C LIBRARY DESTINATION vllm_fl COMPONENT _C)
