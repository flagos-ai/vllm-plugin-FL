# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

set(ASCEND_TOOLKIT_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
if(NOT ASCEND_TOOLKIT_PATH)
    set(ASCEND_TOOLKIT_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

if(NOT EXISTS "${ASCEND_TOOLKIT_PATH}/include/acl/acl.h")
    message(WARNING "Ascend CANN not found at ${ASCEND_TOOLKIT_PATH}. Skipping.")
    return()
endif()

message(STATUS "Ascend CANN: ${ASCEND_TOOLKIT_PATH}")

set(VLLM_FL_ASCEND_SRCS
    weak_ref_tensor.cpp
    torch_bindings.cpp
)

Python_add_library(_C MODULE WITH_SOABI ${VLLM_FL_ASCEND_SRCS})
target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)

target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${ASCEND_TOOLKIT_PATH}/include
    ${TORCH_INCLUDE_DIRS}
)

get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
target_link_directories(_C PRIVATE
    ${TORCH_LIB_DIR}
    ${ASCEND_TOOLKIT_PATH}/lib64
)

target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})

set_target_properties(_C PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

install(TARGETS _C LIBRARY DESTINATION vllm_fl COMPONENT _C)
