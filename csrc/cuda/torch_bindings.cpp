// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project
//
// CUDA torch bindings for vLLM-FL operators

#include <torch/torch.h>
#include <torch/library.h>

#include "registration.h"

namespace vllm_fl {

// Forward declarations of CUDA implementations
torch::Tensor weak_ref_tensor_cuda(torch::Tensor& tensor);

}  // namespace vllm_fl

// Register extension for Python import
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

// Define operators using the extension name
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", c10::kCUDA, &vllm_fl::weak_ref_tensor_cuda);

    // Add more operators here:
    // ops.def("another_op(Tensor input) -> Tensor");
    // ops.impl("another_op", c10::kCUDA, &vllm_fl::another_op_cuda);
}
