// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project
//
// Ascend torch bindings for vLLM-FL operators

#include <torch/torch.h>
#include <torch/library.h>

#include "registration.h"

namespace vllm_fl {

// Forward declarations of Ascend implementations
torch::Tensor weak_ref_tensor_ascend(const torch::Tensor& tensor);

}  // namespace vllm_fl

// Register extension for Python import
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

// Define operators using the extension name
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", c10::kPrivateUse1, &vllm_fl::weak_ref_tensor_ascend);
}
