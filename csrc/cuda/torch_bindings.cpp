// Copyright (c) 2026 BAAI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include "registration.h"

namespace vllm_fl {

torch::Tensor weak_ref_tensor_cuda(torch::Tensor& tensor);

}  // namespace vllm_fl

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

TORCH_LIBRARY_FRAGMENT_EXPAND(TORCH_EXTENSION_NAME, ops) {
  const auto existing = c10::Dispatcher::singleton().findSchema(
      c10::OperatorName("_C::weak_ref_tensor", ""));
  if (!existing) {
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  }
  // Stable-ABI and legacy vLLM builds may already own the CUDA implementation.
  // Empty builds can instead expose only the Python schema/CPU fallback.
  if (!existing ||
      !existing->hasKernelForDispatchKey(c10::DispatchKey::CUDA)) {
    ops.impl("weak_ref_tensor", c10::kCUDA, &vllm_fl::weak_ref_tensor_cuda);
  }
}
