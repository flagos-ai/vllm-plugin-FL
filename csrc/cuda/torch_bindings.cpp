// Copyright (c) 2026 BAAI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "registration.h"

namespace vllm_fl {

torch::Tensor weak_ref_tensor_cuda(torch::Tensor& tensor);
int64_t enqueue_cuda_eventfd_completion(int64_t stream_ptr, int64_t event_fd);

}  // namespace vllm_fl

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

// vLLM may already own the `_C` namespace. A fragment lets the plugin add its
// compatibility op without requiring this extension to be the namespace owner.
TORCH_LIBRARY_FRAGMENT_EXPAND(TORCH_EXTENSION_NAME, ops) {
  const auto existing = c10::Dispatcher::singleton().findSchema(
      c10::OperatorName("_C::weak_ref_tensor", ""));
  if (!existing.has_value()) {
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", c10::kCUDA, &vllm_fl::weak_ref_tensor_cuda);
  }
}

// Completion notification has no Tensor argument, so register a catch-all
// implementation in a plugin-owned namespace instead of relying on dispatch.
TORCH_LIBRARY_FRAGMENT(vllm_fl, ops) {
  ops.def("enqueue_cuda_eventfd_completion(int stream_ptr, int event_fd) -> int");
  ops.impl("enqueue_cuda_eventfd_completion",
           &vllm_fl::enqueue_cuda_eventfd_completion);
}
