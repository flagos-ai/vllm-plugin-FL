// Copyright (c) 2026 BAAI. All rights reserved.

#include <torch/torch.h>

namespace vllm_fl {

torch::Tensor weak_ref_tensor_ascend(torch::Tensor& tensor) {
  if (!tensor.is_privateuseone()) {
    throw std::runtime_error("Tensor must be on NPU device");
  }

  void* data_ptr = tensor.data_ptr();
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();
  auto options = tensor.options();

  return torch::from_blob(data_ptr, sizes, strides, options);
}

}  // namespace vllm_fl
