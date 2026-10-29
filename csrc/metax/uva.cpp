// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM v0.28 csrc/libtorch_stable/cuda_view.cu.
// Host runtime only: this file contains no model or device compute kernels.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <mc_runtime_api.h>

at::Tensor accelerator_view(const at::Tensor& cpu) {
  TORCH_CHECK(cpu.device().is_cpu(), "Input tensor must be on CPU");
  int device = 0;
  mcError_t error = mcGetDevice(&device);
  TORCH_CHECK(error == mcSuccess, "mcGetDevice failed: ", mcGetErrorString(error));
  auto options = cpu.options().device(at::Device(at::kCUDA, device));
  if (cpu.numel() == 0) {
    return at::empty(cpu.sizes(), options);
  }
  // Preserve the upstream distinction: pinned inputs alias; unpinned inputs
  // are copied into a new contiguous pinned allocation retained by the view.
  at::Tensor owner = cpu.is_pinned() ? cpu : cpu.contiguous().pin_memory();
  void* pointer = nullptr;
  error = mcHostGetDevicePointer(&pointer, owner.data_ptr(), 0);
  TORCH_CHECK(error == mcSuccess, "mcHostGetDevicePointer failed: ", mcGetErrorString(error));
  return at::from_blob(pointer, owner.sizes(), owner.strides(),
                       [owner](void*) {}, options);
}

TORCH_LIBRARY_IMPL(_C, CPU, library) {
  library.impl("get_cuda_view_from_cpu_tensor", &accelerator_view);
}
