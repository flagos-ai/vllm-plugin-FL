// SPDX-License-Identifier: Apache-2.0
// Host-view ownership follows vLLM csrc/cuda_view.cu.
// MetaX registration, device guards and teardown are handled here.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <mcr/mc_runtime_api.h>
#include <memory>
#include "../registration.h"

namespace {
struct DeviceGuard {
  int previous = -1;
  explicit DeviceGuard(int device) {
    TORCH_CHECK(mcGetDevice(&previous) == mcSuccess, "Cannot get MetaX device");
    auto error = mcSetDevice(device);
    TORCH_CHECK(error == mcSuccess, mcGetErrorString(error));
  }
  ~DeviceGuard() { if (previous >= 0) mcSetDevice(previous); }
};

struct Registration {
  at::Tensor host;
  int device;
  bool registered = false;
  Registration(at::Tensor tensor, int index) : host(std::move(tensor)), device(index) {}
  ~Registration() {
    if (!registered) return;
    int previous = -1;
    if (mcGetDevice(&previous) != mcSuccess || mcSetDevice(device) != mcSuccess) return;
    // Only at final view destruction, never on the step path. Host storage is
    // released after all queued accesses finish and registration is removed.
    if (mcDeviceSynchronize() == mcSuccess) mcHostUnregister(host.data_ptr());
    if (previous >= 0) mcSetDevice(previous);
  }
};

void validate(const at::Tensor& host) {
  TORCH_CHECK(host.device().is_cpu() && host.layout() == at::kStrided,
              "Expected a strided CPU tensor");
  TORCH_CHECK(!host.requires_grad(), "Mapped host views are inference-only");
}

at::Tensor mapped_view(const at::Tensor& host, int64_t device) {
  validate(host);
  DeviceGuard guard(static_cast<int>(device));
  auto options = host.options().device(at::Device(at::kCUDA, device)).pinned_memory(false);
  if (host.numel() == 0) return at::empty_strided(host.sizes(), host.strides(), options);
  TORCH_CHECK(host.is_pinned(), "UVA view requires pinned host storage");
  void* pointer = nullptr;
  auto error = mcHostGetDevicePointer(&pointer, host.data_ptr(), 0);
  TORCH_CHECK(error == mcSuccess, "mcHostGetDevicePointer: ", mcGetErrorString(error));
  return at::from_blob(pointer, host.sizes(), host.strides(),
                       [owner = host](void*) {}, options);
}

at::Tensor register_host(const at::Tensor& host, int64_t device) {
  validate(host);
  TORCH_CHECK(host.is_contiguous(),
              "Host registration requires a contiguous CPU span");
  TORCH_CHECK(host.numel() > 0, "Cannot register an empty allocation");
  DeviceGuard guard(static_cast<int>(device));
  auto owner = std::make_shared<Registration>(host, static_cast<int>(device));
  auto error = mcHostRegister(host.data_ptr(), host.nbytes(),
                             mcHostRegisterPortable | mcHostRegisterMapped);
  TORCH_CHECK(error == mcSuccess, "mcHostRegister: ", mcGetErrorString(error));
  owner->registered = true;
  void* pointer = nullptr;
  error = mcHostGetDevicePointer(&pointer, host.data_ptr(), 0);
  TORCH_CHECK(error == mcSuccess, "mcHostGetDevicePointer: ", mcGetErrorString(error));
  return at::from_blob(pointer, host.sizes(), host.strides(),
                       [owner](void*) {},
                       host.options().device(at::Device(at::kCUDA, device)).pinned_memory(false));
}
}  // namespace

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

TORCH_LIBRARY(vllm_fl_metax, m) {
  m.def("host_view(Tensor host, int device) -> Tensor");
  m.def("register_host(Tensor host, int device) -> Tensor");
}
TORCH_LIBRARY_IMPL(vllm_fl_metax, CPU, m) {
  m.impl("host_view", mapped_view);
  m.impl("register_host", register_host);
}
