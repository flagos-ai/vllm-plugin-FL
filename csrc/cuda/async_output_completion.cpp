// Copyright (c) 2026 BAAI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cerrno>
#include <climits>
#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <sys/eventfd.h>

namespace vllm_fl {
namespace {

// Keep this translation unit buildable with a host C++ compiler. The plugin
// resolves the small CUDA Driver API surface dynamically, so CUDA-like builds
// that lack NVIDIA headers do not accidentally acquire a CUDA link dependency.
using CUresult = int;
using CUstream = void*;
using CUhostFn = void (*)(void*);
using CuLaunchHostFunc = CUresult (*)(CUstream, CUhostFn, void*);

constexpr CUresult kCudaSuccess = 0;
constexpr int64_t kInvalidArgument = -1;
constexpr int64_t kDriverUnavailable = -2;
constexpr int64_t kSymbolUnavailable = -3;

std::once_flag load_driver_once;
void* cuda_driver_handle = nullptr;
CuLaunchHostFunc cu_launch_host_func = nullptr;
int64_t load_status = kDriverUnavailable;

void load_cuda_driver() {
  cuda_driver_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (cuda_driver_handle == nullptr) {
    load_status = kDriverUnavailable;
    return;
  }
  cu_launch_host_func = reinterpret_cast<CuLaunchHostFunc>(
      dlsym(cuda_driver_handle, "cuLaunchHostFunc"));
  if (cu_launch_host_func == nullptr) {
    load_status = kSymbolUnavailable;
    return;
  }
  load_status = kCudaSuccess;
}

// CUDA invokes this function only after all earlier work in the copy stream,
// including D2H copies, has completed. It must remain tiny: no Python, no GIL,
// no CUDA API, no allocation, and no lock. Each fd has at most one outstanding
// callback, so adding one cannot saturate the eventfd counter.
void notify_eventfd(void* user_data) {
  const int event_fd = static_cast<int>(reinterpret_cast<intptr_t>(user_data));
  int result;
  do {
    result = eventfd_write(event_fd, 1);
  } while (result != 0 && errno == EINTR);
}

}  // namespace

int64_t enqueue_cuda_eventfd_completion(int64_t stream_ptr, int64_t event_fd) {
  if (stream_ptr == 0 || event_fd <= 0 || event_fd > INT32_MAX) {
    return kInvalidArgument;
  }
  std::call_once(load_driver_once, load_cuda_driver);
  if (load_status != kCudaSuccess || cu_launch_host_func == nullptr) {
    return load_status;
  }
  const CUresult status = cu_launch_host_func(
      reinterpret_cast<CUstream>(stream_ptr), notify_eventfd,
      reinterpret_cast<void*>(static_cast<intptr_t>(event_fd)));
  return static_cast<int64_t>(status);
}

}  // namespace vllm_fl
