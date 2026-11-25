// Standalone build of vLLM's official Custom AllReduce for Hygon/HIP.
//
// Kernel: vllm-0.24.0/csrc/custom_all_reduce.cuh (自带 USE_ROCM/HIP 支持, 原样使用)
// Bindings: 经典 at::Tensor API 重写 (vllm 0.24 自带的 libtorch_stable 绑定依赖
//           stable-ABI 头链条, 此处改用与 hcu_ops 完全相同的算子签名注册进
//           torch.ops._C_custom_ar, 使 vllm 原生 CustomAllreduce python 类可直接使用).
//
// Build: python3 build.py  (torch.utils.cpp_extension.load, ROCm 下自动 hipcc)

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <tuple>
#include <vector>

#include "custom_all_reduce.cuh"

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

static void _weak_contiguous_check(const at::Tensor& t) {
  // 与 vllm 语义一致: 允许 contiguous 或首维切片
  TORCH_CHECK(t.is_contiguous() ||
                  t.storage().nbytes() - t.storage_offset() * t.element_size() ==
                      t.numel() * t.element_size(),
              "custom allreduce requires weakly contiguous tensors");
}

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      at::Tensor& rank_data, int64_t rank,
                      bool fully_connected) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new vllm::CustomAllreduce(
      ipc_ptrs, rank_data.mutable_data_ptr(), rank_data.numel(), rank,
      world_size, fully_connected);
}

void all_reduce(fptr_t _fa, at::Tensor& inp, at::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  c10::DeviceGuard device_guard(at::Device(at::kCUDA, inp.get_device()));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(inp.get_device());

  TORCH_CHECK(inp.scalar_type() == out.scalar_type());
  TORCH_CHECK(inp.numel() == out.numel());
  _weak_contiguous_check(inp);
  _weak_contiguous_check(out);
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK(input_size <= reg_buffer_sz_bytes);
    cudaMemcpyAsync(reg_buffer, inp.const_data_ptr(), input_size,
                    cudaMemcpyDeviceToDevice, stream);
  } else {
    reg_buffer = inp.mutable_data_ptr();
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.mutable_data_ptr()),
                           out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.mutable_data_ptr()),
                          out.numel());
      break;
// BW1000(gfx936) 原生支持 bf16; dtk hipcc 的 host pass 也会定义 __CUDA_ARCH__,
// 原版 guard 会误排除 bf16, 此处对 HIP 放开.
#if (defined(USE_ROCM) || __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), out.numel());
      break;
#endif
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
}

void dispose(fptr_t _fa) { delete reinterpret_cast<vllm::CustomAllreduce*>(_fa); }

int64_t meta_size() { return sizeof(vllm::Signal); }

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  fa->register_graph_buffers(bytes, offsets);
}

std::tuple<fptr_t, at::Tensor> allocate_shared_buffer_and_handle(int64_t size) {
  int device_index;
  cudaGetDevice(&device_index);
  c10::DeviceGuard device_guard(at::Device(at::kCUDA, device_index));
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index);
  cudaThreadExchangeStreamCaptureMode(&mode);

#if defined(USE_ROCM)
  // 与 vllm 一致: signal buffer 在 MI 系列需要 uncached 分配
  hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached);
#else
  cudaMalloc((void**)&buffer, size);
#endif
  cudaMemsetAsync(buffer, 0, size, stream);
  cudaStreamSynchronize(stream);
  cudaThreadExchangeStreamCaptureMode(&mode);

  auto handle = torch::empty(
      {static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.mutable_data_ptr(), buffer);
  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(at::Tensor& mem_handle) {
  void* ipc_ptr;
  cudaIpcOpenMemHandle((void**)&ipc_ptr,
                       *((const cudaIpcMemHandle_t*)mem_handle.const_data_ptr()),
                       cudaIpcMemLazyEnablePeerAccess);
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
  cudaFree(reinterpret_cast<void*>(buffer));
}

// DEBUG: 从任意设备指针读 n 个 bf16 (验证 IPC 映射内容)
std::vector<double> debug_peek_bf16(int64_t ptr, int64_t n) {
  std::vector<at::BFloat16> tmp(n);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(tmp.data(), (void*)ptr, n * sizeof(at::BFloat16),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::vector<double> out;
  for (auto v : tmp) out.push_back(static_cast<double>(v));
  return out;
}

// DEBUG: 从任意设备指针读 n 个 bf16 (验证 IPC 映射内容)

// DEBUG: 微内核, 向 out 写 42.0
__global__ void debug_fill_kernel(float* out) {
  out[threadIdx.x + blockIdx.x * blockDim.x] = 42.0f;
}

// DEBUG: 变体1 同 kernel 但用 hipLaunchKernel(void** args) 启动
static void launch_fill_variant(void* fn, dim3 g, dim3 b, void** args,
                                cudaStream_t stream) {
  hipLaunchKernel(fn, g, b, args, 0, stream);
}

// DEBUG: eager 填充 (variant 0=hipLaunchKernelGGL, 1=hipLaunchKernel); 捕获由 python 侧 torch.cuda.graph 驱动
void debug_fill(at::Tensor& out, int64_t variant) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  float* outp = (float*)out.mutable_data_ptr();
  void* args[1] = {&outp};
  if (variant == 0)
    hipLaunchKernelGGL(debug_fill_kernel, dim3(2), dim3(128), 0, stream, outp);
  else
    hipLaunchKernel((void*)debug_fill_kernel, dim3(2), dim3(128), args, 0, stream);
}

// DEBUG: 复制 RankData 表 (从表头 num_rows 行, 每行 world 个对端指针) 到 host
std::vector<int64_t> debug_read_rank_data(fptr_t _fa, int64_t num_rows,
                                          int64_t world) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  num_rows = std::min<int64_t>(num_rows, 8);
  std::vector<vllm::RankData> rows(num_rows);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(rows.data(), fa->rank_data_base_for_debug(),
                  sizeof(vllm::RankData) * num_rows,
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::vector<int64_t> out;
  for (auto& r : rows)
    for (int j = 0; j < world; j++) out.push_back((int64_t)r.ptrs[j]);
  return out;
}

TORCH_LIBRARY(_C_custom_ar, m) {
  m.def("init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
        "int rank, bool fully_connected) -> int");
  m.def("all_reduce(int fa, Tensor inp, Tensor(a!) out, int reg_buffer, "
        "int reg_buffer_sz_bytes) -> ()");
  m.def("dispose(int fa) -> ()");
  m.def("meta_size() -> int");
  m.def("register_buffer(int fa, int[] ipc_tensors) -> ()");
  m.def("get_graph_buffer_ipc_meta(int fa) -> (int[], int[])");
  m.def("register_graph_buffers(int fa, int[][] handles, int[][] offsets) -> ()");
  m.def("allocate_shared_buffer_and_handle(int size) -> (int, Tensor)");
  m.def("open_mem_handle(Tensor mem_handle) -> int");
  m.def("free_shared_buffer(int ptr) -> ()");
  m.def("debug_read_rank_data(int fa, int num_rows, int world) -> int[]");
  m.def("debug_peek_bf16(int ptr, int n) -> float[]");
  m.def("debug_fill(Tensor(a!) out, int variant) -> ()");
}

TORCH_LIBRARY_IMPL(_C_custom_ar, CUDA, m) {
  m.impl("init_custom_ar", &init_custom_ar);
  m.impl("all_reduce", &all_reduce);
}

TORCH_LIBRARY_IMPL(_C_custom_ar, CompositeExplicitAutograd, m) {
  m.impl("dispose", &dispose);
  m.impl("meta_size", &meta_size);
  m.impl("register_buffer", &register_buffer);
  m.impl("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.impl("register_graph_buffers", &register_graph_buffers);
  m.impl("allocate_shared_buffer_and_handle",
         &allocate_shared_buffer_and_handle);
  m.impl("open_mem_handle", &open_mem_handle);
  m.impl("free_shared_buffer", &free_shared_buffer);
  m.impl("debug_read_rank_data", &debug_read_rank_data);
  m.impl("debug_peek_bf16", &debug_peek_bf16);
  m.impl("debug_fill", &debug_fill);
  m.impl("debug_peek_bf16", &debug_peek_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("meta_size", &meta_size, "vllm custom AR standalone build");
}
