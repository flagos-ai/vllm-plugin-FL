// Copyright (c) 2026 BAAI. All rights reserved.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" int gqa_decode_native_launch(int mode, const void* q, const void* kc, const void* vc, void* out,
                   const int* seq_lens, const int* bt, float* ws_acc, float* ws_m, float* ws_l, float scale_log2e,
                   int S, int Hkv, int H, int G, int bt_stride, int max_splits, int min_pages,
                   long sqs, long sqh, long sos, long soh, long skb, long sks, long skh, cudaStream_t stream);

// mode: 0 = BF16 [blocks, 16, Hkv, 128], 1 = I8G16 uint8 [blocks, 16, Hkv, 144],
//       2 = K8V5 bf16/uint8 [blocks, 2, 16, Hkv, 58] (k_cache passes full kv_cache).
void gqa_decode_native(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor out,
                       torch::Tensor seq_lens, torch::Tensor block_table,
                       torch::Tensor ws_acc, torch::Tensor ws_m, torch::Tensor ws_l,
                       double scale_log2e, int64_t max_splits, int64_t min_pages, int64_t mode)
{
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.size(2) == 128 && q.stride(2) == 1, "q must be bf16 [S, H, 128]");
    const int S = q.size(0), H = q.size(1);
    int Hkv = 0;
    long skb = 0, sks = 0, skh = 0;
    const void* k_ptr = k_cache.data_ptr();
    const void* v_ptr = v_cache.data_ptr();

    if (mode == 2) {
        TORCH_CHECK(k_cache.dim() == 5 && k_cache.size(1) == 2 && k_cache.size(2) == 16 && k_cache.size(4) == 58,
                    "K8V5 kv_cache must be [blocks, 2, 16, Hkv, 58] bf16");
        Hkv = k_cache.size(3);
        skb = k_cache.stride(0) * (long)k_cache.element_size();
    } else if (mode == 1) {
        TORCH_CHECK(k_cache.dtype() == torch::kUInt8 && k_cache.size(1) == 16 && k_cache.size(3) == 144,
                    "I8G16 cache rows must be 144 bytes");
        Hkv = k_cache.size(2);
        skb = k_cache.stride(0); sks = k_cache.stride(1); skh = k_cache.stride(2);
    } else {
        TORCH_CHECK(k_cache.dtype() == torch::kBFloat16 && k_cache.size(1) == 16 && k_cache.size(3) == 128,
                    "bf16 cache rows must be 128 elements");
        Hkv = k_cache.size(2);
        skb = k_cache.stride(0) * 2; sks = k_cache.stride(1) * 2; skh = k_cache.stride(2) * 2;
    }

    const int G = H / Hkv;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int rc = gqa_decode_native_launch((int)mode, q.data_ptr(), k_ptr, v_ptr, out.data_ptr(),
        seq_lens.data_ptr<int>(), block_table.data_ptr<int>(),
        ws_acc.data_ptr<float>(), ws_m.data_ptr<float>(), ws_l.data_ptr<float>(), (float)scale_log2e,
        S, Hkv, H, G, (int)block_table.stride(0), (int)max_splits, (int)min_pages,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        skb, sks, skh, stream);
    TORCH_CHECK(rc == 0, "gqa_decode_native_launch failed: ", rc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gqa_decode_native", &gqa_decode_native, "GQA paged decode attention on the matrix cores (BF16 / I8G16 / K8V5)");
}
