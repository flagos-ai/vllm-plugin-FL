// Copyright (c) 2026 BAAI. All rights reserved.
// GQA-packed paged decode attention on the matrix cores of the targeted accelerator.
//
// Supports three cache formats via MODE:
//   MODE = 0 (BF16) : kc/vc rows of 128 bf16 (vLLM layout [blocks, 16, Hkv, 128])
//   MODE = 1 (I8G16): kc/vc rows of 144 bytes (128 int8 + 8 bf16 scales)
//   MODE = 2 (K8V5) : asymmetric split-bitplane block inside kv_cache [blocks, 2, 16, Hkv, 58]:
//                     K in I8G16 (144B/row) at block+0, V in I5G32 (88B/row: 64B lo4 + 16B hi1 + 8B scales)
//                     at block + 16*Hkv*144, executed via zero-spill two-phase software pipelining.
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <math.h>

using namespace nvcuda;

#define WARP 64
#define HEAD_DIM 128
#define PAGE 16
#define FULL_MASK (~0ULL)

__device__ __forceinline__ float shfl_xor_f(float v, int m) { return __shfl_xor_sync(FULL_MASK, v, m, WARP); }
__device__ __forceinline__ float shfl_f(float v, int src) { return __shfl_sync(FULL_MASK, v, src, WARP); }
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }
__device__ __forceinline__ void unpack8(const uint4 u, float* f) {
    f[0] = __uint_as_float(u.x << 16); f[1] = __uint_as_float(u.x & 0xffff0000u);
    f[2] = __uint_as_float(u.y << 16); f[3] = __uint_as_float(u.y & 0xffff0000u);
    f[4] = __uint_as_float(u.z << 16); f[5] = __uint_as_float(u.z & 0xffff0000u);
    f[6] = __uint_as_float(u.w << 16); f[7] = __uint_as_float(u.w & 0xffff0000u);
}
__device__ __forceinline__ void unpack8_i8(const uint2 u, float s, float* f) {
    f[0] = (float)(int8_t)(u.x & 0xff) * s;         f[1] = (float)(int8_t)((u.x >> 8) & 0xff) * s;
    f[2] = (float)(int8_t)((u.x >> 16) & 0xff) * s; f[3] = (float)(int8_t)((u.x >> 24) & 0xff) * s;
    f[4] = (float)(int8_t)(u.y & 0xff) * s;         f[5] = (float)(int8_t)((u.y >> 8) & 0xff) * s;
    f[6] = (float)(int8_t)((u.y >> 16) & 0xff) * s; f[7] = (float)(int8_t)((u.y >> 24) & 0xff) * s;
}

__device__ __forceinline__ uint16_t f2bf_bits(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ __forceinline__ uint32_t pack2(float a, float b) { return (uint32_t)f2bf_bits(a) | ((uint32_t)f2bf_bits(b) << 16); }

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, float, wmma::row_major> FragA;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, float, wmma::row_major> FragB;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> FragC;

__global__ void __launch_bounds__(WARP)
gqa_decode_k8v5_pipe_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc,
    uint16_t* __restrict__ out, const int* __restrict__ seq_lens, const int* __restrict__ block_table,
    float* __restrict__ ws_acc, float* __restrict__ ws_m, float* __restrict__ ws_l,
    const float scale_log2e, const int H, const int Hkv, const int G, const int bt_stride,
    const int max_splits, const int min_pages,
    const long stride_qs, const long stride_qh, const long stride_os, const long stride_oh,
    const long stride_kb)
{
    __shared__ float pbuf[16][17];
    const int seq = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z;
    const int lane = threadIdx.x;
    const int c = lane & 15, r = lane >> 4;
    const int h0 = kvh * G + r, h1 = h0 + 4;
    const int L = seq_lens[seq];
    if (L <= 0) {
        if (split == 0) {
            uint16_t* o0 = out + (long)seq * stride_os + (long)h0 * stride_oh;
            uint16_t* o1 = out + (long)seq * stride_os + (long)h1 * stride_oh;
            *reinterpret_cast<uint4*>(o0 + 8 * c) = make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(o1 + 8 * c) = make_uint4(0u, 0u, 0u, 0u);
        }
        return;
    }
    const int n_pages = (L + PAGE - 1) / PAGE;
    int n_splits = n_pages / min_pages;
    n_splits = n_splits < 1 ? 1 : (n_splits > max_splits ? max_splits : n_splits);
    if (split >= n_splits) return;
    const int pps = (n_pages + n_splits - 1) / n_splits;
    const int p0 = split * pps;
    int p1 = p0 + pps; if (p1 > n_pages) p1 = n_pages;
    if (p0 >= p1) {
        const long row0 = ((long)seq * H + h0) * max_splits + split;
        const long row1 = ((long)seq * H + h1) * max_splits + split;
        float* a0 = ws_acc + row0 * HEAD_DIM;
        float* a1 = ws_acc + row1 * HEAD_DIM;
        reinterpret_cast<float4*>(a0 + 8 * c)[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        reinterpret_cast<float4*>(a0 + 8 * c)[1] = make_float4(0.f, 0.f, 0.f, 0.f);
        reinterpret_cast<float4*>(a1 + 8 * c)[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        reinterpret_cast<float4*>(a1 + 8 * c)[1] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (c == 0) {
            ws_m[row0] = -INFINITY; ws_l[row0] = 0.f;
            ws_m[row1] = -INFINITY; ws_l[row1] = 0.f;
        }
        return;
    }

    float qb[8][4];
    if (c < 8) {
        const uint16_t* qp = q + (long)seq * stride_qs + (long)(kvh * G + c) * stride_qh;
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            float f[8];
            unpack8(*reinterpret_cast<const uint4*>(qp + 8 * (r + 4 * qq)), f);
            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) qb[ks][qq] = f[ks] * scale_log2e;
        }
    } else {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) { qb[ks][0] = 0.f; qb[ks][1] = 0.f; qb[ks][2] = 0.f; qb[ks][3] = 0.f; }
    }
    const float head_ok = (c < 8) ? 1.f : 0.f;
    float m0 = -INFINITY, l0 = 0.f;
    FragC o[8];
    #pragma unroll
    for (int ns = 0; ns < 8; ++ns) wmma::fill_fragment(o[ns], 0.f);
    const int* bt = block_table + (long)seq * bt_stride;
    const int kend = (p1 * PAGE < L) ? p1 * PAGE : L;
    const int v_blk_off = PAGE * Hkv * 144;
    const int k_grp = c >> 1;
    const int v_grp = c >> 2;
    const int hi_sh = 8 * (c & 3);

    uint2 kreg[4];
    uint16_t ksc_raw[4];
    long cur_page = bt[p0];

    auto prefetch_k = [&](int tt, long pg) {
        const uint8_t* kp = kc + pg * stride_kb;
        const int lim = kend - 1 - tt * PAGE;
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            int key = r + 4 * qq;
            key = key < lim ? key : lim;
            const int idx = key * Hkv + kvh;
            const uint8_t* kr = kp + idx * 144;
            kreg[qq]    = *reinterpret_cast<const uint2*>(kr + 8 * c);
            ksc_raw[qq] = *reinterpret_cast<const uint16_t*>(kr + 128 + 2 * k_grp);
        }
    };

    prefetch_k(p0, cur_page);

    for (int t = p0; t < p1; ++t) {
        // Phase 1: Prefetch V of current tile t while K is already in registers.
        const uint8_t* vp = kc + cur_page * stride_kb + v_blk_off;
        const int lim = kend - 1 - t * PAGE;
        uint32_t vlo[4];
        uint32_t vhi_w[4];
        uint16_t vsc_raw[4];
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            int key = r + 4 * qq;
            key = key < lim ? key : lim;
            const int idx = key * Hkv + kvh;
            const uint8_t* vr = vp + idx * 88;
            vlo[qq]     = *reinterpret_cast<const uint32_t*>(vr + 4 * c);
            vhi_w[qq]   = *reinterpret_cast<const uint32_t*>(vr + 64 + 4 * v_grp);
            vsc_raw[qq] = *reinterpret_cast<const uint16_t*>(vr + 80 + 2 * v_grp);
        }

        // Compute K WMMA (overlapped with V HBM load, zero kf[4][8] array!).
        float ksc[4];
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) ksc[qq] = bf2f(ksc_raw[qq]);
        FragC s;
        wmma::fill_fragment(s, 0.f);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            FragA a; FragB b;
            const int sh = (ks & 3) * 8;
            #pragma unroll
            for (int qq = 0; qq < 4; ++qq) {
                uint32_t w32 = (ks < 4) ? kreg[qq].x : kreg[qq].y;
                a.x[qq] = (float)(int8_t)((w32 >> sh) & 0xffu) * ksc[qq];
                b.x[qq] = qb[ks][qq];
            }
            wmma::mma_sync(s, a, b, s);
        }

        // Phase 2: kreg is consumed -> prefetch K of next tile t+1 BEFORE Softmax & V WMMA!
        if (t + 1 < p1) {
            cur_page = bt[t + 1];
            prefetch_k(t + 1, cur_page);
        }

        float sv[4];
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) sv[qq] = (t * PAGE + r + 4 * qq < kend) ? s.x[qq] : -INFINITY;
        float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
        mx = fmaxf(mx, shfl_xor_f(mx, 16));
        mx = fmaxf(mx, shfl_xor_f(mx, 32));
        const float mn = fmaxf(m0, mx);
        const float mn_safe = (mn == -INFINITY) ? 0.f : mn;
        const float al = exp2f(m0 - mn_safe);
        m0 = mn;
        float psum = 0.f;
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            const float pv = exp2f(sv[qq] - mn_safe) * head_ok;
            psum += pv;
            pbuf[r + 4 * qq][c] = pv;
        }
        l0 = l0 * al + psum;
        const float al0 = shfl_f(al, r), al1 = shfl_f(al, r + 4);
        #pragma unroll
        for (int ns = 0; ns < 8; ++ns) { o[ns].x[0] *= al0; o[ns].x[1] *= al1; }
        __syncwarp();
        FragA pa;
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) pa.x[qq] = pbuf[c][r + 4 * qq];

        float vsc[4];
        uint32_t vhi_b[4];
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            vsc[qq]   = bf2f(vsc_raw[qq]);
            vhi_b[qq] = (vhi_w[qq] >> hi_sh) & 0xffu;
        }
        #pragma unroll
        for (int ns = 0; ns < 8; ++ns) {
            FragB b;
            #pragma unroll
            for (int qq = 0; qq < 4; ++qq) {
                int u5 = (int)(((vlo[qq] >> (4 * ns)) & 0xfu) | (((vhi_b[qq] >> ns) & 1u) << 4));
                b.x[qq] = (float)(u5 - 16) * vsc[qq];
            }
            wmma::mma_sync(o[ns], pa, b, o[ns]);
        }
        __syncwarp();
    }
    l0 += shfl_xor_f(l0, 16);
    l0 += shfl_xor_f(l0, 32);
    const float inv = 1.f / fmaxf(l0, 1e-30f);
    const float inv0 = shfl_f(inv, r), inv1 = shfl_f(inv, r + 4);
    const float mo0 = shfl_f(m0, r), mo1 = shfl_f(m0, r + 4);
    const float lo0 = shfl_f(l0, r), lo1 = shfl_f(l0, r + 4);
    if (n_splits == 1) {
        uint16_t* o0 = out + (long)seq * stride_os + (long)h0 * stride_oh;
        uint16_t* o1 = out + (long)seq * stride_os + (long)h1 * stride_oh;
        uint4 pk;
        pk.x = pack2(o[0].x[0] * inv0, o[1].x[0] * inv0); pk.y = pack2(o[2].x[0] * inv0, o[3].x[0] * inv0);
        pk.z = pack2(o[4].x[0] * inv0, o[5].x[0] * inv0); pk.w = pack2(o[6].x[0] * inv0, o[7].x[0] * inv0);
        *reinterpret_cast<uint4*>(o0 + 8 * c) = pk;
        pk.x = pack2(o[0].x[1] * inv1, o[1].x[1] * inv1); pk.y = pack2(o[2].x[1] * inv1, o[3].x[1] * inv1);
        pk.z = pack2(o[4].x[1] * inv1, o[5].x[1] * inv1); pk.w = pack2(o[6].x[1] * inv1, o[7].x[1] * inv1);
        *reinterpret_cast<uint4*>(o1 + 8 * c) = pk;
    } else {
        const long row0 = ((long)seq * H + h0) * max_splits + split;
        const long row1 = ((long)seq * H + h1) * max_splits + split;
        float* a0 = ws_acc + row0 * HEAD_DIM;
        float* a1 = ws_acc + row1 * HEAD_DIM;
        reinterpret_cast<float4*>(a0 + 8 * c)[0] = make_float4(o[0].x[0], o[1].x[0], o[2].x[0], o[3].x[0]);
        reinterpret_cast<float4*>(a0 + 8 * c)[1] = make_float4(o[4].x[0], o[5].x[0], o[6].x[0], o[7].x[0]);
        reinterpret_cast<float4*>(a1 + 8 * c)[0] = make_float4(o[0].x[1], o[1].x[1], o[2].x[1], o[3].x[1]);
        reinterpret_cast<float4*>(a1 + 8 * c)[1] = make_float4(o[4].x[1], o[5].x[1], o[6].x[1], o[7].x[1]);
        if (c == 0) { ws_m[row0] = mo0; ws_l[row0] = lo0; ws_m[row1] = mo1; ws_l[row1] = lo1; }
    }
}

template <int MODE>
__global__ void __launch_bounds__(WARP)
gqa_decode_kernel(const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                  uint16_t* __restrict__ out, const int* __restrict__ seq_lens, const int* __restrict__ block_table,
                  float* __restrict__ ws_acc, float* __restrict__ ws_m, float* __restrict__ ws_l,
                  const float scale_log2e, const int H, const int Hkv, const int G, const int bt_stride,
                  const int max_splits, const int min_pages,
                  const long stride_qs, const long stride_qh, const long stride_os, const long stride_oh,
                  const long stride_kb, const long stride_ks, const long stride_kh)
{
    __shared__ float pbuf[16][17];
    const int seq = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z;
    const int lane = threadIdx.x;
    const int c = lane & 15, r = lane >> 4;
    const int h0 = kvh * G + r, h1 = h0 + 4;
    const int L = seq_lens[seq];
    if (L <= 0) {
        if (split == 0) {
            uint16_t* o0 = out + (long)seq * stride_os + (long)h0 * stride_oh;
            uint16_t* o1 = out + (long)seq * stride_os + (long)h1 * stride_oh;
            *reinterpret_cast<uint4*>(o0 + 8 * c) = make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(o1 + 8 * c) = make_uint4(0u, 0u, 0u, 0u);
        }
        return;
    }
    const int n_pages = (L + PAGE - 1) / PAGE;
    int n_splits = n_pages / min_pages;
    n_splits = n_splits < 1 ? 1 : (n_splits > max_splits ? max_splits : n_splits);
    if (split >= n_splits) return;
    const int pps = (n_pages + n_splits - 1) / n_splits;
    const int p0 = split * pps;
    int p1 = p0 + pps; if (p1 > n_pages) p1 = n_pages;
    if (p0 >= p1) {
        const long row0 = ((long)seq * H + h0) * max_splits + split;
        const long row1 = ((long)seq * H + h1) * max_splits + split;
        float* a0 = ws_acc + row0 * HEAD_DIM;
        float* a1 = ws_acc + row1 * HEAD_DIM;
        reinterpret_cast<float4*>(a0 + 8 * c)[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        reinterpret_cast<float4*>(a0 + 8 * c)[1] = make_float4(0.f, 0.f, 0.f, 0.f);
        reinterpret_cast<float4*>(a1 + 8 * c)[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        reinterpret_cast<float4*>(a1 + 8 * c)[1] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (c == 0) {
            ws_m[row0] = -INFINITY; ws_l[row0] = 0.f;
            ws_m[row1] = -INFINITY; ws_l[row1] = 0.f;
        }
        return;
    }

    float qb[8][4];
    if (c < 8) {
        const uint16_t* qp = q + (long)seq * stride_qs + (long)(kvh * G + c) * stride_qh;
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            float f[8];
            unpack8(*reinterpret_cast<const uint4*>(qp + 8 * (r + 4 * qq)), f);
            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) qb[ks][qq] = f[ks] * scale_log2e;
        }
    } else {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) { qb[ks][0] = 0.f; qb[ks][1] = 0.f; qb[ks][2] = 0.f; qb[ks][3] = 0.f; }
    }
    const float head_ok = (c < 8) ? 1.f : 0.f;
    float m0 = -INFINITY, l0 = 0.f;
    FragC o[8];
    #pragma unroll
    for (int ns = 0; ns < 8; ++ns) wmma::fill_fragment(o[ns], 0.f);
    const int* bt = block_table + (long)seq * bt_stride;
    const int kend = (p1 * PAGE < L) ? p1 * PAGE : L;
    const uint8_t* kbase = kc + (long)kvh * stride_kh;
    const uint8_t* vbase = vc + (long)kvh * stride_kh;
    const int k_grp = c >> 1;

    uint4 kreg16[4], vreg16[4];
    uint2 kreg8[4], vreg8[4];
    uint16_t ksc[4], vsc[4];

    auto load_tile = [&](int tt) {
        const long page = bt[tt];
        const uint8_t* kp = kbase + page * stride_kb;
        const uint8_t* vp = vbase + page * stride_kb;
        const int lim = kend - 1 - tt * PAGE;
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            int key = r + 4 * qq;
            key = key < lim ? key : lim;
            const uint8_t* kr = kp + (long)key * stride_ks;
            const uint8_t* vr = vp + (long)key * stride_ks;
            if (MODE == 1) {
                kreg8[qq] = *reinterpret_cast<const uint2*>(kr + 8 * c);
                ksc[qq]   = *reinterpret_cast<const uint16_t*>(kr + 128 + 2 * k_grp);
                vreg8[qq] = *reinterpret_cast<const uint2*>(vr + 8 * c);
                vsc[qq]   = *reinterpret_cast<const uint16_t*>(vr + 128 + 2 * k_grp);
            } else {
                kreg16[qq] = *reinterpret_cast<const uint4*>(kr + 16 * c);
                vreg16[qq] = *reinterpret_cast<const uint4*>(vr + 16 * c);
            }
        }
    };

    load_tile(p0);
    for (int t = p0; t < p1; ++t) {
        float kf[4][8];
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            if (MODE == 1) unpack8_i8(kreg8[qq], bf2f(ksc[qq]), kf[qq]);
            else           unpack8(kreg16[qq], kf[qq]);
        }
        FragC s;
        wmma::fill_fragment(s, 0.f);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            FragA a; FragB b;
            a.x[0] = kf[0][ks]; a.x[1] = kf[1][ks]; a.x[2] = kf[2][ks]; a.x[3] = kf[3][ks];
            b.x[0] = qb[ks][0]; b.x[1] = qb[ks][1]; b.x[2] = qb[ks][2]; b.x[3] = qb[ks][3];
            wmma::mma_sync(s, a, b, s);
        }
        float sv[4];
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) sv[qq] = (t * PAGE + r + 4 * qq < kend) ? s.x[qq] : -INFINITY;
        float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
        mx = fmaxf(mx, shfl_xor_f(mx, 16));
        mx = fmaxf(mx, shfl_xor_f(mx, 32));
        const float mn = fmaxf(m0, mx);
        const float mn_safe = (mn == -INFINITY) ? 0.f : mn;
        const float al = exp2f(m0 - mn_safe);
        m0 = mn;
        float psum = 0.f;
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            const float pv = exp2f(sv[qq] - mn_safe) * head_ok;
            psum += pv;
            pbuf[r + 4 * qq][c] = pv;
        }
        l0 = l0 * al + psum;
        const float al0 = shfl_f(al, r), al1 = shfl_f(al, r + 4);
        #pragma unroll
        for (int ns = 0; ns < 8; ++ns) { o[ns].x[0] *= al0; o[ns].x[1] *= al1; }
        __syncwarp();
        FragA pa;
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) pa.x[qq] = pbuf[c][r + 4 * qq];
        float vf[4][8];
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            if (MODE == 1) unpack8_i8(vreg8[qq], bf2f(vsc[qq]), vf[qq]);
            else           unpack8(vreg16[qq], vf[qq]);
        }
        #pragma unroll
        for (int ns = 0; ns < 8; ++ns) {
            FragB b;
            b.x[0] = vf[0][ns]; b.x[1] = vf[1][ns]; b.x[2] = vf[2][ns]; b.x[3] = vf[3][ns];
            wmma::mma_sync(o[ns], pa, b, o[ns]);
        }
        __syncwarp();
        if (t + 1 < p1) load_tile(t + 1);
    }
    l0 += shfl_xor_f(l0, 16);
    l0 += shfl_xor_f(l0, 32);
    const float inv = 1.f / fmaxf(l0, 1e-30f);
    const float inv0 = shfl_f(inv, r), inv1 = shfl_f(inv, r + 4);
    const float mo0 = shfl_f(m0, r), mo1 = shfl_f(m0, r + 4);
    const float lo0 = shfl_f(l0, r), lo1 = shfl_f(l0, r + 4);
    if (n_splits == 1) {
        uint16_t* o0 = out + (long)seq * stride_os + (long)h0 * stride_oh;
        uint16_t* o1 = out + (long)seq * stride_os + (long)h1 * stride_oh;
        uint4 pk;
        pk.x = pack2(o[0].x[0] * inv0, o[1].x[0] * inv0); pk.y = pack2(o[2].x[0] * inv0, o[3].x[0] * inv0);
        pk.z = pack2(o[4].x[0] * inv0, o[5].x[0] * inv0); pk.w = pack2(o[6].x[0] * inv0, o[7].x[0] * inv0);
        *reinterpret_cast<uint4*>(o0 + 8 * c) = pk;
        pk.x = pack2(o[0].x[1] * inv1, o[1].x[1] * inv1); pk.y = pack2(o[2].x[1] * inv1, o[3].x[1] * inv1);
        pk.z = pack2(o[4].x[1] * inv1, o[5].x[1] * inv1); pk.w = pack2(o[6].x[1] * inv1, o[7].x[1] * inv1);
        *reinterpret_cast<uint4*>(o1 + 8 * c) = pk;
    } else {
        const long row0 = ((long)seq * H + h0) * max_splits + split;
        const long row1 = ((long)seq * H + h1) * max_splits + split;
        float* a0 = ws_acc + row0 * HEAD_DIM;
        float* a1 = ws_acc + row1 * HEAD_DIM;
        reinterpret_cast<float4*>(a0 + 8 * c)[0] = make_float4(o[0].x[0], o[1].x[0], o[2].x[0], o[3].x[0]);
        reinterpret_cast<float4*>(a0 + 8 * c)[1] = make_float4(o[4].x[0], o[5].x[0], o[6].x[0], o[7].x[0]);
        reinterpret_cast<float4*>(a1 + 8 * c)[0] = make_float4(o[0].x[1], o[1].x[1], o[2].x[1], o[3].x[1]);
        reinterpret_cast<float4*>(a1 + 8 * c)[1] = make_float4(o[4].x[1], o[5].x[1], o[6].x[1], o[7].x[1]);
        if (c == 0) { ws_m[row0] = mo0; ws_l[row0] = lo0; ws_m[row1] = mo1; ws_l[row1] = lo1; }
    }
}

extern "C" int gqa_decode_native_launch(int mode, const void* q, const void* kc, const void* vc, void* out,
                   const int* seq_lens, const int* bt, float* ws_acc, float* ws_m, float* ws_l, float scale_log2e,
                   int S, int Hkv, int H, int G, int bt_stride, int max_splits, int min_pages,
                   long sqs, long sqh, long sos, long soh, long skb, long sks, long skh, cudaStream_t stream)
{
    if (G != 8) return -1;
    dim3 grid(S, Hkv, max_splits);
    if (mode == 2)
        gqa_decode_k8v5_pipe_kernel<<<grid, WARP, 0, stream>>>((const uint16_t*)q, (const uint8_t*)kc, (uint16_t*)out,
            seq_lens, bt, ws_acc, ws_m, ws_l, scale_log2e, H, Hkv, G, bt_stride, max_splits, min_pages, sqs, sqh, sos, soh, skb);
    else if (mode == 1)
        gqa_decode_kernel<1><<<grid, WARP, 0, stream>>>((const uint16_t*)q, (const uint8_t*)kc, (const uint8_t*)vc, (uint16_t*)out,
            seq_lens, bt, ws_acc, ws_m, ws_l, scale_log2e, H, Hkv, G, bt_stride, max_splits, min_pages, sqs, sqh, sos, soh, skb, sks, skh);
    else
        gqa_decode_kernel<0><<<grid, WARP, 0, stream>>>((const uint16_t*)q, (const uint8_t*)kc, (const uint8_t*)vc, (uint16_t*)out,
            seq_lens, bt, ws_acc, ws_m, ws_l, scale_log2e, H, Hkv, G, bt_stride, max_splits, min_pages, sqs, sqh, sos, soh, skb, sks, skh);
    return 0;
}
