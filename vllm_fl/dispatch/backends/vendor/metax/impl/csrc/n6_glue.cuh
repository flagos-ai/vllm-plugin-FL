// SPDX-License-Identifier: Apache-2.0
// 2026 - FlagOS S2 contest, R7/N6: own glue kernels of a Llama-style decoder layer for
// the MetaX C500, native MACA C++ (our design; not a vendor kernel).
//
// One header, two builds:
//   * device (mxcc -x maca, n6_capi.cu -> libn6_glue.so, C ABI, loaded with ctypes by the
//     plugin's impl/mx_glue_native.py and by the GPU bench);
//   * CPU emulator (-DN6_EMU, ../emu/n1_emu.h, clang++): every GPU thread is an OS thread
//     -> offline checks of all index math, the reduction and the rounding.
//
// Why: between the four projection GEMMs of a layer the served graph runs small
// memory-bound kernels.  Measured per layer on the served MiniCPM5-2B (R6/G1 trace,
// hidden 2048, 16 q / 2 kv heads of 128, intermediate 6144, page 16):
//
//   kernel (deployed origin)             M=2048 mixed step    M=64 decode (in graph)
//   add+RMSNorm x2 (Inductor, 2 red.)    30.2 + 46.3 us       6.5 + 6.6 us
//   RoPE q / k    (Inductor "cat" x2)    25.9 + 6.9 us        4.6 + 4.3 us
//   KV write      (vendor reshape_cache) 18.9 us              4.5 us
//   SiLU*mul      (Inductor)             62.2 us              6.0 us
//
// i.e. 7.2 ms of an 84 ms mixed step and 1.18 ms of a 17.9 ms decode step, in 6 launches
// per layer (13 of the step's 13.5 kernels per layer).  R6/G1 fused them in Triton and
// won on the graph-captured decode steps (-0.32 ms/step) but LOST the mixed step: at
// M=2048 a FlagGems/Triton launch costs 61-116 us of HOST time against 22-30 us of GPU
// time, and mixed steps are not graph-captured, so the host became the critical path
// (4k mean TTFT 1904 -> 2055 ms).  These kernels are the mixed-step answer: one launch
// each, every tensor read and written once, ctypes -> ~4 us of host time.
//
// Kernels
//   n6_norm_kernel   residual add + RMSNorm, one CTA per row, the row held in registers:
//                    s = x + r (fp32), ro = bf16(s), y = bf16(s * rsqrt(mean(s^2)+eps) * w).
//                    fp32 accumulation (never bf16), one rounding per stored tensor.
//   n6_rope_kernel   neox RoPE of q and of k + the paged KV write in ONE kernel: a CTA
//                    takes TPB tokens and either HPG q heads (-> q_out) or the HK k heads
//                    (rotated into the cache) and v heads (copied into the cache).
//                    cos/sin are loaded once per token and reused over the head loop.
//   n6_silu_kernel   SiLU(gate) * up of the [M, 2I] gate_up GEMM output.
//
// All three use 128-bit (8 bf16) loads and stores only, keep the arithmetic in fp32 and
// round once (RTNE, like torch / Triton .to(bf16)).  Built with -ffp-contract=off so that
// x1*cos - x2*sin is NOT contracted into an fma: that keeps the rotated q and the cache
// contents bitwise equal to the deployed Inductor RoPE and the vendor cache write.
// Nothing allocates, synchronises or reads the host: every launch is CUDA-graph safe.
#pragma once
#include <stdint.h>

#define N6_VERSION 2
#define N6_VEC 8        // bf16 elements per 128-bit access
#define N6_MAXN 8192    // hidden size the norm configs cover
#define N6_MAXVPT 32    // fp32 values held per thread by the norm kernel

#ifndef N6_SAFE
#define N6_SAFE 0       // 1: conservative build (no vector ext types, scalar accesses)
#endif

#ifdef N6_EMU
#include "n1_emu.h"
#define N6_DEV N1_DEV
#define N6_GLOBAL
#define N6_LAUNCH_BOUNDS(n)
#define N6_SMEM(T, name) N1_SMEM(T, name)
#define N6_SYNC() N1_SYNC()
#define N6_SHFL_XOR_F(v, m) N1_SHFL_XOR_F(v, m)
#define N6_TID() N1_TID()
#define N6_BID() N1_BID()
#else
#define N6_DEV __device__ __forceinline__
#define N6_GLOBAL __global__
#define N6_LAUNCH_BOUNDS(n) __launch_bounds__(n)
#define N6_SMEM(T, name) __shared__ T name
#define N6_SYNC() __syncthreads()
#define N6_SHFL_XOR_F(v, m) __shfl_xor_sync((uint64_t)(-1), (float)(v), (m))
#define N6_TID() ((int)threadIdx.x)
#define N6_BID() ((int)blockIdx.x)
#if !defined(N6_STREAM_T) || !defined(N6_LAST_ERROR)
#error "front-end must define N6_STREAM_T and N6_LAST_ERROR() before including n6_glue.cuh"
#endif
#endif

// 128-bit access type (the clang vector extension, as in attention/ops/csrc/n1_dec_attn.cuh:
// every load and store of these kernels is one 16-byte access)
typedef uint32_t n6u4 __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
N6_DEV float n6_u2f(uint32_t u) { return __builtin_bit_cast(float, u); }
N6_DEV uint32_t n6_f2u(float f) { return __builtin_bit_cast(uint32_t, f); }
N6_DEV float n6_bflo(uint32_t w) { return n6_u2f(w << 16); }
N6_DEV float n6_bfhi(uint32_t w) { return n6_u2f(w & 0xffff0000u); }
// element e (0..7) of a 128-bit bf16 vector
N6_DEV float n6_bfe(n6u4 v, int e) {
  const uint32_t w = v[e >> 1];
  return (e & 1) ? n6_bfhi(w) : n6_bflo(w);
}
// fp32 -> bf16 bits, round to nearest even, NaN kept quiet (same as torch .to(bf16))
N6_DEV uint32_t n6_f2bf(float f) {
  uint32_t u = n6_f2u(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (u >> 16) | ((u & 0xffffu) ? 0x40u : 0u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
N6_DEV uint32_t n6_pack2(float lo, float hi) { return n6_f2bf(lo) | (n6_f2bf(hi) << 16); }
N6_DEV float n6_rsqrt(float x) { return 1.0f / __builtin_sqrtf(x); }
N6_DEV long long n6_ld_idx(const void* p, int i, int wide) {
  return wide ? ((const long long*)p)[i] : (long long)((const int*)p)[i];
}

// ---------------------------------------------------------------------------
// residual add + RMSNorm
// ---------------------------------------------------------------------------
// y[m, :] = bf16(s * rsqrt(sum(s^2)/N + eps) * w),  ro[m, :] = bf16(s),  s = x + r (fp32)
// One CTA per row, NT = 64 * W threads, VPT fp32 values per thread, N == NT * VPT.
// Thread tid owns the 128-bit chunks tid, tid + NT, ... (fully coalesced per instruction).
struct N6NormParams {
  const uint16_t* x;
  long long x_rs;
  const uint16_t* r;
  long long r_rs;
  const uint16_t* w;
  uint16_t* y;
  long long y_rs;
  uint16_t* ro;
  long long ro_rs;
  int m, n;
  float eps, inv_n;
};

struct N6Red {
  float w[16];
};

template <int W, int VPT>
N6_GLOBAL void N6_LAUNCH_BOUNDS(64 * W) n6_norm_kernel(N6NormParams p) {
  const int NT = 64 * W, VC = VPT / N6_VEC;
  const int tid = N6_TID(), row = N6_BID();
  const uint16_t* xp = p.x + (long long)row * p.x_rs;
  const uint16_t* rp = p.r + (long long)row * p.r_rs;
  n6u4 xv[VC], rv[VC];
#pragma unroll
  for (int j = 0; j < VC; ++j) {
    const long long off = (long long)(tid + j * NT) * N6_VEC;
    xv[j] = *(const n6u4*)(xp + off);
    rv[j] = *(const n6u4*)(rp + off);
  }
  float s[VPT];
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < VC; ++j) {
#pragma unroll
    for (int e = 0; e < N6_VEC; ++e) {
      const float v = n6_bfe(xv[j], e) + n6_bfe(rv[j], e);
      s[j * N6_VEC + e] = v;
      acc += v * v;   // fp32 accumulation (-ffp-contract=off: no fma)
    }
  }
  // the new residual does not depend on the reduction: store it first
  uint16_t* rop = p.ro + (long long)row * p.ro_rs;
#pragma unroll
  for (int j = 0; j < VC; ++j) {
    n6u4 o;
#pragma unroll
    for (int t = 0; t < 4; ++t)
      o[t] = n6_pack2(s[j * N6_VEC + 2 * t], s[j * N6_VEC + 2 * t + 1]);
    *(n6u4*)(rop + (long long)(tid + j * NT) * N6_VEC) = o;
  }
  // sum of squares over the row: 64-lane warp tree, then the W warps through shared memory
#pragma unroll
  for (int d = 32; d; d >>= 1) acc += N6_SHFL_XOR_F(acc, d);
  if (W > 1) {
    N6_SMEM(N6Red, red);
    if ((tid & 63) == 0) red.w[tid >> 6] = acc;
    N6_SYNC();
    acc = 0.f;
#pragma unroll
    for (int i = 0; i < W; ++i) acc += red.w[i];
  }
  const float rstd = n6_rsqrt(acc * p.inv_n + p.eps);
  uint16_t* yp = p.y + (long long)row * p.y_rs;
#pragma unroll
  for (int j = 0; j < VC; ++j) {
    const long long off = (long long)(tid + j * NT) * N6_VEC;
    const n6u4 wv = *(const n6u4*)(p.w + off);
    n6u4 o;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int e = 2 * t;
      o[t] = n6_pack2(s[j * N6_VEC + e] * rstd * n6_bfe(wv, e),
                      s[j * N6_VEC + e + 1] * rstd * n6_bfe(wv, e + 1));
    }
    *(n6u4*)(yp + off) = o;
  }
}

// ---------------------------------------------------------------------------
// neox RoPE of q and k + paged KV write
// ---------------------------------------------------------------------------
// q [T, HQ, D] -> qo [T, HQ, D] rotated; k [T, HK, D] rotated into kc, v copied into vc at
// slot_mapping[t] (slot < 0 or t >= n_slots: skipped, like reshape_and_cache_flash).
// rope_only == 1 (R10/F): the SAME rotation without the cache write -- k is rotated into
// ko [T, HK, D] (ko may alias k: every thread reads the two halves it owns before it
// writes them back) and v / kc / vc / slot are unused.  That is the form the INT8 KV path
// needs, because its pages are written later by the quantising kernel, and it is the only
// RoPE in this file that is legal inside a CUDA-graph capture: the torch fallback it
// replaces reaches FlagGems' generic broadcast multiply, which copies its shape/stride
// meta tensor from the host at every call.
// D = 2 * half, rotary_dim == D (neox): out[d] = x[d]*cos[d] - x[d+half]*sin[d],
// out[d+half] = x[d+half]*cos[d] + x[d]*sin[d], cos/sin from cs[pos] = (cos | sin).
// Grid is flat: bid = tile * (NQG + 1) + g; g < NQG does q heads [g*HPG, (g+1)*HPG),
// g == NQG does the k and v heads of the same TPB tokens.
struct N6RopeParams {
  const uint16_t* q;
  long long q_rs, q_hs;
  const uint16_t* k;
  long long k_rs, k_hs;
  const uint16_t* v;
  long long v_rs, v_hs;
  const void* pos;
  const uint16_t* cs;
  long long cs_rs;
  uint16_t* qo;
  long long qo_rs, qo_hs;
  uint16_t* kc;
  uint16_t* vc;
  long long c_bs, c_ts, c_hs;
  uint16_t* ko;                 // rope_only: rotated k destination (may alias k)
  long long ko_rs, ko_hs;
  const void* slot;
  int rope_only;
  int t, hq, hk, half;
  int n_pos, pos64, n_slots, slot64;
  int block_size, block_shift;
  int hpg, nqg, tpb, nsub;
  int max_pos;
};

template <int W>
N6_GLOBAL void N6_LAUNCH_BOUNDS(64 * W) n6_rope_kernel(N6RopeParams p) {
  const int tid = N6_TID();
  const int sub = tid % p.nsub, lt = tid / p.nsub;   // 8 dims per thread, p.nsub per token
  const int ngrp = p.nqg + 1, bid = N6_BID();
  const int tile = bid / ngrp, g = bid - tile * ngrp;
  const int t = tile * p.tpb + lt;
  if (t >= p.t || lt >= p.tpb) return;   // no barrier in this kernel: early exit is safe
  const int d0 = sub * N6_VEC;

  long long pv = (t < p.n_pos) ? n6_ld_idx(p.pos, t, p.pos64) : 0;
  pv = pv < 0 ? 0 : (pv >= p.max_pos ? p.max_pos - 1 : pv);
  const uint16_t* cp = p.cs + pv * p.cs_rs + d0;
  const n6u4 cv = *(const n6u4*)cp;
  const n6u4 sv = *(const n6u4*)(cp + p.half);
  float co[N6_VEC], si[N6_VEC];
#pragma unroll
  for (int e = 0; e < N6_VEC; ++e) {
    co[e] = n6_bfe(cv, e);
    si[e] = n6_bfe(sv, e);
  }

  if (g < p.nqg) {
    const uint16_t* src = p.q + (long long)t * p.q_rs + (long long)g * p.hpg * p.q_hs + d0;
    uint16_t* dst = p.qo + (long long)t * p.qo_rs + (long long)g * p.hpg * p.qo_hs + d0;
    for (int i = 0; i < p.hpg; ++i) {
      const n6u4 a = *(const n6u4*)src;
      const n6u4 b = *(const n6u4*)(src + p.half);
      n6u4 o1, o2;
#pragma unroll
      for (int t2 = 0; t2 < 4; ++t2) {
        const int e = 2 * t2;
        const float x1 = n6_bfe(a, e), y1 = n6_bfe(b, e);
        const float x2 = n6_bfe(a, e + 1), y2 = n6_bfe(b, e + 1);
        o1[t2] = n6_pack2(x1 * co[e] - y1 * si[e], x2 * co[e + 1] - y2 * si[e + 1]);
        o2[t2] = n6_pack2(y1 * co[e] + x1 * si[e], y2 * co[e + 1] + x2 * si[e + 1]);
      }
      *(n6u4*)dst = o1;
      *(n6u4*)(dst + p.half) = o2;
      src += p.q_hs;
      dst += p.qo_hs;
    }
    return;
  }
  // kv group
  const uint16_t* ks = p.k + (long long)t * p.k_rs + d0;
  const uint16_t* vs;
  uint16_t* kd;
  uint16_t* vd;
  long long kd_st, vd_st;
  if (p.rope_only) {
    // rotate k in place (or into ko); no slot lookup, no v, no cache -- so this branch
    // touches nothing that a CUDA-graph replay could have moved.
    vs = ks;
    kd = p.ko + (long long)t * p.ko_rs + d0;
    vd = kd;
    kd_st = p.ko_hs;
    vd_st = 0;
  } else {
    if (t >= p.n_slots) return;
    const long long slot = n6_ld_idx(p.slot, t, p.slot64);
    if (slot < 0) return;
    const long long blk = p.block_shift >= 0 ? (slot >> p.block_shift) : (slot / p.block_size);
    const long long off = slot - blk * p.block_size;
    vs = p.v + (long long)t * p.v_rs + d0;
    kd = p.kc + blk * p.c_bs + off * p.c_ts + d0;
    vd = p.vc + blk * p.c_bs + off * p.c_ts + d0;
    kd_st = p.c_hs;
    vd_st = p.c_hs;
  }
  for (int h = 0; h < p.hk; ++h) {
    const n6u4 a = *(const n6u4*)ks;
    const n6u4 b = *(const n6u4*)(ks + p.half);
    n6u4 o1, o2;
#pragma unroll
    for (int t2 = 0; t2 < 4; ++t2) {
      const int e = 2 * t2;
      const float x1 = n6_bfe(a, e), y1 = n6_bfe(b, e);
      const float x2 = n6_bfe(a, e + 1), y2 = n6_bfe(b, e + 1);
      o1[t2] = n6_pack2(x1 * co[e] - y1 * si[e], x2 * co[e + 1] - y2 * si[e + 1]);
      o2[t2] = n6_pack2(y1 * co[e] + x1 * si[e], y2 * co[e + 1] + x2 * si[e + 1]);
    }
    *(n6u4*)kd = o1;
    *(n6u4*)(kd + p.half) = o2;
    if (!p.rope_only) {
      *(n6u4*)vd = *(const n6u4*)vs;                    // v is copied, not rotated
      *(n6u4*)(vd + p.half) = *(const n6u4*)(vs + p.half);
      vs += p.v_hs;
      vd += vd_st;
    }
    ks += p.k_hs;
    kd += kd_st;
  }
}

// ---------------------------------------------------------------------------
// SiLU(gate) * up
// ---------------------------------------------------------------------------
// y[m, i] = bf16(g * sigmoid(g) * u), g = x[m, i], u = x[m, I + i]  (fp32 math, as the
// deployed Inductor kernel).  Grid is flat: bid = row * chunks + c, a CTA takes
// NT * 8 consecutive columns of one row.
struct N6SiluParams {
  const uint16_t* x;
  long long x_rs;
  uint16_t* y;
  long long y_rs;
  int m, i, chunks;
};

N6_DEV float n6_silu(float g) { return g / (1.0f + __builtin_expf(-g)); }

template <int W>
N6_GLOBAL void N6_LAUNCH_BOUNDS(64 * W) n6_silu_kernel(N6SiluParams p) {
  const int NT = 64 * W;
  const int tid = N6_TID(), bid = N6_BID();
  const int row = bid / p.chunks, c = bid - row * p.chunks;
  const long long col = (long long)c * NT * N6_VEC + (long long)tid * N6_VEC;
  if (col >= p.i) return;
  const uint16_t* src = p.x + (long long)row * p.x_rs + col;
  const n6u4 gv = *(const n6u4*)src;
  const n6u4 uv = *(const n6u4*)(src + p.i);
  n6u4 o;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int e = 2 * t;
    o[t] = n6_pack2(n6_silu(n6_bfe(gv, e)) * n6_bfe(uv, e),
                    n6_silu(n6_bfe(gv, e + 1)) * n6_bfe(uv, e + 1));
  }
  *(n6u4*)(p.y + (long long)row * p.y_rs + col) = o;
}

// ---------------------------------------------------------------------------
// configs: (id, W, VPT) for the norm, (id, W) for rope / silu
// ---------------------------------------------------------------------------
#define N6_NORM_CFG_LIST(X)                                                        \
  X(0, 1, 8) X(1, 1, 16) X(2, 1, 32) X(3, 2, 8) X(4, 2, 16) X(5, 2, 32)            \
  X(6, 4, 8) X(7, 4, 16) X(8, 4, 32) X(9, 8, 8) X(10, 8, 16) X(11, 16, 8)
#define N6_NORM_NUM_CFGS 12

#define N6_W_CFG_LIST(X) X(0, 1) X(1, 2) X(2, 4) X(3, 8)
#define N6_W_NUM_CFGS 4

struct N6NormCfg {
  int w, vpt;
};
static const N6NormCfg n6_norm_cfgs[N6_NORM_NUM_CFGS] = {
#define N6_X(id, W_, V_) {W_, V_},
    N6_NORM_CFG_LIST(N6_X)
#undef N6_X
};
static const int n6_w_cfgs[N6_W_NUM_CFGS] = {
#define N6_X(id, W_) W_,
    N6_W_CFG_LIST(N6_X)
#undef N6_X
};

// ---------------------------------------------------------------------------
// argument checks (shared by the device and emulator front ends)
// ---------------------------------------------------------------------------
static inline int n6_aligned(const void* p) { return ((uintptr_t)p % 16) == 0; }

static inline int n6_check_norm(const N6NormParams& p, int cfg) {
  if (cfg < 0 || cfg >= N6_NORM_NUM_CFGS) return -10;
  const N6NormCfg c = n6_norm_cfgs[cfg];
  if (p.n != 64 * c.w * c.vpt) return -11;              // config does not cover this N
  if (p.n <= 0 || p.n > N6_MAXN || p.m <= 0) return -12;
  if (p.x_rs % N6_VEC || p.r_rs % N6_VEC || p.y_rs % N6_VEC || p.ro_rs % N6_VEC) return -13;
  if (!n6_aligned(p.x) || !n6_aligned(p.r) || !n6_aligned(p.w) || !n6_aligned(p.y) ||
      !n6_aligned(p.ro))
    return -14;
  if (p.eps < 0.f) return -15;
  return 0;
}

static inline int n6_check_rope(const N6RopeParams& p, int cfg) {
  if (cfg < 0 || cfg >= N6_W_NUM_CFGS) return -10;
  const int nt = 64 * n6_w_cfgs[cfg];
  if (p.half <= 0 || p.half % N6_VEC || (p.half & (p.half - 1))) return -11;
  if (p.nsub != p.half / N6_VEC || p.nsub > nt) return -12;
  if (p.tpb != nt / p.nsub || p.tpb <= 0) return -13;
  if (p.hq <= 0 || p.hk <= 0 || p.hpg <= 0 || p.hq % p.hpg || p.nqg != p.hq / p.hpg) return -14;
  if (p.t <= 0 || p.max_pos <= 0) return -15;
  if (p.q_hs % N6_VEC || p.k_hs % N6_VEC || p.qo_hs % N6_VEC) return -16;
  if (p.q_rs % N6_VEC || p.k_rs % N6_VEC || p.qo_rs % N6_VEC) return -17;
  if (!n6_aligned(p.q) || !n6_aligned(p.k) || !n6_aligned(p.qo) || !n6_aligned(p.cs))
    return -19;
  if (p.cs_rs % N6_VEC) return -20;
  if (p.rope_only) {
    if (p.ko == 0 || !n6_aligned(p.ko)) return -23;
    if (p.ko_rs % N6_VEC || p.ko_hs % N6_VEC) return -24;
    return 0;
  }
  if (p.v_hs % N6_VEC || p.v_rs % N6_VEC) return -16;
  if (p.c_bs % N6_VEC || p.c_ts % N6_VEC || p.c_hs % N6_VEC) return -18;
  if (!n6_aligned(p.v) || !n6_aligned(p.kc) || !n6_aligned(p.vc)) return -19;
  if (p.block_size <= 0 || (p.block_shift >= 0 && (1 << p.block_shift) != p.block_size))
    return -21;
  if (p.n_slots > 0 && p.slot == 0) return -22;
  return 0;
}

static inline int n6_check_silu(const N6SiluParams& p, int cfg) {
  if (cfg < 0 || cfg >= N6_W_NUM_CFGS) return -10;
  if (p.m <= 0 || p.i <= 0 || p.i % N6_VEC) return -11;
  if (p.x_rs % N6_VEC || p.y_rs % N6_VEC) return -12;
  if (!n6_aligned(p.x) || !n6_aligned(p.y)) return -13;
  if (p.chunks != (p.i + 64 * n6_w_cfgs[cfg] * N6_VEC - 1) / (64 * n6_w_cfgs[cfg] * N6_VEC))
    return -14;
  return 0;
}
