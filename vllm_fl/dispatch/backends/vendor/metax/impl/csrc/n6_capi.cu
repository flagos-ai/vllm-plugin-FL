// SPDX-License-Identifier: Apache-2.0
// R7/N6 C-ABI front-end: built straight with mxcc (no torch headers), loaded with ctypes
// by vllm_fl .../metax/impl/mx_glue_native.py (plugin) and by bench_n6.py:
//
//   mxcc -x maca --offload-arch=xcore1000 -O3 -std=c++17 -ffp-contract=off -fPIC -shared \
//        n6_capi.cu -o libn6_glue.so --maca-path=/opt/maca -L/opt/maca/lib -lmcruntime
//
// -DN6_EMU builds the same file as plain C++ against ../emu/n1_emu.h (CPU, offline).
// Every entry point takes the stream as void* (torch.cuda.current_stream().cuda_stream),
// only enqueues work (CUDA-graph capturable: no allocation, no sync, no host reads) and
// returns 0 on success, <0 for a rejected argument, >= 1000 for a launch error
// (1000 + runtime error code).
#ifndef N6_EMU
#include <mc_runtime.h>
#define N6_STREAM_T mcStream_t
#define N6_LAST_ERROR() ((int)mcGetLastError())
#endif
#include "n6_glue.cuh"

#define N6_API extern "C" __attribute__((visibility("default")))

#ifdef N6_EMU
typedef void* n6_stream_t;
#else
typedef N6_STREAM_T n6_stream_t;
#endif

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------
#ifdef N6_EMU
#define N6_LAUNCH(kernel, grid, nt, stream, params)                           \
  do {                                                                        \
    (void)(stream);                                                           \
    n1emu_launch((unsigned)(grid), (unsigned)(nt), [=] { kernel(params); });   \
  } while (0)
#define N6_LAUNCH_RC() 0
#else
#define N6_LAUNCH(kernel, grid, nt, stream, params)                             \
  kernel<<<dim3((unsigned)(grid), 1, 1), dim3((unsigned)(nt), 1, 1), 0,         \
           (n6_stream_t)(stream)>>>(params)
#define N6_LAUNCH_RC()                 \
  ({                                   \
    const int e_ = N6_LAST_ERROR();    \
    e_ ? 1000 + e_ : 0;                \
  })
#endif

template <int W, int VPT>
static int n6_norm_launch(const N6NormParams& p, void* stream) {
  N6_LAUNCH((n6_norm_kernel<W, VPT>), p.m, 64 * W, stream, p);
  return N6_LAUNCH_RC();
}

template <int W>
static int n6_rope_launch(const N6RopeParams& p, void* stream) {
  const long long tiles = (p.t + p.tpb - 1) / p.tpb;
  N6_LAUNCH((n6_rope_kernel<W>), tiles * (p.nqg + 1), 64 * W, stream, p);
  return N6_LAUNCH_RC();
}

template <int W>
static int n6_silu_launch(const N6SiluParams& p, void* stream) {
  N6_LAUNCH((n6_silu_kernel<W>), (long long)p.m * p.chunks, 64 * W, stream, p);
  return N6_LAUNCH_RC();
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------
N6_API int n6_version(void) { return N6_VERSION * 100 + (N6_SAFE ? 1 : 0); }
N6_API int n6_num_cfgs(int kernel) { return kernel == 0 ? N6_NORM_NUM_CFGS : N6_W_NUM_CFGS; }

// info[0] = threads/CTA, info[1] = VPT (norm) or 0, info[2] = N covered (norm) or 0
N6_API int n6_cfg_info(int kernel, int cfg, int* info) {
  if (kernel == 0) {
    if (cfg < 0 || cfg >= N6_NORM_NUM_CFGS) return -1;
    info[0] = 64 * n6_norm_cfgs[cfg].w;
    info[1] = n6_norm_cfgs[cfg].vpt;
    info[2] = info[0] * info[1];
    return 0;
  }
  if (cfg < 0 || cfg >= N6_W_NUM_CFGS) return -1;
  info[0] = 64 * n6_w_cfgs[cfg];
  info[1] = 0;
  info[2] = 0;
  return 0;
}

#ifndef N6_EMU
template <class F>
static int n6_attr_f(F f, int nt, int* out) {
  mcFuncAttributes a;
  const void* fp = (const void*)f;
  int e = (int)mcFuncGetAttributes(&a, fp);
  if (e) return 1000 + e;
  out[0] = a.numRegs;
  out[1] = (int)a.localSizeBytes;
  out[2] = a.maxThreadsPerBlock;
  out[3] = (int)a.sharedSizeBytes;
  int occ = 0;
  e = (int)mcOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fp, nt, 0);
  out[4] = e ? -1 : occ;
  return 0;
}
#endif

// out[0..4] = registers, local (spill) bytes, max threads / CTA, static smem, CTAs per AP
N6_API int n6_attr(int kernel, int cfg, int* out) {
#ifdef N6_EMU
  (void)kernel;
  (void)cfg;
  out[0] = out[1] = out[2] = out[3] = 0;
  out[4] = 8;
  return 0;
#else
  if (kernel == 0) {
    switch (cfg) {
#define N6_X(id, W_, V_)                                              \
  case id:                                                            \
    return n6_attr_f(n6_norm_kernel<W_, V_>, 64 * W_, out);
      N6_NORM_CFG_LIST(N6_X)
#undef N6_X
      default:
        return -10;
    }
  }
  if (kernel == 1) {
    switch (cfg) {
#define N6_X(id, W_)                                          \
  case id:                                                    \
    return n6_attr_f(n6_rope_kernel<W_>, 64 * W_, out);
      N6_W_CFG_LIST(N6_X)
#undef N6_X
      default:
        return -10;
    }
  }
  switch (cfg) {
#define N6_X(id, W_)                                          \
  case id:                                                    \
    return n6_attr_f(n6_silu_kernel<W_>, 64 * W_, out);
    N6_W_CFG_LIST(N6_X)
#undef N6_X
    default:
      return -10;
  }
#endif
}

// ---------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------
// residual add + RMSNorm.  Strides in elements.  y / ro may alias x / r only if the
// strides are identical (the kernel reads a row before it writes it).
N6_API int n6_add_rms_norm(int cfg, const void* x, long long x_rs, const void* r,
                           long long r_rs, const void* w, void* y, long long y_rs, void* ro,
                           long long ro_rs, int m, int n, float eps, void* stream) {
  N6NormParams p;
  p.x = (const uint16_t*)x;
  p.x_rs = x_rs;
  p.r = (const uint16_t*)r;
  p.r_rs = r_rs;
  p.w = (const uint16_t*)w;
  p.y = (uint16_t*)y;
  p.y_rs = y_rs;
  p.ro = (uint16_t*)ro;
  p.ro_rs = ro_rs;
  p.m = m;
  p.n = n;
  p.eps = eps;
  p.inv_n = 1.0f / (float)n;
  const int rc = n6_check_norm(p, cfg);
  if (rc) return rc;
  switch (cfg) {
#define N6_X(id, W_, V_)                      \
  case id:                                    \
    return n6_norm_launch<W_, V_>(p, stream);
    N6_NORM_CFG_LIST(N6_X)
#undef N6_X
    default:
      return -10;
  }
}

// neox RoPE of q and k + the paged KV write.  pos / slot are int64 when pos64 / slot64,
// else int32.  block_shift = log2(block_size) or -1.  hpg must divide hq.
N6_API int n6_rope_kv(int cfg, const void* q, long long q_rs, long long q_hs, const void* k,
                      long long k_rs, long long k_hs, const void* v, long long v_rs,
                      long long v_hs, const void* pos, int n_pos, int pos64, const void* cs,
                      long long cs_rs, int max_pos, void* qo, long long qo_rs, long long qo_hs,
                      void* kc, void* vc, long long c_bs, long long c_ts, long long c_hs,
                      const void* slot, int n_slots, int slot64, int block_size,
                      int block_shift, int t, int hq, int hk, int half, int hpg,
                      void* stream) {
  N6RopeParams p;
  p.q = (const uint16_t*)q;
  p.q_rs = q_rs;
  p.q_hs = q_hs;
  p.k = (const uint16_t*)k;
  p.k_rs = k_rs;
  p.k_hs = k_hs;
  p.v = (const uint16_t*)v;
  p.v_rs = v_rs;
  p.v_hs = v_hs;
  p.pos = pos;
  p.cs = (const uint16_t*)cs;
  p.cs_rs = cs_rs;
  p.qo = (uint16_t*)qo;
  p.qo_rs = qo_rs;
  p.qo_hs = qo_hs;
  p.kc = (uint16_t*)kc;
  p.vc = (uint16_t*)vc;
  p.c_bs = c_bs;
  p.c_ts = c_ts;
  p.c_hs = c_hs;
  p.ko = 0;
  p.ko_rs = 0;
  p.ko_hs = 0;
  p.slot = slot;
  p.rope_only = 0;
  p.t = t;
  p.hq = hq;
  p.hk = hk;
  p.half = half;
  p.n_pos = n_pos;
  p.pos64 = pos64;
  p.n_slots = n_slots;
  p.slot64 = slot64;
  p.block_size = block_size;
  p.block_shift = block_shift;
  p.hpg = hpg;
  p.nqg = hpg > 0 ? hq / hpg : 0;
  p.nsub = half / N6_VEC;
  p.tpb = (cfg >= 0 && cfg < N6_W_NUM_CFGS && p.nsub > 0) ? (64 * n6_w_cfgs[cfg]) / p.nsub : 0;
  p.max_pos = max_pos;
  const int rc = n6_check_rope(p, cfg);
  if (rc) return rc;
  switch (cfg) {
#define N6_X(id, W_)                          \
  case id:                                    \
    return n6_rope_launch<W_>(p, stream);
    N6_W_CFG_LIST(N6_X)
#undef N6_X
    default:
      return -10;
  }
}

// RoPE only, no cache write (R10/F): q -> qo and k -> ko rotated in place-capable form.
// Same kernel, same rounding as n6_rope_kv; the INT8 KV path needs this half because its
// pages are written by the quantising kernel.  CUDA-graph safe like every entry here.
N6_API int n6_rope_only(int cfg, const void* q, long long q_rs, long long q_hs,
                        const void* k, long long k_rs, long long k_hs, const void* pos,
                        int n_pos, int pos64, const void* cs, long long cs_rs, int max_pos,
                        void* qo, long long qo_rs, long long qo_hs, void* ko,
                        long long ko_rs, long long ko_hs, int t, int hq, int hk, int half,
                        int hpg, void* stream) {
  N6RopeParams p;
  p.q = (const uint16_t*)q;
  p.q_rs = q_rs;
  p.q_hs = q_hs;
  p.k = (const uint16_t*)k;
  p.k_rs = k_rs;
  p.k_hs = k_hs;
  p.v = 0;
  p.v_rs = 0;
  p.v_hs = 0;
  p.pos = pos;
  p.cs = (const uint16_t*)cs;
  p.cs_rs = cs_rs;
  p.qo = (uint16_t*)qo;
  p.qo_rs = qo_rs;
  p.qo_hs = qo_hs;
  p.kc = 0;
  p.vc = 0;
  p.c_bs = 0;
  p.c_ts = 0;
  p.c_hs = 0;
  p.ko = (uint16_t*)ko;
  p.ko_rs = ko_rs;
  p.ko_hs = ko_hs;
  p.slot = 0;
  p.rope_only = 1;
  p.t = t;
  p.hq = hq;
  p.hk = hk;
  p.half = half;
  p.n_pos = n_pos;
  p.pos64 = pos64;
  p.n_slots = 0;
  p.slot64 = 0;
  p.block_size = 0;
  p.block_shift = -1;
  p.hpg = hpg;
  p.nqg = hpg > 0 ? hq / hpg : 0;
  p.nsub = half / N6_VEC;
  p.tpb = (cfg >= 0 && cfg < N6_W_NUM_CFGS && p.nsub > 0) ? (64 * n6_w_cfgs[cfg]) / p.nsub : 0;
  p.max_pos = max_pos;
  const int rc = n6_check_rope(p, cfg);
  if (rc) return rc;
  switch (cfg) {
#define N6_X(id, W_)                          \
  case id:                                    \
    return n6_rope_launch<W_>(p, stream);
    N6_W_CFG_LIST(N6_X)
#undef N6_X
    default:
      return -10;
  }
}

// SiLU(gate) * up of a [m, 2 * i] input -> [m, i]
N6_API int n6_silu_mul(int cfg, const void* x, long long x_rs, void* y, long long y_rs, int m,
                       int i, void* stream) {
  N6SiluParams p;
  p.x = (const uint16_t*)x;
  p.x_rs = x_rs;
  p.y = (uint16_t*)y;
  p.y_rs = y_rs;
  p.m = m;
  p.i = i;
  const int nt = (cfg >= 0 && cfg < N6_W_NUM_CFGS) ? 64 * n6_w_cfgs[cfg] : 0;
  p.chunks = nt ? (i + nt * N6_VEC - 1) / (nt * N6_VEC) : 0;
  const int rc = n6_check_silu(p, cfg);
  if (rc) return rc;
  switch (cfg) {
#define N6_X(id, W_)                          \
  case id:                                    \
    return n6_silu_launch<W_>(p, stream);
    N6_W_CFG_LIST(N6_X)
#undef N6_X
    default:
      return -10;
  }
}

#ifdef N6_EMU
// emulator controls: 0 = all CTAs concurrently, 1 = one at a time in order, 2 = reversed
N6_API void n6_emu_set_mode(int mode) { n1emu_mode() = mode; }
#endif
