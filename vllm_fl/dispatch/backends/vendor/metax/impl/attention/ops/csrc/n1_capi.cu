// SPDX-License-Identifier: Apache-2.0
// R6/N1 C-ABI front-end: built straight with mxcc (no torch headers), loaded with ctypes
// by vllm_fl .../attention/ops/mx_dec_native.py (plugin) and by bench_n1.py:
//
//   mxcc -x maca --offload-arch=xcore1000 -O3 -std=c++17 -fPIC -shared \
//        n1_capi.cu -o libn1_dec_attn.so --maca-path=/opt/maca -L/opt/maca/lib -lmcruntime
//
// -DN1_EMU builds the same file as plain C++ against ../emu/n1_emu.h (CPU, offline).
// Every entry point takes the stream as void* (torch.cuda.current_stream().cuda_stream),
// only enqueues work (CUDA-graph capturable) and returns 0 on success, <0 for a rejected
// argument, >=1000 for a launch error (1000 + runtime error code).
#ifndef N1_EMU
#include <mc_runtime.h>
#define N1_STREAM_T mcStream_t
#define N1_LAST_ERROR() ((int)mcGetLastError())
#endif
#include "n1_dec_attn.cuh"

#define N1_API extern "C" __attribute__((visibility("default")))

N1_API int n1_version(void) { return N1_VERSION * 100 + N1_G; }

// build flags: bit 0 = byte-permute V transpose, bit 1 = conservative (N1_SAFE) build
N1_API int n1_features(void) { return (N1_USE_BYTE_PERM ? 1 : 0) | (N1_SAFE ? 2 : 0); }

N1_API int n1_num_cfgs(void) { return N1_NUM_CFGS; }

// info[0..6] = W, PF, LDK, MODE, KL, SCHED, Q8
N1_API int n1_cfg_info(int cfg, int* info) {
  if (cfg < 0 || cfg >= N1_NUM_CFGS) return -1;
  info[0] = n1_cfgs[cfg].w;
  info[1] = n1_cfgs[cfg].pf;
  info[2] = n1_cfgs[cfg].ldk;
  info[3] = n1_cfgs[cfg].mode;
  info[4] = n1_cfgs[cfg].kl;
  info[5] = n1_cfgs[cfg].sched;
  info[6] = n1_cfgs[cfg].q8;
  return 0;
}

#ifndef N1_EMU
template <int W, int PF, int LDK, int MODE, int KL, int SCHED, int Q8>
static int n1_attr_t(int* out) {
  mcFuncAttributes a;
  const void* f = (const void*)n1_dec_attn_kernel<N1_G, W, PF, LDK, MODE, KL, SCHED, Q8>;
  int e = (int)mcFuncGetAttributes(&a, f);
  if (e) return 1000 + e;
  out[0] = a.numRegs;
  out[1] = (int)a.localSizeBytes;
  out[2] = a.maxThreadsPerBlock;
  out[3] = (int)a.sharedSizeBytes;
  int occ = 0;
  e = (int)mcOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, W * 64, 0);
  out[4] = e ? -1 : occ;
  return 0;
}
#endif

// out[0..4] = registers, local (spill) bytes, max threads / CTA, static smem bytes,
// co-resident CTAs per AP
N1_API int n1_attr(int cfg, int* out) {
#ifdef N1_EMU
  (void)cfg;
  out[0] = out[1] = out[2] = out[3] = 0;
  out[4] = 4;
  return 0;
#else
  switch (cfg) {
#define N1_X(id, W_, PF_, LDK_, MODE_, KL_, SC_, Q8_) \
  case id:                                            \
    return n1_attr_t<W_, PF_, LDK_, MODE_, KL_, SC_, Q8_>(out);
    N1_CFG_LIST(N1_X)
#undef N1_X
    default:
      return -10;
  }
#endif
}

// Decode attention for q_len-1 rows.  Strides in elements.  ws_o: >= max_items * G * 128
// fp32; ws_ml: >= max_items * 32 fp32; ctr: >= 2 + rows * hkv int32, all ZERO before the
// first launch (every launch leaves them zero again).  max_items >= rows * hkv + 2 * grid
// (SCHED 0 also wants >= target_items + rows * hkv to avoid widening its chunks).
// q8 = 0 (bf16 cache) or 1 (R8/Q1 INT8 cache: kc / vc are then read as int8 base pointers
// of the same allocation and q8_sc is the byte offset of the fp32 scale block inside a half
// page; the cfg's own Q8 field must match).
// scale = softmax scale (e.g. 1 / sqrt(128)).  Only enqueues work: CUDA-graph capturable,
// no host-visible state, nothing allocated, nothing read from the host at launch except
// the (capture-time constant) shapes, strides, pointers and grid.
N1_API int n1_dec_attn(int cfg, const void* q, long long q_rs, long long q_hs, const void* kc,
                       const void* vc, long long c_bs, long long c_ts, long long c_hs,
                       const void* bt, long long bt_rs, const void* seq, int rows, void* out,
                       long long o_rs, long long o_hs, void* ws_o, void* ws_ml, void* ctr,
                       int max_items, int hq, int hkv, int bs_tiles, float scale, int grid,
                       int target_items, int min_ct, int claim_lead, void* sink, void* stream,
                       int q8, int q8_sc) {
  N1Params p;
  p.q = (const uint16_t*)q;
  p.q_rs = q_rs;
  p.q_hs = q_hs;
  p.kc = (const uint16_t*)kc;
  p.vc = (const uint16_t*)vc;
  p.c_bs = c_bs;
  p.c_ts = c_ts;
  p.c_hs = c_hs;
  p.bt = (const int*)bt;
  p.bt_rs = bt_rs;
  p.seq = (const int*)seq;
  p.out = (uint16_t*)out;
  p.o_rs = o_rs;
  p.o_hs = o_hs;
  p.ws_o = (float*)ws_o;
  p.ws_ml = (float*)ws_ml;
  p.ctr = (int*)ctr;
  p.rows = rows;
  p.hkv = hkv;
  p.bs_tiles = bs_tiles;
  p.max_items = max_items;
  p.target_items = target_items;
  p.min_ct = min_ct;
  p.claim_lead = claim_lead < 0 ? 0 : claim_lead;
  p.pad0 = 0;
  p.scale_log2 = scale * 1.4426950408889634f;
  p.sink = (unsigned*)sink;
  p.q8 = q8;
  p.q8_sc = q8_sc;
  if (cfg < 0 || cfg >= N1_NUM_CFGS) return -10;
  const int rc = n1_check(p, hq, grid);
  if (rc) return rc;
  return n1_dispatch(cfg, p, grid, (n1_stream_t)stream);
}

// fragment-model probe: out[512] fp32 (see n1_probe_mma_kernel)
N1_API int n1_probe_mma(void* out, void* stream) {
#ifdef N1_EMU
  (void)stream;
  float* o = (float*)out;
  n1emu_launch(1u, 64u, [=] { n1_probe_mma_kernel(o); });
  return 0;
#else
  n1_probe_mma_kernel<<<dim3(1, 1, 1), dim3(64, 1, 1), 0, (n1_stream_t)stream>>>((float*)out);
  const int e = (int)N1_LAST_ERROR();
  return e ? 1000 + e : 0;
#endif
}

#ifdef N1_EMU
// emulator controls: 0 = all CTAs concurrently, 1 = one at a time in order, 2 = reversed
N1_API void n1_emu_set_mode(int mode) { n1emu_mode() = mode; }
#endif
