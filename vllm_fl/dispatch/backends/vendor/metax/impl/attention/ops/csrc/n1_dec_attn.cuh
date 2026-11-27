// SPDX-License-Identifier: Apache-2.0
// 2026 - FlagOS S2 contest, R6/N1: own paged decode attention for the MetaX C500,
// native MACA C++ on the 16x16x16 bf16 MMA unit (our design; not a vendor kernel).
//
// One header, two builds:
//   * device (mxcc -x maca, n1_capi.cu -> libn1_dec_attn.so, C ABI, loaded with ctypes
//     by the plugin's ops/mx_dec_native.py and by the GPU bench)
//   * CPU emulator (-DN1_EMU, ../emu/n1_emu.h): every GPU thread is an OS thread, the
//     CTAs of a grid run concurrently (or one by one, in any order), warps of 64 lanes
//     exchange MMA fragments / shuffles through a barrier -> offline checks of all
//     index math, the online softmax, the work claiming and the cross-CTA merge.
//
// Problem: decode rows (q_len 1) of a GQA layer, bf16 paged KV cache
//   q   [rows, HQ, 128]          (row / head strides given)
//   K,V [blocks, BS, HKV, 128]   (block / token / head strides given, BS % 16 == 0)
//   out [rows, HQ, 128] bf16     (row / head strides given)
//   G = HQ / HKV = 8 q heads per kv head (template; MiniCPM5: 16 q / 2 kv heads).
//
// Two schedules (template parameter SCHED), both rebuilt inside the kernel from seq_lens
// in DEVICE memory, so one launch configuration captured in a CUDA graph stays correct for
// every later replay (growing contexts, padded seq_len-0 rows):
//
// SCHED 0 (R6/N1, kept bit-identical -- serves the decode rows of overlapped mixed steps):
// an item is (row, kv head, chunk of the row's 16-token tiles).  A persistent grid
// (APs x CTAs/AP) claims items with one atomic counter; every CTA first rebuilds the same
// item table: chunk length CT = max(min_ct, ceil(total_tiles / target_items)) tiles,
// n_r = ceil(tiles_r / CT) chunks per row (1 for seq_len <= 0, which writes zeros),
// exclusive prefix over rows in shared memory.  CTAs that start late (e.g. next to the
// prefill kernel of a mixed step) simply claim fewer items.
//
// SCHED 1 (R7/N4, for the decode-only steps -- inside the FULL CUDA graphs and eager):
// a static, perfectly balanced stream-K partition of TILE space, no atomic claiming.  The
// (row, kv head) pairs are "segments" of tiles_r tiles each, laid out row-major; with
// T = sum of their lengths and Geff = min(gridDim.x, T), CTA c owns the contiguous tile
// range [floor(c*T/Geff), floor((c+1)*T/Geff)) -- equal to +-1 tile for every CTA, so the
// makespan has no claim granularity and no tail wave (the measured 18% (4k) / 9% (16k) gap
// of SCHED 0 to the paged-read floor).  A CTA walks the segments its range intersects; a
// segment it covers completely is written straight out, and only the <= 2 segments that
// straddle its range ends produce fp32 partials, in slots 2c (first) and 2c+1 (last).
// A segment is split over nr = cta_of(last tile) - cta_of(first tile) + 1 <= few CTAs; the
// one that arrives last on the segment counter merges slots 2c + (start(c) < seg_start) for
// c ascending, i.e. in tile order -> the same reduction order for the same seq_lens, so the
// output is bit-identical across replays.  cta_of(t) = ((t+1)*Geff - 1)/T inverts the
// partition exactly.  Partials: <= 2*gridDim.x slots instead of ~2.5x more for SCHED 0.
//
// CTA = W warps; warp w takes the tiles tb+w, tb+w+W, ... of the chunk.  Per 16-token
// tile (= one page of one kv head) a warp issues 8 x 128-bit loads per lane:
//   K: lane (r, g) <- K[token r][dims 32u + 8g .. +7], u = 0..3   (64 B runs per token)
//   V: lane (r, g) <- V[token 4g + j][dims 8r .. 8r+7], j = 0..3  (256 B runs per token)
// and runs 16 MMAs (fragment model below):
//   S^T = K . Q^T   8 MMAs (K-dim = head dims, permuted identically in K and Q:
//                   MMA step (u, h) uses dims 32u + 8g + 4h + j for lane group g);
//                   lane (r, g) receives S[head r][token 4g + i], i = 0..3 -- exactly the
//                   A-fragment layout P[head r][token 4g + j] of the PV MMA, so the
//                   probabilities never move between lanes;
//   O += P . V      8 MMAs, one per output-column tile k = 0..7 mapping column n -> dim
//                   8n + k, so the B fragment {V[4g+j][8r+k]}_j is a 16-bit transpose of
//                   the four V registers inside the lane (no shared memory, no shuffle);
//                   lane (r, g) accumulates O[head 4g + i][dim 8r + k].
// GQA: the G = 8 q heads fill rows 0..7 of the 16-row MMA (rows 8..15 compute zeros).
// Softmax: fp32, base 2 (scale * log2 e folded into the scores), running max per head
// reduced over the 4 lanes that share a head (2 xor shuffles), P rounded to bf16 (RTNE)
// for the MMA and the denominator summed from the SAME rounded values.
// Pipelining: PF tiles of K/V per warp in flight in registers (PF = 1: double buffer);
// page ids of the next 64 tiles of a warp sit in one register per lane (shuffle lookup).
// Merge: the W warps of a CTA merge through shared memory (fixed order); a segment split
// over n_r > 1 CTAs writes fp32 partials (o unnormalised, m, l) to its slot, and the CTA
// that arrives last on the segment counter (atomic, after a device fence) merges the
// partials in chunk order -> deterministic output, no second launch.  Counters reset
// themselves: the merging CTA zeroes its segment counter, and (SCHED 0 only) the last CTA
// to exit zeroes the claim / exit counters, so the next launch -- the next layer, the next
// graph replay -- starts clean with no host-side zeroing and no memset node.
//
// MMA fragment model (MACA WMMA header __clang_maca_mma_functions.h, confirmed on the
// C500 by R6/X4 x4_probe_mma and R6/X1 sec_mma): d = __builtin_mxc_mma_16x16x16bf16(x, y, c)
// with lane l, r = l % 16, g = l / 16:
//   x (2 x u32 = 4 bf16): X[r][4g + j], j = 0..3       (X: 16 x 16, M x K)
//   y (2 x u32 = 4 bf16): Y[4g + j][r]                  (Y: 16 x 16, K x N)
//   c/d (4 x f32):        D[4g + i][r], i = 0..3        (D = X . Y + C)
// n1_probe_mma re-checks the model on the device before the bench trusts the kernel.

// R8/Q1 (template parameter Q8): the same tile code reads a SELF-DEVELOPED INT8 KV cache
// (int8 data + one fp32 scale per (token, kv head) for K and for V, carved out of the same
// bf16 pages so the KV capacity is unchanged).  Q8 = 0 is the bf16 path, bit-identical to
// R6/N1 + R7/N4; Q8 = 1 reads the "Q1-L1" layout documented in q1_int8.cuh, dequantises in
// registers before the bf16 MMA (K's scale after the QK^T MMA, V's scale folded into P) and
// needs no identity-MMA transpose (16 MMAs per tile instead of 24).  528 B per token per
// layer instead of 1024.
#pragma once
#include <stdint.h>

#define N1_VERSION 4
#define N1_D 128      // head size
#define N1_TILE 16    // tokens per tile (one MMA K-step of PV)
#define N1_MAXR 512   // rows per launch (shared-memory item table)
#define N1_SMB 4      // O merge buffers in shared memory (warps 4.. add in rounds)
#define N1_ML 32      // floats per partial slot for (m[16], l[16])
#ifndef N1_USE_BYTE_PERM
#define N1_USE_BYTE_PERM 0   // 1: V transpose with the byte-permute instruction (variant build)
#endif

// N1_SAFE=1: conservative build used by the build script when the default does not
// compile (no lambda attributes, no non-temporal / byte-permute / vote builtins)
#ifndef N1_SAFE
#define N1_SAFE 0
#endif
#if N1_SAFE
#undef N1_USE_BYTE_PERM
#define N1_USE_BYTE_PERM 0
#endif

#ifdef N1_EMU
#include "n1_emu.h"
#else
#define N1_DEV __device__ __forceinline__
#if N1_SAFE
#define N1_INLINE_LAMBDA
#else
#define N1_INLINE_LAMBDA __attribute__((always_inline))
#endif
#define N1_GLOBAL __global__
#define N1_LAUNCH_BOUNDS(n) __launch_bounds__(n)
#define N1_SMEM(T, name) __shared__ T name
#define N1_MMA(x, y, c) __builtin_mxc_mma_16x16x16bf16((x), (y), (c))
// R9/R: same 16x16x16 shape, same <4 x half> / <4 x float> operands and the same 10
// MT registers as the bf16 MMA (declared in __clang_maca_mma_functions.h:2440-2446);
// the int8 one takes 4 int8 in ONE u32 and accumulates int32 (:2477-2479).  The K = 32
// int8 shape is NOT available on xcore1000, so nothing here may assume it.
#define N1_MMA_F16(x, y, c) __builtin_mxc_mma_16x16x16f16((x), (y), (c))
#define N1_MMA_I8(x, y, c) __builtin_mxc_mma_16x16x16i8((x), (y), (c))
#define N1_SYNC() __syncthreads()
#define N1_SHFL_XOR_F(v, m) __shfl_xor_sync((uint64_t)(-1), (float)(v), (m))
#define N1_SHFL_XOR_I(v, m) __shfl_xor_sync((uint64_t)(-1), (int)(v), (m))
#define N1_SHFL_F(v, s) __shfl_sync((uint64_t)(-1), (float)(v), (s))
#define N1_SHFL_I(v, s) __shfl_sync((uint64_t)(-1), (int)(v), (s))
#if N1_SAFE
#define N1_ANY(pred) n1_any_shfl((int)(pred))
#else
#define N1_ANY(pred) __any_sync((unsigned long long)(-1), (int)(pred))
#endif
#define N1_FENCE() __threadfence()
#define N1_ATOMIC_ADD(p, v) atomicAdd((p), (v))
#define N1_ATOMIC_EXCH(p, v) atomicExch((p), (v))
#define N1_EXP2(x) __builtin_exp2f(x)
#define N1_TID() ((int)threadIdx.x)
#define N1_BID() ((int)blockIdx.x)
#define N1_NBLK() ((int)gridDim.x)
#if !defined(N1_STREAM_T) || !defined(N1_LAST_ERROR)
#error "front-end must define N1_STREAM_T and N1_LAST_ERROR() before including n1_dec_attn.cuh"
#endif
#endif

typedef uint32_t n1u2 __attribute__((ext_vector_type(2)));
typedef uint32_t n1u4 __attribute__((ext_vector_type(4)));
typedef float n1f4 __attribute__((ext_vector_type(4)));
typedef int n1i4 __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------------------
// launch arguments (passed by value)
// ---------------------------------------------------------------------------
struct N1Params {
  const uint16_t* q;    // [rows, HQ, 128]
  long long q_rs, q_hs;
  const uint16_t* kc;   // [blocks, BS, HKV, 128]
  const uint16_t* vc;   // same strides as kc
  long long c_bs, c_ts, c_hs;
  const int* bt;        // [rows, bt_rs] int32 block ids
  long long bt_rs;
  const int* seq;       // [rows] int32 context lengths (<= 0: row skipped, zeros written)
  uint16_t* out;        // [rows, HQ, 128]
  long long o_rs, o_hs;
  float* ws_o;          // [max_items, G, 128] fp32 partial o (unnormalised)
  float* ws_ml;         // [max_items, 32] fp32 partial m (0..15) and l (16..31)
  int* ctr;             // [2 + rows * hkv] int32: claim, exit, per (row, kv head) arrivals
  int rows, hkv, bs_tiles, max_items;
  int target_items, min_ct, claim_lead, pad0;
  float scale_log2;     // softmax scale * log2(e)
  unsigned* sink;       // MODE 1 (read-only floor): keeps the loads alive
  // R8/Q1 INT8 KV (used only by the Q8 != 0 instantiations; kc / vc are then the int8
  // base pointers of the same allocation and c_bs / c_ts / c_hs still describe the bf16
  // pages, so the page is 2 * c_bs bytes).
  int q8;               // 0 bf16 cache, 1 int8 "Q1-L1" layout (q1_int8.cuh)
  int q8_sc;            // byte offset of the fp32 scale block inside a half page
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
N1_DEV float n1_u2f(uint32_t u) { return __builtin_bit_cast(float, u); }
N1_DEV uint32_t n1_f2u(float f) { return __builtin_bit_cast(uint32_t, f); }
N1_DEV float n1_bflo(uint32_t w) { return n1_u2f(w << 16); }
N1_DEV float n1_bfhi(uint32_t w) { return n1_u2f(w & 0xffff0000u); }
N1_DEV float n1_max(float a, float b) { return a > b ? a : b; }
N1_DEV int n1_imin(int a, int b) { return a < b ? a : b; }
N1_DEV int n1_imax(int a, int b) { return a > b ? a : b; }
N1_DEV float n1_ninf() { return -__builtin_huge_valf(); }

// ---- SCHED 1 (stream-K) tile-space partition, all integer, all device-side -----------
// CTA c owns tiles [n1_cta_start(c), n1_cta_start(c+1)); n1_cta_of inverts it exactly.
N1_DEV int n1_cta_start(int c, int total, int geff) {
  return (int)(((long long)c * total) / geff);
}
N1_DEV int n1_cta_of(int t, int total, int geff) {
  return (int)(((long long)(t + 1) * geff - 1) / total);
}
// partial slot of CTA c for the segment starting at seg_start: 2c for the first segment c
// touches, 2c+1 for its last one (no other segment of c can be incomplete)
N1_DEV int n1_slot(int c, int seg_start, int total, int geff) {
  return 2 * c + (n1_cta_start(c, total, geff) < seg_start ? 1 : 0);
}

// fp32 -> bf16 bits, round to nearest even, NaN kept quiet (outputs)
N1_DEV uint32_t n1_f2bf(float f) {
  uint32_t u = n1_f2u(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (u >> 16) | ((u & 0xffffu) ? 0x40u : 0u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
// finite non-negative p (softmax weights): RTNE without the NaN branch, result stays in
// the HIGH half (low 16 bits are garbage) so packing is one shift / mask
N1_DEV uint32_t n1_rtne_hi(float f) {
  const uint32_t u = n1_f2u(f);
  return u + 0x7fffu + ((u >> 16) & 1u);
}
N1_DEV uint32_t n1_pack_p(float lo, float hi) {
  return (n1_rtne_hi(lo) >> 16) | (n1_rtne_hi(hi) & 0xffff0000u);
}
N1_DEV uint32_t n1_pack_out(float lo, float hi) { return n1_f2bf(lo) | (n1_f2bf(hi) << 16); }

N1_DEV n1u4 n1_z4() { return (n1u4){0u, 0u, 0u, 0u}; }

// element c (0..7, bf16) of a and of b packed as {a[c] (low half), b[c] (high half)}
N1_DEV uint32_t n1_pick2(n1u4 a, n1u4 b, int c) {
  const int wd = c >> 1;
#if N1_USE_BYTE_PERM
  return __byte_perm(a[wd], b[wd], (c & 1) ? 0x7632u : 0x5410u);
#else
  return (c & 1) ? ((a[wd] >> 16) | (b[wd] & 0xffff0000u)) : ((a[wd] & 0xffffu) | (b[wd] << 16));
#endif
}
N1_DEV n1f4 n1_f4z() { return (n1f4){0.f, 0.f, 0.f, 0.f}; }

// global loads: 0 plain, 1 non-temporal (streamed KV)
template <int LDK>
N1_DEV n1u4 n1_ld(const uint16_t* p) {
#if !defined(N1_EMU) && !N1_SAFE
  if (LDK == 1) return __builtin_nontemporal_load((const n1u4*)p);
#endif
  return *(const n1u4*)p;
}
#if N1_SAFE && !defined(N1_EMU)
// warp vote through xor shuffles (N1_SAFE builds only)
N1_DEV int n1_any_shfl(int pred) {
  int v = pred ? 1 : 0;
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v |= N1_SHFL_XOR_I(v, o);
  return v;
}
#endif
// partials written by other CTAs of the same launch: volatile (no stale L1 lines)
N1_DEV n1f4 n1_ldv4(const float* p) { return *(const volatile n1f4*)p; }
N1_DEV float n1_ldv(const float* p) { return *(const volatile float*)p; }

// block-wide sum (all threads get it); red[] has >= W ints
template <int W>
N1_DEV int n1_block_sum(int v, int* red) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v += N1_SHFL_XOR_I(v, o);
  const int tid = N1_TID(), lane = tid & 63, w = tid >> 6;
  N1_SYNC();
  if (lane == 0) red[w] = v;
  N1_SYNC();
  int s = 0;
#pragma unroll
  for (int i = 0; i < W; ++i) s += red[i];
  return s;
}

// block-wide exclusive scan of one int per thread (thread order); *total = sum
template <int W>
N1_DEV int n1_block_excl_scan(int v, int* red, int* total) {
  const int tid = N1_TID(), lane = tid & 63, w = tid >> 6;
  int x = v;
#pragma unroll
  for (int d = 1; d < 64; d <<= 1) {
    const int y = N1_SHFL_I(x, (lane - d) & 63);
    if (lane >= d) x += y;
  }
  N1_SYNC();
  if (lane == 63) red[w] = x;
  N1_SYNC();
  int off = 0, tot = 0;
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const int t = red[i];
    if (i < w) off += t;
    tot += t;
  }
  *total = tot;
  return off + x - v;
}

// ---------------------------------------------------------------------------
// shared memory of one CTA
// ---------------------------------------------------------------------------
template <int G, int W>
struct N1Smem {
  n1f4 o[N1_SMB][G * N1_D / 4];   // per-warp-group O (scaled to the CTA max)
  float m[W][16], l[W][16];       // per warp, per head
  int cum[N1_MAXR + 1];           // SCHED 0: item prefix over rows; SCHED 1: tile prefix
  int red[16];
  int item, next, flag, total_items, ct, pad[3];
};

// one 16-token tile of one kv head in registers
struct N1Tile {
  n1u4 k[4];
  n1u4 v[4];
};

// R8/Q1: INT8 storage layout, register dequantisation and Q1Tile
#include "q1_int8.cuh"

template <int Q8>
struct N1TileSel {
  typedef N1Tile type;
};
template <>
struct N1TileSel<1> {
  typedef Q1Tile type;
};
template <>
struct N1TileSel<2> {
  typedef Q1Tile type;
};
template <>
struct N1TileSel<3> {
  typedef Q1Tile type;
};

// ---------------------------------------------------------------------------
// the kernel
//   MODE 0: attention; MODE 1: same loads / claiming, no math (read floor)
// ---------------------------------------------------------------------------
//   KL 0: K loaded as MMA x fragments (lane (r, g) <- 64 B runs of token r);
//   KL 1: K loaded like V (16 lanes <- one token's 256 B) and turned into x fragments with an
//         identity MMA per 16-dim column tile (exact: bf16 values x 1.0)
//   SCHED 0: atomic-claimed (row, kv head, chunk) items; SCHED 1: static stream-K
//            partition of tile space (see the top of this file)
//   Q8 0: bf16 cache (R6/N1 + R7/N4, unchanged); 1: INT8 cache, layout "Q1-L1"
//         (q1_int8.cuh), dequantised in registers, K's scale folded into the score and
//         V's into P.  Q8 >= 1 implies the KL = 0 register mapping (the layout delivers
//         the x fragments directly) and a page of exactly one tile.
//         R9/R: 2 = the same cache read with the FP16 magic-number dequantisation and the
//         fp16 MMA (25 + 56 ALU ops per lane per tile instead of 161 + 160, see
//         q1_int8.cuh "R9/R"); 3 = 2 with QK^T on the INT8 MMA, which needs no convert at
//         all on the K side and quantises Q in the prologue (N1_Q1_QPLANES = 2 makes that
//         quantisation exact in 16 bits at the cost of 8 more MMAs per tile).
template <int G, int W, int PF, int LDK, int MODE, int KL, int SCHED, int Q8 = 0>
N1_GLOBAL void N1_LAUNCH_BOUNDS(W * 64) n1_dec_attn_kernel(N1Params p) {
  static_assert(G >= 1 && G <= 16, "G q heads per kv head must fit the 16 MMA rows");
  static_assert(W >= 1 && W <= 16, "warps per CTA");
  static_assert(PF >= 0 && PF <= 3, "prefetch depth");
  static_assert(SCHED == 0 || SCHED == 1, "schedule");
  static_assert(Q8 >= 0 && Q8 <= 3, "KV storage / dequantisation");
  static_assert(Q8 == 0 || KL == 0, "INT8 loads the x fragments straight from memory");
  static_assert(N1_Q1_QPLANES == 1 || N1_Q1_QPLANES == 2, "int8 Q planes");
  typedef N1Smem<G, W> SmemT;
  N1_SMEM(SmemT, sm);
  constexpr int NT = W * 64;
  constexpr int NBUF = PF + 1;
  const int tid = N1_TID(), lane = tid & 63, w = tid >> 6, r = lane & 15, g = lane >> 4;
  const int bid = N1_BID(), nblk = N1_NBLK();
  int* const ctr = p.ctr;
  const int rows = p.rows, hkv = p.hkv;

  // (0) first claim, issued before the table build so its latency is hidden (SCHED 0)
  int claim = 0;
  if (SCHED == 0 && tid == 0) claim = N1_ATOMIC_ADD(&ctr[0], 1);

  // (1) work table, rebuilt from p.seq (DEVICE memory) by every CTA of every launch, so
  //     one captured launch configuration is valid for every replay
  const int per_t = (rows + NT - 1) / NT;   // rows per thread (<= N1_MAXR / NT)
  const int r0 = n1_imin(tid * per_t, rows), r1 = n1_imin(r0 + per_t, rows);
  int my_tiles = 0;
  for (int i = r0; i < r1; ++i) {
    const int s = p.seq[i];
    my_tiles += s > 0 ? (s + N1_TILE - 1) / N1_TILE : 0;
  }
  int t_cur = 0, t_end = 0, n_tiles = 1, geff = 1;   // SCHED 1 (1 = never divide by 0)
  if (SCHED == 1) {
    // cum[i] = prefix over rows of hkv * tiles(i) in TILE space (segments row-major)
    int total = 0;
    int off = n1_block_excl_scan<W>(my_tiles * hkv, sm.red, &total);
    for (int i = r0; i < r1; ++i) {
      sm.cum[i] = off;
      const int s = p.seq[i];
      off += hkv * (s > 0 ? (s + N1_TILE - 1) / N1_TILE : 0);
    }
    if (tid == 0) {
      sm.cum[rows] = total;
      sm.total_items = total;
    }
    N1_SYNC();
    n_tiles = n1_imax(1, sm.total_items);
    geff = n1_imin(nblk, n_tiles);                 // every CTA < geff owns >= 1 tile
    if (sm.total_items > 0 && bid < geff) {
      t_cur = n1_cta_start(bid, n_tiles, geff);
      t_end = n1_cta_start(bid + 1, n_tiles, geff);
    }
  } else {
    // (1a) item table: CT tiles per chunk, n_r chunks per row, prefix over rows
    const int total_tiles = n1_block_sum<W>(my_tiles, sm.red) * hkv;
    int ct = n1_imax(p.min_ct, (total_tiles + p.target_items - 1) / n1_imax(p.target_items, 1));
    ct = n1_imax(ct, 1);
    int total_items = 0;
    for (int tries = 0; tries < 32; ++tries) {
      int cnt = 0;
      for (int i = r0; i < r1; ++i) {
        const int s = p.seq[i];
        const int tl = s > 0 ? (s + N1_TILE - 1) / N1_TILE : 0;
        cnt += hkv * (tl > 0 ? (tl + ct - 1) / ct : 1);
      }
      int off = n1_block_excl_scan<W>(cnt, sm.red, &total_items);
      if (total_items <= p.max_items || tries == 31) {
        for (int i = r0; i < r1; ++i) {
          sm.cum[i] = off;
          const int s = p.seq[i];
          const int tl = s > 0 ? (s + N1_TILE - 1) / N1_TILE : 0;
          off += hkv * (tl > 0 ? (tl + ct - 1) / ct : 1);
        }
        break;
      }
      ct *= 2;   // block-uniform decision (total_items is the same in every thread)
    }
    if (tid == 0) {
      sm.cum[rows] = total_items;
      sm.total_items = total_items <= p.max_items ? total_items : 0;   // overflow: do nothing
      sm.ct = ct;
      sm.item = claim;
    }
    N1_SYNC();
  }
  int item = SCHED == 0 ? sm.item : 0;
  const int n_items = SCHED == 0 ? sm.total_items : 0;

  const float NEG_INF = n1_ninf();
  unsigned sink_acc = 0u;

  for (;;) {
    int row, kvh, nr, tb, te, slot, seg_start = 0;
    if (SCHED == 0) {
      if (item >= n_items) break;
      // ---- locate the item: row = last r with cum[r] <= item ------------------------
      int lo = 0, hi = rows;
      while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (sm.cum[mid] <= item) lo = mid; else hi = mid;
      }
      row = lo;
      nr = (sm.cum[row + 1] - sm.cum[row]) / hkv;
      const int jj = item - sm.cum[row];
      const int c = jj / hkv;
      kvh = jj - c * hkv;
      const int sq = p.seq[row];
      const int tl = sq > 0 ? (sq + N1_TILE - 1) / N1_TILE : 0;
      const int per = (tl + nr - 1) / nr;
      tb = c * per;
      te = n1_imin(tl, tb + per);
      slot = item;
    } else {
      if (t_cur >= t_end) break;
      // ---- locate the tile cursor: row = last r with cum[r] <= t_cur ----------------
      int lo = 0, hi = rows;
      while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (sm.cum[mid] <= t_cur) lo = mid; else hi = mid;
      }
      row = lo;
      const int sq = p.seq[row];
      const int tl = (sq + N1_TILE - 1) / N1_TILE;   // > 0: cum[row] <= t_cur < cum[row+1]
      const int in_row = t_cur - sm.cum[row];
      kvh = in_row / tl;
      seg_start = sm.cum[row] + kvh * tl;
      tb = t_cur - seg_start;
      te = n1_imin(tl, tb + (t_end - t_cur));
      // nr = number of CTAs this segment is split over; cta_of(t) = ((t+1)*geff-1)/T
      const int c0 = n1_cta_of(seg_start, n_tiles, geff);
      const int c1 = n1_cta_of(seg_start + tl - 1, n_tiles, geff);
      nr = c1 - c0 + 1;
      // slot 2c for the first segment this CTA touches, 2c+1 for its last one (only those
      // two can be incomplete, every segment strictly inside the range has nr == 1)
      slot = n1_slot(bid, seg_start, n_tiles, geff);
    }
    const int seq = p.seq[row];
    const int ntw = te - tb > w ? (te - tb - w + W - 1) / W : 0;   // tiles of this warp

    // ---- Q fragments (MMA y operand): lane (r, g) <- Q[head r][dims 32u + 8g .. +7] --
    n1u4 qf[4] = {n1_z4(), n1_z4(), n1_z4(), n1_z4()};
    n1u2 qy[8];   // KL 1: y fragment of column tile c = {Q[head r][32g + 8j + c]}_j
    if (MODE == 0) {
      if (KL == 0) {
        const uint16_t* qp =
            p.q + (long long)row * p.q_rs + (long long)(kvh * G + r) * p.q_hs + 8 * g;
#pragma unroll
        for (int u = 0; u < 4; ++u) qf[u] = r < G ? *(const n1u4*)(qp + 32 * u) : n1_z4();
      } else {
        const uint16_t* qp =
            p.q + (long long)row * p.q_rs + (long long)(kvh * G + r) * p.q_hs + 32 * g;
#pragma unroll
        for (int j = 0; j < 4; ++j) qf[j] = r < G ? *(const n1u4*)(qp + 8 * j) : n1_z4();
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          qy[c].x = n1_pick2(qf[0], qf[1], c);
          qy[c].y = n1_pick2(qf[2], qf[3], c);
        }
      }
    }
    // ---- R9/R (Q8 >= 2): Q into the MMA's own type, ONCE per item ---------------------
    // Everything here is amortised over the item's 70..1000 tiles, so the per-tile cost of
    // the fp16 path is only the convert itself.  Two exact, cost-free normalisations:
    //   * |q| <= 1 by dividing by the binade above the head's max (fp16's range is 2^-14
    //     .. 65504, so an unnormalised bf16 Q could overflow), with 2^Eq multiplied back
    //     into the score scale;
    //   * the magic-number offset of K, which contributes Q1_HBIAS * sum_d q_d to EVERY
    //     logit of head r, planted in the QK accumulator's init so no tile pays for it.
    //     The sum is taken over the fp16 values the MMA will actually see, so it cancels
    //     to fp32 precision; lane (r, g) owns dims {32u + 8g + j} of head r, and the two
    //     xor shuffles over g complete the 128-dim sum (the same reduction the running max
    //     uses).  Rows r >= G hold q = 0, so their bias is 0 as well.
    float sclane = p.scale_log2;
    n1f4 qb4 = n1_f4z();
    n1u2 qi8[4], qi8h[4];   // Q8 == 3: the int8 Q planes (low, high)
    if (MODE == 0) {
      if constexpr (Q8 >= 2) {
        float am = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            am = n1_max(am, __builtin_fabsf(n1_bflo(qf[u][j])));
            am = n1_max(am, __builtin_fabsf(n1_bfhi(qf[u][j])));
          }
        am = n1_max(am, N1_SHFL_XOR_F(am, 16));
        am = n1_max(am, N1_SHFL_XOR_F(am, 32));
        if constexpr (Q8 == 2) {
          const int eq = q1_ilog2(am, -60, 60) + 1;   // 2^eq > am >= 2^(eq-1)
          const float qn = q1_pow2i(-eq);
          sclane = p.scale_log2 * q1_pow2i(eq);
          float qs = 0.f;
#pragma unroll
          for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t h = q1_pack_h(n1_bflo(qf[u][j]) * qn, n1_bfhi(qf[u][j]) * qn);
              qf[u][j] = h;   // bf16 -> fp16 in the same registers
              qs += q1_h_lo(h) + q1_h_hi(h);
            }
          qs += N1_SHFL_XOR_F(qs, 16);
          qs += N1_SHFL_XOR_F(qs, 32);
          const float qb = -Q1_HBIAS * qs;
          qb4 = (n1f4){qb, qb, qb, qb};
        } else {
          // Q8 == 3: q -> int8 (QPLANES 1) or an exact 16-bit pair of int8 planes
          // (QPLANES 2: q16 = 256 * qh + ql), so the int8 MMA reads K with NO convert at
          // all and the accumulator is exact in int32.  No magic offset exists here.
          const float qmax = (float)(127 * (N1_Q1_QPLANES == 2 ? 256 : 1));
          const float qsc = am > 0.f ? qmax / am : 0.f;
          sclane = p.scale_log2 * (am > 0.f ? am / qmax : 0.f);
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            uint32_t lo[2] = {0u, 0u}, hi[2] = {0u, 0u};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float f[2] = {n1_bflo(qf[u][j]) * qsc, n1_bfhi(qf[u][j]) * qsc};
#pragma unroll
              for (int t = 0; t < 2; ++t) {
                const int d = 2 * j + t;           // dim offset 0..7 inside this u
                const int q16 = (int)__builtin_rintf(f[t]);
                int ql = q16, qh = 0;
                if (N1_Q1_QPLANES == 2) {
                  qh = (q16 + 128) >> 8;
                  ql = q16 - 256 * qh;
                }
                lo[d >> 2] |= ((uint32_t)(ql & 0xff)) << (8 * (d & 3));
                hi[d >> 2] |= ((uint32_t)(qh & 0xff)) << (8 * (d & 3));
              }
            }
            qi8[u] = (n1u2){lo[0], lo[1]};
            qi8h[u] = (n1u2){hi[0], hi[1]};
          }
        }
      }
    }
    // identity y fragment: Y[4g + j][r] = (4g + j == r)
    n1u2 eye;
    eye.x = (4 * g + 0 == r ? 0x3F80u : 0u) | (4 * g + 1 == r ? 0x3F800000u : 0u);
    eye.y = (4 * g + 2 == r ? 0x3F80u : 0u) | (4 * g + 3 == r ? 0x3F800000u : 0u);

    // ---- KV tile ids of this warp's first 64 tiles: lane L <- tile tb + w + L * W ----
    // (tile id = block id * bs_tiles + tile within the block; K/V tile base = id * 16 * c_ts)
    const int* __restrict__ btr = p.bt + (long long)row * p.bt_rs;
    const int bst = p.bs_tiles;
    int pid_lane = 0;
    {
      const int t = tb + w + lane * W;
      if (lane < ntw) pid_lane = bst == 1 ? btr[t] : btr[t / bst] * bst + t % bst;
    }
    const long long hoff = (long long)kvh * p.c_hs;
    const long long tstride = (long long)N1_TILE * p.c_ts;
    const long long koff = (long long)r * p.c_ts + 8 * g;            // + 32u
    const long long voff = (long long)(4 * g) * p.c_ts + 8 * r;      // + j * c_ts
    const long long cts = p.c_ts;
    // Q8 1 (INT8, layout Q1-L1 in q1_int8.cuh): byte offsets inside a half page.  A page
    // is one tile (bs_tiles == 1, checked by n1_check), 2 * c_bs bytes; a kv head owns
    // 16 * 128 B of data at kvh * 2048 and 16 fp32 scales at q8_sc + 64 * kvh.
    const long long pb8 = 2 * p.c_bs;
    const long long hoff8 = (long long)kvh * (N1_TILE * N1_D);
    const long long koff8 = (long long)g * (N1_TILE * 8) + 8 * r;    // + (4u)*128
    const long long voff8 = (long long)(4 * g) * N1_D + 8 * r;       // + j * 128
    const long long soff8 = (long long)p.q8_sc + 4 * (kvh * N1_TILE + 4 * g);
    // R9/R (Q8 >= 2): Lv shifts the softmax exponent so the peak weight P * v_scale lands
    // near 2^Q1_HTARGET instead of ~2^-7, which is what keeps it out of fp16's subnormal
    // range (see q1_int8.cuh "RANGE").  It is a CONSTANT FACTOR on this item's numerator
    // AND denominator, so it cancels in the final divide and is never undone -- but for
    // that it must be the same number in every warp and every CTA of the segment, hence
    // the V scale of the ROW's first token (one broadcast 4-byte load) and not the warp's.
    float lvf = 0.f;
    if (MODE == 0 && Q8 >= 2 && seq > 0) {
      const float vs0 = *(const float*)((const int8_t*)p.vc + (long long)btr[0] * pb8 +
                                        (long long)p.q8_sc + 4 * (kvh * N1_TILE));
      lvf = (float)(Q1_HTARGET - q1_ilog2(vs0, -20, 40));
    }

    float m_run = NEG_INF, l_run = 0.f;
#if N1_Q1_PV_FOLD
    float ls_run = 0.f;   // sum of the fp16 P actually fed to the MMA (PV bias fold)
#endif
    n1f4 o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = n1_f4z();

    typedef typename N1TileSel<Q8>::type TileT;
    TileT buf[NBUF];
    int next_claim = -1;
    const int claim_at = n1_imax(0, ntw - p.claim_lead);

    // load tile number ii (0-based in this warp's list) into b
    auto load = [&](TileT& b, int ii) N1_INLINE_LAMBDA {
      if ((ii & 63) == 0 && ii > 0) {   // next batch of tile ids (warp-uniform branch)
        const int t = tb + w + (ii + lane) * W;
        pid_lane = (ii + lane < ntw) ? (bst == 1 ? btr[t] : btr[t / bst] * bst + t % bst) : 0;
      }
      const int tid_kv = N1_SHFL_I(pid_lane, ii & 63);
      if constexpr (Q8 >= 1) {
        // one 8-byte load per MMA x half / V token; over the warp each instruction covers
        // 512 B of the page (K: contiguous, V: 4 x 128 B runs)
        const int8_t* kp = (const int8_t*)p.kc + (long long)tid_kv * pb8;
        const int8_t* vp = (const int8_t*)p.vc + (long long)tid_kv * pb8;
#pragma unroll
        for (int u = 0; u < 4; ++u)
          b.k[u] = q1_ld64<LDK>(kp + hoff8 + koff8 + (long long)(4 * u) * N1_D);
#pragma unroll
        for (int j = 0; j < 4; ++j)
          b.v[j] = q1_ld64<LDK>(vp + hoff8 + voff8 + (long long)j * N1_D);
        b.ks = *(const n1f4*)(kp + soff8);
        b.vs = *(const n1f4*)(vp + soff8);
      } else {
        const long long base = (long long)tid_kv * tstride + hoff;
        const uint16_t* vp = p.vc + base + voff;
        if (KL == 0) {
          const uint16_t* kp = p.kc + base + koff;
#pragma unroll
          for (int u = 0; u < 4; ++u) b.k[u] = n1_ld<LDK>(kp + 32 * u);
        } else {
          const uint16_t* kp = p.kc + base + voff;   // V-like: token 4g + j, dims 8r .. 8r+7
#pragma unroll
          for (int j = 0; j < 4; ++j) b.k[j] = n1_ld<LDK>(kp + j * cts);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) b.v[j] = n1_ld<LDK>(vp + j * cts);
      }
    };

    auto compute = [&](TileT& b, int ii) N1_INLINE_LAMBDA {
      if (MODE == 1) {
        if constexpr (Q8 >= 1) {
#pragma unroll
          for (int u = 0; u < 4; ++u) sink_acc += b.k[u].x ^ b.k[u].y;
#pragma unroll
          for (int j = 0; j < 4; ++j) sink_acc += b.v[j].x ^ b.v[j].y;
          sink_acc += n1_f2u(b.ks.x) ^ n1_f2u(b.vs.x);
        } else {
#pragma unroll
          for (int u = 0; u < 4; ++u) sink_acc += b.k[u].x ^ b.k[u].y ^ b.k[u].z ^ b.k[u].w;
#pragma unroll
          for (int j = 0; j < 4; ++j) sink_acc += b.v[j].x ^ b.v[j].y ^ b.v[j].z ^ b.v[j].w;
        }
        return;
      }
      const int tok0 = (tb + w + ii * W) * N1_TILE;
      const bool full = tok0 + N1_TILE <= seq;   // warp-uniform
      // tail tile: zero K / V rows of tokens past the end of the row.  P is 0 there anyway;
      // this keeps NaN/Inf garbage of unused cache slots out of the MMAs (KL 1: the identity
      // MMA mixes all 16 tokens of a column tile, so K must be clean too)
      // Q8 1: the int8 data of invalid tokens is harmless (an int8 can never be NaN / Inf,
      // and K's MMA keeps token rows separate), but the fp32 SCALES of never-written cache
      // slots can be any bit pattern, so the tail zeroes those instead (q1_int8.cuh).
      n1f4 ksc = n1_f4z(), vsc = n1_f4z();   // dead (folded away) for Q8 == 0
      if constexpr (Q8 >= 1) {
        ksc = b.ks;
        vsc = b.vs;
        if (!full) {
#pragma unroll
          for (int i = 0; i < 4; ++i)
            if (tok0 + 4 * g + i >= seq) {
              ksc[i] = 0.f;
              vsc[i] = 0.f;
            }
        }
      } else if (!full) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (tok0 + 4 * g + j >= seq) {
            b.v[j] = n1_z4();
            if (KL == 1) b.k[j] = n1_z4();
          }
      }
      // S^T = K . Q^T: lane (r, g) gets S[head r][token tok0 + 4g + i]
      // (two independent accumulation chains of 4 MMAs each)
      n1f4 s;
      if constexpr (Q8 == 3) {
        // int8 MMA: the x fragment IS the loaded K word (4 int8 = dims 32u + 8g + 0..3 of
        // token r), so the K side of the consume loop costs ZERO ALU ops; int32 is exact
        // and the two scales are applied afterwards (sclane carries Q's, ksc[i] K's).
        n1i4 ia = (n1i4){0, 0, 0, 0}, ib = ia, ja = ia, jb = ia;
#pragma unroll
        for (int u = 0; u < 4; u += 2) {
          ia = N1_MMA_I8(b.k[u].x, qi8[u].x, ia);
          ib = N1_MMA_I8(b.k[u + 1].x, qi8[u + 1].x, ib);
          ia = N1_MMA_I8(b.k[u].y, qi8[u].y, ia);
          ib = N1_MMA_I8(b.k[u + 1].y, qi8[u + 1].y, ib);
          if constexpr (N1_Q1_QPLANES == 2) {
            ja = N1_MMA_I8(b.k[u].x, qi8h[u].x, ja);
            jb = N1_MMA_I8(b.k[u + 1].x, qi8h[u + 1].x, jb);
            ja = N1_MMA_I8(b.k[u].y, qi8h[u].y, ja);
            jb = N1_MMA_I8(b.k[u + 1].y, qi8h[u + 1].y, jb);
          }
        }
        const n1i4 it = N1_Q1_QPLANES == 2 ? (ia + ib) + 256 * (ja + jb) : ia + ib;
        s = __builtin_convertvector(it, n1f4);
      } else {
      n1f4 sa = Q8 == 2 ? qb4 : n1_f4z(), sb = n1_f4z();
      if constexpr (Q8 == 2) {
        // fp16 magic convert: 1 XOR per 4 bytes + 1 permute per 16-bit pair = 3 ALU ops
        // per loaded word, against 24 for the bf16 route below
#pragma unroll
        for (int u = 0; u < 4; u += 2) {
          const uint32_t a0 = q1_x80(b.k[u].x), a1 = q1_x80(b.k[u].y);
          const uint32_t c0 = q1_x80(b.k[u + 1].x), c1 = q1_x80(b.k[u + 1].y);
          // qf now holds fp16: elements 0,1 of a word are dims +2j, +2j+1 (q1_h4 takes
          // the two words).  NB: __builtin_bit_cast of a SWIZZLE (qf[u].zw) silently
          // ignores the swizzle offset in clang 19/22 -- it casts the vector's first
          // lanes -- so the words are named explicitly here.  q1_int8.cuh "SWIZZLE TRAP".
          sa = N1_MMA_F16(q1_h4(q1_h2lo(a0), q1_h2hi(a0)), q1_h4(qf[u][0], qf[u][1]), sa);
          sb = N1_MMA_F16(q1_h4(q1_h2lo(c0), q1_h2hi(c0)),
                          q1_h4(qf[u + 1][0], qf[u + 1][1]), sb);
          sa = N1_MMA_F16(q1_h4(q1_h2lo(a1), q1_h2hi(a1)), q1_h4(qf[u][2], qf[u][3]), sa);
          sb = N1_MMA_F16(q1_h4(q1_h2lo(c1), q1_h2hi(c1)),
                          q1_h4(qf[u + 1][2], qf[u + 1][3]), sb);
        }
      } else if constexpr (Q8 == 1) {
        // the x fragments come straight from the layout; dequantise 4 bytes at a time
#pragma unroll
        for (int u = 0; u < 4; u += 2) {
          const n1u2 a0 = q1_cvt4(b.k[u].x), a1 = q1_cvt4(b.k[u].y);
          const n1u2 c0 = q1_cvt4(b.k[u + 1].x), c1 = q1_cvt4(b.k[u + 1].y);
          sa = N1_MMA(a0, qf[u].xy, sa);
          sb = N1_MMA(c0, qf[u + 1].xy, sb);
          sa = N1_MMA(a1, qf[u].zw, sa);
          sb = N1_MMA(c1, qf[u + 1].zw, sb);
        }
      } else if (KL == 0) {
#pragma unroll
        for (int u = 0; u < 4; u += 2) {
          sa = N1_MMA(b.k[u].xy, qf[u].xy, sa);
          sb = N1_MMA(b.k[u + 1].xy, qf[u + 1].xy, sb);
          sa = N1_MMA(b.k[u].zw, qf[u].zw, sa);
          sb = N1_MMA(b.k[u + 1].zw, qf[u + 1].zw, sb);
        }
      } else {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          // x' = {K[4g + j][8r + c]}_j; D = x' . I -> lane (r, g) gets K[token r][8(4g+i) + c]
          n1u2 xt;
          xt.x = n1_pick2(b.k[0], b.k[1], c);
          xt.y = n1_pick2(b.k[2], b.k[3], c);
          const n1f4 d = N1_MMA(xt, eye, n1_f4z());
          n1u2 kx;   // exact: the values are bf16
          kx.x = (n1_f2u(d.x) >> 16) | (n1_f2u(d.y) & 0xffff0000u);
          kx.y = (n1_f2u(d.z) >> 16) | (n1_f2u(d.w) & 0xffff0000u);
          if (c & 1)
            sb = N1_MMA(kx, qy[c], sb);
          else
            sa = N1_MMA(kx, qy[c], sa);
        }
      }
      s = sa + sb;
      }
      float x[4];
      float mt = NEG_INF;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const bool ok = full || tok0 + 4 * g + i < seq;
        // Q8 1: S = softmax_scale * k_scale[token] * (q . k_int), the per-token K scale
        // folded into the score of exactly the tokens this lane owns
        const float sc = Q8 == 0 ? p.scale_log2
                                 : (Q8 == 1 ? p.scale_log2 * ksc[i] : sclane * ksc[i]);
        x[i] = ok ? s[i] * sc : NEG_INF;
        mt = n1_max(mt, x[i]);
      }
      mt = n1_max(mt, N1_SHFL_XOR_F(mt, 16));
      mt = n1_max(mt, N1_SHFL_XOR_F(mt, 32));
      const float m_new = n1_max(m_run, mt);   // finite: token tok0 of the tile is valid
      const float alpha = m_run == NEG_INF ? 0.f : N1_EXP2(m_run - m_new);
      m_run = m_new;
      // Q8 >= 2: the SAME exp2 also applies the exponent shift Lv (one add per tile), so
      // the fp16 P operand is 2^Lv times the true weight -- a constant this item's
      // denominator carries as well, so it cancels at the divide
      float mpv = m_new;
      if constexpr (Q8 >= 2) mpv = m_new - lvf;
      const float e0 = N1_EXP2(x[0] - mpv), e1 = N1_EXP2(x[1] - mpv);
      const float e2 = N1_EXP2(x[2] - mpv), e3 = N1_EXP2(x[3] - mpv);
      n1u2 pa;
      if constexpr (Q8 >= 2) {
        // P * v_scale as fp16 (the V scale is on a contracted dimension, so it cannot be
        // applied after the dot); the denominator sums the fp32 values, which is both
        // cheaper than unpacking and more accurate than the bf16 path's rounded ones
        const uint32_t w01 = q1_pack_h(e0 * vsc[0], e1 * vsc[1]);
        const uint32_t w23 = q1_pack_h(e2 * vsc[2], e3 * vsc[3]);
        l_run = l_run * alpha + ((e0 + e1) + (e2 + e3));
#if N1_Q1_PV_FOLD
        ls_run = ls_run * alpha +
                 ((q1_h_lo(w01) + q1_h_hi(w01)) + (q1_h_lo(w23) + q1_h_hi(w23)));
#endif
        pa.x = r < G ? w01 : 0u;
        pa.y = r < G ? w23 : 0u;
      } else {
        const uint32_t p01 = n1_pack_p(e0, e1);
        const uint32_t p23 = n1_pack_p(e2, e3);
        // the denominator always sums the UNSCALED rounded P (see q1_int8.cuh "V")
        l_run = l_run * alpha + ((n1_bflo(p01) + n1_bfhi(p01)) + (n1_bflo(p23) + n1_bfhi(p23)));
        if constexpr (Q8 == 1) {
          // Q8 1: O += sum_tok (P * v_scale[tok]) * V_int[tok], the per-token V scale folded
          // into the MMA's P operand (a contracted dimension cannot be scaled afterwards)
          const uint32_t q01 = n1_pack_p(e0 * vsc[0], e1 * vsc[1]);
          const uint32_t q23 = n1_pack_p(e2 * vsc[2], e3 * vsc[3]);
          pa.x = r < G ? q01 : 0u;
          pa.y = r < G ? q23 : 0u;
        } else {
          pa.x = r < G ? p01 : 0u;   // rows >= G are not heads of this kv head
          pa.y = r < G ? p23 : 0u;
        }
      }
      // rescale O rows 4g + i by alpha of head 4g + i (held by lane 4g + i); skipped when
      // no head's running max moved (alpha == 1 everywhere: the product would be exact)
      if (N1_ANY(alpha != 1.f)) {
        n1f4 al;
        al.x = N1_SHFL_F(alpha, 4 * g + 0);
        al.y = N1_SHFL_F(alpha, 4 * g + 1);
        al.z = N1_SHFL_F(alpha, 4 * g + 2);
        al.w = N1_SHFL_F(alpha, 4 * g + 3);
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] *= al;
      }
      // O += P . V, column tile k: n -> dim 8n + k; y = {V[4g+j][8r+k]}_j
      if constexpr (Q8 >= 2) {
        // Per 16-bit element pair wd of the loaded V words: 4 permutes turn the four
        // tokens' bytes 2wd / 2wd+1 into fp16 pairs, 4 more transpose them into the two
        // column tiles' y fragments.  The ^0x80 is one XOR per loaded word, done in place
        // (each pipeline slot is reloaded before its next compute).
        const n1h4 ph = q1_h4(pa.x, pa.y);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          b.v[j].x = q1_x80(b.v[j].x);
          b.v[j].y = q1_x80(b.v[j].y);
        }
#pragma unroll
        for (int wd = 0; wd < 4; ++wd) {
          const int sw = wd >> 1;
          uint32_t h[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t t = (wd & 1) ? q1_h2hi(b.v[j][sw]) : q1_h2lo(b.v[j][sw]);
            h[j] = N1_Q1_PV_FOLD ? t : q1_h2_unbias(t);
          }
          o[2 * wd] = N1_MMA_F16(
              ph, q1_h4(q1_pick2_h(h[0], h[1], 0), q1_pick2_h(h[2], h[3], 0)), o[2 * wd]);
          o[2 * wd + 1] = N1_MMA_F16(
              ph, q1_h4(q1_pick2_h(h[0], h[1], 1), q1_pick2_h(h[2], h[3], 1)), o[2 * wd + 1]);
        }
      } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        n1u2 vy;
        if constexpr (Q8 == 1) {
          vy.x = q1_pick2_i8(b.v[0], b.v[1], k);
          vy.y = q1_pick2_i8(b.v[2], b.v[3], k);
        } else {
          vy.x = n1_pick2(b.v[0], b.v[1], k);
          vy.y = n1_pick2(b.v[2], b.v[3], k);
        }
        o[k] = N1_MMA(pa, vy, o[k]);
      }
      }
    };

    // ---- software-pipelined tile loop (PF tiles in flight per warp) -----------------
#pragma unroll
    for (int s0 = 0; s0 < PF; ++s0)
      if (s0 < ntw) load(buf[s0], s0);
    for (int i0 = 0; i0 < ntw; i0 += NBUF) {
#pragma unroll
      for (int b = 0; b < NBUF; ++b) {
        const int ii = i0 + b;
        if (ii + PF < ntw) load(buf[(b + PF) % NBUF], ii + PF);
        if (ii < ntw) {
          if (SCHED == 0 && w == 0 && lane == 0 && ii == claim_at)
            next_claim = N1_ATOMIC_ADD(&ctr[0], 1);
          compute(buf[b], ii);
        }
      }
    }
    if (SCHED == 0) {
      if (tid == 0 && next_claim < 0) next_claim = N1_ATOMIC_ADD(&ctr[0], 1);
      if (tid == 0) sm.next = next_claim;
    }

    if (MODE == 0) {
      // ---- merge the W warps of this CTA ---------------------------------------------
      l_run += N1_SHFL_XOR_F(l_run, 16);
      l_run += N1_SHFL_XOR_F(l_run, 32);
#if N1_Q1_PV_FOLD
      if constexpr (Q8 >= 2) {
        // O[head 4g+i][*] carries Q1_HBIAS * sum_tok P_fed[head 4g+i][tok]; ls_run holds
        // that sum for head r, so it is completed over g and then transposed from lane
        // 4g+i exactly like alpha.  Linear, and applied BEFORE any merge, so the warp,
        // CTA and cross-CTA reductions and the partial slots all stay unchanged.
        ls_run += N1_SHFL_XOR_F(ls_run, 16);
        ls_run += N1_SHFL_XOR_F(ls_run, 32);
        n1f4 lsv;
        lsv.x = N1_SHFL_F(ls_run, 4 * g + 0);
        lsv.y = N1_SHFL_F(ls_run, 4 * g + 1);
        lsv.z = N1_SHFL_F(ls_run, 4 * g + 2);
        lsv.w = N1_SHFL_F(ls_run, 4 * g + 3);
        const n1f4 cor = Q1_HBIAS * lsv;
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] -= cor;
      }
#endif
      if (g == 0 && r < G) {
        sm.m[w][r] = m_run;
        sm.l[w][r] = l_run;
      }
      N1_SYNC();
      {   // scale own O rows (heads 4g + i) to the CTA max of that head
        n1f4 sc;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int h = 4 * g + i;
          float f = 0.f;
          if (h < G) {
            float mx = NEG_INF;
#pragma unroll
            for (int v = 0; v < W; ++v) mx = n1_max(mx, sm.m[v][h]);
            const float mw = sm.m[w][h];
            f = mw == NEG_INF ? 0.f : N1_EXP2(mw - mx);
          }
          sc[i] = f;
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] *= sc;
      }
#pragma unroll
      for (int rd = 0; rd < (W + N1_SMB - 1) / N1_SMB; ++rd) {
        if (w / N1_SMB == rd) {
          n1f4* dst = sm.o[w % N1_SMB];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int h = 4 * g + i;
            if (h < G) {
              const n1f4 a = (n1f4){o[0][i], o[1][i], o[2][i], o[3][i]};
              const n1f4 bb = (n1f4){o[4][i], o[5][i], o[6][i], o[7][i]};
              const int ix = h * (N1_D / 4) + 2 * r;   // dims 8r .. 8r+3, 8r+4 .. 8r+7
              if (rd == 0) {
                dst[ix] = a;
                dst[ix + 1] = bb;
              } else {
                dst[ix] += a;
                dst[ix + 1] += bb;
              }
            }
          }
        }
        N1_SYNC();
      }
      // tasks (head hh, 4 dims q4): CTA result -> final output or partial slot
      const long long orow = (long long)row * p.o_rs;
      for (int task = tid; task < G * (N1_D / 4); task += NT) {
        const int hh = task / (N1_D / 4), q4 = task - hh * (N1_D / 4);
        n1f4 acc = sm.o[0][task];
#pragma unroll
        for (int bb = 1; bb < (W < N1_SMB ? W : N1_SMB); ++bb) acc += sm.o[bb][task];
        float mx = NEG_INF;
#pragma unroll
        for (int v = 0; v < W; ++v) mx = n1_max(mx, sm.m[v][hh]);
        float lsum = 0.f;
#pragma unroll
        for (int v = 0; v < W; ++v) {
          const float mw = sm.m[v][hh];
          lsum += mw == NEG_INF ? 0.f : N1_EXP2(mw - mx) * sm.l[v][hh];
        }
        if (nr == 1) {
          const float inv = lsum > 0.f ? 1.f / lsum : 0.f;
          uint16_t* op = p.out + orow + (long long)(kvh * G + hh) * p.o_hs + 4 * q4;
          *(n1u2*)op = (n1u2){n1_pack_out(acc.x * inv, acc.y * inv),
                              n1_pack_out(acc.z * inv, acc.w * inv)};
        } else {
          *(n1f4*)(p.ws_o + (long long)slot * (G * N1_D) + 4 * task) = acc;
          if (q4 == 0) {
            p.ws_ml[(long long)slot * N1_ML + hh] = mx;
            p.ws_ml[(long long)slot * N1_ML + 16 + hh] = lsum;
          }
        }
      }
      if (nr > 1) {
        // ---- cross-CTA merge: the last CTA to arrive on (row, kvh) combines ---------
        N1_FENCE();
        N1_SYNC();
        if (tid == 0) {
          const int old = N1_ATOMIC_ADD(&ctr[2 + row * hkv + kvh], 1);
          sm.flag = old == nr - 1;
        }
        N1_SYNC();
        if (sm.flag) {
          N1_FENCE();
          // slot of the cc-th chunk, in ascending tile order (deterministic reduction):
          //   SCHED 0: item index of chunk cc of this (row, kv head)
          //   SCHED 1: 2 * c + (start(c) < seg_start) for the cc-th CTA c = c0 + cc
          const int base = SCHED == 0 ? sm.cum[row] + kvh : 0;
          const int c0 = SCHED == 0 ? 0 : n1_cta_of(seg_start, n_tiles, geff);
          for (int task = tid; task < G * (N1_D / 4); task += NT) {
            const int hh = task / (N1_D / 4), q4 = task - hh * (N1_D / 4);
            float mx = NEG_INF;
            for (int cc = 0; cc < nr; ++cc) {
              const long long sl0 = SCHED == 0 ? base + cc * hkv : n1_slot(c0 + cc, seg_start,
                                                                          n_tiles, geff);
              mx = n1_max(mx, n1_ldv(p.ws_ml + sl0 * N1_ML + hh));
            }
            n1f4 acc = n1_f4z();
            float lsum = 0.f;
            for (int cc = 0; cc < nr; ++cc) {
              const long long sl = SCHED == 0 ? base + cc * hkv : n1_slot(c0 + cc, seg_start,
                                                                         n_tiles, geff);
              const float mc = n1_ldv(p.ws_ml + sl * N1_ML + hh);
              const float e = mc == NEG_INF ? 0.f : N1_EXP2(mc - mx);
              lsum += e * n1_ldv(p.ws_ml + sl * N1_ML + 16 + hh);
              acc += e * n1_ldv4(p.ws_o + sl * (G * N1_D) + 4 * task);
            }
            const float inv = lsum > 0.f ? 1.f / lsum : 0.f;
            uint16_t* op = p.out + orow + (long long)(kvh * G + hh) * p.o_hs + 4 * q4;
            *(n1u2*)op = (n1u2){n1_pack_out(acc.x * inv, acc.y * inv),
                                n1_pack_out(acc.z * inv, acc.w * inv)};
          }
          if (tid == 0) N1_ATOMIC_EXCH(&ctr[2 + row * hkv + kvh], 0);
        }
      }
    }
    N1_SYNC();   // sm.next written; sm.o / sm.m / sm.flag reads of this item done
    if (SCHED == 0) {
      item = sm.next;
      N1_SYNC();   // everyone has read sm.next before thread 0 may write it again
    } else {
      t_cur += te - tb;   // thread-local, identical in every thread of the CTA
    }
  }

  if (MODE == 1 && sink_acc == 0x9E3779B9u && p.sink != nullptr) p.sink[0] = sink_acc;

  // ---- rows with seq_len <= 0 (CUDA-graph batch padding): write zeros ----------------
  // SCHED 0 gives them one empty item each; SCHED 1 has no tiles for them, so the
  // segments are swept here, grid-strided, after the real work.
  if (SCHED == 1 && MODE == 0) {
    for (int sg = bid; sg < rows * hkv; sg += nblk) {
      const int row = sg / hkv, kvh = sg - row * hkv;
      if (p.seq[row] > 0) continue;
      const long long orow = (long long)row * p.o_rs;
      for (int task = tid; task < G * (N1_D / 4); task += NT) {
        const int hh = task / (N1_D / 4), q4 = task - hh * (N1_D / 4);
        uint16_t* op = p.out + orow + (long long)(kvh * G + hh) * p.o_hs + 4 * q4;
        *(n1u2*)op = (n1u2){0u, 0u};
      }
    }
  }

  // ---- exit: the last CTA out resets the claim / exit counters for the next launch ----
  // SCHED 1 uses neither counter, so it leaves ctr[0..1] untouched (its only device state
  // is ctr[2 + segment], zeroed by the CTA that merged that segment).
  if (SCHED == 0 && tid == 0) {
    N1_FENCE();
    const int done = N1_ATOMIC_ADD(&ctr[1], 1);
    if (done == nblk - 1) {
      N1_ATOMIC_EXCH(&ctr[0], 0);
      N1_ATOMIC_EXCH(&ctr[1], 0);
    }
  }
}

// ---------------------------------------------------------------------------
// MMA fragment probe (same tests as R6/X4): lane l, element i = 16 * (4 * (l / 16) + i) + l % 16
//   test 0: x = I, y[k][n] = 16k + n  -> D = y;   test 1: x[m][k] = 16m + k, y = I -> D = x
// ---------------------------------------------------------------------------
N1_GLOBAL void N1_LAUNCH_BOUNDS(64) n1_probe_mma_kernel(float* out) {
  const int l = N1_TID() & 63, r = l & 15, g = l >> 4;
  float a0[4], b0[4], a1[4], b1[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int k = 4 * g + j;
    a0[j] = (r == k) ? 1.f : 0.f;
    b0[j] = (float)(16 * k + r);
    a1[j] = (float)(16 * r + k);
    b1[j] = (k == r) ? 1.f : 0.f;
  }
  const n1f4 zero = n1_f4z();
  const n1f4 d0 = N1_MMA(((n1u2){n1_pack_out(a0[0], a0[1]), n1_pack_out(a0[2], a0[3])}),
                         ((n1u2){n1_pack_out(b0[0], b0[1]), n1_pack_out(b0[2], b0[3])}), zero);
  const n1f4 d1 = N1_MMA(((n1u2){n1_pack_out(a1[0], a1[1]), n1_pack_out(a1[2], a1[3])}),
                         ((n1u2){n1_pack_out(b1[0], b1[1]), n1_pack_out(b1[2], b1[3])}), zero);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[l * 4 + i] = d0[i];
    out[256 + l * 4 + i] = d1[i];
  }
}

// ---------------------------------------------------------------------------
// host side: configuration table and launchers (0 ok, <0 rejected, >=1000 launch error)
// ---------------------------------------------------------------------------
#ifndef N1_EMU
typedef N1_STREAM_T n1_stream_t;
#else
typedef void* n1_stream_t;
#endif

#define N1_G 8   // q heads per kv head built into the library (MiniCPM5: 16 / 2)

// id, W (warps / CTA), PF (tiles in flight / warp), LDK (0 plain, 1 non-temporal),
// MODE (0 attention, 1 read floor), KL (K load: 0 fragment layout, 1 coalesced + MMA
// transpose), SCHED (0 atomic-claimed items = R6/N1, 1 static stream-K = R7/N4).
// Ids 0..17 are the R6/N1 table, unchanged (cfg 13 is the deployed mixed-step policy);
// 18..27 are the stream-K configs the decode-only / in-graph path chooses from; 28..41 are
// R8/Q1's int8 configs (bf16-magic dequant, cfg 28 decode-only / 38 side stream); 42..52
// are R9/R's fp16-magic (Q8 = 2) and int8-MMA-QK (Q8 = 3) configs over the SAME cache --
// cfg 42 mirrors 28 and cfg 47 mirrors 38 exactly, so they are the A/B pairs to time, and
// cfg 49 mirrors the MODE 1 read floor of cfg 35 (it must measure the same: the loads are
// byte-for-byte the same, only the consume loop changed).
#define N1_CFG_LIST(X)     \
  X(0, 4, 1, 0, 0, 0, 0, 0)   \
  X(1, 4, 0, 0, 0, 0, 0, 0)   \
  X(2, 4, 2, 0, 0, 0, 0, 0)   \
  X(3, 4, 1, 1, 0, 0, 0, 0)   \
  X(4, 2, 1, 0, 0, 0, 0, 0)   \
  X(5, 2, 2, 0, 0, 0, 0, 0)   \
  X(6, 8, 1, 0, 0, 0, 0, 0)   \
  X(7, 8, 0, 0, 0, 0, 0, 0)   \
  X(8, 4, 2, 1, 0, 0, 0, 0)   \
  X(9, 8, 1, 1, 0, 0, 0, 0)   \
  X(10, 2, 2, 1, 0, 0, 0, 0)  \
  X(11, 4, 1, 0, 1, 0, 0, 0)  \
  X(12, 4, 2, 0, 1, 0, 0, 0)  \
  X(13, 4, 1, 0, 0, 1, 0, 0)  \
  X(14, 4, 2, 0, 0, 1, 0, 0)  \
  X(15, 2, 2, 0, 0, 1, 0, 0)  \
  X(16, 8, 1, 0, 0, 1, 0, 0)  \
  X(17, 4, 1, 0, 1, 1, 0, 0)  \
  X(18, 4, 1, 0, 0, 1, 1, 0)  \
  X(19, 4, 2, 0, 0, 1, 1, 0)  \
  X(20, 4, 0, 0, 0, 1, 1, 0)  \
  X(21, 4, 1, 0, 0, 0, 1, 0)  \
  X(22, 8, 1, 0, 0, 1, 1, 0)  \
  X(23, 2, 1, 0, 0, 1, 1, 0)  \
  X(24, 2, 2, 0, 0, 1, 1, 0)  \
  X(25, 4, 1, 1, 0, 1, 1, 0)  \
  X(26, 4, 1, 0, 1, 1, 1, 0)  \
  X(27, 8, 1, 0, 0, 0, 1, 0)  \
  X(28, 4, 1, 0, 0, 0, 1, 1)  \
  X(29, 4, 2, 0, 0, 0, 1, 1)  \
  X(30, 4, 3, 0, 0, 0, 1, 1)  \
  X(31, 8, 1, 0, 0, 0, 1, 1)  \
  X(32, 8, 2, 0, 0, 0, 1, 1)  \
  X(33, 2, 2, 0, 0, 0, 1, 1)  \
  X(34, 4, 2, 1, 0, 0, 1, 1)  \
  X(35, 4, 1, 0, 1, 0, 1, 1)  \
  X(36, 4, 2, 0, 1, 0, 1, 1)  \
  X(37, 4, 3, 0, 1, 0, 1, 1)  \
  X(38, 4, 1, 0, 0, 0, 0, 1)  \
  X(39, 4, 2, 0, 0, 0, 0, 1)  \
  X(40, 4, 3, 0, 0, 0, 0, 1)  \
  X(41, 4, 2, 0, 1, 0, 0, 1)  \
  X(42, 4, 1, 0, 0, 0, 1, 2)  \
  X(43, 4, 2, 0, 0, 0, 1, 2)  \
  X(44, 2, 2, 0, 0, 0, 1, 2)  \
  X(45, 4, 0, 0, 0, 0, 1, 2)  \
  X(46, 8, 1, 0, 0, 0, 1, 2)  \
  X(47, 4, 1, 0, 0, 0, 0, 2)  \
  X(48, 4, 2, 0, 0, 0, 0, 2)  \
  X(49, 4, 1, 0, 1, 0, 1, 2)  \
  X(50, 4, 1, 0, 0, 0, 1, 3)  \
  X(51, 4, 0, 0, 0, 0, 1, 3)  \
  X(52, 4, 1, 0, 0, 0, 0, 3)

struct N1CfgInfo {
  int w, pf, ldk, mode, kl, sched, q8;
};
static const N1CfgInfo n1_cfgs[] = {
#define N1_X(id, W_, PF_, LDK_, MODE_, KL_, SC_, Q8_) \
  {W_, PF_, LDK_, MODE_, KL_, SC_, Q8_},
    N1_CFG_LIST(N1_X)
#undef N1_X
};
#define N1_NUM_CFGS ((int)(sizeof(n1_cfgs) / sizeof(n1_cfgs[0])))

template <int W, int PF, int LDK, int MODE, int KL, int SCHED, int Q8>
static inline int n1_launch_t(const N1Params& prm, int grid, n1_stream_t stream) {
  const N1Params pp = prm;
#ifdef N1_EMU
  n1emu_launch((unsigned)grid, (unsigned)(W * 64), [=] {
    n1_dec_attn_kernel<N1_G, W, PF, LDK, MODE, KL, SCHED, Q8>(pp);
  });
  (void)stream;
#else
  n1_dec_attn_kernel<N1_G, W, PF, LDK, MODE, KL, SCHED, Q8>
      <<<dim3((unsigned)grid, 1, 1), dim3(W * 64, 1, 1), 0, stream>>>(pp);
  const int e = (int)N1_LAST_ERROR();
  if (e) return 1000 + e;
#endif
  return 0;
}

// argument checks shared by the front-ends
static inline int n1_check(const N1Params& p, int hq, int grid) {
  if (p.hkv < 1 || hq != N1_G * p.hkv) return -2;            // GQA group != N1_G
  if (p.rows < 1 || p.rows > N1_MAXR) return -3;
  if (grid < 1 || grid > 65535) return -4;
  if (p.bs_tiles < 1) return -5;
  if ((p.q_rs % 8) || (p.q_hs % 8) || (p.c_bs % 8) || (p.c_ts % 8) || (p.c_hs % 8) ||
      (p.o_rs % 4) || (p.o_hs % 4))
    return -6;                                               // 16-byte (8-byte out) vectors
  if ((((uintptr_t)p.q) | ((uintptr_t)p.kc) | ((uintptr_t)p.vc)) & 15u) return -7;
  if (((uintptr_t)p.out) & 7u) return -7;
  if (p.c_bs != (long long)p.bs_tiles * N1_TILE * p.c_ts) return -5;   // pages = token runs
  if (p.q8) {
    // R8/Q1 INT8 cache (layout Q1-L1): one page = one tile, int8 data then fp32 scales,
    // everything inside the bf16 page the engine allocated (2 * c_bs bytes).
    if (p.q8 != 1) return -20;
    if (p.bs_tiles != 1) return -21;                                    // page must be 16
    if (p.q8_sc < p.hkv * N1_TILE * N1_D) return -22;                   // data / scale overlap
    if (p.q8_sc & 15) return -23;                                       // 16 B scale vectors
    if (2 * p.c_bs < (long long)p.q8_sc + 4 * p.hkv * N1_TILE) return -24;   // scales fit
    if (N1_D != 128) return -25;
  }
  // SCHED 0 needs one slot per (row, kv head); SCHED 1 needs 2 per CTA.  One workspace
  // sized for both serves every captured batch size, so the launcher asks for both.
  if (p.max_items < p.rows * p.hkv + 2 * grid) return -8;
  if (p.target_items < 1 || p.min_ct < 1) return -9;
  return 0;
}

static inline int n1_dispatch(int cfg, const N1Params& p, int grid, n1_stream_t s) {
  // the cfg's dequantisation flavour (1 bf16 magic, 2 fp16 magic, 3 + int8 MMA) must agree
  // with the CACHE FORMAT the caller passes (p.q8: 0 bf16, 1 int8), not with its own id
  if (cfg >= 0 && cfg < N1_NUM_CFGS && (n1_cfgs[cfg].q8 != 0) != (p.q8 != 0)) return -11;
  switch (cfg) {
#define N1_X(id, W_, PF_, LDK_, MODE_, KL_, SC_, Q8_) \
  case id:                                            \
    return n1_launch_t<W_, PF_, LDK_, MODE_, KL_, SC_, Q8_>(p, grid, s);
    N1_CFG_LIST(N1_X)
#undef N1_X
    default:
      return -10;
  }
}
