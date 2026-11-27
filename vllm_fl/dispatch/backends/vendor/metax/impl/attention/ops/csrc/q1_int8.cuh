// SPDX-License-Identifier: Apache-2.0
// 2026 - FlagOS S2 contest work, track R8/Q1: self-developed INT8 KV cache for the
// MetaX C500.  Storage layout and the register dequantisation used by the paged decode
// attention kernel (n1_dec_attn.cuh, template parameter Q8) and mirrored by
// ops/mx_kv_int8.py (write kernel, prefill gather, torch references).
//
// LAYOUT "Q1-L1" (one vLLM page = 16 tokens = exactly one MMA tile)
// ----------------------------------------------------------------
// The engine keeps allocating the bf16 cache (2, NB, 16, HKV, 128), 8192 B per half page
// (the K or the V part of one block).  INT8 mode reinterprets each half page as
//
//   [0, HKV*2048)            int8 data, one 2048 B block per kv head
//   [sc, sc + 4*HKV*16)      fp32 scales, scale[h][t] at sc + 4*(h*16 + t)
//                            (sc = HKV*2048 for the in-place layout; 4096 for HKV = 2)
//   rest                     unused (the page keeps its bf16 size, so the KV capacity
//                            vLLM computed is unchanged -- see DESIGN.md "capacity")
//
// Inside a head's 2048 B block (t = token in the page, d = head dim):
//   V (token-major)     off_v(t, d) = t*128 + d
//   K (dimgroup-major)  off_k(t, d) = (d >> 3)*128 + t*8 + (d & 7)
//
// Why K is swizzled: lane (r, g) of a warp needs, as the MMA x fragment, dims
// 32u + 8g .. +7 of token r (u = 0..3).  In the K layout above those 8 bytes are
// contiguous at (4u + g)*128 + r*8, and over the 64 lanes of a warp one instruction
// covers the contiguous 512 B [(4u)*128, (4u)*128 + 512).  So the x fragments are loaded
// straight from memory with perfect coalescing and the identity-MMA transpose the bf16
// path needs (KL = 1, 8 extra MMAs per tile = +50%) disappears: 16 MMAs per tile instead
// of 24, which matters because an int8 tile is half the bytes and therefore twice as
// compute-dense per byte.  V keeps the token-major layout, whose read pattern (4 x 128 B
// segments per instruction) is the one R6/N1 measured at the 1.47-1.55 TB/s floor.
//
// Dequantisation, per (token, kv head) scale:
//   K: the scale multiplies the SCORE, so it is folded after the QK^T MMA
//      (S[head][tok] = scale_softmax * k_scale[tok] * (q . k_int)) -- exact, no per
//      element work, and the lane that owns S[head r][token 4g+i] also owns k_scale of
//      tokens 4g+i (the output fragment and the scale vector have the same token set).
//   V: the scale multiplies a contracted dimension, so it is folded into P
//      (P'[head][tok] = P[head][tok] * v_scale[tok]) before the rounding to bf16.  The
//      softmax denominator keeps summing the UNSCALED rounded P, so the numerator and
//      the denominator differ by one extra bf16 rounding (2^-9 relative, random per
//      token) -- see DESIGN.md "accuracy".
//   The int8 values themselves must still reach the MMA as bf16: q1_cvt4() turns 4 bytes
//   into 4 bf16 with the fp32 magic-number trick.  Integers of magnitude <= 255 are exact
//   in bf16 and their fp32 image has a zero low half-word, so the pack is a shift / mask
//   with no rounding step.
#pragma once

// 4 int8 bytes packed in a u32 -> value of the byte at shift sh, as an exact float
//   u8 = (byte + 128) in [0, 255];  bitcast(0x4B000000 | u8) = 8388608 + u8
N1_DEV float q1_i8f(uint32_t w, int sh) {
  return n1_u2f(0x4B000000u | ((((uint32_t)(w >> sh)) & 0xffu) ^ 0x80u)) - 8388736.f;
}

// {bf16(a) in the low half, bf16(b) in the high half}: exact for integers |x| <= 255
// (their fp32 bits have a zero low half-word, so no rounding is needed)
N1_DEV uint32_t q1_pack_bf(float a, float b) {
  return (n1_f2u(a) >> 16) | (n1_f2u(b) & 0xffff0000u);
}

// 4 int8 bytes -> 4 bf16 (order b0, b1 | b2, b3), the x-fragment half the MMA wants
N1_DEV n1u2 q1_cvt4(uint32_t w) {
  n1u2 o;
  o.x = q1_pack_bf(q1_i8f(w, 0), q1_i8f(w, 8));
  o.y = q1_pack_bf(q1_i8f(w, 16), q1_i8f(w, 24));
  return o;
}

// byte c (0..7) of the int8 groups a and b -> {bf16(a[c]), bf16(b[c])}: the y fragment
// element of the P . V MMA (the int8 counterpart of n1_pick2)
N1_DEV uint32_t q1_pick2_i8(n1u2 a, n1u2 b, int c) {
  const int wd = c >> 2, sh = 8 * (c & 3);
  return q1_pack_bf(q1_i8f(a[wd], sh), q1_i8f(b[wd], sh));
}

// 8-byte global load (one int8 dim-group of one token, or 4 tokens' worth of one
// dim-group); LDK 1 = non-temporal, like n1_ld
template <int LDK>
N1_DEV n1u2 q1_ld64(const void* p) {
#if !defined(N1_EMU) && !N1_SAFE
  if (LDK == 1) return __builtin_nontemporal_load((const n1u2*)p);
#endif
  return *(const n1u2*)p;
}

// one 16-token int8 tile of one kv head in registers (half the registers of N1Tile)
struct Q1Tile {
  n1u2 k[4];   // K dims 32u + 8g .. +7 of token r      (u = 0..3)
  n1u2 v[4];   // V dims 8r .. 8r+7 of token 4g + j     (j = 0..3)
  n1f4 ks;     // K scales of tokens 4g + 0 .. 4g + 3
  n1f4 vs;     // V scales of the same tokens
};

// byte offsets inside a half page
N1_DEV long long q1_head_off(int kvh, int tile_dim) { return (long long)kvh * tile_dim; }

// ===========================================================================
// R9/R: int8 -> FP16 magic-number dequantisation (template parameter Q8 >= 2)
// ===========================================================================
// WHY fp16 AND NOT bf16.  bf16 carries 8 mantissa bits, so an 8-bit integer does not fit
// below the implicit 1 of ANY exponent: the magic-number image of a byte has to be built
// in fp32 (0x4B000000 | u8 = 8388608 + u8) and then brought back to a small magnitude by
// an explicit per-element subtract before it can be rounded to bf16 at all -- 4.5 ALU ops
// per byte plus 3 per pair (q1_i8f / q1_pack_bf above).  fp16 carries 10 mantissa bits, so
// a byte DOES fit:
//
//   bitcast_f16(0x6400 | u8) = 2^10 * (1 + u8 / 1024) = 1024 + u8      exact, u8 <= 255
//
// (0x6400 = exponent 25, mantissa 0 = 1024; the mantissa field is 10 bits and u8 occupies
// its low 8, so the step is exactly 1).  With u8 = byte ^ 0x80 = v + 128 (v the signed
// int8) the image is v + 1152 -- an AFFINE function of v with ONE constant offset, which
// means the offset folds out of both dot products ANALYTICALLY instead of being paid per
// element (see "BIAS" below).  Building the two fp16 of one 16-bit pair is then a single
// byte permute against the constant 0x64646464, and the ^0x80 of four bytes is one XOR:
//
//   wx = w ^ 0x80808080                     1 op per 4 bytes
//   {f16(v0), f16(v1)} = perm(wx, 0x64..., 0x5140)   1 op
//   {f16(v2), f16(v3)} = perm(wx, 0x64..., 0x5342)   1 op
//
// Measured on the real compiler (mxcc -O3, device LLVM IR of
// work/R9/R-q1-dequant/offline/r_probe*.cu): the QK convert of one tile drops from 161 to
// 25 ALU ops per lane and the PV convert from 160 to 56 (40 with the bias folded on the V
// side as well), with 0 bytes of stack frame in every case.
//
// BIAS.  The MMA sees f16(v) = v + Q1_HBIAS, so each dot product carries a known excess:
//   QK:  sum_d (v_d + B) q_d = sum_d v_d q_d + B * (sum_d q_d).  sum_d q_d is a per (head,
//        lane) constant, computed ONCE per item in the prologue and planted in the MMA's
//        accumulator INIT (c operand), so the correction costs ZERO ops per tile and the
//        accumulator never carries the bias at a larger magnitude than it would anyway.
//   PV:  sum_j w_j (v_j + B) = sum_j w_j v_j + B * (sum_j w_j).  Here the weights w are
//        all POSITIVE, so the bias term grows like sum_j w_j while the signal grows like
//        sqrt(sum_j w_j^2): at 1000 effective tokens the bias is ~730x the signal, and the
//        fp32 accumulation noise of a 1152 * l accumulator is then ~1e-4 of the result
//        (offline/test_r_dequant.py I.8) -- below int8's own ~4e-3, and the CPU emulator
//        measures no change at all.  The default still subtracts the bias explicitly (16
//        v_pk_add_f16 per tile, exact because v + B and v are both representable) because
//        that needs no error argument and no assumption about the hardware MMA's internal
//        accumulation order; N1_Q1_PV_FOLD = 1 folds it analytically, 16 ops per lane per
//        tile cheaper.  R9/G MEASURED BOTH on the C500 and the fold SHIPS (it is now the
//        default below): cfg 42 fold against cfg 42 plain, us/layer over 42 layers,
//        64 decode rows -- 4k 156.8 vs 162.6 (x1.555 / x1.500 of the vendor), 16k 440.3
//        vs 460.5 (x1.782 / x1.704) -- with err_contract against an fp32 reference
//        IDENTICAL to the plain variant's 2.49e-3 at every measured shape and l2 vs the
//        vendor 1.3128e-2 against 1.3129e-2, i.e. the predicted ~1e-4 is invisible under
//        int8's own quantisation error.  Build with -DN1_Q1_PV_FOLD=0 to get the plain
//        variant back (it is what the whole-step table of that run used, so the shipped
//        configuration is FASTER than the projection that was made from it).
//
// RANGE.  fp16 spans 2^-14 (normal) .. 65504, far less than bf16, so both MMA operands are
// normalised by an EXACT power of two that is undone for free:
//   Q: divided by 2^Eq with 2^Eq = the binade above the head's own max |q| (so |q| <= 1),
//      and 2^Eq is multiplied back into the score scale.  Per item, not per tile.
//   P: the V scale is ~2^-7, so P * v_scale alone would sit in fp16's subnormal range for
//      every token whose weight is below ~1% of the peak.  The kernel instead evaluates
//      exp2(x - m + Lv) with Lv = Q1_HTARGET - exponent(v_scale of the row's FIRST token),
//      which puts the peak weight near 2^5.  Lv is a CONSTANT FACTOR on both the numerator
//      and the softmax denominator of the same item, so it cancels in the final divide and
//      never has to be undone -- but exactly because it cancels it must be the same number
//      in every warp and every CTA of one (row, kv head), which is why it is derived from
//      the row's first token and not from the warp's first tile.  Cost: one broadcast
//      4-byte load per item and ONE fp32 add per tile (m_new - Lv).
#ifndef N1_Q1_PV_FOLD
#define N1_Q1_PV_FOLD 1   // 1: fold the PV magic bias analytically instead of subtracting
#endif                    //    (R9/G: the measured winner, see "V" above; 0 = plain)
#ifndef N1_Q1_QPLANES
#define N1_Q1_QPLANES 1   // Q8 == 3 only: int8 planes per Q element (2 = exact 16-bit Q)
#endif

#ifndef N1_Q1_SOFT_F16
#define N1_Q1_SOFT_F16 0   // 1: IEEE fp16 in integer arithmetic (no _Float16 libcalls)
#endif

#define Q1_HMAGIC4 0x64646464u   // the fp16 high byte (0x64 = exponent of 1024) x 4
#define Q1_HBIAS 1152.f          // 1024 + 128: the constant offset of the magic image
#define Q1_HBIAS2 0x64806480u    // {f16(1152), f16(1152)}: 0x6480 = 1024 * 1.125
#define Q1_HTARGET 5             // log2 of the fp16 magnitude the peak softmax weight gets

typedef _Float16 n1h2 __attribute__((ext_vector_type(2)));
typedef _Float16 n1h4 __attribute__((ext_vector_type(4)));

// ^0x80 of four packed bytes at once: byte -> u8 = v + 128
N1_DEV uint32_t q1_x80(uint32_t w) { return w ^ 0x80808080u; }

// the two fp16 of bytes 0,1 (lo) and 2,3 (hi) of a PRE-XORed u32, value = v + Q1_HBIAS
#if N1_SAFE
N1_DEV uint32_t q1_h2lo(uint32_t wx) {
  return (wx & 0x000000ffu) | ((wx & 0x0000ff00u) << 8) | 0x64006400u;
}
N1_DEV uint32_t q1_h2hi(uint32_t wx) {
  return ((wx >> 16) & 0x000000ffu) | ((wx >> 8) & 0x00ff0000u) | 0x64006400u;
}
N1_DEV uint32_t q1_pick2_h(uint32_t a, uint32_t b, int hi) {
  return hi ? ((a >> 16) | (b & 0xffff0000u)) : ((a & 0xffffu) | (b << 16));
}
#else
N1_DEV uint32_t q1_h2lo(uint32_t wx) { return __byte_perm(wx, Q1_HMAGIC4, 0x5140u); }
N1_DEV uint32_t q1_h2hi(uint32_t wx) { return __byte_perm(wx, Q1_HMAGIC4, 0x5342u); }
// {a[c], b[c]} of two converted 16-bit element pairs (hi = 0: element 0, 1: element 1)
N1_DEV uint32_t q1_pick2_h(uint32_t a, uint32_t b, int hi) {
  return __byte_perm(a, b, hi ? 0x7632u : 0x5410u);
}
#endif


// MMA operand (4 fp16) from two packed pairs; fp32 <-> packed fp16
// SWIZZLE TRAP: __builtin_bit_cast of an ext-vector SWIZZLE (v.zw) silently casts the
// vector's FIRST lanes instead of the swizzled ones -- verified on clang 19 (mxcc) and
// clang 22 at -O0 and -O3, and it cost this track a full debug cycle.  Every fp16 operand
// is therefore built from NAMED words through this helper, never by casting a swizzle.
N1_DEV n1h4 q1_h4(uint32_t lo, uint32_t hi) {
  const n1u2 t = {lo, hi};
  return __builtin_bit_cast(n1h4, t);
}
#if defined(N1_EMU) || N1_Q1_SOFT_F16
// The CPU emulator must NOT use the host's _Float16: clang lowers it to the
// __truncsfhf2 / __extendhfsf2 libcalls, whose return ABI differs between compiler-rt
// (_Float16 in xmm0) and libgcc (the bits in eax), so an emulator .so that resolves them
// against libgcc silently computes garbage.  These are IEEE binary16 conversions in
// integer arithmetic instead -- bit-for-bit what the device's v_cvt instructions do, and
// checked against numpy.float16 by offline/test_r_dequant.py part I.
N1_DEV uint32_t q1_f2h(float f) {
  const uint32_t u = n1_f2u(f), sg = (u & 0x80000000u) >> 16, a = u & 0x7fffffffu;
  if (a >= 0x7f800000u) return sg | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u);  // Inf/NaN
  if (a >= 0x477ff000u) return sg | 0x7c00u;        // >= the 65504/Inf midpoint
  if (a < 0x33000000u) return sg;                   // < 2^-25: rounds to +-0
  const int e = (int)(a >> 23) - 127;
  const uint32_t m = (a & 0x7fffffu) | 0x800000u;   // 24-bit 1.m
  const int sh = e >= -14 ? 13 : 13 + (-14 - e);
  const uint32_t ex = e >= -14 ? (uint32_t)(e + 15) : 0u;
  const uint32_t low = m & ((1u << sh) - 1u), hf = 1u << (sh - 1);
  uint32_t r = m >> sh;
  if (low > hf || (low == hf && (r & 1u))) ++r;     // round to nearest even
  return sg | ((ex << 10) + r - (ex ? 0x400u : 0u));   // r == 0x800 carries into ex
}
N1_DEV float q1_h2f(uint32_t h16) {
  const uint32_t h = h16 & 0xffffu, sg = h & 0x8000u, ex = (h >> 10) & 0x1fu, mt = h & 0x3ffu;
  float v;
  if (ex == 0u) v = (float)mt * 5.9604644775390625e-8f;                   // 2^-24
  else if (ex == 31u) v = mt ? __builtin_nanf("") : __builtin_huge_valf();
  else v = (float)(mt | 0x400u) * n1_u2f((uint32_t)((int)ex - 25 + 127) << 23);
  return sg ? -v : v;
}
N1_DEV uint32_t q1_pack_h(float a, float b) { return q1_f2h(a) | (q1_f2h(b) << 16); }
N1_DEV float q1_h_lo(uint32_t w) { return q1_h2f(w); }
N1_DEV float q1_h_hi(uint32_t w) { return q1_h2f(w >> 16); }
#else
N1_DEV uint32_t q1_pack_h(float a, float b) {
  n1h2 h;
  h.x = (_Float16)a;
  h.y = (_Float16)b;
  return __builtin_bit_cast(uint32_t, h);
}
N1_DEV float q1_h_lo(uint32_t w) { return (float)__builtin_bit_cast(n1h2, w).x; }
N1_DEV float q1_h_hi(uint32_t w) { return (float)__builtin_bit_cast(n1h2, w).y; }
#endif

// subtract Q1_HBIAS from two packed fp16 at once (exact: both operands and the result are
// representable, so the fp16 subtraction is error-free)
N1_DEV uint32_t q1_h2_unbias(uint32_t w) {
#if defined(N1_EMU) || N1_Q1_SOFT_F16
  // exact by construction (both operands and the result are representable), so a
  // decode / subtract / re-encode in software is the same value the packed instruction
  // produces -- see q1_f2h above for why the emulator may not call the host's _Float16
  return q1_pack_h(q1_h_lo(w) - Q1_HBIAS, q1_h_hi(w) - Q1_HBIAS);
#else
  return __builtin_bit_cast(uint32_t, __builtin_bit_cast(n1h2, w) -
                                          __builtin_bit_cast(n1h2, (uint32_t)Q1_HBIAS2));
#endif
}

// exact powers of two by exponent arithmetic (no transcendental, no table)
N1_DEV float q1_pow2i(int e) {
  const int b = e + 127;
  return n1_u2f((uint32_t)(b < 1 ? 1 : (b > 254 ? 254 : b)) << 23);
}
// the binade index of |x| (2^e <= |x| < 2^(e+1)), clamped; lo for x == 0 / subnormal
N1_DEV int q1_ilog2(float x, int lo, int hi) {
  const uint32_t u = n1_f2u(x) & 0x7f800000u;
  if (u == 0u) return lo;
  const int e = (int)(u >> 23) - 127;
  return e < lo ? lo : (e > hi ? hi : e);
}
