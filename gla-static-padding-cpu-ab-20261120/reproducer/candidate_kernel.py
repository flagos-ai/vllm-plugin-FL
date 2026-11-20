import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o_tle(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    B_VCHUNK: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_vg, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_v_base = i_vg * B_VCHUNK
    N_VCHUNKS: tl.constexpr = min(B_VCHUNK, triton.cdiv(V, BV))
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    n_k_tiles = tl.cdiv(K, BK)

    if B_VCHUNK <= 1:
        i_v = i_v_base
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(n_k_tiles):
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_g = tl.make_block_ptr(
                g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(
                    h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
                )
            else:
                p_h = tl.make_block_ptr(
                    h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
                )

            b_q = tle.load(p_q, boundary_check=(0, 1), is_async=True)
            b_g = tle.load(p_g, boundary_check=(0, 1), is_async=True).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            b_h = tle.load(p_h, boundary_check=(0, 1), is_async=True)

            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

        b_o *= scale
        p_v = tl.make_block_ptr(
            v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_o = tl.make_block_ptr(
            o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_A = tl.make_block_ptr(
            A, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )

        b_v = tle.load(p_v, boundary_check=(0, 1), is_async=True)
        b_A = tle.load(p_A, boundary_check=(0, 1), is_async=True)
        b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
        b_o += tl.dot(b_A, b_v)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    else:
        b_o_0 = tl.zeros([BT, BV], dtype=tl.float32)
        b_o_1 = tl.zeros([BT, BV], dtype=tl.float32)
        b_o_2 = tl.zeros([BT, BV], dtype=tl.float32)
        b_o_3 = tl.zeros([BT, BV], dtype=tl.float32)

        for i_k in range(n_k_tiles):
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_g = tl.make_block_ptr(
                g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )

            b_q = tle.load(p_q, boundary_check=(0, 1), is_async=True)
            b_g = tle.load(p_g, boundary_check=(0, 1), is_async=True).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)

            for d in tl.static_range(N_VCHUNKS):
                i_v = i_v_base + d

                if STATE_V_FIRST:
                    p_h = tl.make_block_ptr(
                        h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
                    )
                else:
                    p_h = tl.make_block_ptr(
                        h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
                    )

                b_h = tle.load(p_h, boundary_check=(0, 1), is_async=True)

                if STATE_V_FIRST:
                    dot_res = tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
                else:
                    dot_res = tl.dot(b_qg, b_h.to(b_qg.dtype))

                if d == 0:
                    b_o_0 += dot_res
                elif d == 1:
                    b_o_1 += dot_res
                elif d == 2:
                    b_o_2 += dot_res
                elif d == 3:
                    b_o_3 += dot_res

        p_A = tl.make_block_ptr(
            A, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        b_A = tle.load(p_A, boundary_check=(0, 1), is_async=True)
        b_A = tl.where(m_s, b_A, 0.0)

        for d in tl.static_range(N_VCHUNKS):
            i_v = i_v_base + d

            if d == 0:
                b_o_final = b_o_0
            elif d == 1:
                b_o_final = b_o_1
            elif d == 2:
                b_o_final = b_o_2
            elif d == 3:
                b_o_final = b_o_3
            else:
                b_o_final = b_o_0

            b_o_final = b_o_final * scale

            p_v = tl.make_block_ptr(
                v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            b_v = tle.load(p_v, boundary_check=(0, 1), is_async=True)
            b_o_final += tl.dot(b_A.to(b_v.dtype), b_v)

            p_o = tl.make_block_ptr(
                o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_o, b_o_final.to(p_o.dtype.element_ty), boundary_check=(0, 1))
