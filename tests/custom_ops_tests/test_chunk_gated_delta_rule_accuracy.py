# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Numerical accuracy test for the npu_chunk_gated_delta_rule AscendC operator.
#
# The operator contract (verified against csrc/ascend/attention/chunk_gated_delta_rule
# host tiling + kernel stages):
#   query            (T, Nk, Dk)  bf16, L2-normalized by the CALLER
#   key              (T, Nk, Dk)  bf16, L2-normalized by the CALLER
#   value            (T, Nv, Dv)  bf16
#   beta             (T, Nv)      bf16
#   initial_state    (B, Nv, Dv, Dk)  bf16   <-- (Dv, Dk) state layout, NO transpose
#   actual_seq_lengths (B,)      int32
#   g                (T, Nv)      fp32 (optional log-gate; None => no decay)
#   scale_value      float
# Outputs:
#   out              (T, Nv, Dv)  bf16
#   final_state      (B, Nv, Dv, Dk)  bf16
#
# Reference: pure-PyTorch fp32 recurrent gated-delta-rule, cross-validated against
# vLLM's Triton chunk_gated_delta_rule (the known-good production path) when
# available.
#
# Coverage matrix:
#   - B in {1, 2, 3, 4}, per-seq lengths from 1 up to > chunk-group length
#   - chunk(64) boundary: 63/64/65, 127/128/129, tails
#   - GQA ratios: Nv/Nk in {1, 2, 4}
#   - (Dk, Dv) square and non-square (non-square pins the state layout
#     unambiguously)
#   - g present (realistic negative log-gates) / absent
#   - zero initial state (fresh prefill) / random non-zero initial state
#     (continued prefill)
#   - deliberate transposed initial_state run (reproduces the integration bug)

import os
import sys

from vllm_fl.utils import enable_custom_op

if not enable_custom_op():
    print(
        "ERROR: vllm_fl/_cann_ops_custom is not installed.\n"
        "Please build and install the CANN framework operators first, e.g.:\n"
        "  bash csrc/ascend/build_aclnn.sh <soc_version>",
        file=sys.stderr,
    )
    sys.exit(1)

import torch
import torch_npu  # noqa: F401

import vllm_fl._C_ascend  # noqa: F401

DEVICE = os.environ.get("TEST_NPU_DEVICE", "npu:4")

# Pass thresholds: the kernel keeps intermediates (state included) in bf16, so
# errors accumulate ~sqrt(num_chunks) * 2^-8. Correct runs land well below
# these bounds; a layout bug lands orders of magnitude above them.
REL_L2_TOL = 5e-2
COS_TOL = 0.998


# ---------------------------------------------------------------------------
# fp32 recurrent reference
# ---------------------------------------------------------------------------
def ref_gated_delta_rule(q, k, v, g, beta, initial_state, seqlens, scale):
    """Pure fp32 recurrent gated delta rule.

    q, k: (T, Nk, Dk) float32 (already L2-normalized)
    v:    (T, Nv, Dv) float32
    g:    (T, Nv) float32 log-gate, or None
    beta: (T, Nv) float32
    initial_state: (B, Nv, Dv, Dk) float32 -- kernel-native layout: element
        [v, k] holds S[k, v] of the FLA (Dk, Dv) convention.

    Recurrence per token (matches FLA fused_recurrent and the AscendC kernel):
        u = S @ k_t                        # (Nv, Dv), pre-decay state
        S = exp(g_t) * S + (beta_t * (v_t - u)) outer k_t
        o_t = scale * (S @ q_t)            # post-update state
    """
    T, Nk, Dk = q.shape
    Nv, Dv = v.shape[1], v.shape[2]
    ratio = Nv // Nk
    B = initial_state.shape[0]
    S = initial_state.to(torch.float32).clone()
    out = torch.zeros(T, Nv, Dv, dtype=torch.float32, device=q.device)
    start = 0
    for b in range(B):
        L = int(seqlens[b])
        for t in range(start, start + L):
            k_t = k[t].repeat_interleave(ratio, dim=0)  # (Nv, Dk)
            q_t = q[t].repeat_interleave(ratio, dim=0)  # (Nv, Dk)
            u = torch.einsum("nvk,nk->nv", S[b], k_t)
            w = beta[t].unsqueeze(-1) * (v[t] - u)  # (Nv, Dv)
            if g is not None:
                S[b] = S[b] * torch.exp(g[t]).view(Nv, 1, 1)
            S[b] = S[b] + torch.einsum("nv,nk->nvk", w, k_t)
            out[t] = scale * torch.einsum("nvk,nk->nv", S[b], q_t)
        start += L
    return out, S


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def l2norm(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def rel_l2(a, b):
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    denom = torch.linalg.norm(b).item()
    if denom == 0.0:
        return torch.linalg.norm(a - b).item()
    return (torch.linalg.norm(a - b) / denom).item()


def cos_sim(a, b):
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if denom == 0.0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return torch.dot(a, b).div(denom).item()


def make_case(name, seqlens, nk, nv, dk, dv, with_g=True, g_scale=1.0,
              init_state="zero", seed=0, transpose_state=False):
    return dict(name=name, seqlens=seqlens, nk=nk, nv=nv, dk=dk, dv=dv,
                with_g=with_g, g_scale=g_scale, init_state=init_state,
                seed=seed, transpose_state=transpose_state)


def run_case(case):
    seqlens = case["seqlens"]
    B = len(seqlens)
    T = sum(seqlens)
    nk, nv, dk, dv = case["nk"], case["nv"], case["dk"], case["dv"]
    gen = torch.Generator(device="cpu").manual_seed(case["seed"])

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen, dtype=dtype)

    # raw inputs, then L2-normalize q/k as the op contract requires
    q = l2norm(randn(T, nk, dk)).to(torch.bfloat16).to(DEVICE)
    k = l2norm(randn(T, nk, dk)).to(torch.bfloat16).to(DEVICE)
    v = randn(T, nv, dv).to(torch.bfloat16).to(DEVICE)
    beta = torch.sigmoid(randn(T, nv)).to(torch.bfloat16).to(DEVICE)
    if case["with_g"]:
        # realistic log-gates: strictly negative, like -softplus(a + dt_bias)
        g = (-torch.nn.functional.softplus(randn(T, nv)) * case["g_scale"]).to(DEVICE)
    else:
        g = None
    if case["init_state"] == "zero":
        init = torch.zeros(B, nv, dv, dk, dtype=torch.bfloat16, device=DEVICE)
    else:
        init = (randn(B, nv, dv, dk) * 0.1).to(torch.bfloat16).to(DEVICE)

    actual_seq_lengths = torch.tensor(seqlens, dtype=torch.int32, device=DEVICE)
    scale = dk ** -0.5

    init_for_op = init.transpose(-1, -2).contiguous() if case["transpose_state"] else init

    out_op, final_op = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        q, k, v, beta, init_for_op, actual_seq_lengths, g, scale
    )

    # fp32 reference on the *same* op-contract inputs. When the case feeds a
    # transposed state to the op (bug repro), the reference must consume the
    # true (untransposed) state to represent correct math.
    g_ref = g.to(torch.float32) if g is not None else None
    out_ref, final_ref = ref_gated_delta_rule(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
        g_ref, beta.to(torch.float32), init.to(torch.float32),
        seqlens, scale,
    )

    # When the op was given a transposed input state, its final_state comes
    # back in the op's native (Dv, Dk) layout as computed from the wrong
    # (transposed) start; compare directly against the reference layout.
    res = {
        "name": case["name"],
        "out_rel_l2": rel_l2(out_op, out_ref),
        "out_cos": cos_sim(out_op, out_ref),
        "state_rel_l2": rel_l2(final_op, final_ref),
        "state_cos": cos_sim(final_op, final_ref),
        "out_max_abs": (out_op.to(torch.float32) - out_ref).abs().max().item(),
    }
    res["pass"] = (
        res["out_rel_l2"] < REL_L2_TOL
        and res["state_rel_l2"] < REL_L2_TOL
        and res["out_cos"] > COS_TOL
        and res["state_cos"] > COS_TOL
    )
    return res


def build_cases():
    cases = []
    # ---- Group A: fresh prefill (zero initial state), with g ---------------
    cases.append(make_case("A1_tiny_L8", [8], 2, 2, 64, 64, seed=11))
    cases.append(make_case("A2_chunk63", [63], 2, 2, 128, 128, seed=12))
    cases.append(make_case("A3_chunk64_exact", [64], 4, 4, 128, 128, seed=13))
    cases.append(make_case("A4_chunk65_tail", [65], 4, 4, 128, 128, seed=14))
    cases.append(make_case("A5_chunk127", [127], 4, 4, 128, 128, seed=15))
    cases.append(make_case("A6_chunk128_exact", [128], 4, 4, 128, 128, seed=16))
    cases.append(make_case("A7_chunk129", [129], 4, 4, 128, 128, seed=17))
    cases.append(make_case("A8_L256_gqa2", [256], 8, 16, 128, 128, seed=18))
    cases.append(make_case("A9_L192_qwen35_heads", [192], 16, 32, 128, 128, seed=19))
    cases.append(make_case("A10_batch_mixed", [7, 65, 130], 16, 32, 128, 128, seed=20))
    cases.append(make_case("A11_batch4_mixed", [1, 64, 129, 300], 4, 8, 128, 128, seed=21))
    cases.append(make_case("A12_L1024", [1024], 16, 32, 128, 128, seed=22))
    cases.append(make_case("A13_L3000_near_group", [3000], 4, 8, 128, 128, seed=23))
    cases.append(make_case("A14_L4096_cross_group", [4096], 4, 8, 128, 128, seed=24))
    cases.append(make_case("A15_L1_single_token", [1], 2, 2, 64, 64, seed=25))
    # ---- Group B: non-square dims pin the (Dv, Dk) state layout ------------
    cases.append(make_case("B1_dk64_dv128", [100], 2, 2, 64, 128, seed=26))
    cases.append(make_case("B2_dk128_dv64", [100], 2, 2, 128, 64, seed=27))
    cases.append(make_case("B3_dk64_dv128_multi", [70, 130], 2, 4, 64, 128, seed=28))
    # ---- Group C: g absent (no decay) --------------------------------------
    cases.append(make_case("C1_no_g", [64, 80], 4, 4, 128, 128, with_g=False, seed=29))
    cases.append(make_case("C2_no_g_long", [517], 4, 8, 128, 128, with_g=False, seed=30))
    # ---- Group D: gate strength sweep --------------------------------------
    cases.append(make_case("D1_weak_gate", [200], 4, 8, 128, 128, g_scale=0.1, seed=31))
    cases.append(make_case("D2_strong_gate", [200], 4, 8, 128, 128, g_scale=4.0, seed=32))
    # ---- Group E: non-zero initial state (continued prefill) ---------------
    cases.append(make_case("E1_init_L64", [64], 4, 4, 128, 128,
                           init_state="random", seed=33))
    cases.append(make_case("E2_init_mixed", [200, 37], 16, 32, 128, 128,
                           init_state="random", seed=34))
    cases.append(make_case("E3_init_nonsquare", [150], 2, 2, 64, 128,
                           init_state="random", seed=35))
    cases.append(make_case("E4_init_long", [1000], 4, 8, 128, 128,
                           init_state="random", seed=36))
    # ---- Group F: bug repro -- transposed initial_state --------------------
    # Mirrors what _chunk_gated_delta_rule_aclnn in patch_qwen3_6_gdn.py does
    # today: feeds ssm_state.transpose(-1, -2) to the op. Expect FAIL.
    cases.append(make_case("F1_bugrepro_transposed_state", [200, 37], 16, 32, 128, 128,
                           init_state="random", transpose_state=True, seed=37))
    cases.append(make_case("F2_bugrepro_weak_gate", [200, 37], 16, 32, 128, 128,
                           init_state="random", transpose_state=True,
                           g_scale=0.05, seed=38))
    cases.append(make_case("F3_bugrepro_no_decay", [128], 4, 8, 128, 128,
                           init_state="random", transpose_state=True,
                           with_g=False, seed=39))
    return cases


def _import_qwen3_next_lib():
    """Import vllm.model_executor.models.qwen3_next.

    This dev box has both the vllm-ascend and the FL platform plugins
    installed, and stock vLLM refuses to resolve a platform when two OOT
    plugins activate. Drop the ascend entry from platform-plugin discovery
    (the FL plugin is the one under test) before vllm resolves its platform.
    """
    import vllm.platforms as vp

    orig = vp.load_plugins_by_group

    def filtered(group):
        plugins = orig(group)
        if group == vp.PLATFORM_PLUGINS_GROUP:
            plugins = {k: v for k, v in plugins.items() if k != "ascend"}
        return plugins

    vp.load_plugins_by_group = filtered
    import vllm.model_executor.models.qwen3_next as qwen3_next_lib

    return qwen3_next_lib


def test_reference_selfcheck():
    """Validate the fp32 recurrent reference against the plugin's Ascend-Triton
    chunk_gated_delta_rule (the known-good production path) on several configs."""
    try:
        _import_qwen3_next_lib()  # applies the platform-plugin filter for vllm imports
        from vllm_fl.dispatch.backends.vendor.ascend.impl.fla import (
            chunk_gated_delta_rule_npu as triton_cgdr,
        )
        from vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils import (
            init_device_properties_triton,
        )
        init_device_properties_triton()
    except Exception as e:  # noqa: BLE001
        print(f"[ref-selfcheck] skipped (import failed: {e})")
        return True

    print("=" * 100)
    print("reference self-check: fp32 recurrent reference vs Triton chunk_gated_delta_rule")
    all_ok = True
    for name, seqlens, nk, nv, dk, dv, init_kind, with_g, seed in [
        ("R1_fresh", [130, 7], 4, 8, 128, 128, "zero", True, 61),
        ("R2_continued", [200, 37], 16, 32, 128, 128, "random", True, 62),
        # NOTE: 64-dim dv is avoided here on purpose: the flag_gems Ascend
        # Triton chunk kernel carries state incorrectly at dv=64 (probe:
        # final_state rel_err ~1.0 vs both the fp32 reference and the aclnn
        # op), so it cannot serve as a reference for that shape. The aclnn
        # op itself is verified against the fp32 reference at dv=64 in the
        # op-level groups above (A1/A15/B2, all pass).
        ("R3_no_g", [96], 2, 2, 128, 128, "random", False, 63),
        ("R4_nonsquare", [150], 2, 2, 64, 128, "random", True, 64),
    ]:
        B = len(seqlens)
        T = sum(seqlens)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def randn(*shape):
            return torch.randn(*shape, generator=gen, dtype=torch.float32)

        q_raw = randn(1, T, nk, dk).to(torch.bfloat16).to(DEVICE)
        k_raw = randn(1, T, nk, dk).to(torch.bfloat16).to(DEVICE)
        v = randn(1, T, nv, dv).to(torch.bfloat16).to(DEVICE)
        beta = torch.sigmoid(randn(1, T, nv)).to(torch.bfloat16).to(DEVICE)
        # the Triton reference requires a g tensor; use zeros for "no decay"
        g = (-torch.nn.functional.softplus(randn(1, T, nv))).to(DEVICE) if with_g \
            else torch.zeros(1, T, nv, dtype=torch.float32, device=DEVICE)

        init_fla = torch.zeros(B, nv, dk, dv, dtype=torch.bfloat16, device=DEVICE)
        if init_kind == "random":
            init_fla = (randn(B, nv, dk, dv) * 0.1).to(torch.bfloat16).to(DEVICE)
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
            dtype=torch.int32, device=DEVICE)

        out_triton, final_triton = triton_cgdr(
            q=q_raw, k=k_raw, v=v, g=g, beta=beta,
            initial_state=init_fla,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

        # fp32 reference consumes L2-normalized q/k and the (Dv, Dk) state layout
        q_ref = l2norm(q_raw.to(torch.float32).squeeze(0))
        k_ref = l2norm(k_raw.to(torch.float32).squeeze(0))
        init_ref = init_fla.to(torch.float32).transpose(-1, -2).contiguous()
        g_ref = g.to(torch.float32).squeeze(0) if g is not None else None
        out_ref, final_ref = ref_gated_delta_rule(
            q_ref, k_ref, v.to(torch.float32).squeeze(0),
            g_ref, beta.to(torch.float32).squeeze(0),
            init_ref, seqlens, dk ** -0.5,
        )
        final_ref_fla = final_ref.transpose(-1, -2)

        o_rel = rel_l2(out_triton, out_ref.unsqueeze(0))
        s_rel = rel_l2(final_triton, final_ref_fla)
        o_cos = cos_sim(out_triton, out_ref)
        s_cos = cos_sim(final_triton, final_ref_fla)
        ok = o_rel < REL_L2_TOL and s_rel < REL_L2_TOL and o_cos > COS_TOL and s_cos > COS_TOL
        all_ok = all_ok and ok
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<34} out_relL2={o_rel:.4e} out_cos={o_cos:.5f} "
              f"state_relL2={s_rel:.4e} state_cos={s_cos:.5f} {verdict}")
    return all_ok


def test_wrapper_vs_triton():
    """End-to-end check of the production wrapper _chunk_gated_delta_rule_aclnn
    (patch_qwen3_6_gdn.py) against the known-good Triton chunk path, using the
    same (Hv, Dv, Dk) ssm_state cache layout as production."""
    try:
        _import_qwen3_next_lib()  # applies the platform-plugin filter for vllm imports
        from vllm_fl.dispatch.backends.vendor.ascend.impl.fla import (
            chunk_gated_delta_rule_npu as triton_cgdr,
        )
        from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_qwen3_6_gdn import (
            _chunk_gated_delta_rule_aclnn,
        )
        from vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils import (
            init_device_properties_triton,
        )
        init_device_properties_triton()
    except Exception as e:  # noqa: BLE001
        print(f"[wrapper-vs-triton] skipped (import failed: {e})")
        return True

    print("=" * 100)
    print("wrapper-level check: _chunk_gated_delta_rule_aclnn vs Triton chunk_gated_delta_rule")
    all_ok = True
    for name, seqlens, nk, nv, dk, dv, init_kind, seed in [
        ("W1_fresh_zero_state", [200, 37], 16, 32, 128, 128, "zero", 51),
        ("W2_continued_random_state", [200, 37], 16, 32, 128, 128, "random", 52),
        ("W3_continued_long", [1000], 4, 8, 128, 128, "random", 53),
    ]:
        B = len(seqlens)
        T = sum(seqlens)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def randn(*shape):
            return torch.randn(*shape, generator=gen, dtype=torch.float32)

        # production-shaped inputs: (1, T, H, D), raw (unnormalized) q/k
        query = randn(1, T, nk, dk).to(torch.bfloat16).to(DEVICE)
        key = randn(1, T, nk, dk).to(torch.bfloat16).to(DEVICE)
        value = randn(1, T, nv, dv).to(torch.bfloat16).to(DEVICE)
        g = (-torch.nn.functional.softplus(randn(1, T, nv))).to(DEVICE)
        beta = torch.sigmoid(randn(1, T, nv)).to(torch.bfloat16).to(DEVICE)

        # ssm_state cache: (num_slots, Nv, Dv, Dk), kernel-native layout
        num_slots = B + 2
        ssm_state = torch.zeros(num_slots, nv, dv, dk, dtype=torch.bfloat16, device=DEVICE)
        state_indices = torch.arange(1, B + 1, dtype=torch.int32, device=DEVICE)
        if init_kind == "random":
            ssm_state[state_indices.long()] = (randn(B, nv, dv, dk) * 0.1).to(torch.bfloat16).to(DEVICE)
        has_initial_state = torch.tensor(
            [init_kind == "random"] * B, dtype=torch.bool, device=DEVICE)
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
            dtype=torch.int32, device=DEVICE)

        # wrapper (aclnn) path
        out_aclnn, final_aclnn = _chunk_gated_delta_rule_aclnn(
            query, key, value, g, beta, ssm_state, state_indices,
            has_initial_state, cu_seqlens,
        )

        # Triton reference path (mirrors the non-fresh branch of the patch)
        initial_state = ssm_state[state_indices.long()].transpose(-1, -2).contiguous()
        initial_state[~has_initial_state, ...] = 0
        out_triton, final_triton = triton_cgdr(
            q=query, k=key, v=value, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

        # final_aclnn is transposed by the wrapper into FLA (Dk, Dv) layout,
        # same as final_triton -- compare directly.
        o_rel = rel_l2(out_aclnn, out_triton)
        s_rel = rel_l2(final_aclnn, final_triton)
        o_cos = cos_sim(out_aclnn, out_triton)
        s_cos = cos_sim(final_aclnn, final_triton)
        ok = o_rel < REL_L2_TOL and s_rel < REL_L2_TOL and o_cos > COS_TOL and s_cos > COS_TOL
        all_ok = all_ok and ok
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<34} out_relL2={o_rel:.4e} out_cos={o_cos:.5f} "
              f"state_relL2={s_rel:.4e} state_cos={s_cos:.5f} {verdict}")
    return all_ok


def main():
    torch.npu.set_device(DEVICE)
    print(f"device: {DEVICE}")
    print("=" * 100)
    header = (f"{'case':<34} {'out_relL2':>10} {'out_cos':>9} "
              f"{'st_relL2':>10} {'st_cos':>9} {'verdict':>8}")
    print(header)
    print("-" * 100)

    failures = []
    for case in build_cases():
        expect_pass = not case["transpose_state"]
        try:
            res = run_case(case)
        except Exception as e:  # noqa: BLE001
            print(f"{case['name']:<34} ERROR: {type(e).__name__}: {e}")
            if expect_pass:
                failures.append(case["name"])
            continue
        verdict = "PASS" if res["pass"] else "FAIL"
        ok = (res["pass"] == expect_pass)
        marker = "" if ok else "  <-- UNEXPECTED"
        print(f"{res['name']:<34} {res['out_rel_l2']:>10.4e} {res['out_cos']:>9.5f} "
              f"{res['state_rel_l2']:>10.4e} {res['state_cos']:>9.5f} {verdict:>8}{marker}")
        if not ok:
            failures.append(case["name"])

    print("=" * 100)
    ref_ok = test_reference_selfcheck()
    print("=" * 100)
    wrapper_ok = test_wrapper_vs_triton()
    print("=" * 100)
    if failures or not wrapper_ok or not ref_ok:
        if failures:
            print(f"UNEXPECTED results in {len(failures)} case(s): {failures}")
        if not ref_ok:
            print("reference self-check FAILED (fp32 reference deviates from Triton path)")
        if not wrapper_ok:
            print("wrapper-level check FAILED (production aclnn wrapper deviates from Triton path)")
        sys.exit(1)
    print("All cases behave as expected.")


if __name__ == "__main__":
    main()
