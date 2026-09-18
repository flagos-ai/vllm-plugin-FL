# SPDX-License-Identifier: Apache-2.0
"""Regression cases from the validated initial M3 MetaX adaptation."""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("GEMS_VENDOR") != "metax" or not torch.cuda.is_available(),
        reason="requires the matched MetaX runtime and GEMS_VENDOR=metax",
    ),
]


@pytest.fixture(autouse=True)
def metax_runtime():
    import flag_gems

    from vllm_fl.ops.minimax_m3.integration import install
    from vllm_fl.utils import get_flag_gems_whitelist_blacklist, use_flaggems

    assert use_flaggems()
    install()
    _, blacklist = get_flag_gems_whitelist_blacklist()
    old_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(), flag_gems.use_gems(unused=blacklist or []):
            yield
    finally:
        torch.set_num_threads(old_threads)


def test_initial_baseline_cases():
    import math

    import torch

    torch.set_num_threads(2)
    torch.manual_seed(702)

    from vllm_fl.dispatch import resolve_op
    from vllm_fl.ops.minimax_m3.msa import (
        minimax_m3_index_decode,
        minimax_m3_index_score,
        minimax_m3_index_topk,
        minimax_m3_sparse_attn,
        minimax_m3_sparse_attn_decode,
    )

    for name in (
        "index_score",
        "index_topk",
        "index_decode",
        "sparse_attn",
        "sparse_decode",
    ):
        assert resolve_op("m3_" + name).__module__ == "vllm_fl.ops.minimax_m3.torch_ops"
    report = []

    # Fixed thresholds before execution: score rtol .001/atol .01;
    # BF16 attention max_abs .03125, relative-L2 .008; discrete indices exact.
    def attn_check(label, out, ref):
        a = out.cpu().float()
        b = ref.float()
        err = (a - b).abs().max().item()
        rel = ((a - b).norm() / b.norm().clamp_min(1e-8)).item()
        assert torch.isfinite(a).all()
        assert err <= 0.03125 and rel <= 0.008, (label, err, rel)
        report.append(dict(case=label, max_abs=err, relative_l2=rel))

    def i32(v):
        return torch.tensor(v, device="cuda", dtype=torch.int32)

    for prefix, qlen in ((0, 3), (126, 4), (128, 3), (2050, 3), (4096, 3), (8448, 3)):
        seq = prefix + qlen
        nb = math.ceil(seq / 128)
        perm = torch.randperm(nb)
        idxk = (torch.randn(nb, 128, 128) * 0.05).bfloat16()
        iq = (torch.randn(qlen, 1, 128) * 0.1).bfloat16()
        iq[:, :, 0] = 1.0
        # Distinct block-score margin for strict index comparison.
        for b in range(nb):
            idxk[perm[b], :, 0] = float(b) * 0.5
        kv = (torch.randn(nb, 2, 128, 1, 128) * 0.2).bfloat16()
        q = (torch.randn(qlen, 4, 128) * 0.2).bfloat16()
        bt = i32([perm.tolist()])
        cu = i32([0, qlen])
        lens = i32([seq])
        pre = i32([prefix])
        ikg = idxk.cuda()
        qg = q.cuda()
        kvg = kv.cuda()
        iqg = iq.cuda()
        scores = minimax_m3_index_score(iqg, ikg, bt, cu, lens, pre, qlen, seq, 1)
        expected = torch.full((1, qlen, 16), -1, dtype=torch.int32)
        refs = []
        for t in range(qlen):
            visible = prefix + t + 1
            valid = math.ceil(visible / 128)
            sv = []
            for b in range(valid):
                count = min(128, visible - b * 128)
                sv.append(
                    (iq[t, 0].float() @ idxk[perm[b], :count].float().T).max().item()
                )
            torch.testing.assert_close(
                scores[0, t, :valid].cpu(), torch.tensor(sv), rtol=0.001, atol=0.01
            )
            sv[-1] = 1e29  # local_blocks=1 is mandatory, within the top16 budget.
            chosen = sorted(range(valid), key=lambda b: sv[b], reverse=True)[:16]
            expected[0, t, : len(chosen)] = torch.tensor(chosen)
            tokens = []
            for b in chosen:
                tokens.extend((b, j) for j in range(min(128, visible - b * 128)))
            k = torch.stack([kv[perm[b], 0, j, 0] for b, j in tokens]).float()
            v = torch.stack([kv[perm[b], 1, j, 0] for b, j in tokens]).float()
            refs.append(
                (torch.softmax(q[t].float() @ k.T / (128**0.5), -1) @ v).bfloat16()
            )
        buf = torch.full((1, qlen + 2, 16), -777, device="cuda", dtype=torch.int32)
        top = minimax_m3_index_topk(scores, cu, pre, qlen, 16, 0, 1, out=buf)
        assert top.data_ptr() == buf.data_ptr() and (buf[:, qlen:] == -777).all()
        assert torch.equal(top.cpu(), expected), (
            "prefill indices",
            prefix,
            top.cpu(),
            expected,
        )
        out = torch.empty_like(qg)
        minimax_m3_sparse_attn(qg, kvg, top, bt, cu, lens, pre, qlen, 1, 128**-0.5, out)
        attn_check(f"prefill_prefix{prefix}", out, torch.stack(refs))
        dec = minimax_m3_index_decode(iqg[-1:], ikg, bt, lens, seq, 16, 0, 1, 1, 1, 1)
        assert torch.equal(dec.cpu(), expected[:, -1:]), (
            "decode indices",
            prefix,
            dec.cpu(),
            expected[:, -1:],
        )
        dout = torch.empty_like(qg[-1:])
        minimax_m3_sparse_attn_decode(
            qg[-1:], kvg, dec, bt, lens, 1, 128**-0.5, dout, 1
        )
        attn_check(f"decode_prefix{prefix}", dout, torch.stack(refs)[-1:])
        print("CASE_PASS", prefix, qlen, flush=True)
    torch.cuda.synchronize()
    print(
        "MSA_PASS",
        len(report),
        "attention checks; score and exact topk checks passed",
        flush=True,
    )
