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
    import torch

    from vllm_fl.ops.minimax_m3.layers import qknorm_rope_insert as op

    torch.manual_seed(704)
    torch.set_num_threads(2)
    t = torch.arange(4096).float()
    freq = 1 / (5000000.0 ** (torch.arange(0, 64, 2).float() / 64))
    cs = torch.cat(
        ((t[:, None] * freq).cos(), (t[:, None] * freq).sin()), -1
    ).bfloat16()

    def ref(x, w, pos):
        f = x.float()
        z = (
            (
                f
                * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + w.float())
            )
            .bfloat16()
            .float()
        )
        out = z.clone()
        c = cs[pos, :32].float()[:, None, :]
        s = cs[pos, 32:].float()[:, None, :]
        out[:, :, :32] = z[:, :, :32] * c - z[:, :, 32:64] * s
        out[:, :, 32:64] = z[:, :, 32:64] * c + z[:, :, :32] * s
        return out.bfloat16()

    results = []
    for m in (1, 7, 129):
        for nq, nk in ((4, 1), (8, 2)):
            for sparse in (False, True):
                ni = 1 if sparse else 0
                qs = nq * 128
                ks = nk * 128
                isz = ni * 128
                original = torch.randn(
                    m, qs + 2 * ks + (isz + 128 if sparse else 0)
                ).bfloat16()
                w = [(torch.randn(128) * 0.2).bfloat16() for _ in range(4)]
                pos = (torch.arange(m) * 31 % 4096).long()
                a = original.cuda()
                wg = [v.cuda() for v in w]
                cg = cs.cuda()
                pg = pos.cuda()
                qr = ref(original[:, :qs].view(m, nq, 128), w[0], pos).view(m, qs)
                kr = ref(original[:, qs : qs + ks].view(m, nk, 128), w[1], pos).view(
                    m, ks
                )
                v = original[:, qs + ks : qs + 2 * ks]
                if not sparse:
                    op(a, wg[0], wg[1], cg, pg, nq, nk, 64, 1e-6)
                    qo = a[:, :qs]
                else:
                    cap = 1024
                    slots = (torch.arange(m) * 17) % cap
                    islots = (slots * 7 + 13) % cap
                    invalid = torch.arange(m) % 11 == 0
                    slots[invalid] = -1
                    islots[invalid] = -1
                    cache = torch.full(
                        (8, 2, 128, nk, 128), -77.0, device="cuda", dtype=torch.bfloat16
                    )
                    icache = torch.full(
                        (8, 128, 128), -77.0, device="cuda", dtype=torch.bfloat16
                    )
                    qo = torch.empty(m, qs, device="cuda", dtype=torch.bfloat16)
                    iqo = torch.empty(m, 128, device="cuda", dtype=torch.bfloat16)
                    op(
                        a,
                        wg[0],
                        wg[1],
                        cg,
                        pg,
                        nq,
                        nk,
                        64,
                        1e-6,
                        wg[2],
                        wg[3],
                        ni,
                        slots.cuda(),
                        islots.cuda(),
                        cache,
                        icache,
                        128,
                        qo,
                        iqo,
                        "bfloat16",
                    )
                    iqr = ref(
                        original[:, qs + 2 * ks : qs + 2 * ks + 128].view(m, 1, 128),
                        w[2],
                        pos,
                    ).view(m, 128)
                    ikr = ref(original[:, -128:].view(m, 1, 128), w[3], pos).view(
                        m, 128
                    )
                    torch.testing.assert_close(iqo.cpu(), iqr, rtol=0.01, atol=0.01)
                    torch.testing.assert_close(
                        a[:, -128:].cpu(), ikr, rtol=0.01, atol=0.01
                    )
                    assert torch.equal(a[:, :qs].cpu(), original[:, :qs])
                    expected = torch.full(
                        cache.shape, -77.0, device="cpu", dtype=torch.bfloat16
                    )
                    iexpected = torch.full(
                        icache.shape, -77.0, device="cpu", dtype=torch.bfloat16
                    )
                    for i in range(m):
                        if slots[i] >= 0:
                            sl = int(slots[i])
                            expected[sl // 128, 0, sl % 128] = kr[i].view(nk, 128)
                            expected[sl // 128, 1, sl % 128] = v[i].view(nk, 128)
                            si = int(islots[i])
                            iexpected[si // 128, si % 128] = ikr[i]
                    torch.testing.assert_close(
                        cache.cpu(), expected, rtol=0.01, atol=0.01
                    )
                    torch.testing.assert_close(
                        icache.cpu(), iexpected, rtol=0.01, atol=0.01
                    )
                    assert torch.equal(cache[:, 1].cpu(), expected[:, 1])
                torch.testing.assert_close(qo.cpu(), qr, rtol=0.01, atol=0.01)
                torch.testing.assert_close(
                    a[:, qs : qs + ks].cpu(), kr, rtol=0.01, atol=0.01
                )
                assert torch.equal(a[:, qs + ks : qs + 2 * ks].cpu(), v)
                results.append(
                    {
                        "tokens": m,
                        "q_heads": nq,
                        "kv_heads": nk,
                        "sparse": sparse,
                        "q_max_abs": (qo.cpu().float() - qr.float()).abs().max().item(),
                    }
                )
                print("QKV_CASE_PASS", results[-1], flush=True)
    print("QKV_PASS", len(results), flush=True)
