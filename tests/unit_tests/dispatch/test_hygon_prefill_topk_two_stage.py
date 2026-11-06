# Copyright (c) 2026 BAAI. All rights reserved.
"""Production dispatch, independent selection oracle and rollback coverage."""
import numpy as np
import pytest
import torch

from tests.unit_tests.dispatch.test_hygon_prefill_indexer import (
    _oracle, test_chunk_metadata_and_workspace_graph as check_chunk,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='HIP GPU required')


@pytest.mark.parametrize('n', [4097, 8191, 8192, 8193, 16381, 16384])
@pytest.mark.parametrize('strided', [False, True])
def test_enabled_graph_cpu_oracle(n, strided, monkeypatch):
    import flag_gems
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import prefill_topk as impl
    monkeypatch.setenv('VLLM_FL_HYGON_PREFILL_TOPK_TWO_STAGE', '1')
    # Any silent fallback would invalidate the new-path numerical check.
    monkeypatch.setattr(impl, '_convert', None)
    m = 129
    rng = np.random.default_rng(917)
    backing = torch.empty((m, 2*n+128 if strided else n), device='cuda')
    x = backing[:, :2*n:2] if strided else backing
    out_backing = torch.full((m, 1040), -777, device='cuda', dtype=torch.int32)
    out = out_backing[:, 5:1029:2]
    s = np.resize(np.array([0, 1, -3, n+1, 63, 0, 17, n-7], np.int32), m)
    e = np.resize(np.array([0, 512, n+7, n+9, 575, n, 16, n], np.int32), m)
    ss, ee = torch.from_numpy(s).cuda(), torch.from_numpy(e).cuda()
    with flag_gems.use_gems(include=['topk']):
        def run():
            impl.prefill_topk(x, ss, ee, out, optimize_wide=True)
        x.zero_();run();torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for stage in range(6):
            values = rng.normal(size=(m, n)).astype(np.float32)
            if stage == 1:values = np.round(values)
            if stage == 2:values = np.where(values < 0, -0., 0.).astype(np.float32)
            if stage == 3:values = np.float32(1) + values * np.float32(2**-21)
            if stage == 4:
                values *= np.float32(1e37)
                values[:, ::17] = np.inf
                values[:, ::19] = -np.inf
            if stage == 5:
                # Update the causal bounds on replay; all rows now use the
                # entire width, including the large-tie case at the cutoff.
                s[:] = 0;e[:] = n;values.fill(.25)
            for row in range(m):
                lo = min(n, max(0, int(s[row])))
                hi = min(n, max(lo, int(e[row])))
                values[row, :lo] = np.nan;values[row, hi:] = np.nan
            x.copy_(torch.from_numpy(values));ss.copy_(torch.from_numpy(s));ee.copy_(torch.from_numpy(e))
            out_backing.fill_(-777);graph.replay()
            _oracle(values, s, e, out.cpu().numpy())
            sentinel = out_backing.clone();sentinel[:, 5:1029:2] = -777
            assert bool((sentinel == -777).all())
            np.testing.assert_array_equal(x.cpu().numpy(), values)


@pytest.mark.parametrize('case', ['off', 'baseline', 'reference', 'small_m', 'too_wide'])
def test_disabled_and_unmeasured_routes(case, monkeypatch):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import prefill_topk as impl
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import prefill_topk_two_stage as leaf
    def forbidden(*args, **kwargs):
        pytest.fail('Baseline/reference/unsupported shape entered two-stage TopK')
    monkeypatch.setattr(leaf, 'prefill_topk_two_stage', forbidden)
    monkeypatch.setenv('VLLM_FL_HYGON_PREFILL_TOPK_TWO_STAGE', '0' if case == 'off' else '1')
    m, n = (127 if case == 'small_m' else 128), (16385 if case == 'too_wide' else 8192)
    x = torch.zeros((m, n), device='cuda')
    s = torch.zeros(m, device='cuda', dtype=torch.int32);e = torch.full_like(s, n)
    out = torch.empty((m, 512), device='cuda', dtype=torch.int32)
    impl.prefill_topk(x, s, e, out, already_masked=True,
                     optimize_wide=case != 'baseline', reference=case == 'reference')
    _oracle(x.cpu().numpy(), s.cpu().numpy(), e.cpu().numpy(), out.cpu().numpy())


@pytest.mark.parametrize('mode', ['auto', 'baseline', 'reference'])
def test_model_indexer_route(mode, monkeypatch):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import prefill_topk_two_stage as leaf
    original = leaf.prefill_topk_two_stage
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[0].shape)
        assert mode == 'auto'
        return original(*args, **kwargs)
    monkeypatch.setattr(leaf, 'prefill_topk_two_stage', counted)
    monkeypatch.setenv('VLLM_FL_HYGON_PREFILL_TOPK_TWO_STAGE', '1')
    check_chunk(monkeypatch, mode, ([129], [65531], None))
    assert bool(calls) == (mode == 'auto')


def test_dynamic_lengths_keep_bounded_compilation():
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import prefill_topk_two_stage as leaf
    x = torch.randn((192, 16512), device='cuda')
    s = torch.zeros(192, device='cuda', dtype=torch.int32);e = torch.full_like(s, 16384)
    out = torch.empty((192, 512), device='cuda', dtype=torch.int32)
    def sizes():
        return [len(k.device_caches[torch.cuda.current_device()][0])
                for k in (leaf._compact, leaf._finish)]
    leaf.prefill_topk_two_stage(x[:, :16384], s, e, out)
    torch.cuda.synchronize();initial = sizes()
    for step in range(64):
        m, n = 192-step, 16384-step
        e.fill_(n)
        leaf.prefill_topk_two_stage(x[:m, :n], s[:m], e[:m], out[:m])
    torch.cuda.synchronize()
    assert sizes() == initial
