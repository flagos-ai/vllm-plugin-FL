"""Compare row-parallel sampling with the existing Triton sampler."""

import pytest
import torch

from vllm_fl.ops import sampling


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_row_parallel_sampling_preserves_upstream_outputs(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_triton as upstream

    monkeypatch.setattr(upstream, "num_compute_units", lambda _: 2)
    candidate = sampling._make_wrapper(upstream, upstream.apply_top_k_top_p_triton)
    generator = torch.Generator(device="cuda").manual_seed(72931)
    for batch in (2, 7, 3):
        logits = torch.randn((batch, 257), device="cuda", generator=generator)
        k = torch.full((batch,), 37, dtype=torch.int32, device="cuda")
        p = torch.full((batch,), 0.85, device="cuda")
        expected = upstream.apply_top_k_top_p_triton(logits.clone(), k, p)
        actual = candidate(logits.clone(), k, p)
        assert torch.equal(actual, expected)
