# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Unit tests for the kunlunxin penalties_compat torch replacement.

These tests run on CPU with small tensors and verify that the torch
implementations produce the same results as a reference written in plain Python
(no Triton).  They do not require a GPU or the Kunlunxin runtime.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_bincount(expanded_idx_mapping, all_token_ids, prompt_len, prefill_len,
                   prompt_bin_mask, output_bin_counts):
    """Pure-python reference for _bincount_kernel."""
    idx = expanded_idx_mapping.tolist()
    prompt_bin_mask.index_fill_(0, expanded_idx_mapping.long(), 0)
    output_bin_counts.index_fill_(0, expanded_idx_mapping.long(), 0)
    for req in idx:
        p = int(prompt_len[req])
        pf = int(prefill_len[req])
        for pos in range(pf):
            tok = int(all_token_ids[req, pos])
            if pos < p:
                word = tok >> 5
                bit = 1 << (tok & 31)
                # Treat int32 overflow the same as C/Triton: mask to 32 bits
                val = int(prompt_bin_mask[req, word])
                val = (val | bit) & 0xFFFFFFFF
                if val >= 0x80000000:
                    val -= 0x100000000
                prompt_bin_mask[req, word] = val
            else:
                output_bin_counts[req, tok] += 1


def _ref_apply_penalties(logits, expanded_idx_mapping, token_ids,
                          expanded_local_pos, repetition_penalty,
                          frequency_penalty, presence_penalty,
                          prompt_bin_mask, output_bin_counts):
    """Pure-python reference for _penalties_kernel."""
    logits = logits.clone().float()
    num_tokens, vocab_size = logits.shape
    num_words = prompt_bin_mask.shape[1]
    for ti in range(num_tokens):
        req = int(expanded_idx_mapping[ti])
        rep = float(repetition_penalty[req])
        freq = float(frequency_penalty[req])
        pres = float(presence_penalty[req])
        if rep == 1.0 and freq == 0.0 and pres == 0.0:
            continue
        pos = int(expanded_local_pos[ti])
        start = ti - pos
        counts = output_bin_counts[req].clone().int()
        for prev in range(pos):
            prev_tok = int(token_ids[start + prev + 1])
            counts[prev_tok] += 1
        output_mask = counts > 0
        # Unpack prompt bitmask
        prompt_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for w in range(num_words):
            word = int(prompt_bin_mask[req, w])
            for b in range(32):
                tok_id = w * 32 + b
                if tok_id < vocab_size and (word >> b) & 1:
                    prompt_mask[tok_id] = True
        combined = prompt_mask | output_mask
        for v in range(vocab_size):
            if rep != 1.0 and combined[v]:
                if logits[ti, v] > 0:
                    logits[ti, v] /= rep
                else:
                    logits[ti, v] *= rep
            logits[ti, v] -= freq * int(counts[v])
            logits[ti, v] -= pres * float(output_mask[v])
    return logits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_state():
    """A tiny penalties state with 3 requests, 2 penalty reqs."""
    torch.manual_seed(42)
    MAX_REQS, MAX_LEN, VOCAB = 4, 20, 64
    all_token_ids = torch.randint(0, VOCAB, (MAX_REQS, MAX_LEN), dtype=torch.int32)
    prompt_len = torch.tensor([10, 8, 12, 5], dtype=torch.int32)
    prefill_len = torch.tensor([15, 12, 18, 7], dtype=torch.int32)
    idx = torch.tensor([0, 2], dtype=torch.int64)
    num_words = (VOCAB + 31) // 32
    pbm = torch.zeros(MAX_REQS, num_words, dtype=torch.int32)
    obc = torch.zeros(MAX_REQS, VOCAB, dtype=torch.int32)
    return {
        "idx": idx, "all_token_ids": all_token_ids,
        "prompt_len": prompt_len, "prefill_len": prefill_len,
        "pbm": pbm, "obc": obc,
        "VOCAB": VOCAB, "MAX_REQS": MAX_REQS,
    }


# ---------------------------------------------------------------------------
# Tests: penalties_compat module
# ---------------------------------------------------------------------------

def test_import():
    """Module must import cleanly and expose the two public functions."""
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import penalties_compat
    assert callable(penalties_compat._bincount)
    assert callable(penalties_compat._apply_penalties)
    assert callable(penalties_compat.apply_penalties_compat)


def test_bincount_matches_reference(small_state):
    """_bincount must match the pure-python reference on small tensors."""
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.penalties_compat import (
        _bincount,
    )

    s = small_state
    max_pf = int(s["prefill_len"][s["idx"]].max())

    pbm_ref = s["pbm"].clone()
    obc_ref = s["obc"].clone()
    _ref_bincount(s["idx"], s["all_token_ids"], s["prompt_len"], s["prefill_len"],
                  pbm_ref, obc_ref)

    pbm_got = s["pbm"].clone()
    obc_got = s["obc"].clone()
    _bincount(s["idx"], s["all_token_ids"], s["prompt_len"], s["prefill_len"],
              pbm_got, obc_got, max_pf)

    assert torch.equal(pbm_got, pbm_ref), "prompt_bin_mask mismatch"
    assert torch.equal(obc_got, obc_ref), "output_bin_counts mismatch"


def test_bincount_empty_idx():
    """_bincount with an empty idx mapping should not raise or modify tensors."""
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.penalties_compat import (
        _bincount,
    )
    pbm = torch.zeros(4, 2, dtype=torch.int32)
    obc = torch.zeros(4, 64, dtype=torch.int32)
    idx = torch.zeros(0, dtype=torch.int64)
    all_ids = torch.zeros(4, 10, dtype=torch.int32)
    plen = torch.zeros(4, dtype=torch.int32)
    pflen = torch.zeros(4, dtype=torch.int32)
    _bincount(idx, all_ids, plen, pflen, pbm, obc, 0)
    assert pbm.sum() == 0 and obc.sum() == 0


def test_apply_penalties_no_penalty():
    """_apply_penalties must leave logits unchanged when no penalties are active."""
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.penalties_compat import (
        _apply_penalties,
    )
    NT, VOCAB, MAXR = 4, 32, 8
    logits = torch.randn(NT, VOCAB)
    orig = logits.clone()
    eim = torch.zeros(NT, dtype=torch.int64)
    tok = torch.randint(0, VOCAB, (NT,), dtype=torch.int32)
    pos = torch.zeros(NT, dtype=torch.int32)
    rep = torch.ones(MAXR)
    freq = torch.zeros(MAXR)
    pres = torch.zeros(MAXR)
    pbm = torch.zeros(MAXR, (VOCAB + 31) // 32, dtype=torch.int32)
    obc = torch.zeros(MAXR, VOCAB, dtype=torch.int32)
    _apply_penalties(logits, eim, tok, pos, rep, freq, pres, pbm, obc)
    assert torch.allclose(logits, orig), "logits must not change when no penalty is active"


def test_apply_penalties_matches_reference():
    """_apply_penalties must match the pure-python reference."""
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.penalties_compat import (
        _apply_penalties,
    )
    torch.manual_seed(7)
    NT, VOCAB, MAXR = 6, 64, 8
    logits = torch.randn(NT, VOCAB, dtype=torch.float32)
    eim = torch.randint(0, MAXR, (NT,), dtype=torch.int64)
    tok = torch.randint(0, VOCAB, (NT,), dtype=torch.int32)
    pos = torch.randint(0, 3, (NT,), dtype=torch.int32)
    rep = torch.rand(MAXR) * 0.5 + 0.5   # 0.5..1.0 — some ≠ 1.0
    freq = torch.rand(MAXR) * 0.1
    pres = torch.rand(MAXR) * 0.1
    pbm = torch.randint(0, 2**31, (MAXR, (VOCAB + 31) // 32), dtype=torch.int32)
    obc = torch.randint(0, 5, (MAXR, VOCAB), dtype=torch.int32)

    logits_got = logits.clone()
    _apply_penalties(logits_got, eim, tok, pos, rep, freq, pres, pbm, obc)
    logits_ref = _ref_apply_penalties(logits.clone(), eim, tok, pos, rep, freq, pres, pbm, obc)

    assert torch.allclose(logits_got, logits_ref, atol=1e-4, rtol=1e-4), \
        f"max diff: {(logits_got - logits_ref).abs().max()}"


def test_apply_penalties_compat_patches_module():
    """apply_penalties_compat must replace bincount and apply_penalties in-module."""
    import sys
    # Remove any cached import so we start clean
    for key in list(sys.modules):
        if "vllm.v1.worker.gpu.sample.penalties" in key:
            del sys.modules[key]

    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.penalties_compat import (
        _PATCH_FLAG,
        _apply_penalties,
        _bincount,
        apply_penalties_compat,
    )
    # Reset the flag in case a previous test already patched
    try:
        from vllm.v1.worker.gpu.sample import penalties
        setattr(penalties, _PATCH_FLAG, False)
    except ImportError:
        pytest.skip("vllm.v1.worker.gpu.sample.penalties not installed")

    apply_penalties_compat()
    from vllm.v1.worker.gpu.sample import penalties
    assert penalties.bincount is _bincount
    assert penalties.apply_penalties is _apply_penalties
    assert getattr(penalties, _PATCH_FLAG, False)


def test_apply_penalties_compat_idempotent():
    """Calling apply_penalties_compat twice must not raise or double-patch."""
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.penalties_compat import (
        _bincount,
        apply_penalties_compat,
    )
    try:
        from vllm.v1.worker.gpu.sample import penalties
    except ImportError:
        pytest.skip("vllm.v1.worker.gpu.sample.penalties not installed")
    apply_penalties_compat()
    apply_penalties_compat()
    assert penalties.bincount is _bincount
