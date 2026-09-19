"""Real FlagGems registration and CUDA graph routing, in a fresh process."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="Native CUDA/FlagGems MM integration requires NVIDIA CUDA",
    ),
]


def test_flaggems_mm_identity_and_cuda_graph():
    code = textwrap.dedent("""
        import torch
        import flag_gems
        from vllm_fl.patches import flaggems_mm_shape_aware as m
        import functools
        calls = []
        def enable(lib):
            original_impl = lib.impl
            def observed_impl(name, fn, key, **kwargs):
                if name == 'mm':
                    original_fn = fn
                    @functools.wraps(original_fn)
                    def counted(a, b):
                        calls.append(a.shape[0])
                        return original_fn(a, b)
                    fn = counted
                return original_impl(name, fn, key, **kwargs)
            lib.impl = observed_impl
            flag_gems.only_enable(lib=lib, include=['mm'])
        status = m.configure_flaggems_mm(enable, whitelist=['mm'])
        assert status.status == 'installed'
        assert 'flag_gems/' in status.flaggems
        assert 'RegisterCUDA' in status.native
        state = m._STATE
        # Numerical output is checked against the captured native handle.
        keys = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
        for rows in [2, 3, 64]:
            a = torch.randn(rows, 128, device='cuda', dtype=torch.float16)
            b = torch.randn(128, 64, device='cuda', dtype=torch.float16)
            expected = state.native_mm.call_boxed(keys, a, b)
            calls.clear()
            for _ in range(3): actual = torch.mm(a,b)
            assert calls == ([] if rows <= 2 else [rows] * 3), calls
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): actual = torch.mm(a,b)
            a.add_(1)
            graph.replay()
            expected = state.native_mm.call_boxed(keys, a,b)
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)
        assert m.configure_flaggems_mm(lambda lib: None, whitelist=['mm']).status == 'already_active'
        print(status)
    """)
    env = os.environ.copy()
    env["VLLM_FL_FLAGOS_MM_SHAPE_AWARE"] = "1"
    env["VLLM_FL_FLAGOS_MM_DECODE_MAX_M"] = "2"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
