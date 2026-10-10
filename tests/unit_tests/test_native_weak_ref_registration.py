# SPDX-License-Identifier: Apache-2.0
"""Native registration must coexist with vLLM and schema-only empty builds."""

import importlib.util
import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("owner", ["vllm_native", "schema_only", "existing_cuda"])
def test_native_weak_ref_registration_preserves_owner(owner):
    if importlib.util.find_spec("vllm_fl._C") is None:
        pytest.skip("Requires the built native plugin wheel")
    script = f"""
import torch
owner = {owner!r}
calls = []
if owner == 'vllm_native':
    from vllm.platforms import current_platform
    assert torch._C._dispatch_has_kernel_for_dispatch_key('_C::weak_ref_tensor', 'CUDA')
    before = torch._C._dispatch_dump_table('_C::weak_ref_tensor')
else:
    assert not hasattr(torch.ops._C, 'weak_ref_tensor')
    library = torch.library.Library('_C', 'FRAGMENT')
    library.define('weak_ref_tensor(Tensor input) -> Tensor')
    if owner == 'existing_cuda':
        def existing(value):
            calls.append('original-owner')
            return value
        library.impl('weak_ref_tensor', existing, 'CUDA')
        before = torch._C._dispatch_dump_table('_C::weak_ref_tensor')
import vllm_fl._C
value = torch.empty((2, 3), device='cuda')
alias = torch.ops._C.weak_ref_tensor(value)
assert alias.shape == value.shape and alias.data_ptr() == value.data_ptr()
if owner != 'schema_only':
    assert torch._C._dispatch_dump_table('_C::weak_ref_tensor') == before
if owner == 'existing_cuda':
    assert calls == ['original-owner']
"""
    environment = {**os.environ, "VLLM_PLUGINS": ""}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
