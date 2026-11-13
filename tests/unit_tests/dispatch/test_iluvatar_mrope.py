from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.mrope import _wrap_mrope_forward


@pytest.mark.parametrize(
    "axes,sections",
    [(3, [1, 1, 1]), (4, [1] * 4), (3, [1] * 4), (4, [1] * 3), (3, [1] * 2), (3, [])],
)
def test_mrope_only_routes_three_axes_and_sections(monkeypatch, axes, sections):
    original = Mock(return_value=("native-q", "native-k"))
    original._iluvatar_mrope_patched = False
    op = Mock()
    monkeypatch.setattr(torch.ops.vllm, "mrope", op, raising=False)
    q, k = Mock(), Mock()
    q.device = SimpleNamespace(type="cuda")
    layer = SimpleNamespace(
        mrope_section=sections,
        head_size=6,
        rotary_dim=6,
        mrope_interleaved=False,
        is_neox_style=True,
        _match_cos_sin_cache_dtype=Mock(return_value="cache"),
    )
    positions = torch.zeros((axes, 2), dtype=torch.long)
    result = _wrap_mrope_forward(original)(layer, positions, q, k)
    if axes == 3 and len(sections) == 3:
        original.assert_not_called()
        op.assert_called_once()
        assert result == (q.clone.return_value, k.clone.return_value)
        assert op.call_args.args[6:9] == tuple(sections)
        q.clone.assert_called_once_with(memory_format=torch.contiguous_format)
        k.clone.assert_called_once_with(memory_format=torch.contiguous_format)
    else:
        original.assert_called_once_with(layer, positions, q, k, None)
        op.assert_not_called()
        q.clone.assert_not_called()
        k.clone.assert_not_called()
        assert result == ("native-q", "native-k")
