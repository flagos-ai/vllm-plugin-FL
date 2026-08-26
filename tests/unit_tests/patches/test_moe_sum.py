from types import ModuleType

from vllm_fl.patches import moe_sum


class _FakeTensor:
    def __init__(self, numel, hidden_stride=1):
        self._numel = numel
        self._hidden_stride = hidden_stride

    def numel(self):
        return self._numel

    def stride(self, dim):
        assert dim == -1
        return self._hidden_stride


def test_moe_sum_patch_routes_nonempty_and_guards_empty(monkeypatch):
    calls = []
    ops = ModuleType("fake_vllm_custom_ops")
    ops.moe_sum = lambda input, output: calls.append(("original", input, output))

    monkeypatch.setattr(
        moe_sum, "use_flaggems_op", lambda op_name: op_name == "moe_sum"
    )
    monkeypatch.setattr(
        moe_sum,
        "_dispatch_moe_sum",
        lambda input, output: calls.append(("dispatch", input, output)),
    )

    assert moe_sum.patch_vllm_moe_sum(ops) is True
    assert moe_sum.patch_vllm_moe_sum(ops) is False

    nonempty_input = _FakeTensor(24)
    nonempty_output = _FakeTensor(8)
    ops.moe_sum(nonempty_input, nonempty_output)
    ops.moe_sum(_FakeTensor(0), nonempty_output)

    assert calls == [("dispatch", nonempty_input, nonempty_output)]
    assert ops.moe_sum._vllm_fl_original is not ops.moe_sum


def test_moe_sum_patch_uses_stride_safe_fallback(monkeypatch):
    calls = []
    ops = ModuleType("fake_vllm_custom_ops")
    ops.moe_sum = lambda input, output: None

    monkeypatch.setattr(moe_sum, "use_flaggems_op", lambda op_name: True)
    monkeypatch.setattr(
        moe_sum,
        "_torch_moe_sum",
        lambda input, output: calls.append((input, output)),
    )

    assert moe_sum.patch_vllm_moe_sum(ops) is True
    noncontiguous_input = _FakeTensor(24, hidden_stride=2)
    output = _FakeTensor(8)
    ops.moe_sum(noncontiguous_input, output)

    assert calls == [(noncontiguous_input, output)]


def test_moe_sum_patch_respects_flaggems_disable(monkeypatch):
    ops = ModuleType("fake_vllm_custom_ops")
    original = lambda input, output: None
    ops.moe_sum = original

    monkeypatch.setattr(moe_sum, "use_flaggems_op", lambda op_name: False)

    assert moe_sum.patch_vllm_moe_sum(ops) is False
    assert ops.moe_sum is original
