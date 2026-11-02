from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('compiled', [False, True])
def test_gcu_ir_defaults_reference_registered_providers(monkeypatch, compiled):
    from vllm.config.compilation import CompilationMode
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, 'device_type', 'gcu')
    config = SimpleNamespace(compilation_config=SimpleNamespace(
        backend='inductor',
        mode=CompilationMode.VLLM_COMPILE if compiled else CompilationMode.NONE))
    priorities = PlatformFL.get_default_ir_op_priority(config)
    assert priorities.rms_norm == ['native']
    assert priorities.fused_add_rms_norm == ['native']
    assert priorities.gelu_and_mul_sparse == ['native']
    # Exercise actual target registry validation, not just config construction.
    with priorities.set_priority():
        pass
