"""Exercise the runner's pre-compilation dispatch warmup guard."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('vendor,device,mode,expected', [
    ('iluvatar', 'cuda', 3, True),
    ('iluvatar', 'cuda', 0, False),
    ('ascend', 'npu', 3, True),
    ('nvidia', 'cuda', 3, False),
])
def test_runner_warms_dispatch_before_compile(monkeypatch, vendor, device, mode, expected):
    import vllm_fl.dispatch as dispatch
    calls = []
    monkeypatch.setattr(dispatch, 'warmup_cached_ops', lambda: calls.append('warm'))
    source = Path(__file__).parents[2] / 'vllm_fl/worker/iluvatar_model_runner.py'
    tree = ast.parse(source.read_text())
    load = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == 'load_model')
    branch = next(n for n in load.body if isinstance(n, ast.If)
                  and any(isinstance(c, ast.ImportFrom) and c.module == 'vllm_fl.dispatch'
                          for c in ast.walk(n)))
    code = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    exec(compile(code, str(source), 'exec'), {
        'self': SimpleNamespace(compilation_config=SimpleNamespace(mode=mode)),
        'current_platform': SimpleNamespace(vendor_name=vendor, device_type=device),
        'CompilationMode': SimpleNamespace(NONE=0),
    })
    assert calls == (['warm'] if expected else [])
