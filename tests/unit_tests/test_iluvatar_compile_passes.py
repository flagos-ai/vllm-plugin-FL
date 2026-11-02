"""Check OOT-only fusion defaults without loading model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('vendor,disabled', [('iluvatar', True), ('nvidia', False)])
def test_only_iluvatar_disables_unimported_cuda_passes(vendor, disabled):
    path = Path(__file__).parents[2] / 'vllm_fl/platform.py'
    tree = ast.parse(path.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Compare)
                  and ast.unparse(n.test) == "getattr(cls, 'vendor_name', None) == 'iluvatar'"
                  and any(isinstance(c, ast.Constant) and c.value == 'pass_config'
                          for c in ast.walk(n)))
    flags = ('fuse_rope_kvcache_cat_mla', 'fuse_norm_quant',
             'fuse_act_quant', 'fuse_attn_quant')
    pc = SimpleNamespace(**dict.fromkeys(flags, True))
    config = SimpleNamespace(pass_config=pc)
    code = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    exec(compile(code, str(path), 'exec'), {
        'cls': SimpleNamespace(vendor_name=vendor),
        'compilation_config': config,
        'os': SimpleNamespace(environ={}),
    })
    assert all(getattr(pc, name) is (not disabled) for name in flags)
