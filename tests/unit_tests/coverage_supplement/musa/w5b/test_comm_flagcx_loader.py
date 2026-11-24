"""Loader / package-level contracts for the FlagCX communicator family."""
import sys
from pathlib import Path
from unittest.mock import Mock
import pytest
import torch
import vllm_fl.distributed.device_communicators.flagcx as fx
_FULL_API = 'from dataclasses import dataclass\n\n@dataclass\nclass flagcxUniqueId:\n    internal: bytes = b""\n\nclass FLAGCXLibrary:\n    pass\n\nbuffer_type = flagcxComm_t = flagcxDataTypeEnum = flagcxRedOpTypeEnum = object()\n'

def _write_api(directory: Path, source: str) -> Path:
    path = directory / 'api.py'
    path.write_text(source, encoding='utf-8')
    return path

def test_load_wheel_bindings_success(tmp_path):
    path = _write_api(tmp_path, _FULL_API)
    result = fx._load_flagcx_wheel_bindings(path)
    assert len(result) == 6
    (library, buffer_type, comm_t, dtype, unique, redop) = result
    assert library.__name__ == 'FLAGCXLibrary'
    assert unique.__module__.startswith(fx.__name__)

def test_load_wheel_bindings_missing_file_propagates(tmp_path):
    with pytest.raises(OSError):
        fx._load_flagcx_wheel_bindings(tmp_path / 'absent.py')

def test_load_amd_flagcx_api_success(tmp_path):
    path = _write_api(tmp_path, _FULL_API)
    api = fx._load_amd_flagcx_api(path)
    assert api.FLAGCXLibrary.__name__ == 'FLAGCXLibrary'
    assert sys.modules[f'{fx.__name__}._amd_api'] is api

def test_load_amd_flagcx_api_failure_cleans_module(tmp_path):
    path = _write_api(tmp_path, "raise ImportError('nope')\n")
    with pytest.raises(ImportError):
        fx._load_amd_flagcx_api(path)
    assert f'{fx.__name__}._amd_api' not in sys.modules

def test_distributed_package_lazy_attribute():
    import vllm_fl.distributed as pkg
    pkg.__dict__.pop('communicator', None)
    module = pkg.__getattr__('communicator')
    assert module is sys.modules['vllm_fl.distributed.communicator']
    assert pkg.communicator is module
    with pytest.raises(AttributeError, match='has no attribute'):
        pkg.__getattr__('missing')