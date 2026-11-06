"""Small explicit compatibility boundary. No AST rewriting or anonymous providers."""

import hashlib
import importlib.abc
import inspect
import json
import sys
from pathlib import Path


class ForbiddenVendorImport(BaseException):
    pass


class _DenyVendor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm_metax" or fullname.startswith("vllm_metax."):
            raise ForbiddenVendorImport(fullname)


def install_vendor_guard():
    if any(n == "vllm_metax" or n.startswith("vllm_metax.") for n in sys.modules):
        raise ForbiddenVendorImport("Vendor framework imported before initialization")
    if not any(isinstance(x, _DenyVendor) for x in sys.meta_path):
        sys.meta_path.insert(0, _DenyVendor())


def install_serial_execution(expected_source_sha256):
    import vllm.utils.multi_stream_utils as native
    from .serial import run_pair_serially, run_branches_serially

    source = Path(native.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_source_sha256, (
        "Unsupported vLLM multi-stream source"
    )
    pairs = (
        ("maybe_execute_in_parallel", run_pair_serially),
        ("execute_in_parallel", run_branches_serially),
    )
    for name, replacement in pairs:
        previous = getattr(native, name)
        if previous is replacement:
            continue
        assert tuple(inspect.signature(previous).parameters) == tuple(
            inspect.signature(replacement).parameters
        ), (name, "API changed")
        setattr(native, name, replacement)
