# Copyright 2026 FlagOS Contributors

import copy

import pytest
from runtime_stack import (
    ATTENTION,
    STACK,
    audit_plan,
    validate_devices,
    validate_imports,
)


@pytest.fixture
def imports():
    return {
        "python": (3, 12),
        "vllm_module_version": "0.32.0",
        "flagtree_version": STACK["protected"]["flagtree"],
        "triton": {"version": "3.6.0", "path": "/opt/flagtree/triton/__init__.py"},
        "flash_op_registered": True,
        "plugin_registered": True,
        **{
            name: {"version": version}
            for name, version in STACK["protected"].items()
            if name not in ("flagtree", "triton")
        },
        **{
            name: copy.copy(STACK[name])
            for name in ("vllm", "flaggems-vllm", "flash-attn-3", "flagcx")
        },
    }


def test_valid_stack(imports):
    validate_imports(imports)
    validate_devices(
        {
            "vendor": "thead",
            "distributed_backend": "flagcx",
            "attention": ATTENTION,
            "devices": ["PPU-ZW810E"] * 4,
        }
    )


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("vllm", {"version": "0.32.0+empty"}, "Wrong vllm wheel version"),
        ("vllm", {"sha256": None}, "Wrong vllm wheel sha256"),
        ("torch", {"version": "2.10.0"}, "Wrong torch version"),
        ("flag-gems", {"version": "5.5.0"}, "Wrong flag-gems version"),
        ("triton", {"path": "/opt/triton/triton/__init__.py"}, "Runtime FlagTree"),
        ("flagtree_version", None, "Runtime FlagTree"),
        ("flash-attn-3", {"sha256": None}, "Wrong flash-attn-3 wheel sha256"),
        ("flagcx", {"sha256": None}, "Wrong flagcx wheel sha256"),
        ("flash_op_registered", False, "forward op is missing"),
        ("plugin_registered", False, "plugin entry point is missing"),
    ],
)
def test_wrong_import_is_rejected(imports, name, change, message):
    if isinstance(change, dict):
        imports[name].update(change)
    else:
        imports[name] = change
    with pytest.raises(RuntimeError, match=message):
        validate_imports(imports)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"vendor": "nvidia"}, "Expected T-Head"),
        ({"distributed_backend": "nccl"}, "requires FlagCX"),
        ({"attention": "upstream.cuda.Backend"}, "Wrong attention backend"),
        ({"devices": ["PPU-ZW810E"]}, "Expected four ZW810E"),
    ],
)
def test_wrong_devices_are_rejected(change, message):
    facts = {
        "vendor": "thead",
        "distributed_backend": "flagcx",
        "attention": ATTENTION,
        "devices": ["PPU-ZW810E"] * 4,
    }
    with pytest.raises(RuntimeError, match=message):
        validate_devices(facts | change)


@pytest.mark.parametrize("name", STACK["protected"])
def test_no_replacement_of_runtime_packages(name):
    replacement = {"metadata": {"name": name.upper().replace("-", "_")}}
    with pytest.raises(RuntimeError, match=f"pip would modify {name}"):
        audit_plan({"install": [replacement]})


def test_pip_plan_requires_hash():
    entry = {"metadata": {"name": "other"}, "download_info": {"archive_info": {}}}
    with pytest.raises(RuntimeError, match="did not verify a wheel hash"):
        audit_plan({"install": [entry]})
    entry["download_info"]["archive_info"]["hashes"] = {"sha256": "a" * 64}
    audit_plan({"install": [entry]})
