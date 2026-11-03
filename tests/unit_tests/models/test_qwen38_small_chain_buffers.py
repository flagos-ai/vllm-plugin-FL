# SPDX-License-Identifier: Apache-2.0
"""Model output-initialization and HC callback contracts, without a GPU."""

import __future__

import ast
from functools import lru_cache

from _qwen38_cpu import ROOT


def fn(path, name, ns):
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    scope = dict(ns)
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    return scope[name]


def test_hc_row_settings_cache_is_bounded_and_reuses_same_shape():
    path = (
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/hc_policy.py"
    )
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_row_settings"
    )
    scope = {
        "lru_cache": lru_cache,
        "_settings": lambda: {"gate": [{"max_rows": 8192, "block": 256}]},
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    get = scope["_row_settings"]
    assert get("gate", 8192) is get("gate", 8192)
    for rows in range(1, 80):
        get("gate", rows)
    assert get.cache_info().maxsize == 64 and get.cache_info().currsize == 64
