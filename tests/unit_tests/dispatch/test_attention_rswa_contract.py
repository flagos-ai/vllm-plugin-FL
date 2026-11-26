# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unsupported capability hints must fail without loading vendor libraries."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "vllm_fl/dispatch/backends/vendor"
VENDORS = [
    "amd",
    "gcu",
    "thead",
    "musa",
    "txda",
    "kunlunxin",
    "iluvatar",
    "cuda",
    "ascend",
    "sunrise",
]


@pytest.mark.parametrize("vendor", VENDORS)
def test_optional_rswa_hint_rejects_unsupported_vendor_before_imports(vendor):
    path = ROOT / vendor / f"{vendor}.py"
    tree = ast.parse(path.read_text())
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "attention_backend"
    ]
    assert len(functions) == 1
    function = functions[0]
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    # No imports or device globals are supplied to the isolated method. A
    # misplaced guard would attempt the real vendor path and fail this check.
    with pytest.raises(NotImplementedError, match="R-SWA selection"):
        namespace["attention_backend"](None, use_rswa=True)
