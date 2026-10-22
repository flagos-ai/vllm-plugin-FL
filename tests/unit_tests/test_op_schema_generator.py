# Copyright (c) 2026 BAAI. All rights reserved.

"""The EMPTY schema shim must preserve the upstream operator namespace."""

import importlib.util
from pathlib import Path


def _load_generator():
    path = Path(__file__).resolve().parents[2] / "tools" / "generate_op_schemas.py"
    spec = importlib.util.spec_from_file_location("op_schema_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_namespace_does_not_leak_into_c_schemas(tmp_path):
    csrc = tmp_path / "csrc"
    csrc.mkdir()
    (csrc / "torch_bindings.cpp").write_text(
        """
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("norm(Tensor input, "
          "Tensor! out) -> ()");
}
STABLE_TORCH_LIBRARY_FRAGMENT(_C_cache_ops, ops) {
  ops.def("reshape_and_cache(Tensor input, Tensor! cache) -> ()");
}
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("activation(Tensor input) -> Tensor");
}
"""
    )

    assert _load_generator().extract_schemas(tmp_path) == [
        "norm(Tensor input, Tensor! out) -> ()",
        "activation(Tensor input) -> Tensor",
    ]


def test_schema_overloads_survive_duplicate_source_definitions(tmp_path):
    csrc = tmp_path / "csrc"
    stable = csrc / "libtorch_stable"
    stable.mkdir(parents=True)
    source = """
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("norm(Tensor input) -> Tensor");
  ops.def("norm.out(Tensor input, Tensor! out) -> ()");
}
TORCH_LIBRARY_FRAGMENT(_C_other, ops) {
  ops.def("unrelated(Tensor input) -> Tensor");
}
"""
    (csrc / "torch_bindings.cpp").write_text(source)
    (stable / "torch_bindings.cpp").write_text(source)

    assert _load_generator().extract_schemas(tmp_path) == [
        "norm(Tensor input) -> Tensor",
        "norm.out(Tensor input, Tensor! out) -> ()",
    ]
