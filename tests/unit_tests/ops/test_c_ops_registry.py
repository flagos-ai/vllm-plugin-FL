# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import vllm_fl
from vllm_fl.ops import _C_ops_registry as registry


def test_loads_legacy_and_stable_extensions_on_every_platform(monkeypatch):
    calls = []
    monkeypatch.setattr(registry, "_import_extension", calls.append)

    assert registry.load_vllm_native_extensions()
    assert calls == [
        registry._LEGACY_C_EXTENSION,
        registry._STABLE_C_EXTENSION,
    ]


def test_loads_stable_extension_without_legacy(monkeypatch):
    calls = []

    def import_extension(module_name):
        calls.append(module_name)
        if module_name == registry._LEGACY_C_EXTENSION:
            raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(registry, "_import_extension", import_extension)

    assert registry.load_vllm_native_extensions()
    assert calls == [
        registry._LEGACY_C_EXTENSION,
        registry._STABLE_C_EXTENSION,
    ]


def test_loads_legacy_extension_without_stable(monkeypatch):
    calls = []

    def import_extension(module_name):
        calls.append(module_name)
        if module_name == registry._STABLE_C_EXTENSION:
            raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(registry, "_import_extension", import_extension)

    assert registry.load_vllm_native_extensions()
    assert calls == [
        registry._LEGACY_C_EXTENSION,
        registry._STABLE_C_EXTENSION,
    ]


def test_fallback_schema_registration_skipped_when_native_extension_loaded(
    monkeypatch,
):
    monkeypatch.delattr(registry.register_op_schemas, "_lib", raising=False)
    monkeypatch.setattr(
        registry,
        "load_vllm_native_extensions",
        lambda: True,
    )

    def fail_if_fallback_library_is_created(*args, **kwargs):
        raise AssertionError("fallback schemas must not precede native CUDA ops")

    monkeypatch.setattr(
        registry.torch.library,
        "Library",
        fail_if_fallback_library_is_created,
    )

    registry.register_op_schemas()


def test_plugin_initialization_loads_native_ops_before_fallback(monkeypatch):
    monkeypatch.setattr(
        registry,
        "load_vllm_native_extensions",
        lambda: True,
    )

    def fail_if_fallback_is_registered():
        raise AssertionError("native ops must bypass fallback schema registration")

    monkeypatch.setattr(
        registry,
        "register_op_schemas",
        fail_if_fallback_is_registered,
    )

    vllm_fl._patch_custom_ops()
