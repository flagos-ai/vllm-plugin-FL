# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for dispatch discovery module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, NonCallableMagicMock

from vllm_fl.dispatch.discovery import (
    discover_plugins,
    discover_from_env_modules,
    get_discovered_plugins,
    clear_discovered_plugins,
    _call_register_function,
    PLUGIN_MODULES_ENV,
)


class TestCallRegisterFunction:
    def test_direct_callable(self):
        registry = MagicMock()
        fn = MagicMock()

        result = _call_register_function(fn, registry, "test")

        assert result is True
        fn.assert_called_once_with(registry)

    def test_module_with_register_function(self):
        registry = MagicMock()
        module = NonCallableMagicMock(spec=["register"])  # Only has register attr
        module.register = MagicMock()

        result = _call_register_function(module, registry, "test")

        assert result is True
        module.register.assert_called_once_with(registry)

    def test_module_with_vllm_fl_register(self):
        registry = MagicMock()
        module = NonCallableMagicMock(spec=["vllm_fl_register"])  # Only has vllm_fl_register attr
        module.vllm_fl_register = MagicMock()

        result = _call_register_function(module, registry, "test")

        assert result is True
        module.vllm_fl_register.assert_called_once_with(registry)

    def test_callable_raises_exception(self):
        registry = MagicMock()
        fn = MagicMock(side_effect=Exception("test error"))

        result = _call_register_function(fn, registry, "test")

        assert result is False

    def test_no_register_function(self):
        registry = MagicMock()
        module = NonCallableMagicMock(spec=[])  # Not callable, no register function

        result = _call_register_function(module, registry, "test")

        assert result is False


class TestDiscoverFromEnvModules:
    @pytest.fixture(autouse=True)
    def clear_plugins(self):
        clear_discovered_plugins()
        yield
        clear_discovered_plugins()

    def test_empty_env_var(self):
        with patch.dict(os.environ, {PLUGIN_MODULES_ENV: ""}):
            registry = MagicMock()
            result = discover_from_env_modules(registry)
            assert result == 0

    def test_no_env_var(self):
        env = os.environ.copy()
        env.pop(PLUGIN_MODULES_ENV, None)
        with patch.dict(os.environ, env, clear=True):
            registry = MagicMock()
            result = discover_from_env_modules(registry)
            assert result == 0

    def test_import_error_handling(self):
        with patch.dict(os.environ, {PLUGIN_MODULES_ENV: "nonexistent_module"}):
            registry = MagicMock()
            result = discover_from_env_modules(registry)
            assert result == 0
            plugins = get_discovered_plugins()
            assert len(plugins) == 1
            assert plugins[0][2] is False  # success = False


class TestDiscoverPlugins:
    @pytest.fixture(autouse=True)
    def clear_plugins(self):
        clear_discovered_plugins()
        yield
        clear_discovered_plugins()

    def test_none_registry(self):
        result = discover_plugins(None)
        assert result == 0

    def test_empty_discovery(self):
        with patch.dict(os.environ, {PLUGIN_MODULES_ENV: ""}):
            with patch(
                "vllm_fl.dispatch.discovery._get_entry_points", return_value=[]
            ):
                registry = MagicMock()
                result = discover_plugins(registry)
                assert result == 0


class TestGetDiscoveredPlugins:
    @pytest.fixture(autouse=True)
    def clear_plugins(self):
        clear_discovered_plugins()
        yield
        clear_discovered_plugins()

    def test_returns_copy(self):
        plugins1 = get_discovered_plugins()
        plugins2 = get_discovered_plugins()
        assert plugins1 is not plugins2

    def test_initially_empty(self):
        plugins = get_discovered_plugins()
        assert plugins == []
