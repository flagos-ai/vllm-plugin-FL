# Copyright (c) 2026 BAAI. All rights reserved.

"""
Tests for dispatch logger_manager module.

Covers:
  - get_logger: returns a Logger, caches by name
  - set_log_level: sets level by name or all loggers
"""

import logging

import pytest


class TestGetLogger:
    """Tests for get_logger()."""

    def setup_method(self):
        # Import fresh each test to avoid cross-test state leakage
        from vllm_fl.dispatch.logger_manager import _loggers
        _loggers.clear()

    def test_returns_logger_instance(self):
        from vllm_fl.dispatch.logger_manager import get_logger
        logger = get_logger("test_logger_a")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self):
        from vllm_fl.dispatch.logger_manager import get_logger
        logger = get_logger("my_component")
        assert logger.name == "my_component"

    def test_same_name_returns_same_object(self):
        from vllm_fl.dispatch.logger_manager import get_logger
        l1 = get_logger("shared_logger")
        l2 = get_logger("shared_logger")
        assert l1 is l2

    def test_different_names_return_different_objects(self):
        from vllm_fl.dispatch.logger_manager import get_logger
        l1 = get_logger("component_x")
        l2 = get_logger("component_y")
        assert l1 is not l2

    def test_logger_is_registered(self):
        from vllm_fl.dispatch.logger_manager import get_logger, _loggers
        get_logger("registered_comp")
        assert "registered_comp" in _loggers

    def test_multiple_loggers_all_registered(self):
        from vllm_fl.dispatch.logger_manager import get_logger, _loggers
        for name in ("alpha", "beta", "gamma"):
            get_logger(name)
        for name in ("alpha", "beta", "gamma"):
            assert name in _loggers


class TestSetLogLevel:
    """Tests for set_log_level()."""

    def setup_method(self):
        from vllm_fl.dispatch.logger_manager import _loggers
        _loggers.clear()

    def test_set_level_by_name_debug(self):
        from vllm_fl.dispatch.logger_manager import get_logger, set_log_level
        get_logger("lvl_test")
        set_log_level("DEBUG", name="lvl_test")
        from vllm_fl.dispatch.logger_manager import _loggers
        assert _loggers["lvl_test"].level == logging.DEBUG

    def test_set_level_by_name_warning(self):
        from vllm_fl.dispatch.logger_manager import get_logger, set_log_level
        get_logger("lvl_warn")
        set_log_level("WARNING", name="lvl_warn")
        from vllm_fl.dispatch.logger_manager import _loggers
        assert _loggers["lvl_warn"].level == logging.WARNING

    def test_set_level_by_name_error(self):
        from vllm_fl.dispatch.logger_manager import get_logger, set_log_level
        get_logger("lvl_err")
        set_log_level("ERROR", name="lvl_err")
        from vllm_fl.dispatch.logger_manager import _loggers
        assert _loggers["lvl_err"].level == logging.ERROR

    def test_set_level_nonexistent_name_no_crash(self):
        """set_log_level with unknown name should not raise."""
        from vllm_fl.dispatch.logger_manager import set_log_level
        set_log_level("DEBUG", name="does_not_exist")

    def test_set_level_all_loggers(self):
        """set_log_level without name sets all registered loggers."""
        from vllm_fl.dispatch.logger_manager import get_logger, set_log_level, _loggers
        get_logger("all_a")
        get_logger("all_b")
        set_log_level("ERROR")
        for logger in _loggers.values():
            assert logger.level == logging.ERROR

    def test_set_level_all_then_specific(self):
        """Setting all then one specific name works independently."""
        from vllm_fl.dispatch.logger_manager import get_logger, set_log_level, _loggers
        get_logger("spec_a")
        get_logger("spec_b")
        set_log_level("ERROR")
        set_log_level("DEBUG", name="spec_a")
        assert _loggers["spec_a"].level == logging.DEBUG
        assert _loggers["spec_b"].level == logging.ERROR

    def test_set_level_info(self):
        from vllm_fl.dispatch.logger_manager import get_logger, set_log_level, _loggers
        get_logger("info_logger")
        set_log_level("INFO", name="info_logger")
        assert _loggers["info_logger"].level == logging.INFO
