# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for the dispatch logger manager."""

import importlib
import logging

import pytest

import vllm_fl.dispatch.logger_manager as logger_manager
from vllm_fl.dispatch.logger_manager import get_logger, set_log_level


@pytest.fixture(autouse=True)
def _restore_logger_state():
    """Snapshot and restore the logger cache plus real logging state."""
    saved_cache = dict(logger_manager._loggers)
    saved_default = logger_manager._DEFAULT_LOG_LEVEL
    yield
    # Detach handlers we added so other tests never see duplicate output.
    for logger in logger_manager._loggers.values():
        if logger.name.startswith("vllm_fl.dispatch.test."):
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
    logger_manager._loggers.clear()
    logger_manager._loggers.update(saved_cache)
    logger_manager._DEFAULT_LOG_LEVEL = saved_default


class TestGetLogger:
    """get_logger caching and per-logger handler configuration."""

    @pytest.mark.supplementary
    def test_logger_is_cached_per_name(self):
        """Repeated calls return the identical Logger object."""
        first = get_logger("vllm_fl.dispatch.test.cache")
        second = get_logger("vllm_fl.dispatch.test.cache")
        assert first is second
        assert logger_manager._loggers["vllm_fl.dispatch.test.cache"] is first

    @pytest.mark.supplementary
    def test_new_logger_gets_stream_handler_and_no_propagation(self):
        """A fresh logger gets one StreamHandler and propagate=False."""
        logger = get_logger("vllm_fl.dispatch.test.handler")
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.propagate is False
        formatter = logger.handlers[0].formatter
        assert "[%(levelname)s] [%(name)s] %(message)s" in formatter._fmt

    @pytest.mark.supplementary
    def test_existing_handlers_are_not_duplicated(self):
        """A logger that already has handlers keeps them untouched."""
        name = "vllm_fl.dispatch.test.existing"
        pre = logging.getLogger(name)
        custom = logging.NullHandler()
        pre.addHandler(custom)
        try:
            logger = get_logger(name)
            assert custom in logger.handlers
            assert len(logger.handlers) == 1  # only the pre-existing one
        finally:
            pre.removeHandler(custom)

    @pytest.mark.supplementary
    def test_default_level_read_at_import_time(self, monkeypatch):
        """VLLM_FL_LOG_LEVEL is baked in at import, not read per call."""
        # A later env change must not alter the already-imported default.
        monkeypatch.setenv("VLLM_FL_LOG_LEVEL", "DEBUG")
        assert logger_manager._DEFAULT_LOG_LEVEL == "INFO"

    @pytest.mark.supplementary
    def test_import_reads_env_level(self, monkeypatch):
        """A reload with the env var set applies that level to new loggers."""
        monkeypatch.setenv("VLLM_FL_LOG_LEVEL", "WARNING")
        reloaded = importlib.reload(logger_manager)
        try:
            logger = reloaded.get_logger("vllm_fl.dispatch.test.envlevel")
            assert logger.level == logging.WARNING
        finally:
            importlib.reload(logger_manager)

    @pytest.mark.supplementary
    def test_invalid_env_level_falls_back_to_info(self, monkeypatch):
        """An unparsable VLLM_FL_LOG_LEVEL degrades to INFO after reload."""
        monkeypatch.setenv("VLLM_FL_LOG_LEVEL", "NOT_A_LEVEL")
        reloaded = importlib.reload(logger_manager)
        try:
            logger = reloaded.get_logger("vllm_fl.dispatch.test.badlevel")
            assert logger.level == logging.INFO
        finally:
            importlib.reload(logger_manager)


class TestSetLogLevel:
    """set_log_level for one cached logger or all of them."""

    @pytest.mark.supplementary
    def test_set_level_for_named_logger(self):
        """set_log_level with a name updates only that cached logger."""
        target = get_logger("vllm_fl.dispatch.test.target")
        other = get_logger("vllm_fl.dispatch.test.other")
        set_log_level("ERROR", name="vllm_fl.dispatch.test.target")
        assert target.level == logging.ERROR
        assert other.level != logging.ERROR

    @pytest.mark.supplementary
    def test_set_level_for_all_cached_loggers(self):
        """set_log_level without a name updates every cached logger."""
        first = get_logger("vllm_fl.dispatch.test.all1")
        second = get_logger("vllm_fl.dispatch.test.all2")
        set_log_level("CRITICAL")
        assert first.level == logging.CRITICAL
        assert second.level == logging.CRITICAL

    @pytest.mark.supplementary
    def test_set_level_is_case_insensitive(self):
        """Lowercase level strings are accepted."""
        logger = get_logger("vllm_fl.dispatch.test.case")
        set_log_level("debug", name="vllm_fl.dispatch.test.case")
        assert logger.level == logging.DEBUG

    @pytest.mark.supplementary
    def test_invalid_level_falls_back_to_info(self):
        """An unknown level string degrades to INFO, not an error."""
        logger = get_logger("vllm_fl.dispatch.test.invalid")
        set_log_level("NO_SUCH_LEVEL", name="vllm_fl.dispatch.test.invalid")
        assert logger.level == logging.INFO

    @pytest.mark.supplementary
    def test_unknown_name_is_ignored(self):
        """Setting a level for an uncached name does not raise."""
        set_log_level("DEBUG", name="vllm_fl.dispatch.test.never-created")
