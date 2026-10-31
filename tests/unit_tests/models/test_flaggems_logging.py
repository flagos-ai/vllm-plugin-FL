"""CPU contracts for optional FG logging; no device or model imports."""

import importlib.util
import logging
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "fg_logging_policy", ROOT / "vllm_fl/worker/flaggems_logging.py"
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.fixture
def fg_logger():
    log = logging.getLogger("flag_gems")
    previous = log.level
    yield log
    log.setLevel(previous)


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_legacy_and_explicit_recording(rank, fg_logger):
    fg_logger.setLevel(logging.DEBUG)
    assert policy.configure_oplist_recording({}, rank) == (rank == 0)
    assert policy.configure_oplist_recording({"flaggems_record_ops": True}, rank) == (
        rank == 0
    )
    assert fg_logger.level == logging.DEBUG


@pytest.mark.parametrize(
    "level", [logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING]
)
def test_disabled_skips_construction_and_preserves_stricter_levels(
    level, fg_logger, monkeypatch
):
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.DEBUG)
    fg_logger.setLevel(level)
    calls = []
    original = fg_logger.makeRecord

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fg_logger, "makeRecord", counted)
    for rank in [0, 1, 7]:
        assert (
            policy.configure_oplist_recording({"flaggems_record_ops": False}, rank)
            is False
        )
        for _ in range(100):
            fg_logger.debug("probe")
    assert calls == []
    assert fg_logger.level == max(logging.INFO, level)
    assert fg_logger.isEnabledFor(logging.WARNING)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_invalid_value_rejected(value, fg_logger):
    with pytest.raises(ValueError):
        policy.configure_oplist_recording({"flaggems_record_ops": value}, 0)
