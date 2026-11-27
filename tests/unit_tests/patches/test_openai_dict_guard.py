# Copyright (c) 2026 BAAI. All rights reserved.
"""Tests for the OpenAI non-dict payload guard patch.

Covers the behaviours required by the plugin development standard:

* F1 — every target validator accepts non-dict payloads through the real
  request-model entry point, and non-target validators keep upstream behaviour;
* F2 — dict payloads preserve upstream semantics, including upstream's own
  semantic rejections;
* C2 — installation is idempotent and never leaves a half-applied patch;
* C5 — a missing/renamed upstream symbol fails loudly at the earliest point,
  and ``verify_only`` probes without mutating the classes.
"""

import importlib

import pytest

from vllm_fl.patches import openai_dict_guard as guard


def _cls(module_path: str, class_name: str):
    return getattr(importlib.import_module(module_path), class_name)


def _all_targets():
    """Yield (class, validator name) for every patched validator."""
    for module_path, class_name, names in guard._TARGETS:
        cls = _cls(module_path, class_name)
        for name in names:
            yield cls, name


def _snapshot(cls, name):
    """Return the current validator body and decorator function identity."""
    attr = cls.__dict__[name]
    body = attr.__func__ if isinstance(attr, classmethod) else attr
    decorator = cls.__pydantic_decorators__.model_validators[name]
    return body, decorator.func


def _is_patched(cls, name) -> bool:
    body, decorator_func = _snapshot(cls, name)
    return getattr(body, guard._MARKER, False) and getattr(
        decorator_func, guard._MARKER, False
    )


_ORIGINALS: dict = {}


def _capture_originals():
    """Record the untouched upstream callables once, before any patch runs."""
    for cls, name in _all_targets():
        attr = cls.__dict__[name]
        body = attr.__func__ if isinstance(attr, classmethod) else attr
        decorator_func = cls.__pydantic_decorators__.model_validators[name].func
        _ORIGINALS[(cls, name)] = (body, decorator_func)


def _uninstall(cls, name):
    """Restore one validator from the captured originals and rebuild."""
    saved = _ORIGINALS.get((cls, name))
    if saved is None:
        return
    original_body, original_func = saved
    type.__setattr__(cls, name, classmethod(original_body))
    cls.__pydantic_decorators__.model_validators[name].func = original_func


def _rebuild_request_models():
    """Force a schema rebuild for every request model the patch touches."""
    seen = set()
    for module_path, class_name, _ in guard._TARGETS:
        cls = _cls(module_path, class_name)
        if cls in seen:
            continue
        cls.model_rebuild(force=True)
        seen.add(cls)


@pytest.fixture(autouse=True)
def clean_patch_state():
    """Start and end every test from the untouched upstream state (C2)."""
    if not _ORIGINALS:
        _capture_originals()
    for cls, name in _all_targets():
        _uninstall(cls, name)
    _rebuild_request_models()
    yield
    for cls, name in _all_targets():
        _uninstall(cls, name)
    _rebuild_request_models()


@pytest.mark.parametrize("payload", ["bare-string", 12345, ["a", "list"]])
def test_non_dict_payload_no_longer_raises_attribute_error(payload):
    """F1: malformed bodies must surface as normal validation errors."""
    guard.install()
    for module_path, class_name, _ in guard._TARGETS:
        cls = _cls(module_path, class_name)
        with pytest.raises(Exception) as excinfo:
            cls.model_validate(payload)
        assert not isinstance(
            excinfo.value, AttributeError
        ), f"{class_name} still raises AttributeError for {payload!r}"


def test_valid_dict_payload_still_accepted():
    """F2: a well-formed chat request validates exactly as before the patch."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    before = ChatCompletionRequest.model_validate(body)
    guard.install()
    after = ChatCompletionRequest.model_validate(body)
    assert before.model == after.model
    assert len(after.messages) == 1


def test_upstream_semantic_rejection_preserved():
    """F2: contradictory flags are still rejected after installation."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    guard.install()
    with pytest.raises(Exception):
        ChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "continue_final_message": True,
                "add_generation_prompt": True,
            }
        )


def test_install_marks_every_target():
    """F1: installation reaches both the class attribute and the decorator."""
    result = guard.install()
    assert len(result.installed) == sum(len(n) for _, _, n in guard._TARGETS)
    for cls, name in _all_targets():
        assert _is_patched(cls, name), f"{cls.__name__}.{name} not patched"


def test_install_is_idempotent():
    """C2: a second install must not stack wrappers."""
    first = guard.install()
    assert first.installed
    second = guard.install()
    assert not second.installed
    assert set(second.already_patched) == set(first.installed)
    for cls, name in _all_targets():
        body, _ = _snapshot(cls, name)
        inner = getattr(body, "_vllm_fl_original", None)
        assert inner is not None and not getattr(inner, guard._MARKER, False)


def test_missing_validator_fails_before_mutating(monkeypatch):
    """C5: a renamed upstream validator aborts install with nothing applied."""
    # Snapshot the real targets before injecting the bogus symbol.
    real_targets = list(_all_targets())

    broken = list(guard._TARGETS)
    module_path, class_name, names = broken[0]
    broken[0] = (
        module_path,
        class_name,
        tuple(names) + ("this_validator_does_not_exist",),
    )
    monkeypatch.setattr(guard, "_TARGETS", tuple(broken))

    with pytest.raises(RuntimeError, match="not found"):
        guard.install()

    for cls, name in real_targets:
        assert not _is_patched(cls, name), f"{cls.__name__}.{name} was patched"


def test_verify_only_does_not_mutate():
    """C5: probing the symbols must leave the classes untouched."""
    guard.install(verify_only=True)
    for cls, name in _all_targets():
        assert not _is_patched(cls, name)


def test_non_target_validator_untouched():
    """F1: validators outside the target list keep their upstream identity."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    name = "check_system_message_content_type"
    body_before, func_before = _snapshot(ChatCompletionRequest, name)
    guard.install()
    body_after, func_after = _snapshot(ChatCompletionRequest, name)
    assert body_after is body_before
    assert func_after is func_before