# Copyright (c) 2026 BAAI. All rights reserved.
"""Tolerate non-dict OpenAI request payloads in vLLM 0.24 request validators.

Upstream vLLM 0.24.0 declares several ``@model_validator(mode="before")``
callables on the OpenAI request models that read the raw payload with
``data.get(...)`` / ``data[...]`` before checking whether ``data`` is a
mapping. A caller sending a non-dict body (a bare string, a list, an integer)
therefore hits ``AttributeError: 'str' object has no attribute 'get'`` inside
request validation, and the server answers HTTP 500 instead of a normal 4xx
validation error. Evaluation platforms and gateways that probe the endpoint
with malformed payloads trigger this path and can abort a whole evaluation
batch.

Compatibility boundary (C5):

* Relies on upstream private symbols:
  ``vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionRequest``,
  ``vllm.entrypoints.openai.completion.protocol.CompletionRequest`` and
  ``vllm.entrypoints.openai.responses.protocol.ResponsesRequest``, plus their
  ``__pydantic_decorators__.model_validators`` mapping and
  ``BaseModel.model_rebuild``.
* Verified against: vLLM 0.24.0 (``vllm-0.24.0+cu129-cp38-abi3``).
* Why a schema rebuild is required: pydantic v2 compiles ``model_validator``
  callables into ``__pydantic_core_schema__`` when the class is created, and
  the compiled validator keeps a direct reference to the original function.
  Replacing the class attribute or the decorator entry alone does not affect
  validation; ``model_rebuild(force=True)`` is needed so the new callables are
  picked up.
* Expected invalidation: a target validator is renamed or removed (the patch
  fails loudly), upstream gains its own ``isinstance(data, dict)`` guard (the
  patch becomes a no-op for that target and logs it), or pydantic changes the
  decorator/schema API (``model_rebuild`` raises and installation is rolled
  back).
* Long-term replacement: file the guard upstream and delete this module once a
  vLLM release carrying the fix becomes the pinned baseline.

Installation is idempotent and never edits files under ``site-packages/vllm``:
class attributes and decorator entries are replaced in memory and every
installed wrapper carries a marker so repeated initialisation is a no-op.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

_MARKER = "_vllm_fl_dict_guard_patch"

_TARGETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "vllm.entrypoints.openai.chat_completion.protocol",
        "ChatCompletionRequest",
        (
            "validate_response_format",
            "validate_stream_options",
            "check_logprobs",
            "check_structured_outputs_count",
            "check_generation_prompt",
            "check_cache_salt_support",
        ),
    ),
    (
        "vllm.entrypoints.openai.completion.protocol",
        "CompletionRequest",
        (
            "validate_response_format",
            "check_structured_outputs_count",
            "check_logprobs",
            "validate_stream_options",
            "validate_prompt_and_prompt_embeds",
            "check_cache_salt_support",
        ),
    ),
    (
        "vllm.entrypoints.openai.responses.protocol",
        "ResponsesRequest",
        (
            "validate_background",
            "validate_prompt",
            "check_cache_salt_support",
            "input_item_parsing",
        ),
    ),
)


class PatchResult(NamedTuple):
    """Outcome of one :func:`install` call."""

    installed: tuple[str, ...]
    already_patched: tuple[str, ...]
    upstream_guarded: tuple[str, ...]


def _make_guard(func, *, bound_method: bool):
    """Wrap a validator body so non-mapping payloads pass straight through.

    pydantic invokes ``mode="before"`` model validators in two shapes depending
    on how they are reached:

    * through the class attribute (``classmethod``) — the first argument is the
      class and the second is the payload;
    * through the compiled core schema, which binds the class already and
      passes the payload as the first argument (optionally followed by a
      ``ValidationInfo``).

    ``bound_method`` selects which shape to accept so both paths stay correct.
    """

    if bound_method:
        cls_holder: dict[str, Any] = {}

        def guarded(data, *args, **kwargs):
            if not isinstance(data, dict):
                return data
            return func(cls_holder["cls"], data, *args, **kwargs)

        guarded._vllm_fl_bind_cls = cls_holder  # type: ignore[attr-defined]
    else:
        def guarded(cls, data, *args, **kwargs):
            if not isinstance(data, dict):
                return data
            return func(cls, data, *args, **kwargs)

    guarded.__name__ = getattr(func, "__name__", "guarded")
    guarded.__qualname__ = getattr(func, "__qualname__", guarded.__name__)
    guarded.__doc__ = getattr(func, "__doc__", None) or "Non-dict tolerant validator."
    setattr(guarded, _MARKER, True)
    setattr(guarded, "_vllm_fl_original", func)
    return guarded


def _upstream_guarded(func) -> bool:
    """Return True when the validator already short-circuits non-dict payloads."""
    import inspect

    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):  # pragma: no cover - no source available
        return False
    return "isinstance(data, dict)" in source


def _collect_plan() -> tuple[list[tuple[Any, str, Any, Any]], list[str], list[str]]:
    """Validate every dependency and return the replacement plan (C3 phase 1)."""
    import importlib

    plan: list[tuple[Any, str, Any, Any]] = []
    already: list[str] = []
    guarded: list[str] = []

    for module_path, class_name, names in _TARGETS:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise RuntimeError(
                "vllm_fl dict-guard patch: "
                f"{module_path}.{class_name} not found; upstream layout changed, "
                "refusing to install a partial patch"
            )
        validators = getattr(getattr(cls, "__pydantic_decorators__", None), "model_validators", None)
        if validators is None:
            raise RuntimeError(
                "vllm_fl dict-guard patch: "
                f"{class_name} exposes no __pydantic_decorators__.model_validators; "
                "pydantic integration changed, update the patch"
            )
        for name in names:
            attr = cls.__dict__.get(name)
            if attr is None:
                raise RuntimeError(
                    "vllm_fl dict-guard patch: "
                    f"{class_name}.{name} not found; upstream validators were renamed "
                    "or removed, update _TARGETS"
                )
            if not isinstance(attr, classmethod):
                raise RuntimeError(
                    "vllm_fl dict-guard patch: "
                    f"{class_name}.{name} is not a classmethod "
                    f"({type(attr).__name__}); decorator shape changed"
                )
            decorator = validators.get(name)
            if decorator is None:
                raise RuntimeError(
                    "vllm_fl dict-guard patch: "
                    f"{class_name}.{name} has no pydantic model validator entry; "
                    "decorator registration changed"
                )
            body = attr.__func__
            label = f"{class_name}.{name}"
            if getattr(body, _MARKER, False):
                already.append(label)
            elif _upstream_guarded(body):
                guarded.append(label)
            else:
                plan.append((cls, name, decorator, body))
    return plan, already, guarded


def install(verify_only: bool = False) -> PatchResult:
    """Install the non-dict payload guard on the affected OpenAI validators.

    Args:
        verify_only: only probe that every target symbol exists with the
            expected shape; nothing is replaced or rebuilt.

    Returns:
        A :class:`PatchResult` listing installed, already-patched and
        upstream-guarded validators.

    Raises:
        RuntimeError: a target class, validator, or decorator entry is
            missing, or a target is not a ``classmethod`` as vLLM 0.24.0
            constructs it. The affected request models would then be at risk,
            so failing early is required (C5).
    """
    plan, already, guarded = _collect_plan()

    if verify_only:
        pending = tuple(f"{cls.__name__}.{name}" for cls, name, _, _ in plan)
        return PatchResult((), tuple(already), tuple(guarded) + pending)

    if not plan:
        return PatchResult((), tuple(already), tuple(guarded))

    # Phase 2: replace and rebuild atomically. Every target was pre-validated,
    # so a failure here means pydantic changed underneath us; roll back the
    # classes already touched so no half-installed state survives (C3).
    done: list[tuple[Any, str, Any, Any]] = []
    installed: list[str] = []
    try:
        for cls, name, decorator, body in plan:
            original_cm = cls.__dict__[name]
            original_func = decorator.func
            wrapper = _make_guard(body, bound_method=False)
            bound_wrapper = _make_guard(body, bound_method=True)
            bound_wrapper._vllm_fl_bind_cls["cls"] = cls
            type.__setattr__(cls, name, classmethod(wrapper))
            decorator.func = bound_wrapper
            done.append((cls, name, original_cm, (decorator, original_func)))
            installed.append(f"{cls.__name__}.{name}")

        # The compiled core schema holds references to the original callables,
        # so it must be regenerated for the guards to take effect.
        rebuilt: set[type] = set()
        for cls, _, _, _ in plan:
            if cls in rebuilt:
                continue
            cls.model_rebuild(force=True)
            rebuilt.add(cls)
    except Exception:
        for cls, name, original_cm, (decorator, original_func) in reversed(done):
            type.__setattr__(cls, name, original_cm)
            decorator.func = original_func
        for cls, _, _, _ in reversed(done):
            try:
                cls.model_rebuild(force=True)
            except Exception:  # pragma: no cover - best-effort restore
                logger.warning(
                    "vllm_fl dict-guard patch: could not rebuild %s after rollback",
                    cls.__name__,
                )
        logger.error(
            "vllm_fl dict-guard patch: install failed, rolled back %d target(s)",
            len(done),
        )
        raise

    logger.info(
        "Installed vLLM OpenAI non-dict payload guard on %d validator(s): %s",
        len(installed),
        ", ".join(installed),
    )
    if guarded:
        logger.info(
            "vllm_fl dict-guard patch: %d validator(s) already guarded upstream: %s",
            len(guarded),
            ", ".join(guarded),
        )
    return PatchResult(tuple(installed), tuple(already), tuple(guarded))


__all__ = ["PatchResult", "install"]
