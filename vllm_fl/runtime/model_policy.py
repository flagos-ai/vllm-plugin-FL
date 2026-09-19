# SPDX-License-Identifier: Apache-2.0
"""Model-scoped runtime policy plans.

``ActivationPlan`` (:mod:`vllm_fl.activation`) owns the *patch activation* a
model integration needs before its modules are constructed.  This module owns a
separate concern: the *runtime selection* a model needs once the worker has
registered its device and implementations -- the dispatch policy it must run
with, a static attention-backend override, and the native ATen operators that
must survive FlagOS takeover.

The two plans are deliberately kept apart:

* ``ActivationPlan`` -> class-level monkey patches and the model's capability
  requirements (``MoEDispatchDefaults``).  It may also carry a lazy,
  selector-context-dependent attention override callable (GLM5's sparse MLA).
* ``RuntimePlan`` -> the fully resolved ``SelectionPolicy``, an optional
  *static* attention-backend path (``None`` means "use the generic selector"),
  and ``native_aten_ops``.

Responsibilities:

* Model adapters register a :class:`ModelPolicyFactory` keyed by architecture /
  model type.  All model-specific decisions live in the adapter; the common
  layer only matches and calls the registry entry points.
* ``build_model_runtime_plan`` is a pure function of its arguments: no
  environment writes and no global mutation.
* ``validate_model_config`` is the single common entry point for model config
  validation, called late (after vLLM resolves its own final values) from
  ``PlatformFL.check_and_update_config``.  It dispatches to the registered
  factory and runs the final asynchronous-scheduling guard for runners that
  build the PLE n-gram token history on the CPU.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from vllm_fl.dispatch.policy import SelectionPolicy

logger = logging.getLogger(__name__)

__all__ = [
    "ModelPolicyError",
    "ModelPolicyFactory",
    "RuntimePlan",
    "activate_runtime_plan",
    "build_model_runtime_plan",
    "get_active_runtime_plan",
    "preflight_runtime_plan",
    "register_model_policy_factory",
    "reset_model_policy_for_tests",
    "validate_model_config",
]


class ModelPolicyError(RuntimeError):
    """Raised when a model's resolved runtime configuration is unsupported."""


@dataclass(frozen=True)
class RuntimePlan:
    """Resolved, process-global runtime selection for one model.

    Attributes:
        selection_policy: Final dispatch policy, i.e. the user's explicit order
            with the model's own defaults layered only where the user left an
            operator unspecified.
        attention_backend: Attention backend class path that does not depend on
            the selector config, or ``None`` to use the generic selection (or a
            model activation plan's selector-aware override).
        native_aten_ops: ATen operators this process must keep on their native
            implementation (captured before FlagOS takes over).
    """

    selection_policy: SelectionPolicy
    attention_backend: str | None = None
    native_aten_ops: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ModelPolicyFactory:
    """Adapter-provided factory for one model family's :class:`RuntimePlan`.

    Attributes:
        build: ``(vllm_config, device_caps, user_policy) -> RuntimePlan | None``.
            Returning ``None`` falls back to the default plan.
        architectures: ``hf_config.architectures`` values this factory serves.
        model_types: ``hf_text_config.model_type`` values this factory serves,
            used only when no factory matches by architecture.
        validate: Optional ``(vllm_config) -> None`` config-validation hook for
            values that only become final after vLLM's own resolution.
        cpu_token_history: Optional predicate telling whether the model's
            selected runner builds its token history on the CPU.  Required for
            the final asynchronous-scheduling guard; ``None`` means "no CPU
            history" (or "not evaluated here").
        name: Human-readable identifier used in error messages.
    """

    build: Callable[[Any, Any, SelectionPolicy | None], RuntimePlan | None]
    architectures: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ()
    validate: Callable[[Any], None] | None = None
    cpu_token_history: Callable[[Any], bool] | None = None
    name: str = ""


_factories: list[ModelPolicyFactory] = []
_active_runtime_plan: RuntimePlan | None = None
_lock = threading.RLock()


def register_model_policy_factory(
    factory: ModelPolicyFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a model policy factory (benign at plugin import time).

    A factory already registered under the same name is treated as
    idempotent.  Pass ``replace=True`` to clear and re-register (test use).
    """
    with _lock:
        if replace:
            _factories.clear()
        if factory not in _factories:
            _factories.append(factory)


def _model_config(vllm_config: Any) -> Any:
    return getattr(vllm_config, "model_config", None)


def _text_config(vllm_config: Any) -> Any:
    model_config = _model_config(vllm_config)
    if model_config is None:
        return None
    return (
        getattr(model_config, "hf_text_config", None)
        or getattr(model_config, "hf_config", None)
    )


def _architectures(vllm_config: Any) -> set[str]:
    model_config = _model_config(vllm_config)
    if model_config is None:
        return set()
    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    candidates: Sequence[Any] = (
        getattr(model_config, "architectures", None),
        getattr(text_config, "architectures", None),
        getattr(hf_config, "architectures", None),
    )
    result: set[str] = set()
    for candidate in candidates:
        if candidate:
            result.update(str(arch) for arch in candidate)
    return result


def _model_types(vllm_config: Any) -> set[str]:
    model_config = _model_config(vllm_config)
    if model_config is None:
        return set()
    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    candidates = (
        getattr(text_config, "model_type", None),
        getattr(hf_config, "model_type", None),
        getattr(model_config, "model_type", None),
    )
    return {str(model_type) for model_type in candidates if model_type}


def _find_factory(vllm_config: Any) -> ModelPolicyFactory | None:
    """Match on architecture first, then on model type."""
    installed = list(_factories)
    architectures = _architectures(vllm_config)
    if architectures:
        for factory in installed:
            if factory.architectures and architectures.intersection(
                factory.architectures
            ):
                return factory
    model_types = _model_types(vllm_config)
    if model_types:
        for factory in installed:
            if factory.model_types and model_types.intersection(factory.model_types):
                return factory
    return None


def build_model_runtime_plan(
    vllm_config: Any,
    device_caps: Any,
    user_policy: SelectionPolicy | None,
) -> RuntimePlan:
    """Build the :class:`RuntimePlan` for ``vllm_config``.

    Pure with respect to its inputs: the matching factory is looked up from the
    registry and all model-specific decisions are made there.  No environment
    variable is read or written and no global state is mutated.
    """
    base_policy = user_policy if user_policy is not None else SelectionPolicy()
    factory = _find_factory(vllm_config)
    if factory is not None:
        plan = factory.build(vllm_config, device_caps, base_policy)
        if plan is not None:
            return plan
    return RuntimePlan(selection_policy=base_policy)


def preflight_runtime_plan(plan: RuntimePlan) -> None:
    """Reject a conflicting runtime selection before model patches are applied."""
    with _lock:
        if _active_runtime_plan is not None and _active_runtime_plan != plan:
            raise ModelPolicyError(
                "A different RuntimePlan is already active in this process; "
                "start a separate worker process for a different runtime selection."
            )


def activate_runtime_plan(plan: RuntimePlan) -> RuntimePlan | None:
    """Publish a single process plan, allowing only equivalent reactivation."""
    global _active_runtime_plan
    with _lock:
        preflight_runtime_plan(plan)
        previous = _active_runtime_plan
        if previous is None:
            _active_runtime_plan = plan
    return previous


def get_active_runtime_plan() -> RuntimePlan | None:
    return _active_runtime_plan


def _validate_final_async_scheduling(
    vllm_config: Any,
    factory: ModelPolicyFactory | None,
) -> None:
    """Reject async scheduling when PLE history is built on the CPU."""
    if factory is None or factory.cpu_token_history is None:
        return
    text_config = _text_config(vllm_config)
    if text_config is None or not getattr(text_config, "ple_layer_ids", None):
        return
    if not factory.cpu_token_history(vllm_config):
        return
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if (
        scheduler_config is None
        or getattr(scheduler_config, "async_scheduling", None) is not True
    ):
        return
    name = factory.name or "This model"
    raise ModelPolicyError(
        f"{name} enables PLE and its runner builds the PLE n-gram token history "
        "on the CPU, which is incompatible with asynchronous scheduling. "
        "Re-run with --no-async-scheduling."
    )


def validate_model_config(vllm_config: Any) -> None:
    """Validate a model's *resolved* configuration.

    Runs after vLLM's own config resolution, so the final values (including
    ``scheduler_config.async_scheduling``) are visible.  Dispatches to the
    matching factory's validation hook and applies the common late checks.
    It never mutates dispatch state or touches CUDA.
    """
    factory = _find_factory(vllm_config)
    if factory is not None and factory.validate is not None:
        factory.validate(vllm_config)
    _validate_final_async_scheduling(vllm_config, factory)


def reset_model_policy_for_tests() -> None:
    """Clear the registry and the active plan (test-only)."""
    global _active_runtime_plan
    with _lock:
        _active_runtime_plan = None
        _factories.clear()
