# SPDX-License-Identifier: Apache-2.0
"""Process-level model activation plans.

A *plan* is the set of model-scoped, process-global side effects that a model
integration needs before its modules are constructed: class-level monkey
patches, an attention-backend override, and FlagOS dispatch defaults.  Keeping
them here (rather than in plugin registration) means a plain ``import
vllm_fl`` never rewrites the environment or the generic dispatch for models
that did not ask for it.

Rules enforced by this module:

* At most one plan is active per process.  Activating the same plan again is
  idempotent; activating a different plan -- or a model that needs no plan --
  raises before any side effect.  Activation invalidates the cached dispatch
  policy so plan defaults take effect.
* Class-level patches are preflighted before any of them is applied, validated
  with a basic parameter-name check, recorded with the original implementation
  / vLLM version / signature, and rolled back if a later install in the same
  activation fails.
* Plan-provided FlagOS defaults never override an explicit user selection.  A
  user selection whose primary implementation cannot satisfy the plan's
  required semantics raises instead of being rewritten or silently accepted.
"""

from __future__ import annotations

import inspect
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ActivationConflict",
    "ActivationPlan",
    "MoEDispatchDefaults",
    "PendingPatch",
    "activate",
    "activate_for_model",
    "bind_patches",
    "effective_flaggems_whitelist",
    "get_active_plan",
    "get_active_moe_defaults",
    "is_plan_active",
    "merge_per_op_defaults",
    "preflight_activation_config",
    "preflight_patches",
    "patch_inventory",
    "temporary_patches",
    "register_plan_provider",
    "reset_activation_for_tests",
    "resolve_model_plan",
    "validate_flaggems_whitelist",
    "validate_plan_capability",
]


class ActivationConflict(RuntimeError):
    """Raised when a plan or patch would silently replace existing state."""


@dataclass(frozen=True)
class MoEDispatchDefaults:
    """FlagOS dispatch defaults contributed by a model plan.

    Attributes:
        whitelist_ops: Ops the model requires when a flaggems whitelist is in
            force.  A non-empty user whitelist that omits any of them is a
            capability conflict and aborts startup.
        per_op_order: ``op_name -> preferred order``.  Only applied to ops the
            user did not configure explicitly, and takes precedence over an
            auto-detected platform fallback.
        required_impls: ``op_name -> implementation tokens``.  The *primary*
            (first) token of an explicit user order must be one of them; a
            fallback entry does not make an incompatible primary acceptable.
    """

    whitelist_ops: tuple[str, ...] = ()
    per_op_order: tuple[tuple[str, tuple[str, ...]], ...] = ()
    required_impls: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def required_for(self, op_name: str) -> tuple[str, ...] | None:
        for name, required in self.required_impls:
            if name == op_name:
                return required
        return None

    def order_for(self, op_name: str) -> tuple[str, ...] | None:
        for name, order in self.per_op_order:
            if name == op_name:
                return order
        return None

    def is_empty(self) -> bool:
        return not (self.whitelist_ops or self.per_op_order or self.required_impls)


@dataclass(frozen=True)
class ActivationPlan:
    """A single, process-wide model activation."""

    name: str
    fingerprint: str
    apply: Callable[[], None]
    attention_backend: Callable[..., str | None] | None = None
    moe_defaults: MoEDispatchDefaults = field(default_factory=MoEDispatchDefaults)


# ---------------------------------------------------------------------------
# Patch registry: preflight, signature validation, idempotency, rollback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingPatch:
    """A class-attribute patch to install atomically with its siblings."""

    target: str
    owner: Any
    attr: str
    replacement: Any
    fingerprint: str
    pristine: Any
    expected_params: tuple[str, ...] = ()
    # Optional accessors for attributes that are not plain fields (e.g. a
    # classmethod, where every ``getattr`` returns a fresh bound method, so
    # identity must be compared through ``__func__``).
    get_current: Callable[[], Any] | None = None
    undo: Callable[[], None] | None = None
    phase: str = "worker"

    def current(self) -> Any:
        if self.get_current is not None:
            return self.get_current()
        return getattr(self.owner, self.attr)

    def restore(self) -> None:
        if self.undo is not None:
            self.undo()
        else:
            setattr(self.owner, self.attr, self.pristine)


@dataclass(frozen=True)
class _PatchRecord:
    target: str
    fingerprint: str
    original: Any
    installed: Any
    version: str
    signature: str
    phase: str = "worker"


_patch_records: dict[str, _PatchRecord] = {}
_patch_lock = threading.RLock()


def _vllm_version() -> str:
    try:
        return version("vllm")
    except PackageNotFoundError:  # pragma: no cover - vLLM must be installed
        return "unknown"


def _signature_text(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return repr(value)


def _check_signature(patch: PendingPatch) -> None:
    """Check expected parameter names, not full Python calling-convention compatibility."""
    if not patch.expected_params:
        return
    if not callable(patch.replacement):
        raise ActivationConflict(
            f"Patch {patch.target!r} replacement is not callable: {patch.replacement!r}"
        )
    try:
        parameters = inspect.signature(patch.replacement).parameters
    except (TypeError, ValueError):
        return
    for name in patch.expected_params:
        if name not in parameters:
            raise ActivationConflict(
                f"Patch {patch.target!r} replacement {_signature_text(patch.replacement)} "
                f"does not accept required parameter {name!r}"
            )


def preflight_patches(patches: Iterable[PendingPatch]) -> list[PendingPatch]:
    """Validate patches without installing anything.

    Returns the subset that still needs installing.  Raises
    :class:`ActivationConflict` if any target is owned by another fingerprint or
    was already modified away from its pristine value, or if a replacement
    omits an expected parameter name.
    """
    to_apply: list[PendingPatch] = []
    targets: set[str] = set()
    destinations: set[tuple[int, str]] = set()
    with _patch_lock:
        for patch in patches:
            destination = (id(patch.owner), patch.attr)
            if patch.target in targets or destination in destinations:
                raise ActivationConflict(
                    f"Duplicate patch destination for {patch.target!r} in one batch"
                )
            targets.add(patch.target)
            destinations.add(destination)
            record = _patch_records.get(patch.target)
            if record is not None:
                if record.fingerprint == patch.fingerprint:
                    if patch.current() is not record.installed:
                        raise ActivationConflict(
                            f"Patch {patch.target!r} was modified after installation; "
                            "refusing to treat it as an idempotent activation"
                        )
                    continue
                raise ActivationConflict(
                    f"Conflicting activation for {patch.target!r}: already owned "
                    f"by {record.fingerprint!r} (vllm {record.version}, "
                    f"sig={record.signature}); refusing to replace it with "
                    f"{patch.fingerprint!r}"
                )
            current = patch.current()
            if current is not patch.pristine:
                raise ActivationConflict(
                    f"Refusing to patch {patch.target!r}: its current value is "
                    f"not the pristine implementation (got {current!r}); another "
                    f"owner already modified it"
                )
            _check_signature(patch)
            to_apply.append(patch)
    return to_apply


def bind_patches(patches: Iterable[PendingPatch]) -> int:
    """Preflight and install a group of class-attribute patches atomically.

    If an install fails midway, already-installed attributes are restored to
    their pristine values and their records removed, so a failed activation
    leaves no partial side effect.

    Returns the number of attributes installed.
    """
    # Keep preflight, installation and rollback under the same lock. Otherwise
    # another activation can acquire ownership between the check and write.
    with _patch_lock:
        to_apply = preflight_patches(patches)
        applied: list[PendingPatch] = []
        try:
            for patch in to_apply:
                setattr(patch.owner, patch.attr, patch.replacement)
                applied.append(patch)
                _patch_records[patch.target] = _PatchRecord(
                    target=patch.target,
                    fingerprint=patch.fingerprint,
                    original=patch.pristine,
                    installed=patch.current(),
                    version=_vllm_version(),
                    signature=_signature_text(patch.replacement),
                    phase=patch.phase,
                )
        except Exception:
            for patch in reversed(applied):
                patch.restore()
                _patch_records.pop(patch.target, None)
            logger.exception("Rolled back %d partial patch(es)", len(applied))
            raise
    return len(to_apply)


def patch_inventory() -> list[dict[str, str]]:
    """Serializable ownership for registration, construction and worker hooks."""
    with _patch_lock:
        return [
            {
                key: getattr(record, key)
                for key in ("target", "fingerprint", "phase", "version", "signature")
            }
            for record in _patch_records.values()
        ]


@contextmanager
def temporary_patches(patches: Iterable[PendingPatch]):
    """Use the same conflict checks and rollback for constructor-only hooks.

    These process-global hooks are serialized with other patch transactions;
    callers must use them only during single-model worker construction.
    """
    with _patch_lock:
        pending = preflight_patches(patches)
        bind_patches(pending)
        try:
            yield
        finally:
            for patch in reversed(pending):
                patch.restore()
                _patch_records.pop(patch.target, None)


# ---------------------------------------------------------------------------
# Single active plan guard
# ---------------------------------------------------------------------------

_active_plan: ActivationPlan | None = None
_plan_lock = threading.RLock()
_plan_providers: list[Callable[[Any], ActivationPlan | None]] = []

# Sentinel bound when a model needs no plan.  It records that the process has
# been initialized for a model, so a later request for a *different* model
# (including one that does need a plan) is rejected instead of being allowed to
# apply process-global patches on top of a plain model.
_EMPTY_PLAN = ActivationPlan(
    name="<no-plan>", fingerprint="<no-plan>", apply=lambda: None
)


def register_plan_provider(
    provider: Callable[[Any], ActivationPlan | None],
    *,
    replace: bool = False,
) -> None:
    """Register a model plan provider (benign at plugin import time)."""
    with _plan_lock:
        if replace:
            _plan_providers.clear()
        if provider not in _plan_providers:
            _plan_providers.append(provider)


def get_active_plan() -> ActivationPlan | None:
    return _active_plan


def is_plan_active() -> bool:
    return _active_plan is not None


def get_active_moe_defaults() -> MoEDispatchDefaults | None:
    plan = _active_plan
    if plan is None or plan.moe_defaults.is_empty():
        return None
    return plan.moe_defaults


def _invalidate_dispatch_policy() -> None:
    """Rebuild the cached dispatch policy so plan defaults take effect.

    Explicitly configured policies (``set_global_policy``) are preserved and
    re-merged; only the environment-derived cache is dropped.
    """
    from vllm_fl.dispatch.policy import PolicyManager

    PolicyManager.get_instance().invalidate_policy_cache()


def activate(plan: ActivationPlan) -> bool:
    """Activate ``plan`` for this process.

    Returns True if this call installed the plan, False if the same plan was
    already active.  Raises :class:`ActivationConflict` if a different plan is
    active; the caller's plan is not applied in that case.
    """
    global _active_plan
    with _plan_lock:
        if _active_plan is not None:
            if _active_plan.fingerprint == plan.fingerprint:
                return False
            raise ActivationConflict(
                f"Conflicting model plan: {_active_plan.name!r} "
                f"({_active_plan.fingerprint}) is already active; cannot "
                f"activate {plan.name!r} ({plan.fingerprint}) in the same process"
            )
        plan.apply()
        # Cache invalidation is required for the new defaults. A failure must
        # abort startup before the plan can be reported as active/idempotent.
        _invalidate_dispatch_policy()
        _active_plan = plan
        logger.info("Activated model plan %s (%s)", plan.name, plan.fingerprint)
        return True


def resolve_model_plan(vllm_config: Any) -> ActivationPlan | None:
    """Return the plan for ``vllm_config`` without applying any side effect."""
    with _plan_lock:
        providers = list(_plan_providers)
    for provider in providers:
        plan = provider(vllm_config)
        if plan is not None:
            return plan
    return None


def preflight_activation_config(
    vllm_config: Any,
    whitelist: Iterable[str] | None = None,
) -> ActivationPlan | None:
    """Resolve the plan and validate startup config before any side effect.

    Returns the requested plan.  Raises :class:`ActivationConflict` if an
    explicit FlagGems whitelist omits an op the resolved plan requires, so the
    error is raised before the plan is applied (no partial class patches).
    """
    plan = resolve_model_plan(vllm_config)
    validate_flaggems_whitelist(
        whitelist, plan.moe_defaults if plan is not None else None
    )
    return plan


def activate_for_model(vllm_config: Any) -> ActivationPlan | None:
    """Bind the process to ``vllm_config``'s plan, detecting conflicting models.

    The requested plan is resolved *before* comparing with the active plan, so
    a process initialized for one model cannot silently accept another --
    including a change from a plain model (bound to the explicit empty plan) to
    a model that needs global patches.
    """
    requested = resolve_model_plan(vllm_config)
    activate(requested if requested is not None else _EMPTY_PLAN)
    return requested


def _impl_token(impl: Any) -> str:
    """Map a resolved OpImpl to the token used by plan capability requirements."""
    from vllm_fl.dispatch.types import BackendImplKind

    if impl.kind == BackendImplKind.DEFAULT:
        return "flagos"
    if impl.kind == BackendImplKind.REFERENCE:
        return "reference"
    if impl.kind == BackendImplKind.VENDOR:
        return f"vendor:{impl.vendor}"
    return str(impl.kind)


def _reachable(candidates: list[Any], order: Sequence[str]) -> list[Any]:
    from vllm_fl.dispatch.types import match_token

    return [c for c in candidates if any(match_token(c, token) for token in order)]


def validate_plan_capability(
    plan: ActivationPlan | None,
    resolve_candidates: Callable[[str], list[Any]],
    policy_order_for: Callable[[str], Sequence[str] | None] | None = None,
) -> None:
    """Check that the *actually selectable* implementation satisfies the plan.

    Static per-op order checks cannot see blacklists, allow/deny sets or
    availability.  This runs the real resolver (after the registry is built)
    and evaluates the candidates under every order that could select them:
    an explicit user order first (if any), then the plan's own order (which
    overrides a platform fallback).  Under each order the implementation(s)
    that could be selected must map to a token the plan accepts; an order that
    resolves to nothing compatible aborts startup instead of failing or
    silently degrading at runtime.
    """
    if plan is None or plan.moe_defaults.is_empty():
        return
    for op_name, required in plan.moe_defaults.required_impls:
        try:
            candidates = resolve_candidates(op_name)
        except Exception as exc:
            raise ActivationConflict(
                f"The active model requires op {op_name!r}, but no candidate "
                f"could be resolved: {exc}"
            ) from exc
        if not candidates:
            raise ActivationConflict(
                f"The active model requires op {op_name!r}, but the resolver "
                f"returned no candidates"
            )

        orders: list[list[str]] = []
        if policy_order_for is not None:
            explicit = policy_order_for(op_name)
            if explicit:
                orders.append(list(explicit))
        plan_order = plan.moe_defaults.order_for(op_name)
        if plan_order:
            orders.append(list(plan_order))
        if not orders:
            orders.append([])  # empty order -> resolver default: all candidates

        for order in orders:
            reachable = _reachable(candidates, order) if order else list(candidates)
            if not reachable:
                raise ActivationConflict(
                    f"No implementation for required op {op_name!r} under order "
                    f"{order or '<default>'}; candidates="
                    f"{[c.impl_id for c in candidates]}. The order may exclude "
                    f"every implementation that satisfies the active model."
                )
            for candidate in reachable:
                token = _impl_token(candidate)
                if token not in required:
                    raise ActivationConflict(
                        f"Implementation {candidate.impl_id!r} ({token}) is "
                        f"reachable for {op_name!r} (order {order or '<default>'}) "
                        f"but cannot satisfy the active model's required "
                        f"semantics {list(required)}. Adjust the allow/deny, "
                        f"whitelist/blacklist or per-op configuration; it will "
                        f"not be rewritten automatically."
                    )


def reset_activation_for_tests() -> None:
    """Clear worker activation state; retain process bootstrap ownership (test-only)."""
    global _active_plan
    with _plan_lock:
        _active_plan = None
        _plan_providers.clear()
    with _patch_lock:
        for target, record in list(_patch_records.items()):
            if record.phase != "engine/config":
                del _patch_records[target]


# ---------------------------------------------------------------------------
# Plan defaults vs. explicit user selection
# ---------------------------------------------------------------------------


def merge_per_op_defaults(
    defaults: MoEDispatchDefaults,
    explicit_order: Mapping[str, Sequence[str]] | None,
    fallback_order: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Merge plan per-op defaults under explicit/fallback user selections.

    Precedence (highest first): explicit user order, plan default, fallback
    (auto-detected platform config).  For an op the plan marks as
    capability-required, the *primary* (first) token of an explicit order must
    satisfy the requirement; a fallback token later in the list is not enough.
    """
    _raise_on_capability_conflict(defaults, explicit_order)

    merged: dict[str, list[str]] = {
        name: list(order) for name, order in (explicit_order or {}).items()
    }
    for name, order in defaults.per_op_order:
        merged.setdefault(name, list(order))
    for name, order in (fallback_order or {}).items():
        merged.setdefault(name, list(order))
    return merged


def _raise_on_capability_conflict(
    defaults: MoEDispatchDefaults,
    explicit_order: Mapping[str, Sequence[str]] | None,
) -> None:
    if not explicit_order:
        return
    for op_name, required in defaults.required_impls:
        given = explicit_order.get(op_name)
        if not given:
            continue
        if given[0] not in required:
            raise ActivationConflict(
                f"Explicit per-op order for {op_name!r} starts with "
                f"{given[0]!r}, which cannot satisfy the active model's required "
                f"semantics (needs primary one of {list(required)}); a later "
                f"fallback entry does not make it valid. Fix the user "
                f"configuration; it will not be rewritten automatically."
            )


def validate_flaggems_whitelist(
    whitelist: Iterable[str] | None,
    defaults: MoEDispatchDefaults | None = None,
) -> list[str] | None:
    """Validate that a user flaggems whitelist covers the plan's required ops.

    An unset whitelist already enables every op and is returned as-is.  A set
    whitelist that omits a required op raises instead of being silently
    extended, so the FlagGems runtime configuration and the dispatch registry
    always agree.
    """
    if whitelist is None:
        return None
    resolved = list(whitelist)
    active = defaults if defaults is not None else get_active_moe_defaults()
    if active is None:
        return resolved
    missing = [op for op in active.whitelist_ops if op not in resolved]
    if missing:
        raise ActivationConflict(
            "VLLM_FL_FLAGOS_WHITELIST excludes operators the active model "
            f"requires: {', '.join(missing)}. Add them to the whitelist or "
            "unset it (an unset whitelist enables all ops)."
        )
    return resolved


def effective_flaggems_whitelist() -> list[str] | None:
    """Return the env whitelist after plan validation (single source of truth).

    Used by both the FlagGems runtime enable step and the dispatch registry
    filter so an explicit whitelist and the registered implementations cannot
    disagree.
    """
    from vllm_fl.utils import get_flag_gems_whitelist_blacklist

    whitelist, _ = get_flag_gems_whitelist_blacklist()
    return validate_flaggems_whitelist(whitelist)
