# SPDX-License-Identifier: Apache-2.0
"""Model-scoped runtime policy plans.

Public entry points used by the common layer:

* :func:`validate_model_config` -- final per-model config validation.
* :func:`build_model_runtime_plan` -- pure :class:`RuntimePlan` construction.
* :func:`register_model_policy_factory` -- model adapter registration.

See :mod:`vllm_fl.runtime.model_policy` for the ``ActivationPlan`` /
``RuntimePlan`` boundary.
"""

from vllm_fl.runtime.model_policy import (
    ModelPolicyError,
    ModelPolicyFactory,
    RuntimePlan,
    activate_runtime_plan,
    build_model_runtime_plan,
    get_active_runtime_plan,
    preflight_runtime_plan,
    register_model_policy_factory,
    reset_model_policy_for_tests,
    validate_model_config,
)

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
