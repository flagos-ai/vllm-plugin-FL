# SPDX-License-Identifier: Apache-2.0
"""GLM startup before text or vision CustomOp instances are constructed."""

import vllm_fl.activation as activation
import vllm_fl.envs as fl_envs
from vllm_fl.dispatch import get_default_manager
from vllm_fl.dispatch.policy import PolicyManager
from vllm_fl.flaggems_runtime import configure_flaggems
from vllm_fl.ops.custom_ops import register_oot_ops
from vllm_fl.models.glm5.runtime import validate_glm5_config
from vllm_fl.utils import get_flag_gems_whitelist_blacklist


def initialize_glm5_worker(vllm_config, *, rank: int) -> bool:
    """Initialize GLM once; return False for the common worker startup path.

    Validate the requested policy before applying model patches or enabling
    FlagGems. Dispatch continues to read the shared, mutable public policy;
    this hook does not publish a second runtime plan or freeze default orders.
    The empty activation binds ordinary models too, preventing global patches
    from leaking between different models in one process.
    """
    whitelist = None
    if fl_envs.USE_FLAGGEMS:
        whitelist, _ = get_flag_gems_whitelist_blacklist()
    plan = activation.preflight_activation_config(vllm_config, whitelist)
    if plan is None:
        activation.activate_for_model(vllm_config)
        return False

    validate_glm5_config(vllm_config)
    register_oot_ops()
    policy_manager = PolicyManager.get_instance()
    policy = policy_manager.policy_for_plan(plan.moe_defaults)
    with policy_manager.create_policy_context(policy):
        activation.validate_plan_capability(
            plan,
            get_default_manager().resolve_candidates,
            policy_order_for=policy.get_per_op_order,
        )
    activation.activate_for_model(vllm_config)

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    whitelist = activation.validate_flaggems_whitelist(whitelist)

    def enable_flaggems(library):
        import flag_gems

        kwargs = dict(
            record=rank == 0,
            once=True,
            path=fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH,
        )
        if library is not None:
            kwargs["lib"] = library
        if whitelist is not None:
            flag_gems.only_enable(include=whitelist, **kwargs)
        elif blacklist:
            flag_gems.enable(unused=blacklist, **kwargs)
        else:
            flag_gems.enable(**kwargs)

    configure_flaggems(
        enable_flaggems,
        use_flaggems=fl_envs.USE_FLAGGEMS,
        whitelist=whitelist,
        blacklist=blacklist,
    )
    return True
