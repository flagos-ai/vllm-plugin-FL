# SPDX-License-Identifier: Apache-2.0
"""Bind model adapters through the common resolver without hiding GPU failures."""

from .policy import PREFER_DEFAULT, get_policy, get_policy_epoch


class OperatorBinding:
    """A cached common-dispatch binding, refreshed when public policy changes.

    Workload guards select before execution. Every execution error propagates;
    a second implementation must never repeat a cache or state update.
    """

    def __init__(
        self,
        manager,
        op_name,
        *,
        graph_capabilities=None,
        supports=None,
        default_order=(),
    ):
        self.manager = manager
        self.op_name = op_name
        self.graph_capabilities = graph_capabilities or {}
        self.supports = supports or {}
        self.default_order = default_order
        self._cache = None
        self.selected_impl = None

    def preflight(self):
        policy = get_policy()
        epoch = (get_policy_epoch(), self.manager.policy_epoch, policy)
        cached = self._cache
        if cached is None or epoch != cached[0]:
            candidates = self.manager.resolve_candidates(self.op_name)
            failed = self.manager.get_failed_impls(self.op_name).get(
                self.op_name, set()
            )
            candidates = [impl for impl in candidates if impl.impl_id not in failed]
            self.selected_impl = None
            epoch = (get_policy_epoch(), self.manager.policy_epoch, policy)
            self._cache = (epoch, candidates)
        else:
            candidates = cached[1]
        if not candidates:
            raise RuntimeError(
                f"No permitted implementation remains for {self.op_name}"
            )
        return candidates

    def describe(self):
        candidates = self.preflight()
        strict = get_policy().strict
        return dict(
            op=self.op_name,
            selected=self.selected_impl or candidates[0].impl_id,
            candidates=[impl.impl_id for impl in candidates],
            strict=strict,
            fallback_on="never",
            graph_capabilities=self.graph_capabilities,
        )

    def __call__(self, *args, **kwargs):
        candidates = self.preflight()
        policy = get_policy()
        # Workload guards inspect metadata only. Unsupported shapes do not
        # disable an implementation for later calls with a supported shape.
        candidates = [
            impl
            for impl in candidates
            if impl.impl_id not in self.supports
            or self.supports[impl.impl_id](*args, **kwargs)
        ]
        explicit = dict(policy.per_op_order).get(self.op_name)
        if self.default_order and explicit is None and policy.prefer == PREFER_DEFAULT:
            order = {name: i for i, name in enumerate(self.default_order)}
            candidates.sort(key=lambda impl: order.get(impl.impl_id, len(order)))
        if not candidates:
            raise NotImplementedError(
                f"No permitted implementation supports {self.op_name}"
            )
        impl = candidates[0]
        self.manager._record_first_use(self.op_name, impl)
        result = self.manager._call_with_hooks(self.op_name, impl.fn, args, kwargs)
        self.selected_impl = impl.impl_id
        return result
