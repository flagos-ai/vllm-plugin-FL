# SPDX-License-Identifier: Apache-2.0
"""Explicit single-stream execution used by the MetaX DS adapter."""


def run_pair_serially(fn0, fn1, event0=None, event1=None, aux_stream=None):
    return fn0(), fn1()


def run_branches_serially(
    default_fn,
    aux_fns,
    start_event=None,
    done_events=None,
    aux_streams=None,
    enable=False,
):
    default_result = default_fn()
    return default_result, [fn() if fn is not None else None for fn in aux_fns]
