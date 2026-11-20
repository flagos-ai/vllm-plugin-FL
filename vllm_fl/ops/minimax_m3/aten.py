# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Per-worker ATen bindings via FlagGems' public registrar extension point."""

from flag_gems.ops.masked_fill import masked_fill_
from flag_gems.runtime.op_registrar import GeneralOpRegistrar

from .linear import linear

_OVERRIDES = {
    "linear": linear,
    "masked_fill_.Scalar": masked_fill_,
    "masked_fill_.Tensor": masked_fill_,
}


class MiniMaxRegistrar(GeneralOpRegistrar):
    def __init__(self, config, **kwargs):
        def substitute(rows):
            return tuple(
                (row[0], _OVERRIDES.get(row[0], row[1]), *row[2:]) for row in rows
            )

        rows = substitute(config)
        mapping = kwargs.get("full_config_by_func")
        if mapping is not None:
            kwargs["full_config_by_func"] = {
                name: substitute(items) for name, items in mapping.items()
            }
        super().__init__(rows, **kwargs)
        self.selected_functions = {row[0]: row[1] for row in self.config}
        for key, expected in _OVERRIDES.items():
            if key in self.get_all_keys():
                assert self.selected_functions[key] is expected
