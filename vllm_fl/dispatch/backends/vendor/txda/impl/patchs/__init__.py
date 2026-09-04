# Copyright (c) 2026 BAAI. All rights reserved.

# Each patch is imported independently: one failing patch must not abort
# the whole package. An ImportError here propagates to the txda package
# __init__, then to register_ops, which the dispatcher swallows at DEBUG
# level — silently disabling every txda patch and the vendor op
# registrations along with them. Log loudly and keep going instead.

import logging

_logger = logging.getLogger(__name__)

for _patch in (
    "apply_repetition_penalties",  # route sampler to pure-torch penalties
):
    try:
        __import__(f"{__name__}.{_patch}")
    except Exception:
        _logger.exception(
            "Failed to apply txda patch '%s'; continuing with remaining "
            "patches", _patch,
        )
