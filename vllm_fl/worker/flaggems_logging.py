"""Control optional FlagGems op-list logging without changing dispatch."""

import logging


def configure_oplist_recording(additional_config, rank):
    """Keep the legacy default; explicit false skips LogRecord construction."""
    record = (additional_config or {}).get("flaggems_record_ops", True)
    if type(record) is not bool:
        raise ValueError("additional_config.flaggems_record_ops must be a bool")
    if not record:
        # record=False alone does not reset a previously enabled FG DEBUG logger.
        # Preserve stricter user levels and all INFO/WARNING/error diagnostics.
        logger = logging.getLogger("flag_gems")
        if logger.isEnabledFor(logging.DEBUG):
            logger.setLevel(logging.INFO)
    return record and rank == 0
