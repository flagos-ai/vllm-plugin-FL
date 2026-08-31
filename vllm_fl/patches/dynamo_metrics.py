"""Compatibility for Torch serializing FlagGems' logging-function set."""


def patch_dynamo_metrics_serialization() -> None:
    import torch._dynamo.utils as dynamo_utils

    if getattr(dynamo_utils, "_fl_metrics_serialization_patched", False):
        return
    original = dynamo_utils._get_dynamo_config_for_logging

    def get_dynamo_config_for_logging():
        # FlagGems registers a set of logging functions in the Dynamo config;
        # serializing it raises TypeError. The config is logging metadata only,
        # so dropping it is harmless. Wrap unconditionally because the
        # offending entry may be registered after this patch is installed.
        try:
            return original()
        except TypeError:
            return None

    dynamo_utils._get_dynamo_config_for_logging = get_dynamo_config_for_logging
    dynamo_utils._fl_metrics_serialization_patched = True
