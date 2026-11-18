# Copyright (c) 2026 BAAI. All rights reserved.
"""Unlimited-OCR model identity and R-SWA configuration lookup."""

ARCHITECTURE = "UnlimitedOCRForCausalLM"
MODEL_TYPE = "unlimited-ocr"
CHECKPOINT_MODEL_TYPE = "unlimited_ocr"


def is_unlimited_ocr(model_config) -> bool:
    config = getattr(model_config, "hf_config", None)
    return getattr(config, "model_type", None) in (
        MODEL_TYPE,
        CHECKPOINT_MODEL_TYPE,
    ) and ARCHITECTURE in (getattr(config, "architectures", None) or ())


def get_ocr_rswa_window(model_config) -> int | None:
    if not is_unlimited_ocr(model_config):
        return None
    return getattr(model_config.hf_config, "rswa_window", None)
