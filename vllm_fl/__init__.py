# Copyright (c) 2025 BAAI. All rights reserved.

def register():
    """Register the FL platform."""

    return "vllm_fl.platform.PlatformFL"


# def register_connector():
#     from vllm_ascend.distributed import register_connector
#     register_connector()
