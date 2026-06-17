from .glm_moe_dsa import apply_platform_patches as glm5_platform
from .rank_logging import apply_rank_logging_patch

def apply_all_patches():
    glm5_platform()
    apply_rank_logging_patch()
