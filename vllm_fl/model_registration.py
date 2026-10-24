"""General-plugin entry point, loaded after vLLM platform initialization."""

from importlib.metadata import version
from vllm_fl.patches.kimi_k3.profile import is_kimi_k3_profile
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm_fl import register_model as register_common_model_extensions

# Platform loading must not import model kernels before schemas/platform exist.
# vLLM's canonical loader imports exactly the selected backend module here.
_MODEL_EXTENSIONS = {"metax": "vllm_fl.patches.kimi_k3.registration.register"}


def register_model():
    target = _MODEL_EXTENSIONS.get(getattr(current_platform, "vendor_name", None))
    if target is not None and is_kimi_k3_profile():
        installed = version("vllm").split("+", 1)[0]
        if installed != "0.28.0":
            raise RuntimeError(f"K3 integration requires vLLM 0.28.0, got {installed}")
        resolve_obj_by_qualname(target)()
    register_common_model_extensions()
