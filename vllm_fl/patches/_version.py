from importlib.metadata import PackageNotFoundError, version


def is_vllm_024() -> bool:
    """Return whether the installed vLLM belongs to the 0.24 release line."""
    try:
        release = version("vllm").split("+", 1)[0].split(".")
    except PackageNotFoundError:
        return False
    return len(release) >= 2 and release[:2] == ["0", "24"]


__all__ = ["is_vllm_024"]
