"""Read the existing K3 full-graph capture context without GPU data access."""

from vllm.forward_context import get_forward_context, is_forward_context_available


def in_full_graph():
    if not is_forward_context_available():
        return False
    context = get_forward_context()
    # Existing K3 capture wrapper uses inner NONE plus this legacy-named hint.
    return (
        context.additional_kwargs.get(
            "cudagraph_capture_mode", context.cudagraph_runtime_mode.name
        )
        == "FULL"
    )
