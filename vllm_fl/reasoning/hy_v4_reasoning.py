# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

"""Reasoning parser for HYV4's opensource-suffixed thinking tokens."""

from vllm.reasoning.hy_v3_reasoning_parser import HYV3ReasoningParser


class HYV4ReasoningParser(HYV3ReasoningParser):
    """Split HYV4 reasoning from the user-visible final content.

    HYV4 preserves HYV3's ``reasoning_effort`` behavior but its tokenizer and
    chat template use ``<think:opensource>`` tokens. The inherited HYV3 parser
    handles both ordinary responses and the no-think identity path.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        """Match the parser default to HYV4's chat-template default (high)."""
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        if not chat_kwargs.get("reasoning_effort") and not kwargs.get(
            "reasoning_effort"
        ):
            kwargs["reasoning_effort"] = "high"
        super().__init__(tokenizer, *args, **kwargs)

    @property
    def start_token(self) -> str:
        """Token that starts HYV4 reasoning."""
        return "<think:opensource>"

    @property
    def end_token(self) -> str:
        """Token that ends HYV4 reasoning."""
        return "</think:opensource>"
