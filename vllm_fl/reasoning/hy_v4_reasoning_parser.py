# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Reasoning parser for HY4 checkpoints (``hy_v4`` / ``hy_v4_vl``).

The paired vLLM runtime ships a ``hy_v4`` entry in its lazy parser table, but
the module it points at does not exist in the image, so ``--reasoning-parser
hy_v4`` fails with ``KeyError: Reasoning parser 'hy_v4' not found``. This
module ports the reference implementation
(``vllm/reasoning/hy_v4_reasoning_parser.py`` in the upstream HY4 tree) into
plugin-FL and registers it under the same name.

What it does: HY4 wraps its chain of thought in ``<think:SUFFIX>`` /
``</think:SUFFIX>``, where the suffix is a per-checkpoint marker (``:opensource``
for the released weights). The chat template **injects the start token at the
end of the prompt**, so the generated text usually starts *inside* reasoning and
only the end token ever appears. In ``no_think`` mode the whole output is
content.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, TypedDict

import regex as re

from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.tokenizers import TokenizerLike

try:  # vLLM 0.24 lives under ``entrypoints.openai``
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
except ImportError:  # pragma: no cover - newer layouts
    from vllm.entrypoints.generate.base.protocol import DeltaMessage

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)

# Must match the chat template's own default for ``reasoning_effort``.
DEFAULT_REASONING_EFFORT = "high"


class ReasoningDelta(TypedDict):
    """One streaming reasoning delta. Either/both fields may be None."""

    reasoning: str | None
    content: str | None


_STRUCTURAL_TOKEN_RE = re.compile(
    r"<(?:think|tool_calls|tool_call|arg_key|arg_value)(:[^\s>]+)?>"
)


def detect_token_suffix(tokenizer: TokenizerLike) -> str:
    """Detect the per-checkpoint suffix used by HY4 structural tokens.

    Returns the suffix including its leading colon (e.g. ``":opensource"``),
    or ``""`` when the checkpoint uses unsuffixed tokens.
    """
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    think_begin_as_special = init_kwargs.get("model_specific_special_tokens", {}).get(
        "think_begin_token", ""
    )
    if think_begin_as_special:
        raise RuntimeError(
            "This checkpoint declares HYV4 structural tokens (think_begin_token"
            "/toolcalls_begin_token/argkey_begin_token) in "
            "tokenizer_config.json, which transformers 5 no longer supports. "
            "Remove those fields and keep the tokens in the tokenizer's own "
            "token definitions so the suffix can be read from the vocab."
        )

    for token in tokenizer.get_vocab():
        match = _STRUCTURAL_TOKEN_RE.fullmatch(token)
        if match:
            return match.group(1) or ""

    return ""


class HYV4ReasoningExtractor:
    """Reasoning extraction for HY4, on plain data.

    ``thinking`` selects the mode:
    - ``True``: parse ``<think>...</think>``; the start token is injected at the
      end of the prompt and is therefore usually absent from the output.
    - ``False``: ``no_think`` -- the whole output is content, no reasoning.
    """

    def __init__(self, vocab: dict[str, int], token_suffix: str, thinking: bool = True):
        self.thinking = thinking
        self.start_token: str = f"<think{token_suffix}>"
        self.end_token: str = f"</think{token_suffix}>"

        start_token_id = vocab.get(self.start_token)
        end_token_id = vocab.get(self.end_token)
        if thinking and (start_token_id is None or end_token_id is None):
            raise RuntimeError(
                "HYV4 reasoning extractor could not locate think "
                f"start/end tokens ({self.start_token!r}/{self.end_token!r}) "
                "in the tokenizer!"
            )
        # In no_think mode the ids are unused (kept for symmetry / debugging).
        self.start_token_id: int | None = start_token_id
        self.end_token_id: int | None = end_token_id

    def has_reasoning_ended(self, token_ids: Sequence[int]) -> bool:
        """Idempotent state query: has ``</think>`` been emitted yet?"""
        if not self.thinking:
            return True
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == self.start_token_id:
                return False
            if token_ids[i] == self.end_token_id:
                return True
        return False

    def reasoning_ended_in_delta(
        self, token_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """Edge detector: did reasoning end *in this* step?"""
        if not self.thinking:
            return True
        end_token_id = self.end_token_id
        if end_token_id is None:
            return False
        return end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return the token ids after ``</think>`` (all of them in no_think)."""
        if not self.thinking:
            return input_ids
        head = input_ids[: max(0, len(input_ids) - 1)]
        if self.end_token_id not in head:
            return []
        return input_ids[input_ids.index(self.end_token_id) + 1 :]

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count reasoning tokens (0 in no_think)."""
        if not self.thinking:
            return 0
        start = 1 if (token_ids and token_ids[0] == self.start_token_id) else 0
        try:
            return token_ids.index(self.end_token_id, start) - start
        except ValueError:
            # No end token -- reasoning was truncated; everything counts.
            return len(token_ids) - start

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        """Split a complete output into ``(reasoning, content)``."""
        if not self.thinking:
            return None, model_output

        parts = model_output.partition(self.start_token)
        model_output = parts[2] if parts[1] else parts[0]

        if self.end_token not in model_output:
            return model_output, None
        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> ReasoningDelta | None:
        """Classify a streaming delta as reasoning and/or content."""
        if not self.thinking:
            if not delta_text:
                return None
            return ReasoningDelta(reasoning=None, content=delta_text)

        # Skip a delta that is a single lone start/end special token.
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            self.start_token_id,
            self.end_token_id,
        ):
            return None

        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return ReasoningDelta(reasoning=reasoning, content=content or None)
            elif self.end_token_id in previous_token_ids:
                return ReasoningDelta(reasoning=None, content=delta_text)
            else:
                return ReasoningDelta(reasoning=delta_text, content=None)
        elif self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[start_index + len(self.start_token) : end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return ReasoningDelta(reasoning=reasoning, content=content or None)
            else:
                return ReasoningDelta(reasoning=delta_text, content=None)
        else:
            # HY4 injects <think> at the end of the prompt, so it never appears
            # in the output: treat the stream as reasoning until </think>.
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return ReasoningDelta(reasoning=reasoning, content=content or None)
            elif self.end_token_id in previous_token_ids:
                return ReasoningDelta(reasoning=None, content=delta_text)
            else:
                return ReasoningDelta(reasoning=delta_text, content=None)


def build_reasoning_extractor(
    tokenizer: TokenizerLike, *, thinking: bool = True
) -> HYV4ReasoningExtractor:
    return HYV4ReasoningExtractor(
        tokenizer.get_vocab(), detect_token_suffix(tokenizer), thinking
    )


class HYV4ReasoningParser(ReasoningParser):
    """vLLM adapter around :class:`HYV4ReasoningExtractor`.

    Decides ``thinking`` from the request's ``reasoning_effort`` and converts
    the extractor's ``ReasoningDelta`` to ``DeltaMessage``.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # Prefer chat_template_kwargs.reasoning_effort, then the top-level
        # reasoning_effort. The default must match the chat template, which
        # sets reasoning_effort='high' when the request omits it -- defaulting
        # to no_think here would return the whole chain-of-thought (and a stray
        # ``</think:SUF>``) as content.
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        reasoning_effort = chat_kwargs.get("reasoning_effort", None)
        if not reasoning_effort:
            reasoning_effort = (
                kwargs.get("reasoning_effort") or DEFAULT_REASONING_EFFORT
            )
        logger.debug("reasoning_effort for choosing parser: %s", reasoning_effort)
        thinking = reasoning_effort != "no_think"

        self._extractor = build_reasoning_extractor(tokenizer, thinking=thinking)

    # Reported regardless of the per-request thinking mode: ``ReasoningConfig``
    # is engine-wide and needs the checkpoint's delimiters to derive the token
    # ids used for thinking budget enforcement.
    @property
    def reasoning_start_str(self) -> str:
        return self._extractor.start_token

    @property
    def reasoning_end_str(self) -> str:
        return self._extractor.end_token

    def is_reasoning_end(self, input_ids) -> bool:
        return self._extractor.has_reasoning_ended(input_ids)

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return self._extractor.reasoning_ended_in_delta(input_ids, delta_ids)

    def extract_content_ids(self, input_ids) -> list[int]:
        return self._extractor.extract_content_ids(input_ids)

    def count_reasoning_tokens(self, token_ids) -> int:
        return self._extractor.count_reasoning_tokens(token_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        return self._extractor.extract_reasoning(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ) -> DeltaMessage | None:
        delta = self._extractor.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if delta is None:
            return None
        return DeltaMessage(reasoning=delta["reasoning"], content=delta["content"])


_PARSER_NAMES = ("hy_v4", "hy_v4_vl")


def register_hy4_reasoning_parsers() -> bool:
    """Register the HY4 reasoning parser under ``hy_v4`` / ``hy_v4_vl``."""
    from vllm.reasoning import ReasoningParserManager

    ReasoningParserManager.register_module(
        name=list(_PARSER_NAMES),
        module=HYV4ReasoningParser,
        force=True,
    )
    logger.info(
        "Registered HY4 reasoning parser for %s", ", ".join(_PARSER_NAMES)
    )
    return True


__all__ = [
    "HYV4ReasoningExtractor",
    "HYV4ReasoningParser",
    "build_reasoning_extractor",
    "detect_token_suffix",
    "register_hy4_reasoning_parsers",
]
