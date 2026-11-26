"""Avoid priming stock ByteLevel DecodeStream across a complete UTF-8 boundary."""
from functools import wraps

import tokenizers
from tokenizers.decoders import ByteLevel, DecodeStream


def has_complete_tail(tokenizer, ids, skip_special_tokens):
    if not ids:
        return False
    for token_id in reversed(ids[-16:]):
        text = tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens)
        if text:
            return not text.endswith("\ufffd")
    return False


def can_unseed(tokenizer, ids, skip_special_tokens):
    return (tokenizers.__version__ == "0.22.2"
            and type(tokenizer) is tokenizers.Tokenizer
            and type(tokenizer.decoder) is ByteLevel
            and tokenizers.decoders.DecodeStream is DecodeStream
            and DecodeStream.__module__ == "tokenizers.decoders"
            and has_complete_tail(tokenizer, ids, skip_special_tokens))


def install(cls, enabled=lambda: True, witness=lambda _: None):
    original = cls.__init__

    @wraps(original)
    def initialize(self, tokenizer, request):
        original(self, tokenizer, request)
        if enabled() and type(self.stream) is DecodeStream and can_unseed(
                self.tokenizer, request.prompt_token_ids, self.skip_special_tokens):
            self.stream = DecodeStream(skip_special_tokens=self.skip_special_tokens)
            witness(len(request.prompt_token_ids))

    cls.__init__ = initialize


def install_vllm():
    from vllm.logger import init_logger
    from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

    cls = FastIncrementalDetokenizer
    if getattr(cls.__init__, "_fl_exact_bytelevel", False):
        return
    logger = init_logger("vllm.flagos.bytelevel")
    seen = False

    def witness(length):
        nonlocal seen
        if not seen:
            seen = True
            logger.info("FL exact ByteLevel empty stream active prompt_tokens=%d", length)

    install(cls, witness=witness)
    cls.__init__._fl_exact_bytelevel = True
