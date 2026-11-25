# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagEmbedding-compatible BGE-M3 outputs through vLLM's IO plugin API."""

from contextvars import ContextVar
from copy import deepcopy
from threading import Lock

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams


class BgeM3EncodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    sentences: str | list[str] = Field(
        validation_alias=AliasChoices("sentences", "input")
    )
    return_dense: bool | None = True
    return_sparse: bool | None = False
    return_colbert_vecs: bool | None = False
    max_length: int | None = Field(default=512, ge=2, le=8192)

    @model_validator(mode="after")
    def validate_sentences(self):
        if isinstance(self.sentences, list) and not self.sentences:
            raise ValueError("sentences must contain at least one text")
        return self

    @property
    def output_mask(self):
        flags = (
            True if self.return_dense is None else self.return_dense,
            False if self.return_sparse is None else self.return_sparse,
            False if self.return_colbert_vecs is None else self.return_colbert_vecs,
        )
        return sum((1 << i) for i, enabled in enumerate(flags) if enabled)


class BgeM3IOProcessor(IOProcessor):
    def __init__(self, vllm_config, renderer):
        super().__init__(vllm_config, renderer)
        # HF fast tokenizers mutate truncation settings. Keep a private copy so
        # concurrent native embedding requests cannot change this contract.
        self.tokenizer = deepcopy(renderer.get_tokenizer())
        self._tokenizer_lock = Lock()
        self.hidden_size = vllm_config.model_config.hf_config.hidden_size
        self.max_model_len = vllm_config.model_config.max_model_len
        self.unused_token_ids = {
            token_id
            for name in ("cls_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
            if (token_id := getattr(self.tokenizer, name, None)) is not None
        }
        # vLLM invokes pre_process and merge_pooling_params in the same thread.
        # Only two small scalars are retained, never input text or request maps.
        self._selection = ContextVar("bge_m3_selection", default=None)

    def parse_data(self, data):
        self._selection.set(None)
        return BgeM3EncodeRequest.model_validate(data)

    def pre_process(self, prompt, request_id=None, **kwargs):
        single = isinstance(prompt.sentences, str)
        texts = [prompt.sentences] if single else prompt.sentences
        max_length = prompt.max_length if prompt.max_length is not None else 512
        if max_length > self.max_model_len:
            raise ValueError(
                f"max_length={max_length} exceeds model limit {self.max_model_len}"
            )
        # Tokenize here to match HF's right truncation with EOS preserved.
        # Passing token prompts prevents a second tokenization by the renderer.
        with self._tokenizer_lock:
            batch = self.tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
        self._selection.set((prompt.output_mask, single))
        return [{"prompt_token_ids": ids} for ids in batch["input_ids"]]

    def merge_pooling_params(self, params=None):
        selection = self._selection.get()
        if selection is None:
            raise ValueError("BGE-M3 preprocessing must precede pooling parameters")
        params = PoolingParams() if params is None else params.clone()
        if params.dimensions is not None or params.use_activation is False:
            raise ValueError("BGE-M3 requires full dimensions and normalization")
        mask, single = selection
        params.task = "plugin"
        params.use_activation = True
        params.skip_reading_prefix_cache = True
        params.extra_kwargs = {
            **(params.extra_kwargs or {}),
            "bge_m3": {"outputs": mask, "single_input": single},
        }
        return params

    def post_process(self, model_output, request_id=None, **kwargs):
        if not model_output:
            raise ValueError("BGE-M3 received no model outputs")
        result = {"dense_vecs": None, "lexical_weights": None, "colbert_vecs": None}
        keys = tuple(result)
        first_selection = None
        for output in model_output:
            payload = output.outputs.data
            if payload.ndim != 1 or payload.numel() < 2:
                raise ValueError("Invalid BGE-M3 pooled output")
            raw_mask, raw_single = payload[:2].tolist()
            mask, single = int(raw_mask), bool(raw_single)
            if raw_mask != mask or not 0 <= mask <= 7 or raw_single not in (0, 1):
                raise ValueError("Invalid BGE-M3 output selection header")
            if first_selection is None:
                first_selection = (mask, single)
                for i, key in enumerate(keys):
                    if mask & (1 << i):
                        result[key] = []
            elif first_selection != (mask, single):
                raise ValueError("Mixed selections within one BGE-M3 response")
            ids = output.prompt_token_ids
            offset = 2
            if mask & 1:
                dense = payload[offset : offset + self.hidden_size]
                result["dense_vecs"].append(dense.tolist())
                offset += self.hidden_size
            if mask & 2:
                weights = payload[offset : offset + len(ids)].tolist()
                lexical = {}
                for token_id, weight in zip(ids, weights):
                    if token_id not in self.unused_token_ids and weight > 0:
                        key = str(token_id)
                        lexical[key] = max(lexical.get(key, 0.0), weight)
                result["lexical_weights"].append(lexical)
                offset += len(ids)
            if mask & 4:
                count = (len(ids) - 1) * self.hidden_size
                colbert = payload[offset : offset + count].reshape(-1, self.hidden_size)
                result["colbert_vecs"].append(colbert.tolist())
                offset += count
            if offset != payload.numel():
                raise ValueError("BGE-M3 output size does not match input token count")
        if first_selection[1]:
            if len(model_output) != 1:
                raise ValueError("Single-text BGE-M3 response contains multiple inputs")
            result = {
                key: None if value is None else value[0]
                for key, value in result.items()
            }
        return result
