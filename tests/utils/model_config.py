# Copyright (c) 2025 BAAI. All rights reserved.

"""
Shared model configuration loader.

Reads model YAML configs from ``tests/models/<model>/<case>.yaml``
(or legacy ``tests/models/<name>.yaml``) and provides typed interfaces
for engine parameters, test generation, and multimodal asset loading.

Usage::

    from tests.utils.model_config import ModelConfig

    # New directory layout: tests/models/qwen3/06b_tp2.yaml
    cfg = ModelConfig.load("qwen3", "06b_tp2")

    # For offline inference (LLM constructor)
    llm = LLM(**cfg.engine_kwargs())

    # For serving (vllm serve CLI args)
    server = VllmServer(
        model=cfg.model,
        tp_size=cfg.engine.get("tensor_parallel_size", 1),
        extra_args=cfg.serve_args(),
    )

    # Access generate config for smoke tests
    gen = cfg.generate
    print(gen.modality, gen.prompts, gen.assets)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


# ---------------------------------------------------------------------------
# Generate config (test-specific)
# ---------------------------------------------------------------------------


@dataclass
class GenerateConfig:
    """Test generation configuration parsed from the ``generate`` section."""

    modality: str = "text"
    prompts: list[Any] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    sampling: dict[str, Any] = field(default_factory=dict)
    parametrize: dict[str, list] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GenerateConfig:
        return cls(
            modality=raw.get("modality", "text"),
            prompts=raw.get("prompts", []),
            assets=raw.get("assets", []),
            sampling=raw.get("sampling", {}),
            parametrize=raw.get("parametrize", {}),
        )

    def get_parametrize_combos(self) -> list[dict[str, Any]]:
        """Return Cartesian product of parametrize dimensions.

        Example::

            parametrize:
              enforce_eager: [true, false]
              dtype: [bfloat16, float16]

            → [
                {"enforce_eager": True, "dtype": "bfloat16"},
                {"enforce_eager": True, "dtype": "float16"},
                {"enforce_eager": False, "dtype": "bfloat16"},
                {"enforce_eager": False, "dtype": "float16"},
              ]

        Returns a list with one empty dict if no parametrize is defined.
        """
        if not self.parametrize:
            return [{}]

        keys = list(self.parametrize.keys())
        values = [self.parametrize[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Serve config (test-specific)
# ---------------------------------------------------------------------------


@dataclass
class ServeConfig:
    """Serving test configuration parsed from the ``serve`` section.

    Fields:
        api_key: Optional API key for authenticated endpoints.
        extra_engine: Engine param overrides for serving (e.g. dtype).
        endpoints: List of endpoints to test (``"completion"``, ``"chat"``,
                   ``"embedding"``).
        completion_prompt: Prompt string for ``/v1/completions`` endpoint.
        chat_messages: Messages list for ``/v1/chat/completions`` endpoint.
        max_tokens: Max tokens for serving requests.
        served_model_name: Alias passed via ``--served-model-name``.
            Empty string means use the model path directly.
        stream: Whether to test streaming responses (chat endpoint).
        sampling: Sampling parameters (temperature, top_p, etc.) injected
            into the request payload.
        extra_body: Free-form dict passed as ``extra_body`` to the OpenAI SDK
            or merged into the request JSON (e.g. top_k, chat_template_kwargs).
        embedding_input: Input text for ``/v1/embeddings`` endpoint tests.
    """

    api_key: str = ""
    extra_engine: dict[str, Any] = field(default_factory=dict)
    endpoints: list[str] = field(default_factory=list)
    completion_prompt: str = "Hello"
    chat_messages: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 50
    served_model_name: str = ""
    stream: bool = False
    sampling: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    embedding_input: str = ""

    def request_model(self, model_path: str) -> str:
        """Return the model name to use in API requests."""
        return self.served_model_name or model_path

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ServeConfig:
        return cls(
            api_key=raw.get("api_key", ""),
            extra_engine=raw.get("extra_engine", {}),
            endpoints=raw.get("endpoints", []),
            completion_prompt=raw.get("completion_prompt", "Hello"),
            chat_messages=raw.get("chat_messages", []),
            max_tokens=raw.get("max_tokens", 50),
            served_model_name=raw.get("served_model_name", ""),
            stream=raw.get("stream", False),
            sampling=raw.get("sampling", {}),
            extra_body=raw.get("extra_body", {}),
            embedding_input=raw.get("embedding_input", ""),
        )


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Shared model configuration loaded from YAML.

    Supports two YAML layouts:

    **New layout** (``tests/models/<model>/<case>.yaml``)::

        llm:
          model: /data/models/...
          max_model_len: 8192
          ...
        generate:
          prompts: [...]
          sampling: {max_tokens: 5}

    **Legacy layout** (``tests/models/<name>.yaml``)::

        model_path: /data/models/...
        engine:
          max_model_len: 8192
        sampling:
          max_tokens: 5
    """

    model: str
    engine: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)

    @classmethod
    def load(
        cls,
        model: str,
        case: str | None = None,
        models_dir: Path | None = None,
    ) -> ModelConfig:
        """Load model config from YAML.

        Args:
            model: Model family name (e.g. ``"qwen3"``) or legacy flat name.
            case: Case name within the model directory (e.g. ``"06b_tp2"``).
                  If ``None``, falls back to legacy flat file ``<model>.yaml``.
            models_dir: Override directory for model configs.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if models_dir is None:
            models_dir = _MODELS_DIR

        if case is not None:
            path = models_dir / model / f"{case}.yaml"
        else:
            # Legacy flat layout
            path = models_dir / f"{model}.yaml"

        if not path.exists():
            raise FileNotFoundError(
                f"Model config not found: {path}. "
                f"Check tests/models/ directory structure."
            )

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        """Parse a raw YAML dict into ModelConfig.

        Handles both new (llm/generate) and legacy (engine/sampling) layouts.
        """
        # New layout: llm + generate sections
        if "llm" in raw:
            llm_raw = dict(raw.get("llm", {}))
            model = llm_raw.pop("model", "")
            engine = llm_raw
            gen_raw = raw.get("generate", {})
            generate = GenerateConfig.from_dict(gen_raw)
            sampling = gen_raw.get("sampling", {})
        else:
            # Legacy layout: engine + sampling at top level
            model = raw.get("model_path", "")
            engine = raw.get("engine", {})
            sampling = raw.get("sampling", {})
            generate = GenerateConfig(sampling=sampling)

        # Serve config (optional)
        serve_raw = raw.get("serve", {})
        serve = ServeConfig.from_dict(serve_raw) if serve_raw else ServeConfig()

        return cls(
            model=model,
            engine=engine,
            sampling=sampling,
            generate=generate,
            serve=serve,
        )

    def engine_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return engine params as Python kwargs for ``LLM()`` constructor.

        Any keyword arguments override the YAML values.
        """
        params = {"model": self.model, **self.engine}
        params.update(overrides)
        return params

    def serve_args(self, **overrides: Any) -> list[str]:
        """Return engine params as CLI args for ``vllm serve``.

        Converts Python-style kwargs to CLI flags::

            tensor_parallel_size=8 → ["--tensor-parallel-size", "8"]
            enforce_eager=True     → ["--enforce-eager"]

        The ``tensor_parallel_size`` key is excluded because
        ``VllmServer`` handles it via the ``tp_size`` parameter.
        """
        params = {**self.engine}
        params.update(overrides)

        # model and tp_size are handled by VllmServer directly
        params.pop("model", None)
        params.pop("tensor_parallel_size", None)

        args: list[str] = []
        for key, value in params.items():
            flag = "--" + key.replace("_", "-")

            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])

        return args

    def sampling_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return sampling params as Python kwargs for ``SamplingParams()``.

        Any keyword arguments override the YAML values.
        """
        params = {**self.sampling}
        params.update(overrides)
        return params
