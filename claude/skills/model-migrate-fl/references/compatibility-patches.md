# vLLM 0.13.0 Compatibility Patches

Apply these patches after copying upstream model files into the plugin. Not all patches apply to every model — apply only those relevant to the specific model being migrated.

## P1: Relative Imports → Absolute Imports

Upstream files use relative imports that break when moved to the plugin. Convert them:

- **Intra-plugin references** (files that also live in `vllm_fl/models/`) use `vllm_fl.models.*`
- **All other relative imports** use the vLLM 0.13.0 absolute path `vllm.*`

```python
# Before (upstream relative imports):
from .interfaces import ...
from .qwen3_next import ...
from .qwen3_vl import ...
from .qwen2_moe import ...
from .utils import ...

# After (absolute imports):
from vllm.model_executor.models.interfaces import ...
from vllm_fl.models.qwen3_next import ...           # plugin file → vllm_fl
from vllm.model_executor.models.qwen3_vl import ...  # vLLM core → vllm
from vllm.model_executor.models.qwen2_moe import ... # vLLM core → vllm
from vllm.model_executor.models.utils import ...     # vLLM core → vllm
```

**Rule**: If a module also lives in `vllm_fl/models/` (because the plugin may have fixes for it), import from `vllm_fl.models.*`. Otherwise, import from `vllm.model_executor.models.*`.

## P2: Config Imports

Upstream config imports point to paths that don't exist in 0.13.0. Redirect to the plugin's config bridges:

```python
# Before:
from vllm.transformers_utils.configs.qwen3_5 import ...
from vllm.transformers_utils.configs.qwen3_5_moe import ...

# After:
from vllm_fl.configs.qwen3_5 import ...
from vllm_fl.configs.qwen3_5_moe import ...
```

## P3: Remove Missing APIs

These APIs don't exist in 0.13.0 and must be removed:

- `MambaStateCopyFunc` and `MambaStateCopyFuncCalculator` — remove from imports
- `get_mamba_state_copy_func` classmethod — remove the entire method (it references the missing types)

## P4: Replace Context Manager Init Pattern

`_mark_tower_model` and `_mark_language_model` context managers are not available in 0.13.0. Replace with direct initialization:

```python
# Before (upstream context manager pattern):
with self._mark_tower_model(vllm_config, {"image", "video"}):
    self.visual = Qwen3_VisionTransformer(config.vision_config, ...)
with self._mark_language_model(vllm_config):
    self.language_model = XxxForCausalLM(...)

# After (0.13.0 compatible direct init):
if not multimodal_config.get_limit_per_prompt("image") and not multimodal_config.get_limit_per_prompt("video"):
    self.visual = None
else:
    self.visual = Qwen3_VisionTransformer(
        config.vision_config, ...,
        multimodal_config=multimodal_config, ...
    )
self.language_model = XxxForCausalLM(...)
```

Key: add `multimodal_config=multimodal_config` to the `Qwen3_VisionTransformer()` constructor.

## P5: Import Verification

After applying all patches, verify the model imports correctly:

```bash
python3 -c "from vllm_fl.models.{{model_name_lower}} import {{ModelClassName}}; print('OK')"
```

If this fails, fix the specific import error and retry.

## When Patches Aren't Enough

If the model uses APIs not covered above:

1. Test the specific failing import: `python3 -c "from xxx import yyy"`
2. Check if an equivalent exists in 0.13.0 (inspect `/usr/local/lib/python*/dist-packages/vllm/`)
3. If truly missing, stub it out or remove the dependent code
4. Only read the 0.13.0 source file when comparing a specific method signature

## Adding New Patches

When a new incompatibility is discovered, append it as P6, P7, etc. Include:

- **What** to change (before/after code example)
- **Why** it's needed (what's missing or different in 0.13.0)
- **When** this patch applies (which model architectures or patterns trigger it)
