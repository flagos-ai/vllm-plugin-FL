# Claude Code Skills for vllm-plugin-FL

This directory contains [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills and project-level settings for the vllm-plugin-FL project.

> **Note**: This directory is checked in as `claude/` for version control visibility. To use it locally, rename (or symlink) it to `.claude/`:
>
> ```bash
> # Option 1: Rename
> mv claude .claude
>
> # Option 2: Symlink (keeps both accessible)
> ln -s claude .claude
> ```

## Skills

### model-migrate-fl

Migrate a model from the latest vLLM upstream into the vllm-plugin-FL project (pinned at vLLM v0.13.0).

#### When to Use

- Adding support for a new model to vllm-plugin-FL
- Porting / back-porting model code from upstream vLLM (main branch) to the v0.13.0-compatible plugin

#### Invocation

In a Claude Code session inside the project root:

```
/model-migrate-fl <model_name> [upstream_folder] [plugin_folder]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `model_name` | Yes | — | snake_case model identifier, e.g. `qwen3_5`, `kimi_k25`, `deepseek_v4` |
| `upstream_folder` | No | `/tmp/vllm-upstream-ref` | Path to a local clone of the latest vLLM upstream |
| `plugin_folder` | No | current working directory | Path to the vllm-plugin-FL project |

#### Examples

```bash
# Basic — migrate Qwen3.5, auto-clone upstream to /tmp
/model-migrate-fl qwen3_5

# Specify an existing upstream clone
/model-migrate-fl kimi_k25 /path/to/vllm-upstream

# Specify both upstream and plugin paths
/model-migrate-fl deepseek_v4 /path/to/vllm-upstream /path/to/vllm-plugin-FL
```

#### What It Does

The skill executes the following steps automatically:

1. **Baseline unit tests** — runs tests before any changes to establish a clean baseline
2. **Clone / update upstream vLLM** — fetches the latest model code for reference
3. **Investigate model** — resolves model_type, class names, config shape from upstream + HuggingFace
4. **Study existing patterns** — picks the best migration strategy based on similar models already in the plugin
5. **Create config bridge** — adds a `vllm_fl/configs/<model>.py` if the model_type is new to 0.13.0
6. **Create model file (copy-then-patch)** — copies the upstream model file and applies v0.13.0 compatibility patches (P1–P5+)
7. **Register model** — adds config + model class to `vllm_fl/__init__.py`
8. **Regression unit tests** — re-runs tests, compares with baseline, fixes any regressions
9. **Functional tests** — runs functional tests, reports pass/skip/fail
10. **Benchmark verification** — runs `vllm bench throughput` with the new model
11. **Serve + request verification** — starts a vLLM server and sends a test chat completion request

#### Compatibility Patches

The skill maintains a catalog of known vLLM 0.13.0 incompatibilities (`references/compatibility-patches.md`):

| Patch | Description |
|---|---|
| P1 | Relative imports → absolute imports |
| P2 | Config imports → plugin config bridges |
| P3 | Remove APIs missing in 0.13.0 (e.g. `MambaStateCopyFunc`) |
| P4 | Replace `_mark_tower_model` / `_mark_language_model` context managers |
| P5 | Import verification |

New patches (P6, P7, ...) are appended as new incompatibilities are discovered.

#### Prerequisites

- **vLLM 0.13.0** installed in the environment
- **vllm-plugin-FL** installed with `pip install -e .`
- Internet access (to clone upstream vLLM) or a local copy of the latest vLLM source
- Python 3.8+

#### File Structure

```
claude/
  settings.local.json                          # Permission settings for Claude Code
  skills/
    model-migrate-fl/
      SKILL.md                                 # Skill entry point & orchestration logic
      references/
        procedure.md                           # Step-by-step migration procedure
        compatibility-patches.md               # vLLM 0.13.0 patch catalog
```
