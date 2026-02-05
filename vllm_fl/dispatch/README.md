# Dispatch Mechanism

This directory implements the operator dispatch mechanism for vllm-plugin-FL, providing a flexible operator dispatch system that selects between different backend implementations (FlagGems, PyTorch, vendor-specific) based on availability and policy configuration.

## Directory Structure

```
dispatch/
├── __init__.py              # Module entry point, exports public API
├── types.py                 # Core type definitions (OpImpl, BackendImplKind)
├── registry.py              # Thread-safe operator registry
├── policy.py                # Selection policy management
├── manager.py               # Core dispatch manager
├── builtin_ops.py           # Built-in operator registration
├── ops.py                   # Backend base interface
├── discovery.py             # Plugin discovery mechanism
├── logger_manager.py        # Centralized logging configuration
├── config/                  # Platform-specific configurations
│   ├── __init__.py          # Config loader module
│   ├── ascend.yaml          # Ascend NPU default configuration
│   └── cuda.yaml            # CUDA default configuration
└── backends/                # Backend implementations
    ├── base.py              # Backend abstract base class
    ├── flaggems/            # FlagGems backend (DEFAULT, priority 150)
    │   ├── flaggems.py      # Backend class
    │   ├── register_ops.py  # Registration function
    │   └── impl/            # Operator implementations
    │       ├── activation.py
    │       ├── normalization.py
    │       ├── rotary.py
    │       ├── attention.py       # AttentionFLBackend, AttentionFLImpl
    │       ├── mla.py             # MLAFLBackend, MLAFLImpl
    │       └── custom_attention.py # Attention backend registration
    ├── reference/           # Reference backend (PyTorch, priority 50)
    └── vendor/              # Vendor-specific backends (priority 100)
        ├── cuda/            # NVIDIA CUDA backend
        │   └── impl/
        │       ├── activation.py
        │       ├── normalization.py
        │       └── rotary.py
        └── ascend/          # Huawei Ascend NPU backend
            └── impl/
                ├── activation.py
                ├── normalization.py
                ├── rotary.py
                ├── attention.py       # AscendAttentionBackend
                └── attention_mask.py  # Attention mask utilities
```

## Core Concepts

### 1. Backend Implementation Kind

- **DEFAULT**: Default implementation (FlagGems), priority 150
- **VENDOR**: Vendor-specific implementation, priority 100
- **REFERENCE**: Reference implementation (PyTorch native), priority 50

### 2. Operator Implementation (OpImpl)

Each operator implementation contains:
- `op_name`: Operator name (e.g., "silu_and_mul", "rms_norm")
- `impl_id`: Unique implementation identifier (e.g., "default.flagos")
- `kind`: Implementation type
- `fn`: Actual implementation function
- `vendor`: Vendor name (required for VENDOR type)
- `priority`: Selection priority (higher value = preferred)

### 3. Selection Policy

Policy controls operator implementation selection:
- `prefer`: Preferred implementation type
- `strict`: Strict mode, whether to raise error when primary implementation fails
- `per_op_order`: Custom selection order for each operator
- `deny_vendors`: List of denied vendors
- `allow_vendors`: Whitelist of allowed vendors

## Architecture Overview

### Dispatch Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Code                                │
│                 call_op("rms_norm", x, ...)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OpManager                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Check Cache                                            │  │
│  │ 2. Get Policy (from env or context)                      │  │
│  │ 3. Query Registry for all implementations                │  │
│  │ 4. Filter by vendor allow/deny list                      │  │
│  │ 5. Check availability (is_available())                   │  │
│  │ 6. Sort by priority & selection order                    │  │
│  │ 7. Cache & return selected implementation                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OpRegistry                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   FlagGems   │  │    Vendor    │  │  Reference   │         │
│  │ Priority: 150│  │ Priority: 100│  │ Priority: 50 │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Priority Selection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     VLLM_FL_PREFER=flagos                       │
│                    (Default Behavior)                            │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────┴────────────────────┐
        │                                          │
        ▼                                          ▼
┌──────────────┐  Available?  ┌──────────────┐  Available?
│   FlagGems   │─────No──────▶│    Vendor    │─────No──────▶
│ Priority: 150│              │ Priority: 100│
└──────────────┘              └──────────────┘
        │                              │
       Yes                            Yes
        │                              │
        ▼                              ▼
    ✓ Selected                    ✓ Selected

                                                  ┌──────────────┐
                                                  │  Reference   │
                                                  │ Priority: 50 │
                                                  └──────────────┘
                                                         │
                                                        Yes
                                                         │
                                                         ▼
                                                    ✓ Selected
```

### Plugin Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    Plugin Discovery                              │
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │   Built-in     │  │  Entry Points  │  │  Environment   │   │
│  │   backends/    │  │  (setuptools)  │  │  PLUGIN_MODULES│   │
│  │   vendor/      │  │                │  │                │   │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘   │
│           │                   │                    │            │
│           └───────────────────┴────────────────────┘            │
│                               │                                  │
└───────────────────────────────┼──────────────────────────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │   Registry    │
                        │  register()   │
                        └───────────────┘
```

## Quick Start

### Basic Usage

```python
from vllm_fl.dispatch import call_op, resolve_op

# Method 1: Call operator directly
result = call_op("silu_and_mul", x)

# Method 2: Resolve first, then call
fn = resolve_op("rms_norm")
result = fn(x, residual, weight, epsilon)
```

### Using the Manager

```python
from vllm_fl.dispatch import get_default_manager

manager = get_default_manager()

# Resolve operator
fn = manager.resolve("rotary_embedding")
result = fn(query, key, cos, sin, position_ids)

# Or call directly
result = manager.call("silu_and_mul", x)
```

## Configuration

The dispatch system supports multiple ways to configure backend selection:
1. **User-specified configuration file (YAML)** - Complete override
2. **Environment variables** - Override specific items
3. **Platform-specific configuration file** - Auto-detected defaults
4. **Built-in default values**

### Configuration Priority

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Priority                        │
│                  (Highest to Lowest)                             │
├─────────────────────────────────────────────────────────────────┤
│  1. VLLM_FL_CONFIG        │ User config file, complete override │
│  2. Environment Variables │ Override specific items              │
│  3. Platform Config File  │ ascend.yaml / cuda.yaml defaults     │
│  4. Built-in Defaults     │ Code-defined default values          │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Environment variables can override specific items from platform config
- If user doesn't set any environment variable, platform config is used
- Users can also modify platform config files directly

### Platform-Specific Configuration

The system automatically detects hardware and loads the corresponding configuration file from `config/` directory:

| Platform | Config File | Auto-Detection |
|----------|-------------|----------------|
| Ascend NPU | `config/ascend.yaml` | `torch.npu.is_available()` |
| NVIDIA GPU | `config/cuda.yaml` | `torch.cuda.is_available()` |

You can force a specific platform using `VLLM_FL_PLATFORM` environment variable:
```bash
export VLLM_FL_PLATFORM=ascend  # Force Ascend config
export VLLM_FL_PLATFORM=cuda    # Force CUDA config
```

### User-Specified Configuration File (YAML)

Set the `VLLM_FL_CONFIG` environment variable to specify a YAML configuration file that completely overrides all other settings:

```bash
export VLLM_FL_CONFIG=/path/to/vllm_fl_dispatch.yaml
```

#### Example Configuration File

```yaml
# vllm_fl_dispatch.yaml

# Preferred backend type: flagos, vendor, or reference
prefer: vendor

# Strict mode:
#   true  = fail immediately on error, no fallback
#   false = try next backend on failure (default)
strict: false

# Vendor whitelist (optional)
allow_vendors:
  - cuda

# Vendor blacklist (optional)
deny_vendors:
  - ascend

# Per-operator backend selection order (optional)
# Only the backends listed will be tried, in the specified order.
op_backends:
  rms_norm:
    - vendor        # Try any available vendor first
    - flagos        # Then try flagos
    # reference not listed, so it won't be used for rms_norm

  silu_and_mul:
    - vendor:cuda   # Only try CUDA, not other vendors
    - flagos
    - reference

# FlagGems operator blacklist (optional)
# These operators will NOT use FlagGems implementation
flagos_blacklist:
  - to_copy
  - zeros
  - mm

# OOT operator blacklist (optional)
# These operators will NOT be registered as OOT replacements
oot_blacklist:
  - fused_moe
```

#### Token Types Explained

| Token | Description |
|-------|-------------|
| `flagos` | FlagOS default implementation |
| `reference` | PyTorch reference implementation |
| `vendor` | Any available vendor backend (auto-detects hardware) |
| `vendor:cuda` | Only CUDA vendor backend |
| `vendor:ascend` | Only Ascend vendor backend |

**Note**: When using `vendor` (without specifying a vendor name), the system automatically selects an available vendor backend based on hardware detection.

<a id="environment-variables"></a>
### Environment Variables

Environment variables can override specific items from platform config. If not set, values from platform config file are used.

#### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FL_PREFER_ENABLED` | `true` | Global switch. Set `false` to disable all dispatch features |
| `VLLM_FL_CONFIG` | (none) | Path to YAML config file (complete override) |
| `VLLM_FL_PLATFORM` | (auto) | Force platform: `ascend`, `cuda` |

#### Backend Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FL_PREFER` | `flagos` | Preferred backend: `flagos`, `vendor`, `reference` |
| `VLLM_FL_STRICT` | `0` | Strict mode: `1` = fail on error, `0` = try fallback |
| `VLLM_FL_PER_OP` | (none) | Per-operator order: `op1=a\|b\|c;op2=x\|y` |
| `VLLM_FL_ALLOW_VENDORS` | (none) | Vendor whitelist, comma-separated |
| `VLLM_FL_DENY_VENDORS` | (none) | Vendor blacklist, comma-separated |

#### FlagGems Control

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FLAGGEMS` | `true` | Enable/disable FlagGems |
| `VLLM_FL_FLAGOS_WHITELIST` | (none) | FlagGems ops whitelist (mutually exclusive with blacklist) |
| `VLLM_FL_FLAGOS_BLACKLIST` | (none) | FlagGems ops blacklist (mutually exclusive with whitelist) |

**Priority**: `WHITELIST` > `BLACKLIST` (env) > `flagos_blacklist` (config file)

#### OOT Operator Control

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FL_OOT_ENABLED` | `1` | Enable OOT operator registration |
| `VLLM_FL_OOT_WHITELIST` | (none) | OOT ops whitelist |
| `VLLM_FL_OOT_BLACKLIST` | (none) | OOT ops blacklist |

**Priority**: `WHITELIST` > `BLACKLIST` (env) > `oot_blacklist` (config file)

#### Debug & Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FL_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `VLLM_FL_DISPATCH_DEBUG` | `0` | Enable dispatch debug mode |

#### Plugins

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FL_PLUGIN_MODULES` | (none) | External plugin modules, comma-separated |
| `VLLM_FL_OP_CONFIG` | (none) | Operator config JSON file path |

#### Other

| Variable | Default | Description |
|----------|---------|-------------|
| `FLAGCX_PATH` | (none) | FlagCX library path (enables FlagCX communication backend) |
| `FLAGGEMS_ENABLE_OPLIST_PATH` | `/tmp/flaggems_enable_oplist.txt` | FlagGems enabled ops list file |

### Examples

```bash
# Use platform default config (auto-detected)
# Nothing to set - just run your application

# Override only the prefer setting (other items from platform config)
export VLLM_FL_PREFER=vendor

# Override FlagGems blacklist (overrides config file blacklist)
export VLLM_FL_FLAGOS_BLACKLIST="mm,to_copy,zeros"

# Use whitelist instead (completely ignores any blacklist)
export VLLM_FL_FLAGOS_WHITELIST="silu_and_mul,rms_norm"

# Specify per-operator order
export VLLM_FL_PER_OP="rms_norm=vendor|flagos|reference"

# Use completely custom config file
export VLLM_FL_CONFIG=/path/to/my_config.yaml

# Force specific platform
export VLLM_FL_PLATFORM=ascend

# Enable debug logging
export VLLM_FL_LOG_LEVEL=DEBUG
```

#### Op Backends Selection Example

```yaml
op_backends:
  mul:
    - flagos
  silu_and_mul:
    - flagos
    - vendor
    - reference
```

### Configuration Priority Details

The dispatch system applies configuration in the following order:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Configuration Resolution                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  VLLM_FL_CONFIG set?                                                 │
│       │                                                               │
│       ├── Yes ──▶ Use user config file (complete override)           │
│       │                                                               │
│       └── No ──▶ For each setting:                                   │
│                       │                                               │
│                       ├── Env var set? ──▶ Use env var value         │
│                       │                                               │
│                       └── Not set ──▶ Use platform config value      │
│                                              │                        │
│                                              └── Not found ──▶ Default│
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

#### Whitelist vs Blacklist Priority

For FlagGems and OOT operators:

```
WHITELIST (env) ──▶ Completely overrides blacklist
       │
       └── Not set ──▶ BLACKLIST (env) ──▶ Overrides config blacklist
                              │
                              └── Not set ──▶ Config file blacklist
                                                    │
                                                    └── Not set ──▶ Allow all
```

**Important Notes:**
- Whitelist and blacklist environment variables are mutually exclusive (error if both set)
- If whitelist is set, it completely ignores any blacklist (env or config)
- Environment blacklist overrides config file blacklist (not merged)

#### Example: Combined Environment Variables

```bash
# Platform config (ascend.yaml) has:
#   prefer: flagos
#   flagos_blacklist: [to_copy, zeros, mm, ...]

# User overrides only prefer, blacklist still from config
export VLLM_FL_PREFER=vendor

# Result:
#   prefer: vendor (from env)
#   flagos_blacklist: [to_copy, zeros, mm, ...] (from config)
```

```bash
# User wants to override blacklist too
export VLLM_FL_PREFER=vendor
export VLLM_FL_FLAGOS_BLACKLIST="custom_op1,custom_op2"

# Result:
#   prefer: vendor (from env)
#   flagos_blacklist: [custom_op1, custom_op2] (from env, config ignored)
```

#### Important Notes

- **Environment variables override, not merge**: Setting an env var replaces the config value entirely
- **`VLLM_FL_PREFER` sets preference, not exclusivity**: It defines the selection order but will fall back to other backends if the preferred one is unavailable
- **To force a specific backend**: Combine `PREFER` with `DENY_VENDORS` or use `PER_OP` to exclude unwanted backends
- **`VLLM_FL_STRICT=1`**: Enables automatic fallback when the primary implementation fails at runtime

#### Backend Priority Values

Priority values are spaced by 50 to allow future insertion of intermediate priorities:
- `BackendPriority.DEFAULT` = 150 (FlagGems)
- `BackendPriority.VENDOR` = 100 (Vendor-specific)
- `BackendPriority.REFERENCE` = 50 (PyTorch)

## Policy Context Management

Supports temporary policy override in code:

```python
from vllm_fl.dispatch import (
    with_strict_mode,
    with_preference,
    with_allowed_vendors,
    with_denied_vendors,
)

# Temporarily enable strict mode
with with_strict_mode():
    result = call_op("silu_and_mul", x)

# Temporarily switch preferred backend
with with_preference("reference"):
    result = call_op("rms_norm", x, residual, weight, epsilon)

# Temporarily restrict allowed vendors
with with_allowed_vendors("vendor_a"):
    result = call_op("rotary_embedding", query, key, cos, sin, position_ids)
```

## Supported Operators

Currently supported operators:

| Operator | Description | FlagGems | Reference | Vendor |
|----------|-------------|----------|-----------|--------|
| `silu_and_mul` | SiLU activation + element-wise multiplication | ✓ | ✓ | ✓ |
| `rms_norm` | RMS normalization | ✓ | ✓ | ✓ |
| `rotary_embedding` | Rotary position embedding | ✓ | ✓ | ✓ |
| `attention_backend` | Attention backend class path | ✓ | - | ✓ |

## Selection Process

1. **Cache Check**: Check if dispatch cache hits
2. **Get Implementations**: Retrieve all registered implementations from registry
3. **Vendor Filtering**: Filter by policy's allow/deny lists
4. **Availability Check**: Call `is_available()` to check if implementation is available
5. **Priority Sorting**: Select best implementation based on per-op order or default order
6. **Cache Result**: Cache selection result to speed up subsequent calls

## Fallback Mechanism

When `VLLM_FL_STRICT=1`, if the primary implementation fails, the system automatically tries other available implementations:

```
Op 'rms_norm' using 'default.flagos' (kind=flagos, vendor=None)
[WARNING] Implementation 'default.flagos' failed for op 'rms_norm': ...
Op 'rms_norm' fallback to 'reference.torch' (kind=reference, vendor=None)
```

## Extending the System

### Adding New Operators

When adding a new operator, modify these files:
- `backends/flaggems/impl/*.py` - Add FlagGems implementation
- `backends/flaggems/flaggems.py` - Add method to backend class
- `backends/flaggems/register_ops.py` - Register OpImpl
- `backends/reference/impl/*.py` - Add PyTorch implementation (if applicable)
- `backends/reference/reference.py` - Add method to backend class
- `backends/reference/register_ops.py` - Register OpImpl
- `backends/vendor/<vendor>/impl/*.py` - Add vendor-specific implementation (optional)
- `backends/vendor/<vendor>/<vendor>.py` - Add method to vendor backend class
- `backends/vendor/<vendor>/register_ops.py` - Register vendor OpImpl
- `ops.py` - Add abstract method declaration

**Note:** Not all operators require a reference implementation. For example, `attention_backend` only has FlagGems and vendor implementations since it returns a backend class path rather than executing a computation.

### Adding Vendor Backends

The dispatch system supports three ways to integrate vendor backends:

1. **Built-in vendor backends** - Located in `backends/vendor/` (recommended for core vendors)
2. **External plugin packages** - Distributed as separate Python packages
3. **Environment-based plugins** - Loaded via `VLLM_FL_PLUGIN_MODULES`

#### Option 1: Built-in Vendor Backend

Directory structure:
```
backends/vendor/<vendor_name>/
├── __init__.py
├── <vendor_name>.py        # Backend class
├── register_ops.py         # Registration function
└── impl/                   # Operator implementations
    ├── __init__.py
    ├── activation.py
    ├── normalization.py
    ├── rotary.py
    └── attention.py        # (optional) Vendor-specific attention backend
```

**Step 1: Create Backend Class** (`<vendor_name>.py`):

```python
from ...base import Backend

class <VendorName>Backend(Backend):
    _available = None

    @property
    def name(self) -> str:
        return "<vendor_name>"

    @property
    def vendor(self) -> str:
        return "<vendor_name>"  # Required for vendor backends

    def is_available(self) -> bool:
        if <VendorName>Backend._available is None:
            try:
                import <vendor_library>
                <VendorName>Backend._available = True
            except ImportError:
                <VendorName>Backend._available = False
        return <VendorName>Backend._available

    def silu_and_mul(self, x):
        from .impl.activation import silu_and_mul_<vendor>
        return silu_and_mul_<vendor>(x)
```

**Step 2: Create Registration Module** (`register_ops.py`):

```python
from ....types import OpImpl, BackendImplKind, BackendPriority

def register_builtins(registry):
    from .<vendor_name> import <VendorName>Backend
    backend = <VendorName>Backend()

    impls = [
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.<vendor_name>",
            kind=BackendImplKind.VENDOR,
            fn=backend.silu_and_mul,
            vendor="<vendor_name>",
            priority=BackendPriority.VENDOR,  # 100
        ),
    ]
    registry.register_many(impls)
```

**Step 3: Register in builtin_ops.py**:

```python
try:
    from .backends.vendor.<vendor_name>.register_ops import register_builtins as register_<vendor>
    register_<vendor>(registry)
except Exception as e:
    logger.debug(f"<Vendor> operators not available: {e}")
```

#### Option 2: External Plugin Package

Create a separate package with entry points:

```python
# setup.py
setup(
    name="vllm-plugin-<vendor>",
    entry_points={
        "vllm_fl.plugin": [
            "<vendor> = vllm_fl_<vendor>.register_ops:register_builtins",
        ],
    },
)
```

Install and use:
```bash
pip install vllm-plugin-<vendor>
# Plugin auto-discovered via entry points
```

#### Option 3: Environment-based Plugin

```bash
export VLLM_FL_PLUGIN_MODULES=my_custom_backend.register_ops
```

The module should provide a `register_builtins(registry)` function.

#### Priority Levels

Use constants from `types.py`:
- `BackendPriority.DEFAULT` (150) - FlagGems
- `BackendPriority.VENDOR` (100) - Vendor backends
- `BackendPriority.REFERENCE` (50) - PyTorch

#### Testing Your Backend

```python
from vllm_fl.dispatch import get_default_manager

manager = get_default_manager()
manager.ensure_initialized()

# Check registration
snap = manager.registry.snapshot()
for op_name, impls in snap.impls_by_op.items():
    for impl in impls:
        if impl.vendor == "<vendor_name>":
            print(f"{op_name}: {impl.impl_id}, available={impl.is_available()}")
```

Enable debug output:
```bash
export VLLM_FL_LOG_LEVEL=DEBUG
```

#### Vendor Backend Checklist

- [ ] Backend class inherits from `Backend`
- [ ] `vendor` property returns vendor name (not None)
- [ ] `is_available()` checks hardware/library availability
- [ ] `register_ops.py` uses `BackendImplKind.VENDOR`
- [ ] `impl_id` follows format: `vendor.<vendor_name>`
- [ ] Priority set to `BackendPriority.VENDOR` (100)
- [ ] Error handling for missing dependencies
- [ ] (Optional) `attention_backend()` returns vendor-specific attention backend class path

#### Current Vendor Backends

| Vendor | Device | Library | Attention Backend |
|--------|--------|---------|-------------------|
| `cuda` | NVIDIA GPU | `vllm._custom_ops` | - (uses vLLM native) |
| `ascend` | Huawei NPU | `torch_npu` | `AscendAttentionBackend` |

See `backends/vendor/template/` for a template to create new vendor backends.

## Multi-Process Safety

OpManager supports multi-process environments:
- Uses `os.register_at_fork()` to automatically reset state after fork
- PID detection ensures independent initialization per process
- Thread-safe registry and cache operations

## API Reference

### Convenience Functions

- `call_op(op_name, *args, **kwargs)`: Call an operator
- `resolve_op(op_name)`: Resolve operator implementation

### Policy Management

- `get_policy()`: Get current policy
- `set_global_policy(policy)`: Set global policy
- `reset_global_policy()`: Reset to environment variable defaults
- `policy_context(policy)`: Temporary policy context
- `policy_from_config(config_path)`: Create policy from YAML config file

### Manager

- `get_default_manager()`: Get default manager instance
- `reset_default_manager()`: Reset default manager

### Plugin Discovery

- `discover_plugins(registry)`: Discover and load plugins
- `get_discovered_plugins()`: Get list of discovered plugins
- `clear_discovered_plugins()`: Clear discovered plugins list

### Logging

- `get_logger(name)`: Get logger instance
- `set_log_level(level, name)`: Set log level
