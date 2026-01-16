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
└── backends/                # Backend implementations
    ├── base.py              # Backend abstract base class
    ├── flaggems/            # FlagGems backend (DEFAULT, priority 150)
    ├── reference/           # Reference backend (PyTorch, priority 50)
    └── vendor/              # Vendor-specific backends (priority 100)
        └── template/        # Template for creating new vendor backends
```

## Core Concepts

### 1. Backend Implementation Kind

- **DEFAULT**: Default implementation (FlagGems), priority 150
- **VENDOR**: Vendor-specific implementation, priority 100
- **REFERENCE**: Reference implementation (PyTorch native), priority 50

### 2. Operator Implementation (OpImpl)

Each operator implementation contains:
- `op_name`: Operator name (e.g., "silu_and_mul", "rmsnorm")
- `impl_id`: Unique implementation identifier (e.g., "default.flaggems")
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
│                  call_op("rmsnorm", x, ...)                      │
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
│                    VLLM_FL_PREFER=flaggems                       │
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
fn = resolve_op("rmsnorm")
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

## Environment Variables

| Variable | Description | Example | Behavior |
|----------|-------------|---------|----------|
| `VLLM_FL_PREFER` | Preferred backend (sets selection order) | `flaggems`, `vendor`, `reference` | Defines priority order, falls back if unavailable |
| `VLLM_FL_STRICT` | Enable strict mode (auto-fallback on failure) | `1` or `0` | When `1`, tries alternatives if primary fails |
| `VLLM_FL_DENY_VENDORS` | Denied vendors list (blacklist) | `vendor1,vendor2` | Excludes specified vendors from selection |
| `VLLM_FL_ALLOW_VENDORS` | Allowed vendors whitelist | `vendor1,vendor2` | Only allows specified vendors (if set) |
| `VLLM_FL_PER_OP` | Per-operator selection order | `op1=a\|b\|c;op2=x\|y` | Overrides default order for specific ops |
| `VLLM_FL_PLUGIN_MODULES` | Plugin modules to load | `my_plugin,another_plugin` | Loads external plugin modules |
| `VLLM_FL_LOG_LEVEL` | Log level | `DEBUG`, `INFO`, `WARNING`, `ERROR` | Controls logging verbosity |

### Examples

```bash
# Prefer FlagGems implementation
export VLLM_FL_PREFER=flaggems

# Enable strict mode (auto-fallback on failure)
export VLLM_FL_STRICT=1

# Deny specific vendors
export VLLM_FL_DENY_VENDORS=vendor_a,vendor_b

# Specify selection order for specific operator
export VLLM_FL_PER_OP="rmsnorm=vendor|flaggems|reference"

# Load external plugins
export VLLM_FL_PLUGIN_MODULES=my_custom_backend

# Set log level
export VLLM_FL_LOG_LEVEL=DEBUG
```

### Environment Variable Priority

The dispatch system applies environment variables in the following order:

1. **`VLLM_FL_PER_OP`** - Highest priority, overrides default order for specific operators
2. **`VLLM_FL_ALLOW_VENDORS`** - Whitelist filter (if set, only these vendors are allowed)
3. **`VLLM_FL_DENY_VENDORS`** - Blacklist filter (these vendors are excluded)
4. **`VLLM_FL_PREFER`** - Default selection order for all operators
5. **`BackendPriority`** - Code-defined priority (used for tie-breaking within same kind)

**Priority values are spaced by 50 to allow future insertion of intermediate priorities:**
- `BackendPriority.DEFAULT` = 150 (FlagGems)
- `BackendPriority.VENDOR` = 100 (Vendor-specific)
- `BackendPriority.REFERENCE` = 50 (PyTorch)

#### Example: Combined Environment Variables

```bash
export VLLM_FL_PREFER=flaggems                    # Default: flaggems → vendor → reference
export VLLM_FL_DENY_VENDORS=vendor_a              # Exclude vendor_a
export VLLM_FL_PER_OP="rmsnorm=vendor|reference"  # Override for rmsnorm only
```

**Result:**
- **`rmsnorm` operator**: Uses `vendor → reference` order (PER_OP overrides PREFER), excluding vendor_a
- **Other operators** (e.g., `silu_and_mul`): Uses `flaggems → vendor → reference` order (from PREFER), excluding vendor_a

#### Important Notes

- **`VLLM_FL_PREFER` sets preference, not exclusivity**: It defines the selection order but will fall back to other backends if the preferred one is unavailable.
- **To force a specific backend**: Combine `PREFER` with `DENY_VENDORS` or use `PER_OP` to exclude unwanted backends.
- **`VLLM_FL_STRICT=1`**: Enables automatic fallback when the primary implementation fails at runtime (not just unavailable).

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
    result = call_op("rmsnorm", x, residual, weight, epsilon)

# Temporarily restrict allowed vendors
with with_allowed_vendors("vendor_a"):
    result = call_op("rotary_embedding", query, key, cos, sin, position_ids)
```

## Supported Operators

Currently supported operators:

| Operator | Description | FlagGems | Reference | Vendor |
|----------|-------------|----------|-----------|--------|
| `silu_and_mul` | SiLU activation + element-wise multiplication | ✓ | ✓ | ✓ |
| `rmsnorm` | RMS normalization | ✓ | ✓ | ✓ |
| `rotary_embedding` | Rotary position embedding | ✓ | ✓ | ✓ |

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
Op 'rmsnorm' using 'default.flaggems' (kind=flaggems, vendor=None)
[WARNING] Implementation 'default.flaggems' failed for op 'rmsnorm': ...
Op 'rmsnorm' fallback to 'reference.torch' (kind=reference, vendor=None)
```

## Extending the System

### Adding New Operators

When adding a new operator, modify these files:
- `backends/flaggems/impl/*.py` - Add FlagGems implementation
- `backends/flaggems/flaggems.py` - Add method to backend class
- `backends/flaggems/register_ops.py` - Register OpImpl
- `backends/reference/impl/*.py` - Add PyTorch implementation
- `backends/reference/reference.py` - Add method to backend class
- `backends/reference/register_ops.py` - Register OpImpl
- `ops.py` - Add abstract method declaration

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
    └── rotary.py
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

#### Current Vendor Backends

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
