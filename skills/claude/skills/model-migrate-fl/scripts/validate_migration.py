#!/usr/bin/env python3
"""Post-migration code review — deterministic checks for Step 6.

Usage:
    python3 scripts/validate_migration.py <plugin_folder> <model_file> [config_file] [init_file]

Arguments:
    plugin_folder   Path to vllm-plugin-FL root
    model_file      Path to the migrated model .py file
    config_file     (optional) Path to the config bridge .py file
    init_file       (optional) Path to __init__.py, default: <plugin_folder>/vllm_fl/__init__.py

Exit code 0 = all checks passed, 1 = issues found (details printed).
"""

import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

# --- Known APIs missing from vLLM 0.13.0 ---
KNOWN_MISSING_APIS = [
    "_mark_tower_model",
    "_mark_language_model",
    "MambaStateCopyFunc",
    "MambaStateCopyFuncCalculator",
    "get_mamba_state_copy_func",
]

issues = []
warnings = []


def issue(category, msg):
    issues.append(f"[{category}] {msg}")


def warn(category, msg):
    warnings.append(f"[{category}] {msg}")


# ── 6.1 Import Verification ─────────────────────────────────────────────────

def check_imports(filepath):
    """Check for relative imports, unused imports, and missing vllm_fl imports."""
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        issue("IMPORT", f"SyntaxError in {filepath}: {e}")
        return

    for node in ast.walk(tree):
        # Relative imports (level > 0)
        if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
            module = node.module or ""
            names = ", ".join(a.name for a in node.names)
            issue("IMPORT", f"Relative import at line {node.lineno}: from .{module} import {names}")

        # Check vllm_fl imports resolve
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vllm_fl."):
            try:
                spec = importlib.util.find_spec(node.module)
                if spec is None:
                    issue("IMPORT", f"Module not found: {node.module} (line {node.lineno})")
            except (ModuleNotFoundError, ValueError):
                issue("IMPORT", f"Module not found: {node.module} (line {node.lineno})")


# ── 6.2 API Compatibility ────────────────────────────────────────────────────

def check_api_compatibility(filepath):
    """Search for known APIs missing from vLLM 0.13.0."""
    with open(filepath) as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        for api in KNOWN_MISSING_APIS:
            if api in line and not line.lstrip().startswith("#"):
                issue("API", f"0.13.0-missing API '{api}' at line {i}: {line.rstrip()}")


# ── 6.3 Class & Config Consistency ───────────────────────────────────────────

def check_config_consistency(config_file, model_file):
    """Basic check: config bridge defines model_type and constructor."""
    if not config_file:
        return

    with open(config_file) as f:
        config_src = f.read()

    # Check model_type is set
    if "model_type" not in config_src:
        issue("CONFIG", f"No 'model_type' defined in {config_file}")

    # Check it's a PretrainedConfig subclass
    if "PretrainedConfig" not in config_src:
        issue("CONFIG", f"Config does not subclass PretrainedConfig in {config_file}")

    # Check __init__ exists
    if "def __init__" not in config_src:
        warn("CONFIG", f"No __init__ method in config bridge {config_file}")


# ── 6.4 Registration Correctness ─────────────────────────────────────────────

def check_registration(init_file, model_file, config_file):
    """Verify __init__.py registration matches actual class names."""
    with open(init_file) as f:
        init_src = f.read()

    # Extract class names from model file
    with open(model_file) as f:
        model_src = f.read()

    model_classes = set()
    try:
        tree = ast.parse(model_src, filename=model_file)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                model_classes.add(node.name)
    except SyntaxError:
        pass  # Already caught in check_imports

    # Find ModelRegistry.register_model calls in init
    model_basename = Path(model_file).stem
    reg_pattern = rf'register_model\(\s*"(\w+)"'
    for m in re.finditer(reg_pattern, init_src):
        class_name = m.group(1)
        if model_classes and class_name not in model_classes:
            issue("REGISTER", f"Registered class '{class_name}' not found in {model_file}. Available: {model_classes}")

    # Check import path in registration string
    path_pattern = rf'"vllm_fl\.models\.{re.escape(model_basename)}:(\w+)"'
    for m in re.finditer(path_pattern, init_src):
        class_name = m.group(1)
        if model_classes and class_name not in model_classes:
            issue("REGISTER", f"Import path references '{class_name}' but not found in {model_file}")

    # If config file provided, check _CONFIG_REGISTRY
    if config_file:
        config_basename = Path(config_file).stem
        if config_basename not in init_src and f"vllm_fl.configs.{config_basename}" not in init_src:
            warn("REGISTER", f"Config bridge {config_basename} may not be registered in {init_file}")


# ── 6.5 Code Cleanliness ─────────────────────────────────────────────────────

def check_cleanliness(filepath):
    """Check for code smells in migrated file."""
    with open(filepath) as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Empty except blocks
        if stripped == "except:":
            warn("CLEAN", f"Bare 'except:' at line {i} — consider catching specific exceptions")

        # Hardcoded paths outside plugin
        for bad_path in ["/usr/local/lib", "/models/"]:
            if bad_path in line and not stripped.startswith("#"):
                warn("CLEAN", f"Hardcoded path '{bad_path}' at line {i}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    plugin_folder = sys.argv[1]
    model_file = sys.argv[2]
    config_file = sys.argv[3] if len(sys.argv) > 3 else None
    init_file = sys.argv[4] if len(sys.argv) > 4 else os.path.join(plugin_folder, "vllm_fl", "__init__.py")

    # Add plugin to sys.path for import checks
    if plugin_folder not in sys.path:
        sys.path.insert(0, plugin_folder)

    files_to_check = [model_file]
    if config_file:
        files_to_check.append(config_file)

    print(f"Validating migration: {', '.join(files_to_check)}")
    print(f"Against init: {init_file}")
    print()

    for f in files_to_check:
        if not os.path.exists(f):
            issue("FILE", f"File not found: {f}")
            continue
        check_imports(f)
        check_api_compatibility(f)
        check_cleanliness(f)

    if config_file and os.path.exists(config_file):
        check_config_consistency(config_file, model_file)

    if os.path.exists(init_file):
        check_registration(init_file, model_file, config_file)
    else:
        issue("FILE", f"Init file not found: {init_file}")

    # Report
    if issues:
        print(f"ISSUES ({len(issues)}):")
        for i in issues:
            print(f"  {i}")
        print()

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")
        print()

    if not issues and not warnings:
        print("ALL CHECKS PASSED")

    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
