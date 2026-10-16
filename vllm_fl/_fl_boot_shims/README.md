# `_fl_boot_shims` — early-import compatibility shims

These modules patch `torch` / `triton` / `vllm` **before** the patched module is
first imported. That is a hard requirement, not a preference:

* `triton.jit` reads the kernel source from disk at first compile, so a patch
  applied after `import triton` is too late;
* `torch` / `vllm` read several settings at module import time.

The only portable CPython hook that runs early enough is `site`: a `.pth` file
in `site-packages` whose line starts with `import` is exec'd by
`site.addpackage()` before any user code runs.

## Install

```bash
# from the repository root
python scripts/install_fl_boot_shims.py

# verify against the current interpreter (used as a drift gate)
python scripts/install_fl_boot_shims.py --check

# preview without writing
python scripts/install_fl_boot_shims.py --dry-run

# remove only the hooks this script created
python scripts/install_fl_boot_shims.py --uninstall
```

The script is **idempotent** (a `.pth` is rewritten only when its bytes differ)
and **uninstall-safe**: it only deletes files carrying its marker comment, so
hand-written or vendor `.pth` files are never touched.

By default the target is the running interpreter's own `site-packages`
(`sysconfig.get_paths()["purelib"]`); pass `--site-packages` to override — e.g.
to install into a container or a different virtualenv. The shims directory
defaults to the checkout the script lives in, so no machine-specific absolute
path is baked into the repository.

### Naming convention

One hook per shim, mirroring the layout already used on the MetaX boxes:

```
zz_fl_<stem>.pth          # e.g. zz_fl_metax_annassign_compat.pth
```

where `<stem>` is the `_fl_*.py` module name without its leading underscore.
Helper modules that the shims import themselves
(`_kths_triton_unified_attention.py`, `_topk_topp_triton.py`) intentionally get
no hook — they must not be loaded at interpreter start-up.

## Switches

| Variable | Effect |
| --- | --- |
| `FL_BOOT_SHIMS_PATCH=0` | **master switch** — disables every shim at once (baked into each generated `.pth`; any other value, or unset, leaves the shims enabled) |
| `FL_METAX_ANNASSIGN_PATCH=0` | per-shim off switch (see the table below) |

Per-shim switches, each honoured inside its own module:

| Shim | Off switch |
| --- | --- |
| `_fl_metax_annassign_compat` | `FL_METAX_ANNASSIGN_PATCH=0` |
| `_fl_metax_boolchain_compat` | `FL_METAX_BOOLCHAIN_PATCH=0` |
| `_fl_metax_functorch_compat` | `FL_METAX_FUNCTORCH_PATCH=0` |
| `_fl_metax_kths_compat` | `FL_METAX_KTHS_PATCH=0` |
| `_fl_metax_load_ptr_compat` | `FL_METAX_LOAD_PTR_PATCH=0` |
| `_fl_metax_moe_is_monolithic_compat` | `FL_METAX_MOE_IS_MONOLITHIC_PATCH=0` |
| `_fl_metax_sampler_compat` | `FL_METAX_SAMPLER_PATCH=0` |
| `_fl_metax_topk_compat` | `FL_METAX_TOPK_PATCH=0` |
| `_fl_metax_torch_accel` | `FL_METAX_TORCH_ACCEL_PATCH=0` |

The remaining shims (`_fl_metax_fla_shmem_compat`, `_fl_metax_lora_compat`,
`_fl_metax_sparse_mla_impl_compat`, `_fl_metax_supports_combination_compat`,
`_fl_metax_tree_attn_compat`, `_fl_triton_experimental_stub`) currently have no
individual switch; use the master switch to disable them.

## Scope

Each shim self-gates on the active backend (`GEMS_VENDOR` / `plat.is_metax()`)
and is a no-op elsewhere, so installing the hooks is harmless on non-MetaX
hosts. Note that a small number of changes in `vllm_fl/ops/` (the
`FusedMoE`/`FusedMoEFactory` dual-name patch, the `FlashinferMoeBackend`
import fallback, `_OrigFusedMoE`) are **deliberately not gated**: they are
generic vLLM 0.28 API-compatibility shims that every backend passes through.
