"""Install and finalize optional optimizations around model loading.

The worker owns the model-loading lifecycle.  Feature-specific setup lives here
so the upstream-derived worker remains a small integration surface and every
optimization has an explicit fallback path.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.vllm_fl.optimizations.runtime")


class RuntimeOptimizations:
    """Coordinate optional optimizations before and after weight loading."""

    def __init__(self, vllm_config, device: torch.device, model_runner) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.model_runner = model_runner
        self.model_config = vllm_config.model_config

        self._int8_sparse = None
        self._int8_sparse_requested = False
        self._native_sparse = None
        self._native_sparse_enabled = False
        self._prefill_gateup = None
        self._prefill_gateup_enabled = False
        self._single_weight_gateup = None
        self._single_weight_gateup_enabled = False

    def prepare(self, *, load_dummy_weights: bool) -> None:
        """Install wrappers and prepare cache keys before model loading."""
        from vllm_fl.device_profiles import apply_defaults

        apply_defaults(self.device.index, self.vllm_config)
        self._prepare_sparse_int8()
        self._prepare_native_sparse()
        self._install_kernel_option("mm_tuning", "fl_mm_tuning_buckets")
        self._install_kernel_option("mm_large_tiles", "fl_mm_large_tiles")
        self._install_kernel_option(
            "radix_sort",
            "fl_radix_histogram_intrinsic",
            self.model_config.get_vocab_size(),
        )
        self._install_sampling()
        self._prepare_gateup(load_dummy_weights=load_dummy_weights)

    def finalize(self) -> None:
        """Finalize weight-dependent features and account persistent memory."""
        if self._prefill_gateup_enabled:
            self._prefill_gateup.finalize(self.model_runner.get_model())
        if self._single_weight_gateup_enabled:
            self._finalize_single_weight_gateup()
        self._finalize_sparse_int8()
        self._account_native_sparse()

    def _prepare_sparse_int8(self) -> None:
        try:
            from vllm_fl import device_profiles, envs
            from vllm_fl.ops import int8_sparse

            self._int8_sparse = int8_sparse
            profile = device_profiles.lookup(self.device.index) or {}
            self._int8_sparse_requested = envs.switch(
                "VLLM_FL_KV_INT8_SPARSE",
                auto=profile.get("int8_sparse") is True,
            )
            if self._int8_sparse_requested:
                self.vllm_config.additional_config["fl_kv_int8_sparse_hash"] = (
                    int8_sparse.source_hash()
                )
        except Exception:
            if self._int8_sparse is not None:
                self._int8_sparse.disable("could not prepare sparse INT8 source hash")
            self._int8_sparse_requested = False
            self.vllm_config.additional_config.pop("fl_kv_int8_sparse_hash", None)
            logger.warning(
                "Sparse INT8 KV source hash unavailable; retaining the existing KV path",
                exc_info=True,
            )

    def _prepare_native_sparse(self) -> None:
        try:
            from vllm_fl.ops import native_sparse

            self._native_sparse = native_sparse
            self._native_sparse_enabled = native_sparse.install(
                self.vllm_config, self.device
            )
            if self._native_sparse_enabled:
                self.vllm_config.additional_config[
                    "fl_native_sparse_decode_enabled"
                ] = True
                self.vllm_config.additional_config["fl_native_sparse_decode_hash"] = (
                    native_sparse.source_hash()
                )
        except Exception:
            if self._native_sparse is not None:
                self._native_sparse.disable("runtime installation raised")
            self._native_sparse_enabled = False
            self.vllm_config.additional_config.pop(
                "fl_native_sparse_decode_enabled", None
            )
            self.vllm_config.additional_config.pop("fl_native_sparse_decode_hash", None)
            logger.warning(
                "Native sparse decode unavailable; using contiguous KV cache",
                exc_info=True,
            )

    def _install_kernel_option(self, module_name, config_key, *args) -> None:
        """Install an option and key its selected implementation for compilation."""
        import importlib

        module = None
        try:
            module = importlib.import_module(f"vllm_fl.ops.{module_name}")
            if module.install(self.device, *args):
                self.vllm_config.additional_config[f"{config_key}_enabled"] = True
                self.vllm_config.additional_config[f"{config_key}_hash"] = (
                    module.source_hash()
                )
        except Exception:
            self.vllm_config.additional_config.pop(f"{config_key}_enabled", None)
            self.vllm_config.additional_config.pop(f"{config_key}_hash", None)
            if module is not None:
                try:
                    module.disable()
                except Exception:
                    logger.debug("Could not disable %s", module_name, exc_info=True)
            logger.warning(
                "%s unavailable; retaining the original implementation",
                module_name,
                exc_info=True,
            )

    def _install_sampling(self) -> None:
        try:
            from vllm_fl.ops.sampling import install

            install(self.device, self.model_config.get_vocab_size())
        except Exception:
            logger.warning(
                "Optimized sampling unavailable; using upstream sampling",
                exc_info=True,
            )

    def _prepare_gateup(self, *, load_dummy_weights: bool) -> None:
        from vllm_fl.patches import single_weight_gateup

        self._single_weight_gateup = single_weight_gateup
        single_weight_gateup.validate_early(self.vllm_config, self.device)

        try:
            from vllm_fl.patches import prefill_gateup

            self._prefill_gateup = prefill_gateup
            self._prefill_gateup_enabled = prefill_gateup.install(
                self.vllm_config, self.device
            )
        except Exception:
            self._prefill_gateup_enabled = False
            logger.warning(
                "Prefill gate/up INT8 setup failed; using the original path",
                exc_info=True,
            )

        if not load_dummy_weights:
            self._single_weight_gateup_enabled = single_weight_gateup.install(
                self.vllm_config, self.device
            )

    def _finalize_single_weight_gateup(self) -> None:
        previous_usage = int(self.model_runner.model_memory_usage)
        stats = self._single_weight_gateup.finalize(
            self.model_runner.get_model(),
            self.device,
            self.vllm_config,
            model_memory_usage=previous_usage,
        )
        if not stats["delta_bytes"]:
            return
        updated_usage = previous_usage + int(stats["delta_bytes"])
        if updated_usage < 0:
            raise RuntimeError(
                "single-weight gate/up memory adjustment became negative"
            )
        self.model_runner.model_memory_usage = updated_usage
        logger.info(
            "Single-weight gate/up model memory accounted: layers=%d "
            "qweight_bytes=%d scale_bytes=%d bf16_released_bytes=%d "
            "delta_bytes=%d model_memory_usage_bytes=%d",
            stats["layers"],
            stats["qweight_bytes"],
            stats["scale_bytes"],
            stats["bf16_bytes"],
            stats["delta_bytes"],
            updated_usage,
        )

    def _finalize_sparse_int8(self) -> None:
        if not self._int8_sparse_requested or self._int8_sparse is None:
            return
        accounted_bytes = 0
        try:
            enabled = self._int8_sparse.install(self.vllm_config, self.device)
            if not enabled:
                self.vllm_config.additional_config.pop("fl_kv_int8_sparse_hash", None)
                self.vllm_config.additional_config["fl_kv_int8_sparse_enabled"] = False
                return
            workspace_bytes = self._int8_sparse.workspace_nbytes()
            if workspace_bytes <= 0:
                raise RuntimeError(
                    "sparse INT8 KV enabled without persistent workspace"
                )
            self.vllm_config.additional_config["fl_kv_int8_sparse_enabled"] = True
            self.model_runner.model_memory_usage += workspace_bytes
            accounted_bytes = workspace_bytes
            self.model_runner._fl_kv_int8_sparse_accounted_bytes = accounted_bytes
            logger.info(
                "Persistent sparse INT8 KV workspace accounted: "
                "workspace_bytes=%d model_memory_usage_bytes=%d",
                accounted_bytes,
                self.model_runner.model_memory_usage,
            )
        except Exception:
            self._int8_sparse.disable("runtime sparse INT8 installation failed")
            if accounted_bytes:
                self.model_runner.model_memory_usage -= accounted_bytes
                self.model_runner._fl_kv_int8_sparse_accounted_bytes = 0
            self.vllm_config.additional_config.pop("fl_kv_int8_sparse_hash", None)
            self.vllm_config.additional_config["fl_kv_int8_sparse_enabled"] = False
            logger.warning(
                "Sparse INT8 KV unavailable; retaining the existing KV path",
                exc_info=True,
            )

    def _account_native_sparse(self) -> None:
        if not self._native_sparse_enabled:
            return
        workspace_bytes = self._native_sparse.workspace_nbytes()
        if workspace_bytes <= 0:
            raise RuntimeError(
                "native sparse decode enabled without a persistent workspace"
            )
        self.model_runner.model_memory_usage += workspace_bytes
        logger.info(
            "Persistent native sparse decode workspace accounted: "
            "workspace_bytes=%d model_memory_usage_bytes=%d",
            workspace_bytes,
            self.model_runner.model_memory_usage,
        )
