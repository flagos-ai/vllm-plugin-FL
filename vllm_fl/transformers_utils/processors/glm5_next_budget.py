# SPDX-License-Identifier: Apache-2.0
"""One pixel/token contract for GLM preprocessing and serving reservations."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, isqrt


@dataclass(frozen=True)
class VisionBudget:
    min_pixels: int
    max_pixels: int
    pixels_per_token: int
    spatial_factor: int
    temporal_factor: int

    @property
    def max_tokens(self) -> int:
        return self.max_pixels // self.pixels_per_token

    def largest_canvas(self, num_frames=None) -> tuple[int, int]:
        """An aligned canvas maximizing token count, including non-square ones."""
        frames = num_frames or self.temporal_factor
        frames += -frames % self.temporal_factor
        cells = self.max_pixels // (frames * self.spatial_factor**2)
        if cells < 1:
            raise ValueError(
                "GLM5-Next frame count cannot fit one aligned patch per frame"
            )
        choices = [
            (h * min(cells // h, 200 * h), h, min(cells // h, 200 * h))
            for h in range(1, isqrt(cells) + 1)
        ]
        _, h, w = max(choices)
        return h * self.spatial_factor, w * self.spatial_factor


def modality_kwargs(kwargs, modality):
    nested = "images_kwargs" if modality == "image" else "videos_kwargs"
    result = {
        k: v
        for k, v in kwargs.items()
        if k not in ("images_kwargs", "videos_kwargs", "text_kwargs", "common_kwargs")
    }
    result.update(kwargs.get(nested, {}))
    return result


def resolve_vision_budget(processor, overrides=None) -> VisionBudget:
    values = dict(overrides or {})
    if "size" in values:
        raise ValueError("GLM5-Next serving uses max_pixels/max_image_tokens, not size")
    for name in (
        "patch_size",
        "merge_size",
        "temporal_patch_size",
        "patch_expand_factor",
    ):
        if values.get(name, getattr(processor, name)) != getattr(processor, name):
            raise ValueError(f"GLM5-Next serving does not support overriding {name}")
    if values.get("do_resize", True) is not True:
        raise ValueError(
            "GLM5-Next serving requires do_resize=True to enforce its budget"
        )

    def positive(name, default):
        value = values.get(name)
        value = default if value is None else value
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"GLM5-Next {name} must be a positive integer")
        return value

    unit = (
        processor.temporal_patch_size
        * (processor.patch_size * processor.merge_size) ** 2
    )

    def pixels(bound):
        if values.get(bound + "_pixels") is not None:
            return positive(bound + "_pixels", None)
        return (
            positive(
                bound + "_image_tokens", getattr(processor, bound + "_image_tokens")
            )
            * unit
        )

    min_pixels, max_pixels = pixels("min"), pixels("max")
    factor = processor.patch_size * processor.merge_size * processor.patch_expand_factor
    if (
        min_pixels > max_pixels
        or max_pixels < processor.temporal_patch_size * factor**2
    ):
        raise ValueError(
            "GLM5-Next pixel budget is empty or smaller than one aligned patch"
        )
    return VisionBudget(
        min_pixels, max_pixels, unit, factor, processor.temporal_patch_size
    )


def resolve_serving_kwargs(processor, deployment, request):
    """Resolve both modalities and reject requests exceeding reserved resources."""
    merged = deepcopy(deployment)
    merged.update(deepcopy(request))
    for nested in ("images_kwargs", "videos_kwargs"):
        merged[nested] = {**deployment.get(nested, {}), **request.get(nested, {})}
    result = {
        k: v
        for k, v in merged.items()
        if k
        not in (
            "min_pixels",
            "max_pixels",
            "min_image_tokens",
            "max_image_tokens",
            "max_frames",
            "max_frame_count_dynamic",
            "fps",
            "target_fps",
            "fps_interval",
            "do_sample_frames",
            "sampling_policy",
        )
    }
    for modality, nested in (("image", "images_kwargs"), ("video", "videos_kwargs")):
        sub = getattr(processor, modality + "_processor")
        deploy_values = modality_kwargs(deployment, modality)
        values = dict(deploy_values)
        values.update(modality_kwargs(request, modality))
        ceiling = resolve_vision_budget(sub, deploy_values)
        budget = resolve_vision_budget(sub, values)
        if budget.max_pixels > ceiling.max_pixels:
            raise ValueError(
                f"GLM5-Next {modality} request budget exceeds deployment limit "
                f"({budget.max_pixels} > {ceiling.max_pixels} pixels)"
            )
        if modality == "video":
            frame_cap = deploy_values.get(
                "max_frames",
                deploy_values.get(
                    "max_frame_count_dynamic", sub.max_frame_count_dynamic
                ),
            )
            requested = values.get(
                "max_frames", values.get("max_frame_count_dynamic", frame_cap)
            )
            if (
                isinstance(requested, bool)
                or not isinstance(requested, int)
                or requested % sub.temporal_patch_size
                or not sub.temporal_patch_size <= requested <= frame_cap
            ):
                raise ValueError(
                    "GLM5-Next video max_frames exceeds deployment limit or is invalid"
                )
            pixel_frame_cap = budget.max_pixels // budget.spatial_factor**2
            pixel_frame_cap -= pixel_frame_cap % sub.temporal_patch_size
            values["max_frames"] = min(requested, pixel_frame_cap)
            if (
                values.get("sampling_policy", sub.sampling_policy)
                != sub.sampling_policy
            ):
                raise ValueError("GLM5-Next serving cannot override sampling_policy")
        # Pass one normalized result to the actual subprocessor. Non-budget
        # flat kwargs remain flat for the Transformers text/video interfaces.
        scoped = dict(merged.get(nested, {}))
        keys = ["min_image_tokens", "max_image_tokens"]
        if modality == "video":
            keys += ["max_frames", "fps", "target_fps", "do_sample_frames"]
            target_fps = next(
                (
                    values[key]
                    for key in ("fps", "target_fps", "fps_interval")
                    if values.get(key) is not None
                ),
                sub.fps_interval if sub.sampling_policy == "fps_interval" else None,
            )
            if target_fps is not None:
                if (
                    isinstance(target_fps, bool)
                    or not isinstance(target_fps, (int, float))
                    or not isfinite(target_fps)
                    or target_fps <= 0
                ):
                    raise ValueError("GLM5-Next video fps must be finite and positive")
                # BaseVideoProcessor supplies its default fps to sample_frames;
                # normalize aliases to fps too so that default cannot mask them.
                values["fps"] = values["target_fps"] = target_fps
        for key in keys:
            if key in values:
                scoped[key] = values[key]
        scoped.update(min_pixels=budget.min_pixels, max_pixels=budget.max_pixels)
        result[nested] = scoped
    return result
