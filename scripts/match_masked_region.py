#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:
    hydra = None
    DictConfig = None
    OmegaConf = None


def load_patch_feature(path: str, image_name: Optional[str] = None) -> np.ndarray:
    """Load [num_patches, dim] features from a .npy or .npz file."""
    path_obj = Path(path)
    if path_obj.suffix == ".npy":
        feature = np.load(path_obj)
    elif path_obj.suffix == ".npz":
        npz_data = np.load(path_obj)
        keys = list(npz_data.keys())
        if image_name is None:
            if len(keys) != 1:
                raise ValueError(f"{path_obj} has multiple images; pass image name. Available: {keys}")
            image_name = keys[0]
        if image_name not in npz_data:
            raise ValueError(f"Image {image_name} not found in {path_obj}. Available: {keys}")
        feature = npz_data[image_name]
    else:
        raise ValueError(f"Unsupported feature file: {path_obj}")

    if feature.ndim != 2:
        raise ValueError(f"Expected feature shape [num_patches, dim], got {feature.shape}")
    return feature.astype(np.float32, copy=False)


def infer_grid(num_patches: int, grid: Optional[str]) -> Tuple[int, int]:
    if grid:
        h_str, w_str = grid.lower().split("x", 1)
        h_patches, w_patches = int(h_str), int(w_str)
        if h_patches * w_patches != num_patches:
            raise ValueError(f"Grid {grid} does not match {num_patches} patches")
        return h_patches, w_patches

    side = int(np.sqrt(num_patches))
    if side * side == num_patches:
        return side, side

    for h_patches in range(side, 0, -1):
        if num_patches % h_patches == 0:
            return h_patches, num_patches // h_patches
    raise ValueError(f"Cannot infer patch grid for {num_patches} patches")


def load_mask(mask_path: str, grid: Tuple[int, int], threshold: float) -> np.ndarray:
    """Load an RGB/grayscale mask and resize it to the patch grid."""
    h_patches, w_patches = grid
    mask_image = Image.open(mask_path)

    if mask_image.mode == "RGBA":
        mask = np.asarray(mask_image.getchannel("A"), dtype=np.float32) / 255.0
    else:
        mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0

    resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (w_patches, h_patches),
        resample=Image.Resampling.BILINEAR,
    )
    mask_grid = np.asarray(resized, dtype=np.float32) / 255.0
    selected = mask_grid > threshold
    if not selected.any():
        raise ValueError("Mask selected zero patches; lower --mask-threshold or check the mask image")
    return selected


def l2_normalize(feature: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(feature, axis=-1, keepdims=True)
    return feature / np.maximum(norm, eps)


def compute_similarity(
    ref_feature: np.ndarray,
    target_feature: np.ndarray,
    ref_mask: np.ndarray,
) -> np.ndarray:
    """Pool masked ref features into a query and compute target cosine similarity."""
    ref_norm = l2_normalize(ref_feature)
    target_norm = l2_normalize(target_feature)
    query = ref_norm[ref_mask.reshape(-1)].mean(axis=0, keepdims=True)
    query = l2_normalize(query)
    return (target_norm @ query.T).reshape(-1)


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=axis, keepdims=True), 1e-12)


def compute_reverse_affordance_probability(
    ref_feature: np.ndarray,
    target_feature: np.ndarray,
    ref_mask: np.ndarray,
    temperature: float,
    batch_size: int,
) -> np.ndarray:
    """For each target patch, compute probability mass assigned back to the reference affordance mask."""
    if temperature <= 0:
        raise ValueError("--reverse-temperature must be > 0")
    if batch_size <= 0:
        raise ValueError("--reverse-batch-size must be > 0")

    ref_norm = l2_normalize(ref_feature)
    target_norm = l2_normalize(target_feature)
    ref_mask_flat = ref_mask.reshape(-1)
    probabilities = np.empty(target_norm.shape[0], dtype=np.float32)

    for start in range(0, target_norm.shape[0], batch_size):
        end = min(start + batch_size, target_norm.shape[0])
        reverse_logits = (target_norm[start:end] @ ref_norm.T) / temperature
        reverse_probs = softmax(reverse_logits, axis=1)
        probabilities[start:end] = reverse_probs[:, ref_mask_flat].sum(axis=1)

    return probabilities


def compute_functional_correspondence(
    ref_feature: np.ndarray,
    target_feature: np.ndarray,
    ref_mask: np.ndarray,
    reverse_temperature: float,
    reverse_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine forward reference matching with reverse consistency into a stable correspondence score."""
    forward_similarity = compute_similarity(ref_feature, target_feature, ref_mask)
    reverse_probability = compute_reverse_affordance_probability(
        ref_feature,
        target_feature,
        ref_mask,
        reverse_temperature,
        reverse_batch_size,
    )
    correspondence = normalize_map(forward_similarity) * reverse_probability
    return forward_similarity, reverse_probability, correspondence


def normalize_map(values: np.ndarray) -> np.ndarray:
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def resize_normalized_map(values: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    normalized = normalize_map(values)
    image = Image.fromarray((normalized * 255).astype(np.uint8))
    image = image.resize((w, h), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def resize_binary_map(values: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    image = Image.fromarray(values.astype(np.uint8) * 255)
    image = image.resize((w, h), resample=Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def save_heatmap(sim_map: np.ndarray, save_path: Path, cmap: str, dpi: int, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(sim_map, cmap=cmap, interpolation="nearest")
    plt.colorbar(shrink=0.8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {save_path}")


def save_normalized_heatmap(sim_map: np.ndarray, save_path: Path, cmap: str, dpi: int, title: str) -> None:
    save_heatmap(normalize_map(sim_map), save_path, cmap, dpi, title)


def save_overlay(
    sim_map: np.ndarray,
    target_image_path: str,
    save_path: Path,
    percentile: float,
    score_threshold: Optional[float],
    alpha: float,
    cmap: str,
) -> tuple[float, str]:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    target = Image.open(target_image_path).convert("RGB")
    target_array = np.asarray(target, dtype=np.float32) / 255.0
    target_h, target_w = target_array.shape[:2]
    sim_normalized = normalize_map(sim_map)

    if score_threshold is None:
        cutoff = float(np.percentile(sim_map, percentile))
        threshold_type = f"percentile_{percentile:g}"
        selected_grid = sim_map >= cutoff
    else:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("--score-threshold must be in [0, 1] because it is applied to the normalized heatmap")
        cutoff = float(score_threshold)
        threshold_type = "normalized_score_threshold"
        selected_grid = sim_normalized >= cutoff

    selected = resize_binary_map(selected_grid, (target_h, target_w))
    sim_resized = resize_normalized_map(sim_map, (target_h, target_w))

    colormap = plt.get_cmap(cmap)
    color = colormap(sim_resized)[..., :3]
    alpha_map = (selected.astype(np.float32) * alpha)[..., None]
    overlay = target_array * (1.0 - alpha_map) + color * alpha_map

    Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)).save(save_path)
    print(f"Saved target overlay to {save_path}")
    return cutoff, threshold_type


def save_raw_outputs(
    save_path: Path,
    correspondence_map: np.ndarray,
    forward_map: np.ndarray,
    reverse_map: np.ndarray,
    ref_mask: np.ndarray,
    cutoff: float,
    threshold_type: str,
) -> None:
    raw_path = save_path.with_suffix(".npz")
    np.savez_compressed(
        raw_path,
        correspondence=correspondence_map,
        forward_similarity=forward_map,
        reverse_affordance_probability=reverse_map,
        ref_mask=ref_mask,
        overlay_cutoff=cutoff,
        overlay_threshold_type=threshold_type,
    )
    print(f"Saved raw outputs to {raw_path}")


def run_match(args) -> None:
    ref_feature = load_patch_feature(args.ref_feature, args.ref_image_name)
    target_feature = load_patch_feature(args.target_feature, args.target_image_name)
    ref_grid = infer_grid(ref_feature.shape[0], args.ref_grid)
    target_grid = infer_grid(target_feature.shape[0], args.target_grid)
    ref_mask = load_mask(args.ref_mask, ref_grid, args.mask_threshold)

    forward_similarity, reverse_probability, correspondence = compute_functional_correspondence(
        ref_feature,
        target_feature,
        ref_mask,
        args.reverse_temperature,
        args.reverse_batch_size,
    )
    forward_map = forward_similarity.reshape(target_grid)
    reverse_map = reverse_probability.reshape(target_grid)
    correspondence_map = correspondence.reshape(target_grid)

    save_prefix = Path(args.save_prefix)
    forward_heatmap_path = save_prefix.with_name(f"{save_prefix.name}_forward_heatmap.png")
    reverse_heatmap_path = save_prefix.with_name(f"{save_prefix.name}_reverse_consistency_heatmap.png")
    heatmap_path = save_prefix.with_name(f"{save_prefix.name}_heatmap.png")
    normalized_heatmap_path = save_prefix.with_name(f"{save_prefix.name}_heatmap_normalized.png")
    overlay_path = save_prefix.with_name(f"{save_prefix.name}_overlay.png")

    save_heatmap(forward_map, forward_heatmap_path, args.cmap, args.dpi, "Forward Reference-to-Target Similarity")
    save_heatmap(reverse_map, reverse_heatmap_path, args.cmap, args.dpi, "Reverse Consistency Probability")
    save_heatmap(correspondence_map, heatmap_path, args.cmap, args.dpi, "Functional Correspondence Score")
    save_normalized_heatmap(
        correspondence_map,
        normalized_heatmap_path,
        args.cmap,
        args.dpi,
        "Functional Correspondence Score (Normalized)",
    )
    cutoff, threshold_type = save_overlay(
        correspondence_map,
        args.target_image,
        overlay_path,
        args.top_percentile,
        args.score_threshold,
        args.alpha,
        args.cmap,
    )
    save_raw_outputs(
        save_prefix.with_suffix(".png"),
        correspondence_map,
        forward_map,
        reverse_map,
        ref_mask,
        cutoff,
        threshold_type,
    )

    print(f"Reference feature shape: {ref_feature.shape}")
    print(f"Target feature shape: {target_feature.shape}")
    print(f"Reference grid: {ref_grid[0]} x {ref_grid[1]}")
    print(f"Target grid: {target_grid[0]} x {target_grid[1]}")
    print(f"Reference selected patches: {int(ref_mask.sum())}")
    print(f"Forward similarity range: [{forward_map.min():.6f}, {forward_map.max():.6f}]")
    print(f"Reverse consistency range: [{reverse_map.min():.6f}, {reverse_map.max():.6f}]")
    print(f"Correspondence range: [{correspondence_map.min():.6f}, {correspondence_map.max():.6f}]")
    print(f"Overlay threshold ({threshold_type}): {cutoff:.6f}")


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match a masked reference DINOv3 region in a target feature map")
    parser.add_argument("--ref-feature", required=True, help="Reference patch feature .npy/.npz")
    parser.add_argument("--target-feature", required=True, help="Target patch feature .npy/.npz")
    parser.add_argument("--ref-mask", required=True, help="RGB/grayscale/alpha mask for the reference image")
    parser.add_argument("--target-image", required=True, help="Target RGB image for overlay visualization")
    parser.add_argument("--ref-image-name", help="Image key for reference .npz files")
    parser.add_argument("--target-image-name", help="Image key for target .npz files")
    parser.add_argument("--ref-grid", help="Reference patch grid as HxW, for example 160x160")
    parser.add_argument("--target-grid", help="Target patch grid as HxW, for example 160x160")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Threshold after resizing the mask to patch grid")
    parser.add_argument("--top-percentile", type=float, default=90.0,
                        help="Overlay target pixels whose similarity is in this percentile or higher")
    parser.add_argument("--score-threshold", type=float,
                        help="Overlay target pixels whose normalized correspondence score is at least this value")
    parser.add_argument("--reverse-temperature", type=float, default=0.07,
                        help="Softmax temperature for reverse target-to-reference matching")
    parser.add_argument("--reverse-batch-size", type=int, default=1024,
                        help="Number of target patches per reverse-consistency batch")
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay opacity")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap")
    parser.add_argument("--save-prefix", required=True, help="Output prefix, without extension")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def legacy_main() -> None:
    parser = build_legacy_parser()
    run_match(parser.parse_args())


def validate_hydra_config(cfg) -> None:
    required = ["ref_feature", "target_feature", "ref_mask", "target_image", "save_prefix"]
    missing = [name for name in required if not cfg.get(name)]
    if missing:
        raise ValueError(f"Missing required config values: {missing}")


def _run_hydra_main() -> None:
    if hydra is None:
        raise ModuleNotFoundError("Hydra is required for config mode. Install hydra-core or use legacy -- arguments.")

    @hydra.main(version_base=None, config_path="../configs/match_masked_region", config_name="default")
    def hydra_main(cfg) -> None:
        validate_hydra_config(cfg)
        print("Hydra match_masked_region config:")
        print(OmegaConf.to_yaml(cfg))
        run_match(cfg)

    hydra_main()


if __name__ == "__main__":
    if any(arg.startswith("--") for arg in sys.argv[1:]):
        legacy_main()
    else:
        _run_hydra_main()
