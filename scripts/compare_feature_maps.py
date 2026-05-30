#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_feature(path: str, feature_key: str, image_name: Optional[str] = None) -> np.ndarray:
    """Load a feature array from a .npy/.npz file or a feature output directory."""
    path_obj = Path(path)
    if path_obj.is_dir():
        npy_path = path_obj / f"{feature_key}.npy"
        npz_path = path_obj / f"{feature_key}.npz"
        if npy_path.exists():
            path_obj = npy_path
        elif npz_path.exists():
            path_obj = npz_path
        else:
            raise FileNotFoundError(f"Cannot find {feature_key}.npy or {feature_key}.npz in {path_obj}")

    if path_obj.suffix == ".npy":
        feature = np.load(path_obj)
    elif path_obj.suffix == ".npz":
        npz_data = np.load(path_obj)
        keys = list(npz_data.keys())
        if image_name is None:
            if len(keys) != 1:
                raise ValueError(f"{path_obj} contains multiple images; pass --image-name. Available: {keys}")
            image_name = keys[0]
        if image_name not in npz_data:
            raise ValueError(f"Image {image_name} not found in {path_obj}. Available: {keys}")
        feature = npz_data[image_name]
    else:
        raise ValueError(f"Unsupported feature path: {path_obj}")

    if feature.ndim != 2:
        raise ValueError(f"Expected a 2D patch feature array [num_patches, dim], got {feature.shape}")
    return feature.astype(np.float32, copy=False)


def trim_prefix_tokens(feature: np.ndarray, drop_prefix_tokens: int, auto_drop_register_tokens: bool) -> np.ndarray:
    """Drop non-patch prefix tokens when needed."""
    if drop_prefix_tokens:
        if drop_prefix_tokens >= feature.shape[0]:
            raise ValueError(f"Cannot drop {drop_prefix_tokens} tokens from feature with shape {feature.shape}")
        print(f"Dropping first {drop_prefix_tokens} prefix tokens")
        return feature[drop_prefix_tokens:]

    if auto_drop_register_tokens:
        num_patches = feature.shape[0]
        side = int(np.sqrt(num_patches))
        trimmed_side = int(np.sqrt(num_patches - 4)) if num_patches > 4 else 0
        if side * side != num_patches and trimmed_side * trimmed_side == num_patches - 4:
            print("Detected 4 DINOv3 register/storage tokens before patch tokens; dropping them")
            return feature[4:]

    return feature


def infer_grid(num_patches: int, grid: Optional[str] = None) -> Tuple[int, int]:
    """Infer or parse the spatial patch grid."""
    if grid:
        h_str, w_str = grid.lower().split("x", 1)
        h_patches, w_patches = int(h_str), int(w_str)
        if h_patches * w_patches != num_patches:
            raise ValueError(f"--grid {grid} does not match {num_patches} patches")
        return h_patches, w_patches

    side = int(np.sqrt(num_patches))
    if side * side == num_patches:
        return side, side

    for h_patches in range(side, 0, -1):
        if num_patches % h_patches == 0:
            return h_patches, num_patches // h_patches
    raise ValueError(f"Cannot infer patch grid for {num_patches} patches")


def normalize_map(values: np.ndarray) -> np.ndarray:
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.maximum(denom, 1e-12)
    return 1.0 - dot / denom


def compute_diff_maps(reference: np.ndarray, changed: np.ndarray, grid: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != changed.shape:
        raise ValueError(f"Feature shapes must match, got {reference.shape} and {changed.shape}")

    h_patches, w_patches = infer_grid(reference.shape[0], grid)
    cos_map = cosine_distance(reference, changed).reshape(h_patches, w_patches)
    l2_map = np.linalg.norm(reference - changed, axis=1).reshape(h_patches, w_patches)
    return cos_map, l2_map


def save_raw_maps(save_path: Path, cos_map: np.ndarray, l2_map: np.ndarray) -> None:
    raw_path = save_path.with_suffix(".npz")
    np.savez_compressed(raw_path, cosine_distance=cos_map, l2_distance=l2_map)
    print(f"Saved raw diff maps to {raw_path}")


def plot_diff_maps(
    cos_map: np.ndarray,
    l2_map: np.ndarray,
    save_path: Optional[str],
    title: str,
    cmap: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(cos_map, cmap=cmap, interpolation="nearest")
    axes[0].set_title("Cosine Distance\nPatch Resolution")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], shrink=0.75)

    im1 = axes[1].imshow(l2_map, cmap=cmap, interpolation="nearest")
    axes[1].set_title("L2 Distance\nPatch Resolution")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], shrink=0.75)

    combined = 0.5 * normalize_map(cos_map) + 0.5 * normalize_map(l2_map)
    im2 = axes[2].imshow(combined, cmap=cmap, interpolation="bilinear")
    axes[2].set_title("Combined Normalized Change")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], shrink=0.75)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        save_raw_maps(output_path, cos_map, l2_map)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two DINOv3 patch feature maps")
    parser.add_argument("reference", type=str, help="Reference feature directory or .npy/.npz file")
    parser.add_argument("changed", type=str, help="Changed feature directory or .npy/.npz file")
    parser.add_argument("--feature-key", type=str, default="patch_features_layer_-1",
                        help="Feature name to load when inputs are directories")
    parser.add_argument("--image-name", type=str,
                        help="Image key to load when comparing .npz files from directory processing")
    parser.add_argument("--grid", type=str, help="Patch grid as HxW, for example 28x28")
    parser.add_argument("--drop-prefix-tokens", type=int, default=0,
                        help="Drop this many leading non-patch tokens before comparison")
    parser.add_argument("--no-auto-drop-register-tokens", action="store_true",
                        help="Disable automatic removal of 4 DINOv3 register/storage tokens")
    parser.add_argument("--save-path", type=str, help="Where to save the comparison PNG")
    parser.add_argument("--title", type=str, default="DINOv3 Patch Feature Difference")
    parser.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    auto_drop_register_tokens = not args.no_auto_drop_register_tokens and args.grid is None
    reference = load_feature(args.reference, args.feature_key, args.image_name)
    changed = load_feature(args.changed, args.feature_key, args.image_name)
    reference = trim_prefix_tokens(reference, args.drop_prefix_tokens, auto_drop_register_tokens)
    changed = trim_prefix_tokens(changed, args.drop_prefix_tokens, auto_drop_register_tokens)
    cos_map, l2_map = compute_diff_maps(reference, changed, args.grid)

    print(f"Reference feature shape: {reference.shape}")
    print(f"Changed feature shape: {changed.shape}")
    print(f"Patch grid: {cos_map.shape[0]} x {cos_map.shape[1]}")
    print(f"Cosine distance range: [{cos_map.min():.6f}, {cos_map.max():.6f}]")
    print(f"L2 distance range: [{l2_map.min():.6f}, {l2_map.max():.6f}]")

    plot_diff_maps(cos_map, l2_map, args.save_path, args.title, args.cmap, args.dpi)


if __name__ == "__main__":
    main()
