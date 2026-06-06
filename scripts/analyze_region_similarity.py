#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


def load_feature(path: str, feature_key: str, image_name: Optional[str] = None) -> np.ndarray:
    """Load a [num_patches, dim] feature from a feature dir, .npy, or .npz."""
    path_obj = Path(path)
    if path_obj.is_dir():
        path_obj = path_obj / f"{feature_key}.npy"
        if not path_obj.exists():
            npz_path = path_obj.with_suffix(".npz")
            if npz_path.exists():
                path_obj = npz_path

    if path_obj.suffix == ".npy":
        feature = np.load(path_obj)
    elif path_obj.suffix == ".npz":
        npz_data = np.load(path_obj)
        keys = list(npz_data.keys())
        if image_name is None:
            if len(keys) != 1:
                raise ValueError(f"{path_obj} has multiple images; pass image_name. Available: {keys}")
            image_name = keys[0]
        if image_name not in npz_data:
            raise ValueError(f"Image {image_name} not found in {path_obj}. Available: {keys}")
        feature = npz_data[image_name]
    else:
        raise ValueError(f"Unsupported feature path: {path_obj}")

    if feature.ndim != 2:
        raise ValueError(f"Expected [num_patches, dim], got {feature.shape} from {path_obj}")
    return feature.astype(np.float32, copy=False)


def infer_grid(num_patches: int, grid: Optional[str] = None) -> Tuple[int, int]:
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
    h_patches, w_patches = grid
    image = Image.open(mask_path)
    if image.mode == "RGBA":
        mask = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
    else:
        mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (w_patches, h_patches),
        resample=Image.Resampling.BILINEAR,
    )
    mask_grid = np.asarray(resized, dtype=np.float32) / 255.0
    selected = mask_grid > threshold
    if not selected.any():
        raise ValueError(f"Mask selected zero patches: {mask_path}")
    return selected


def normalize(feature: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(feature, axis=-1, keepdims=True)
    return feature / np.maximum(norm, eps)


def build_prototype(feature: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = feature[mask.reshape(-1)]
    prototype = selected.mean(axis=0)
    return normalize(prototype[None, :])[0]


def load_manifest(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    samples = data.get("samples", data if isinstance(data, list) else None)
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest must contain a non-empty 'samples' list")
    return samples


def save_similarity_matrix(
    matrix: np.ndarray,
    names: list[str],
    labels: list[str],
    save_path: Path,
    title: str,
    dpi: int,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(8, 0.45 * len(names))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
    tick_labels = [f"{name}\n{label}" for name, label in zip(names, labels)]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved similarity matrix to {save_path}")


def save_matrix_csv(matrix: np.ndarray, names: list[str], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + [f"{value:.8f}" for value in row])
    print(f"Saved similarity CSV to {save_path}")


def compute_metrics(matrix: np.ndarray, labels: list[str], positive_label: str) -> dict[str, float]:
    labels_array = np.asarray(labels)
    n = len(labels)
    off_diag = ~np.eye(n, dtype=bool)

    pos = labels_array == positive_label
    neg = ~pos

    pos_pos = off_diag & pos[:, None] & pos[None, :]
    pos_neg = pos[:, None] & neg[None, :]
    neg_neg = off_diag & neg[:, None] & neg[None, :]

    def mean_or_nan(mask: np.ndarray) -> float:
        values = matrix[mask]
        return float(values.mean()) if values.size else float("nan")

    pos_pos_mean = mean_or_nan(pos_pos)
    pos_neg_mean = mean_or_nan(pos_neg)
    neg_neg_mean = mean_or_nan(neg_neg)
    return {
        "positive_positive_mean": pos_pos_mean,
        "positive_negative_mean": pos_neg_mean,
        "negative_negative_mean": neg_neg_mean,
        "functional_margin": pos_pos_mean - pos_neg_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze masked region prototype similarity across DINOv3 layers")
    parser.add_argument("--manifest", required=True, help="YAML manifest describing samples")
    parser.add_argument("--layers", nargs="+", default=["-1"], help="Layer suffixes, e.g. -1 -4 -8")
    parser.add_argument("--feature-template", default="patch_features_layer_{layer}",
                        help="Feature key template when feature_path is a directory")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--positive-label", default="handle")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for layer in args.layers:
        feature_key = args.feature_template.format(layer=layer)
        names = []
        labels = []
        prototypes = []

        for sample in samples:
            name = sample["name"]
            label = sample.get("label", "unknown")
            feature_path = sample["feature_path"]
            mask_path = sample["mask_path"]
            image_name = sample.get("image_name")
            grid = sample.get("grid")

            feature = load_feature(feature_path, feature_key, image_name=image_name)
            patch_grid = infer_grid(feature.shape[0], grid)
            mask = load_mask(mask_path, patch_grid, args.mask_threshold)
            prototype = build_prototype(feature, mask)

            names.append(name)
            labels.append(label)
            prototypes.append(prototype)
            print(f"Layer {layer}: {name} label={label} selected_patches={int(mask.sum())}")

        prototypes_array = normalize(np.stack(prototypes, axis=0))
        matrix = prototypes_array @ prototypes_array.T
        metrics = compute_metrics(matrix, labels, args.positive_label)
        metric_rows.append({"layer": layer, **metrics})

        layer_name = str(layer).replace("-", "minus_")
        save_similarity_matrix(
            matrix,
            names,
            labels,
            save_dir / f"similarity_matrix_layer_{layer_name}.png",
            title=f"Region Prototype Similarity Layer {layer}",
            dpi=args.dpi,
        )
        save_matrix_csv(matrix, names, save_dir / f"similarity_matrix_layer_{layer_name}.csv")

    metrics_path = save_dir / "layer_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        fieldnames = ["layer", "positive_positive_mean", "positive_negative_mean", "negative_negative_mean", "functional_margin"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Saved layer metrics to {metrics_path}")


if __name__ == "__main__":
    main()
