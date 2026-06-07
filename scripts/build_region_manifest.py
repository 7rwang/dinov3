#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def find_scene_images(scene_dir: Path) -> list[Path]:
    images = sorted([
        path for path in scene_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not images:
        raise FileNotFoundError(f"No image found in {scene_dir}")
    return images


def find_scene_masks(scene_dir: Path) -> list[Path]:
    mask_dir = scene_dir / "mask"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"No mask directory found in {scene_dir}")

    masks = sorted([
        path for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not masks:
        raise FileNotFoundError(f"No mask image found in {mask_dir}")
    return masks


def normalize_stem(stem: str) -> str:
    normalized = stem.lower()
    for token in ["_mask", "mask_", "mask", "_handle", "handle_", "neg_", "negative_", "_neg", "_negative"]:
        normalized = normalized.replace(token, "")
    return normalized.strip("_-.")


def is_negative_mask(mask_path: Path, negative_prefix: str) -> bool:
    return mask_path.stem.lower().startswith(negative_prefix.lower())


def match_masks_for_image(
    image_path: Path,
    masks: list[Path],
    preferred_mask_name: str | None,
    allow_all_if_ambiguous: bool,
) -> list[Path]:
    if allow_all_if_ambiguous:
        return masks

    if preferred_mask_name and len(masks) == 1 and masks[0].name == preferred_mask_name:
        return masks

    image_stem = normalize_stem(image_path.stem)

    exact_matches = [mask for mask in masks if normalize_stem(mask.stem) == image_stem]
    contains_matches = [
        mask for mask in masks
        if image_stem and (image_stem in normalize_stem(mask.stem) or normalize_stem(mask.stem) in image_stem)
    ]
    matched = sorted(set(exact_matches + contains_matches))
    if matched:
        return matched

    if len(masks) == 1:
        return masks

    raise ValueError(
        f"Cannot match mask for image {image_path.name}. Available masks: {[mask.name for mask in masks]}"
    )


def find_feature_dir(feature_root: Path, scene_id: str) -> Path:
    scene_feature_root = feature_root / scene_id
    if not scene_feature_root.is_dir():
        raise FileNotFoundError(f"No feature directory found for scene {scene_id}: {scene_feature_root}")

    candidates = [
        path / "features"
        for path in scene_feature_root.iterdir()
        if path.is_dir() and (path / "features").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No timestamp/features directory found in {scene_feature_root}")
    return sorted(candidates)[-1]


def build_manifest(
    image_root: Path,
    feature_root: Path,
    mask_name: str,
    label: str,
    negative_label: str,
    negative_prefix: str,
    scene_ids: list[str] | None,
) -> dict:
    if scene_ids:
        scene_dirs = [image_root / scene_id for scene_id in scene_ids]
    else:
        scene_dirs = [path for path in image_root.iterdir() if path.is_dir()]

    samples = []
    for scene_dir in sorted(scene_dirs):
        scene_id = scene_dir.name
        if not scene_dir.is_dir():
            print(f"Skipping missing scene directory: {scene_dir}")
            continue

        image_paths = find_scene_images(scene_dir)
        mask_paths = find_scene_masks(scene_dir)
        feature_dir = find_feature_dir(feature_root, scene_id)

        for image_path in image_paths:
            matched_masks = match_masks_for_image(
                image_path,
                mask_paths,
                mask_name,
                allow_all_if_ambiguous=len(image_paths) == 1,
            )
            for mask_path in matched_masks:
                sample_label = negative_label if is_negative_mask(mask_path, negative_prefix) else label
                sample_name = f"{sample_label}_{scene_id}_{image_path.stem}"
                if sample_label == negative_label:
                    sample_name = f"{sample_name}_{mask_path.stem}"

                samples.append(
                    {
                        "name": sample_name,
                        "label": sample_label,
                        "scene_id": scene_id,
                        "image_name": image_path.stem,
                        "image_path": str(image_path),
                        "feature_path": str(feature_dir),
                        "mask_path": str(mask_path),
                    }
                )
                print(
                    f"Added {sample_name}: label={sample_label}, "
                    f"image={image_path.name}, mask={mask_path.name}, features={feature_dir}"
                )

    if not samples:
        raise ValueError("No samples were added to the manifest")
    return {"samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a region similarity manifest from scene folders")
    parser.add_argument("--image-root", required=True, help="Root containing scene_id/image/mask folders")
    parser.add_argument("--feature-root", required=True, help="Root containing scene_id/timestamp/features folders")
    parser.add_argument("--output", required=True, help="Output YAML manifest path")
    parser.add_argument("--mask-name", default="mask2.png", help="Preferred mask filename inside each mask directory")
    parser.add_argument("--label", default="handle", help="Label to assign to generated samples")
    parser.add_argument("--negative-label", default="negative", help="Label to assign to negative masks")
    parser.add_argument("--negative-prefix", default="neg", help="Mask filename prefix used to identify negatives")
    parser.add_argument("--scene-ids", nargs="*", help="Optional subset of scene IDs to include")
    args = parser.parse_args()

    manifest = build_manifest(
        image_root=Path(args.image_root),
        feature_root=Path(args.feature_root),
        mask_name=args.mask_name,
        label=args.label,
        negative_label=args.negative_label,
        negative_prefix=args.negative_prefix,
        scene_ids=args.scene_ids,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Saved manifest with {len(manifest['samples'])} samples to {output}")


if __name__ == "__main__":
    main()
