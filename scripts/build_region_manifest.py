#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def find_scene_image(scene_dir: Path) -> Path:
    images = [
        path for path in scene_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No image found in {scene_dir}")
    if len(images) > 1:
        print(f"Warning: multiple images in {scene_dir}; using {images[0].name}")
    return sorted(images)[0]


def find_mask(scene_dir: Path, mask_name: str) -> Path:
    mask_path = scene_dir / "mask" / mask_name
    if mask_path.exists():
        return mask_path

    mask_dir = scene_dir / "mask"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"No mask directory found in {scene_dir}")

    masks = [
        path for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not masks:
        raise FileNotFoundError(f"No mask image found in {mask_dir}")
    if len(masks) > 1:
        print(f"Warning: {mask_name} not found in {mask_dir}; using {masks[0].name}")
    return sorted(masks)[0]


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

        image_path = find_scene_image(scene_dir)
        mask_path = find_mask(scene_dir, mask_name)
        feature_dir = find_feature_dir(feature_root, scene_id)

        samples.append(
            {
                "name": f"{label}_{scene_id}",
                "label": label,
                "scene_id": scene_id,
                "image_name": image_path.stem,
                "image_path": str(image_path),
                "feature_path": str(feature_dir),
                "mask_path": str(mask_path),
            }
        )
        print(f"Added {scene_id}: image={image_path.name}, mask={mask_path.name}, features={feature_dir}")

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
    parser.add_argument("--scene-ids", nargs="*", help="Optional subset of scene IDs to include")
    args = parser.parse_args()

    manifest = build_manifest(
        image_root=Path(args.image_root),
        feature_root=Path(args.feature_root),
        mask_name=args.mask_name,
        label=args.label,
        scene_ids=args.scene_ids,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Saved manifest with {len(manifest['samples'])} samples to {output}")


if __name__ == "__main__":
    main()
