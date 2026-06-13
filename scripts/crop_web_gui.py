#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


def default_output_path(image_path: Path) -> str:
    """Return a crop output path next to the source image."""
    suffix = image_path.suffix or ".png"
    return str(image_path.with_name(f"{image_path.stem}_crop{suffix}"))


def load_image(
    image_path: str,
    output_path: str,
) -> tuple[Image.Image, str, Optional[Image.Image], list[tuple[int, int]], str]:
    """Load an image from a server path for interactive cropping."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise gr.Error(f"Image path does not exist: {path}")

    image = Image.open(path).convert("RGB")
    if not output_path:
        output_path = default_output_path(path)

    return image, output_path, None, [], "Loaded image. Click two crop corners."


def parse_select_index(index: Any) -> tuple[int, int]:
    """Parse Gradio image select index as integer x/y coordinates."""
    if isinstance(index, dict):
        x = index.get("x")
        y = index.get("y")
        if x is None or y is None:
            raise gr.Error(f"Unsupported selection index: {index}")
        return int(round(x)), int(round(y))

    if isinstance(index, (tuple, list)) and len(index) >= 2:
        return int(round(index[0])), int(round(index[1]))

    raise gr.Error(f"Unsupported selection index: {index}")


def get_bbox(size: Tuple[int, int], points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return a valid crop box from two corner points."""
    if len(points) < 2:
        raise gr.Error("Click two crop corners first.")

    width, height = size
    (x1, y1), (x2, y2) = points[:2]
    x_min = max(0, min(x1, x2))
    x_max = min(width, max(x1, x2))
    y_min = max(0, min(y1, y2))
    y_max = min(height, max(y1, y2))

    if x_max <= x_min or y_max <= y_min:
        raise gr.Error("Crop box has zero area. Click two different corners.")

    return x_min, y_min, x_max, y_max


def crop_image(image: Image.Image, points: list[tuple[int, int]]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop the loaded image using selected points."""
    bbox = get_bbox(image.size, points)
    return image.crop(bbox), bbox


def draw_crop_preview(image: Image.Image, points: list[tuple[int, int]]) -> Image.Image:
    """Draw clicked points and the current crop preview on an image."""
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)

    for x, y in points[:2]:
        radius = max(3, min(preview.size) // 150)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 0, 0), outline=(255, 255, 255))

    if len(points) >= 2:
        bbox = get_bbox(preview.size, points)
        draw.rectangle(bbox, outline=(255, 0, 0), width=max(2, min(preview.size) // 250))

    return preview


def add_crop_point(
    image_path: str,
    points: list[tuple[int, int]],
    evt: gr.SelectData,
) -> tuple[Image.Image, list[tuple[int, int]], Optional[Image.Image], str]:
    """Add one clicked crop corner and preview the resulting crop."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise gr.Error(f"Image path does not exist: {path}")

    image = Image.open(path).convert("RGB")
    x, y = parse_select_index(evt.index)
    width, height = image.size
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))

    points = list(points or [])
    if len(points) >= 2:
        points = []
    points.append((x, y))

    preview = draw_crop_preview(image, points)
    crop_preview = None
    if len(points) == 2:
        crop_preview, bbox = crop_image(image, points)
        message = f"Crop ready: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}"
    else:
        message = f"First corner selected: x={x}, y={y}. Click the second corner."

    return preview, points, crop_preview, message


def save_crop(
    image_path: str,
    output_path: str,
    points: list[tuple[int, int]],
) -> tuple[str, Image.Image]:
    """Save the selected crop as an RGB image."""
    if not output_path:
        raise gr.Error("Output crop path is required.")

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise gr.Error(f"Image path does not exist: {path}")

    image = Image.open(path).convert("RGB")
    cropped, bbox = crop_image(image, list(points or []))
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output)
    return f"Saved crop to {output} with bbox={bbox}", cropped


def clear_crop(
    image_path: str,
) -> tuple[Optional[Image.Image], Optional[Image.Image], list[tuple[int, int]], str]:
    """Reload the current image and clear the crop preview."""
    if not image_path:
        return None, None, [], ""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise gr.Error(f"Image path does not exist: {path}")
    image = Image.open(path).convert("RGB")
    return image, None, [], "Cleared crop"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Image Cropper") as app:
        gr.Markdown("## Image Cropper\nLoad an image from a server path, click two crop corners, then save the crop.")

        with gr.Row():
            image_path = gr.Textbox(label="Image path on server", placeholder="/nas/qirui/dinov3/src/.../image.jpg")
            output_path = gr.Textbox(label="Output crop path", placeholder="/nas/qirui/dinov3/src/.../image_crop.jpg")

        with gr.Row():
            load_button = gr.Button("Load Image", variant="primary")
            save_button = gr.Button("Save Crop", variant="primary")
            clear_button = gr.Button("Clear")

        crop_points = gr.State([])
        crop_image_view = gr.Image(label="Click two crop corners", type="pil", interactive=True)
        status = gr.Textbox(label="Status", interactive=False)
        crop_preview = gr.Image(label="Crop preview", type="pil")

        load_button.click(
            load_image,
            inputs=[image_path, output_path],
            outputs=[crop_image_view, output_path, crop_preview, crop_points, status],
        )
        crop_image_view.select(
            add_crop_point,
            inputs=[image_path, crop_points],
            outputs=[crop_image_view, crop_points, crop_preview, status],
        )
        save_button.click(
            save_crop,
            inputs=[image_path, output_path, crop_points],
            outputs=[status, crop_preview],
        )
        clear_button.click(
            clear_crop,
            inputs=[image_path],
            outputs=[crop_image_view, crop_preview, crop_points, status],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Web GUI for cropping server images")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true", help="Enable Gradio public share link")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
