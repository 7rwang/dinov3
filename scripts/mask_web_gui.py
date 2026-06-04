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


def load_image(
    image_path: str,
    output_path: str,
) -> tuple[Image.Image, Image.Image, str, Optional[Image.Image], list[tuple[int, int]], str]:
    """Load an image from a server path for interactive mask drawing."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise gr.Error(f"Image path does not exist: {path}")

    image = Image.open(path).convert("RGB")
    if not output_path:
        output_path = str(path.with_name(f"{path.stem}_mask.png"))

    return image, image, output_path, None, [], "Loaded image. Draw a mask or click two bbox corners."


def pil_or_array_to_rgba(layer: Any, size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Convert an editor layer to RGBA numpy array."""
    if layer is None:
        return None

    if isinstance(layer, np.ndarray):
        array = layer
    else:
        array = np.asarray(layer)

    if array.ndim == 2:
        alpha = array.astype(np.uint8)
        return np.dstack([alpha, alpha, alpha, alpha])

    if array.ndim != 3:
        return None

    if array.shape[-1] == 4:
        rgba = array
    elif array.shape[-1] == 3:
        rgb = array
        alpha = (np.any(rgb > 0, axis=-1).astype(np.uint8) * 255)[..., None]
        rgba = np.concatenate([rgb, alpha], axis=-1)
    else:
        return None

    image = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    if image.size != size:
        image = image.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(image)


def extract_mask(editor_value: Any, threshold: int) -> tuple[np.ndarray, Tuple[int, int]]:
    """Extract a binary mask from Gradio ImageEditor data."""
    if not isinstance(editor_value, dict):
        raise gr.Error("No editor data found. Load an image and draw on it first.")

    background = editor_value.get("background")
    composite = editor_value.get("composite")
    layers = editor_value.get("layers") or []

    reference = background if background is not None else composite
    if reference is None:
        raise gr.Error("No image loaded in the editor.")

    reference_image = Image.fromarray(reference.astype(np.uint8)) if isinstance(reference, np.ndarray) else reference
    width, height = reference_image.size
    mask = np.zeros((height, width), dtype=bool)

    for layer in layers:
        rgba = pil_or_array_to_rgba(layer, (width, height))
        if rgba is None:
            continue
        mask |= rgba[..., 3] > threshold

    if not mask.any() and composite is not None and background is not None:
        bg = np.asarray(background.convert("RGB") if isinstance(background, Image.Image) else background[..., :3])
        comp = np.asarray(composite.convert("RGB") if isinstance(composite, Image.Image) else composite[..., :3])
        if bg.shape[:2] != (height, width):
            bg = np.asarray(Image.fromarray(bg.astype(np.uint8)).resize((width, height)))
        if comp.shape[:2] != (height, width):
            comp = np.asarray(Image.fromarray(comp.astype(np.uint8)).resize((width, height)))
        mask = np.any(np.abs(comp.astype(np.int16) - bg.astype(np.int16)) > threshold, axis=-1)

    if not mask.any():
        raise gr.Error("Mask is empty. Draw on the image before saving.")

    return mask.astype(np.uint8) * 255, (width, height)


def save_mask(editor_value: Any, output_path: str, threshold: int) -> tuple[str, Image.Image]:
    """Save the drawn mask as a binary PNG."""
    if not output_path:
        raise gr.Error("Output mask path is required.")

    mask, _ = extract_mask(editor_value, threshold)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    mask_image = Image.fromarray(mask, mode="L")
    mask_image.save(output)
    return f"Saved binary mask to {output}", mask_image


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


def make_bbox_mask(size: Tuple[int, int], points: list[tuple[int, int]]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Create a binary rectangular mask from two corner points."""
    if len(points) < 2:
        raise gr.Error("Click two points on the bbox image first.")

    width, height = size
    (x1, y1), (x2, y2) = points[:2]
    x_min = max(0, min(x1, x2))
    x_max = min(width - 1, max(x1, x2))
    y_min = max(0, min(y1, y2))
    y_max = min(height - 1, max(y1, y2))

    if x_max <= x_min or y_max <= y_min:
        raise gr.Error("BBox has zero area. Click two different corners.")

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
    return mask, (x_min, y_min, x_max, y_max)


def draw_bbox_preview(image: Image.Image, points: list[tuple[int, int]]) -> Image.Image:
    """Draw clicked points and the current bbox preview on an image."""
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)

    for x, y in points[:2]:
        radius = max(3, min(preview.size) // 150)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 0, 0), outline=(255, 255, 255))

    if len(points) >= 2:
        _, bbox = make_bbox_mask(preview.size, points)
        draw.rectangle(bbox, outline=(255, 0, 0), width=max(2, min(preview.size) // 250))

    return preview


def add_bbox_point(
    image_path: str,
    points: list[tuple[int, int]],
    evt: gr.SelectData,
) -> tuple[Image.Image, list[tuple[int, int]], Optional[Image.Image], str]:
    """Add one clicked bbox point and preview the resulting rectangular mask."""
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

    preview = draw_bbox_preview(image, points)
    mask_preview = None
    if len(points) == 2:
        mask_preview, bbox = make_bbox_mask(image.size, points)
        message = f"BBox ready: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}"
    else:
        message = f"First corner selected: x={x}, y={y}. Click the second corner."

    return preview, points, mask_preview, message


def save_bbox_mask(
    image_path: str,
    output_path: str,
    points: list[tuple[int, int]],
) -> tuple[str, Image.Image]:
    """Save a binary rectangular mask from selected bbox points."""
    if not output_path:
        raise gr.Error("Output mask path is required.")

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise gr.Error(f"Image path does not exist: {path}")

    image = Image.open(path).convert("RGB")
    mask, bbox = make_bbox_mask(image.size, list(points or []))
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)
    return f"Saved bbox mask to {output} with bbox={bbox}", mask


def clear_mask(
    image_path: str,
) -> tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], list[tuple[int, int]], str]:
    """Reload the current image and clear the mask preview."""
    if not image_path:
        return None, None, None, [], ""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise gr.Error(f"Image path does not exist: {path}")
    image = Image.open(path).convert("RGB")
    return image, image, None, [], "Cleared mask"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Binary Mask Drawer") as app:
        gr.Markdown("## Binary Mask Drawer\nLoad an image from a server path, draw a mask, then save a 0/255 PNG mask.")

        with gr.Row():
            image_path = gr.Textbox(label="Image path on server", placeholder="/nas/qirui/dinov3/src/.../image.jpg")
            output_path = gr.Textbox(label="Output mask path", placeholder="/nas/qirui/dinov3/src/.../image_mask.png")

        with gr.Row():
            load_button = gr.Button("Load Image", variant="primary")
            save_button = gr.Button("Save Mask", variant="primary")
            clear_button = gr.Button("Clear")

        threshold = gr.Slider(0, 255, value=10, step=1, label="Mask extraction threshold")
        bbox_points = gr.State([])

        with gr.Tab("Brush mask"):
            editor = gr.ImageEditor(label="Draw mask", type="pil", image_mode="RGBA")

        with gr.Tab("BBox mask"):
            gr.Markdown("Click two corners on the image below. The second click creates a rectangular binary mask.")
            bbox_image = gr.Image(label="Click two bbox corners", type="pil", interactive=True)
            save_bbox_button = gr.Button("Save BBox Mask", variant="primary")

        status = gr.Textbox(label="Status", interactive=False)
        mask_preview = gr.Image(label="Saved binary mask preview", type="pil", image_mode="L")

        load_button.click(
            load_image,
            inputs=[image_path, output_path],
            outputs=[editor, bbox_image, output_path, mask_preview, bbox_points, status],
        )
        save_button.click(
            save_mask,
            inputs=[editor, output_path, threshold],
            outputs=[status, mask_preview],
        )
        bbox_image.select(
            add_bbox_point,
            inputs=[image_path, bbox_points],
            outputs=[bbox_image, bbox_points, mask_preview, status],
        )
        save_bbox_button.click(
            save_bbox_mask,
            inputs=[image_path, output_path, bbox_points],
            outputs=[status, mask_preview],
        )
        clear_button.click(
            clear_mask,
            inputs=[image_path],
            outputs=[editor, bbox_image, mask_preview, bbox_points, status],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Web GUI for drawing binary masks on server images")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Enable Gradio public share link")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
