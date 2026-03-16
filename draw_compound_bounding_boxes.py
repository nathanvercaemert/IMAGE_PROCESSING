"""
Draw compound bounding boxes on SKEW-prefixed images using pyvips.

Walks image_dir recursively, reads the matching *.boxes.json file from
data_dir (created by detect_bounding_boxes_with_paddleocr.py), computes
a single compound bounding rectangle around all detected text regions,
adds a 100-pixel buffer, and draws a 30-pixel-wide black rectangle just
outside the buffer (no overlap with the buffer or text area).  Every
image filename must start with the prefix "SKEW"; the script terminates
immediately if any does not.

The compound bounding box coordinates (with buffer, clamped to image
dimensions) are written to a *.compound.txt file under data_dir for
use by a future crop step.  The file contains four lines: left, top,
width, height.

The result is saved with the "BOUND" prefix and the original "SKEW"
file is removed.  Images with no detected bounding boxes are simply
renamed from "SKEW" to "BOUND" (no rectangle drawn, no crop data
written).

WARNING: This script modifies and removes input files.

Usage:
    python draw_compound_bounding_boxes.py <image_dir> <data_dir>

Requires: pyvips (with libvips) installed
"""

import argparse
import json
import os
import sys

import pyvips

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
}

BUFFER_PX = 100
RECT_WIDTH_PX = 30


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths under *directory* (recursive)."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def verify_skew_prefix(images: list[str]) -> None:
    """Terminate if any image filename does not start with 'SKEW'."""
    for image_path in images:
        filename = os.path.basename(image_path)
        if not filename.startswith("SKEW"):
            sys.exit(
                f"ERROR: filename does not start with 'SKEW': {image_path}"
            )


def build_boxes_data_path(
    image_path: str, image_root: str, data_root: str,
) -> str:
    """Map an image path to its bounding box JSON file path under data_root."""
    rel = os.path.relpath(image_path, image_root)
    return os.path.join(data_root, rel + ".boxes.json")


def build_compound_data_path(
    image_path: str, image_root: str, data_root: str,
) -> str:
    """Map an image path to its compound box data file path under data_root."""
    rel = os.path.relpath(image_path, image_root)
    return os.path.join(data_root, rel + ".compound.txt")


def build_bound_path(image_path: str) -> str:
    """Replace the SKEW prefix of the filename with BOUND."""
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    return os.path.join(directory, "BOUND" + filename[4:])


def read_boxes(boxes_path: str) -> list[dict]:
    """Read bounding box data from a JSON file."""
    if not os.path.isfile(boxes_path):
        raise FileNotFoundError(f"Missing bounding box file: {boxes_path}")
    with open(boxes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_compound_box(
    boxes: list[dict],
) -> tuple[float, float, float, float]:
    """Compute the bounding rectangle around all polygon points.

    Returns (min_x, min_y, max_x, max_y).
    """
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for box in boxes:
        for x, y in box["points"]:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    return min_x, min_y, max_x, max_y


def make_solid(width: int, height: int, image: pyvips.Image) -> pyvips.Image:
    """Create a solid black image matching the band count and format of *image*."""
    patch = pyvips.Image.black(width, height, bands=image.bands)
    patch = patch.cast(image.format)
    return patch


def draw_compound_box(
    image_path: str, output_path: str, boxes: list[dict],
) -> tuple[int, int, int, int]:
    """Draw a compound bounding rectangle on the image and save.

    Returns the buffered crop region as (left, top, width, height),
    clamped to image dimensions.
    """
    print(f"\n>> Drawing compound box on '{image_path}'")

    image = pyvips.Image.new_from_file(image_path, access="sequential")
    w, h = image.width, image.height

    min_x, min_y, max_x, max_y = compute_compound_box(boxes)

    # Buffered compound box (the crop region)
    crop_left = max(0, int(min_x) - BUFFER_PX)
    crop_top = max(0, int(min_y) - BUFFER_PX)
    crop_right = min(w, int(max_x) + BUFFER_PX)
    crop_bottom = min(h, int(max_y) + BUFFER_PX)

    # Inner edge of the rectangle = buffered compound box
    inner_left = crop_left
    inner_top = crop_top
    inner_right = crop_right
    inner_bottom = crop_bottom

    # Outer edge of the rectangle = inner edge + rect width
    outer_left = max(0, inner_left - RECT_WIDTH_PX)
    outer_top = max(0, inner_top - RECT_WIDTH_PX)
    outer_right = min(w, inner_right + RECT_WIDTH_PX)
    outer_bottom = min(h, inner_bottom + RECT_WIDTH_PX)

    # Draw 4 filled strips forming the border (black).
    # Top strip: full width, from outer_top to inner_top
    top_h = inner_top - outer_top
    if top_h > 0:
        strip_w = outer_right - outer_left
        if strip_w > 0:
            image = image.insert(make_solid(strip_w, top_h, image), outer_left, outer_top)

    # Bottom strip: full width, from inner_bottom to outer_bottom
    bot_h = outer_bottom - inner_bottom
    if bot_h > 0:
        strip_w = outer_right - outer_left
        if strip_w > 0:
            image = image.insert(make_solid(strip_w, bot_h, image), outer_left, inner_bottom)

    # Left strip: between top and bottom strips
    left_w = inner_left - outer_left
    mid_h = inner_bottom - inner_top
    if left_w > 0 and mid_h > 0:
        image = image.insert(make_solid(left_w, mid_h, image), outer_left, inner_top)

    # Right strip: between top and bottom strips
    right_w = outer_right - inner_right
    if right_w > 0 and mid_h > 0:
        image = image.insert(make_solid(right_w, mid_h, image), inner_right, inner_top)

    xres = image.get("Xres")
    yres = image.get("Yres")
    image = image.copy(xres=xres, yres=yres)

    image.write_to_file(output_path)
    print(f"   Saved '{output_path}'")

    crop_w = crop_right - crop_left
    crop_h = crop_bottom - crop_top
    return crop_left, crop_top, crop_w, crop_h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw compound bounding boxes on SKEW-prefixed images "
                    "using pyvips and saved bounding box JSON files.  "
                    "Replaces SKEW prefix with BOUND and removes the original."
    )
    parser.add_argument("image_dir", help="Directory containing SKEW-prefixed images")
    parser.add_argument("data_dir", help="Directory containing bounding box JSON files")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"ERROR: image directory not found: {args.image_dir}")

    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: data directory not found: {args.data_dir}")

    images = collect_images(args.image_dir)
    if not images:
        sys.exit(f"ERROR: no image files found in {args.image_dir}")

    verify_skew_prefix(images)

    total = len(images)
    failed: list[tuple[str, str]] = []

    print(f"Found {total} image(s) under '{args.image_dir}'")
    print(f"Processing in place (SKEW -> BOUND)\n")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        boxes_data_path = build_boxes_data_path(
            image_path, args.image_dir, args.data_dir
        )
        compound_data_path = build_compound_data_path(
            image_path, args.image_dir, args.data_dir
        )
        bound_path = build_bound_path(image_path)

        print(f"--- [{idx}/{total}] {rel} ---")
        try:
            boxes = read_boxes(boxes_data_path)

            if not boxes:
                os.rename(image_path, bound_path)
                action = "renamed (no bounding boxes detected)"
            else:
                crop_left, crop_top, crop_w, crop_h = draw_compound_box(
                    image_path, bound_path, boxes
                )
                os.remove(image_path)

                os.makedirs(os.path.dirname(compound_data_path), exist_ok=True)
                with open(compound_data_path, "w", encoding="utf-8") as f:
                    f.write(f"{crop_left}\n{crop_top}\n{crop_w}\n{crop_h}\n")

                action = (
                    f"drew compound box around {len(boxes)} region(s), "
                    f"crop={crop_left},{crop_top} {crop_w}x{crop_h}"
                )
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
            print(f"    FAILED: {e}\n", file=sys.stderr)
            failed.append((rel, str(e)))
            continue

        print(f"    OK ({action})\n")

    print("=" * 60)
    print(f"Processed {total - len(failed)}/{total} image(s) successfully.")
    if failed:
        print(f"\n{len(failed)} failure(s):")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
