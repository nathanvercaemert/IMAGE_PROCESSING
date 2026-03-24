"""
Crop BOUND-prefixed images to their compound bounding box using pyvips.

Reads the matching *.compound.txt file for each image in image_dir from
data_dir (created by draw_compound_bounding_boxes.py), and crops each
image to the stored region (left, top, width, height).  Every image
filename must start with the prefix "BOUND"; the script terminates
immediately if any does not.

The cropped result is saved with the "CROP" prefix and the original
"BOUND" file is removed.  Images without a compound data file (i.e.
those that had no detected bounding boxes) are simply renamed from
"BOUND" to "CROP" (no crop applied).

WARNING: This script modifies and removes input files.

Usage:
    python crop_compound_bounding_boxes.py <image_dir> <data_dir>

Requires: pyvips (with libvips) installed
"""

import argparse
import logging
import os
import sys

import pyvips

logger = logging.getLogger("crop_compound_bounding_boxes")


def _configure_logging() -> None:
    """Set up one-line structured logging to stderr."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("pyvips").setLevel(logging.WARNING)


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
}


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths in *directory*."""
    files = []
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(directory, name))
    files.sort()
    return files


def verify_bound_prefix(images: list[str]) -> None:
    """Terminate if any image filename does not start with 'BOUND'."""
    for image_path in images:
        filename = os.path.basename(image_path)
        if not filename.startswith("BOUND"):
            logger.error("filename does not start with 'BOUND': %s", image_path)
            sys.exit(1)


def build_compound_data_path(image_path: str, data_root: str) -> str:
    """Map an image path to its compound box data file path under data_root.

    The compound file was written against the SKEW-prefixed name, so we
    must reconstruct that name.
    """
    filename = os.path.basename(image_path)
    skew_name = "SKEW" + filename[5:]
    return os.path.join(data_root, skew_name + ".compound.txt")


def build_crop_path(image_path: str) -> str:
    """Replace the BOUND prefix of the filename with CROP."""
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    return os.path.join(directory, "CROP" + filename[5:])


def read_compound_data(compound_path: str) -> tuple[int, int, int, int]:
    """Read left, top, width, height from a compound data file."""
    with open(compound_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    if len(lines) < 4:
        raise ValueError(
            f"Compound data file has fewer than 4 lines: {compound_path}"
        )

    try:
        left = int(lines[0])
        top = int(lines[1])
        width = int(lines[2])
        height = int(lines[3])
    except ValueError:
        raise ValueError(
            f"Could not parse crop values from {compound_path}: "
            f"{lines[:4]!r}"
        )

    return left, top, width, height


def crop_image(
    image_path: str, output_path: str,
    left: int, top: int, width: int, height: int,
) -> None:
    """Crop *image_path* to the given region and save to *output_path*."""
    logger.debug("Cropping '%s' to %d,%d %dx%d", image_path, left, top, width, height)

    image = pyvips.Image.new_from_file(image_path, access="sequential")

    cropped = image.crop(left, top, width, height)

    xres = image.get("Xres")
    yres = image.get("Yres")
    cropped = cropped.copy(xres=xres, yres=yres)

    cropped.write_to_file(output_path)
    logger.debug("Saved '%s'", output_path)


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Crop BOUND-prefixed images to their compound bounding "
                    "box using pyvips and saved compound data files.  "
                    "Replaces BOUND prefix with CROP and removes the original."
    )
    parser.add_argument("image_dir", help="Directory containing BOUND-prefixed images")
    parser.add_argument("data_dir", help="Directory containing compound data files")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        logger.error("image directory not found: %s", args.image_dir)
        sys.exit(1)

    if not os.path.isdir(args.data_dir):
        logger.error("data directory not found: %s", args.data_dir)
        sys.exit(1)

    images = collect_images(args.image_dir)
    if not images:
        logger.error("no image files found in %s", args.image_dir)
        sys.exit(1)

    verify_bound_prefix(images)

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, args.image_dir)
    logger.info("Processing in place (BOUND -> CROP)")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        compound_data_path = build_compound_data_path(image_path, args.data_dir)
        crop_path = build_crop_path(image_path)

        try:
            if not os.path.isfile(compound_data_path):
                os.rename(image_path, crop_path)
                action = "renamed (no compound data, no crop applied)"
            else:
                left, top, width, height = read_compound_data(compound_data_path)
                crop_image(image_path, crop_path, left, top, width, height)
                os.remove(image_path)
                action = f"cropped to {left},{top} {width}x{height}"
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        logger.debug("[%d/%d] %s -- OK (%s)", idx, total, rel, action)

    logger.info("Processed %d/%d image(s) successfully.", total - len(failed), total)
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
