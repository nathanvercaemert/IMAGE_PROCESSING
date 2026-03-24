"""
Rotate upside-down images back upright with ImageMagick (in-place).

Reads the matching *.orientation.txt file for each image in image_dir
from data_dir (created by detect_orientation_with_tesseract_osd.py),
and processes each image in place.  Every image filename must start with
the prefix "RAW"; the script terminates immediately if any does not.

For images marked 180, the rotated result is saved with the "ROT" prefix
and the original "RAW" file is removed.  For images marked 0, the file
is simply renamed from "RAW" to "ROT" (no rotation needed).

WARNING: This script modifies and removes input files.

Usage:
    python fix_upside_down_with_magick.py <image_dir> <data_dir>

Requires: ImageMagick (magick) on PATH
"""

import argparse
import logging
import os
import re
import subprocess
import sys

logger = logging.getLogger("fix_upside_down_with_magick")


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


def run(cmd: list[str], description: str) -> None:
    """Run a command, logging what it does and aborting on failure."""
    logger.debug("%s -- $ %s", description, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.debug("%s", result.stdout.rstrip())
    if result.stderr:
        logger.debug("%s", result.stderr.rstrip())
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def get_bit_depth(image_path: str) -> str:
    """Return the bit depth of *image_path* via ImageMagick identify."""
    result = subprocess.run(
        ["magick", "identify", "-format", "%z", image_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not determine bit depth for {image_path}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def build_orientation_path(image_path: str, data_root: str) -> str:
    """Map an image path to its orientation text file path under data_root."""
    filename = os.path.basename(image_path)
    return os.path.join(data_root, filename + ".orientation.txt")


def build_rot_path(image_path: str) -> str:
    """Replace the RAW prefix of the filename with ROT."""
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    return os.path.join(directory, "ROT" + filename[3:])


def read_orientation(orientation_path: str) -> int:
    """Read and validate the orientation value from a text file."""
    if not os.path.isfile(orientation_path):
        raise FileNotFoundError(f"Missing orientation file: {orientation_path}")

    with open(orientation_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if raw not in {"0", "180"}:
        raise ValueError(
            f"Unexpected orientation value in {orientation_path}: {raw!r}. "
            "Expected 0 or 180."
        )
    return int(raw)


def verify_raw_prefix(images: list[str]) -> None:
    """Terminate if any image filename does not start with 'RAW'."""
    for image_path in images:
        filename = os.path.basename(image_path)
        if not filename.startswith("RAW"):
            logger.error("filename does not start with 'RAW': %s", image_path)
            sys.exit(1)


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Fix upside-down images in place using ImageMagick and "
                    "saved 0/180 orientation files.  Replaces RAW prefix "
                    "with ROT and removes the original."
    )
    parser.add_argument("image_dir", help="Directory containing RAW-prefixed images")
    parser.add_argument("data_dir", help="Directory containing orientation files")
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

    verify_raw_prefix(images)

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, args.image_dir)
    logger.info("Processing in place (RAW -> ROT)")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        orientation_path = build_orientation_path(image_path, args.data_dir)
        rot_path = build_rot_path(image_path)

        try:
            orientation = read_orientation(orientation_path)

            if orientation == 180:
                depth = get_bit_depth(image_path)
                run(
                    ["magick", image_path, "-depth", depth, "-rotate", "180", rot_path],
                    f"Rotating 180 (depth={depth}) -> '{os.path.basename(rot_path)}'",
                )
                os.remove(image_path)
                action = "rotated 180, removed original"
            else:
                os.rename(image_path, rot_path)
                action = "renamed (no rotation needed)"
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
