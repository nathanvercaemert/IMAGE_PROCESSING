"""
Detect 0/180 image orientation with Tesseract OSD.

For each image in image_dir, writes a sibling text file
under data_dir that contains either "0" or "180".

Usage:
    python detect_orientation_with_tesseract_osd.py <image_dir> <data_dir>

Requires: tesseract (with osd.traineddata) on PATH
"""

import argparse
import logging
import os
import re
import subprocess
import sys

logger = logging.getLogger("detect_orientation_with_tesseract_osd")


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

ORIENTATION_RE = re.compile(r"Orientation in degrees:\s*(\d+)")


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths in *directory*."""
    files = []
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(directory, name))
    files.sort()
    return files


def detect_orientation(image_path: str) -> int:
    """Run Tesseract OSD on *image_path* and return 0 or 180."""
    cmd = ["tesseract", image_path, "-", "--psm", "0", "-l", "osd"]
    logger.debug("Detecting orientation on '%s' -- $ %s", image_path, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    combined_output = f"{result.stdout}\n{result.stderr}"
    match = ORIENTATION_RE.search(combined_output)
    if match is None:
        raise RuntimeError(
            f"Could not parse orientation for {image_path}.\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout.strip()}\n\n"
            f"stderr:\n{result.stderr.strip()}"
        )

    orientation = int(match.group(1))
    if orientation not in {0, 180}:
        raise RuntimeError(
            f"Unexpected orientation {orientation} for {image_path}. "
            "This workflow only supports 0 or 180."
        )

    return orientation


def build_output_path(image_path: str, data_root: str) -> str:
    """Map an image path to its orientation text file path under data_root."""
    filename = os.path.basename(image_path)
    return os.path.join(data_root, filename + ".orientation.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect image orientation (0 or 180) with Tesseract OSD."
    )
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("data_dir", help="Directory to store orientation files")
    args = parser.parse_args()

    _configure_logging()

    if not os.path.isdir(args.image_dir):
        logger.error("image directory not found: %s", args.image_dir)
        sys.exit(1)

    images = collect_images(args.image_dir)
    if not images:
        logger.error("no image files found in %s", args.image_dir)
        sys.exit(1)

    os.makedirs(args.data_dir, exist_ok=True)

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, args.image_dir)
    logger.info("Data directory: '%s'", args.data_dir)

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        out_path = build_output_path(image_path, args.data_dir)

        try:
            orientation = detect_orientation(image_path)
        except RuntimeError as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{orientation}\n")
        logger.debug("[%d/%d] %s -- OK (%d)", idx, total, rel, orientation)

    logger.info("Processed %d/%d image(s) successfully.", total - len(failed), total)
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  %s: %s", name, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
