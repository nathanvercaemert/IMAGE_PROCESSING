"""
Batch assign a scanner ICC profile and convert to a working colour space
for every image in an input directory.  Processed images are written to
the output directory; originals are never modified.

Usage:
    python batch_assign_convert_icc_profile.py <input_dir> <output_dir> --scanner scanner.icc --working working.icc

Requires: exiftool, ImageMagick (magick) on PATH
"""

import argparse
import logging
import os
import sys

from assign_convert_icc_profile import assign_convert_icc

logger = logging.getLogger("batch_assign_convert_icc_profile")


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


# Extensions recognised as images to process (case-insensitive)
IMAGE_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".psd", ".exr",
}


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths in *directory* (non-recursive)."""
    files = []
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(directory, name))
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch assign + convert ICC profiles for all images in a directory."
    )
    parser.add_argument("input_dir", help="Directory containing source images")
    parser.add_argument("output_dir", help="Directory to write processed images into")
    parser.add_argument("--scanner", required=True,
                        help="Scanner ICC profile to assign")
    parser.add_argument("--working", required=True,
                        help="Working-space ICC profile to convert to")
    args = parser.parse_args()

    _configure_logging()

    if not os.path.isdir(args.input_dir):
        logger.error("input directory not found: %s", args.input_dir)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    images = collect_images(args.input_dir)
    if not images:
        logger.error("no image files found in %s", args.input_dir)
        sys.exit(1)

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) in '%s'", total, args.input_dir)
    logger.info("Output directory: '%s'", args.output_dir)

    for idx, input_path in enumerate(images, 1):
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, filename)

        try:
            assign_convert_icc(
                input_path, output_path, args.scanner, args.working
            )
            logger.info("[%d/%d] %s -- OK", idx, total, filename)
        except (RuntimeError, FileNotFoundError) as e:
            logger.error("[%d/%d] %s -- %s", idx, total, filename, e)
            failed.append((filename, str(e)))

    logger.info("Processed %d/%d image(s) successfully.", total - len(failed), total)
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  %s: %s", name, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
