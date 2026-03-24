"""
Determine skew angle for ROT-prefixed images using leptonica.

For each ROT-prefixed image in image_dir, runs lept_skew
and writes a text file under data_dir containing the skew angle and
confidence.  Every image filename must start with the prefix "ROT";
the script terminates immediately if any does not.

Usage:
    python determine_skew_angle.py <image_dir> <data_dir>

Requires: lept_skew on PATH
"""

import argparse
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger("determine_skew_angle")


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


def detect_skew(image_path: str) -> tuple[float, float]:
    """Run lept_skew on *image_path* and return (angle, confidence)."""
    cmd = ["lept_skew", image_path]
    logger.debug("Detecting skew on '%s' -- $ %s", image_path, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        logger.debug("%s", result.stderr.rstrip())
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse lept_skew output for {image_path}: {e}\n"
            f"stdout:\n{result.stdout.strip()}"
        )

    angle = info["angle"]
    confidence = info["confidence"]
    return float(angle), float(confidence)


def build_output_path(image_path: str, data_root: str) -> str:
    """Map an image path to its skew data file path under data_root."""
    filename = os.path.basename(image_path)
    return os.path.join(data_root, filename + ".skew.txt")


def verify_rot_prefix(images: list[str]) -> None:
    """Terminate if any image filename does not start with 'ROT'."""
    for image_path in images:
        filename = os.path.basename(image_path)
        if not filename.startswith("ROT"):
            logger.error("filename does not start with 'ROT': %s", image_path)
            sys.exit(1)


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Determine skew angle for ROT-prefixed images using leptonica."
    )
    parser.add_argument("image_dir", help="Directory containing ROT-prefixed images")
    parser.add_argument("data_dir", help="Directory to store skew angle files")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        logger.error("image directory not found: %s", args.image_dir)
        sys.exit(1)

    images = collect_images(args.image_dir)
    if not images:
        logger.error("no image files found in %s", args.image_dir)
        sys.exit(1)

    verify_rot_prefix(images)

    os.makedirs(args.data_dir, exist_ok=True)

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, args.image_dir)
    logger.info("Data directory: '%s'", args.data_dir)

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        out_path = build_output_path(image_path, args.data_dir)

        try:
            angle, confidence = detect_skew(image_path)
        except RuntimeError as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{angle}\n{confidence}\n")
        logger.debug("[%d/%d] %s -- OK (angle=%.4f, confidence=%.4f)", idx, total, rel, angle, confidence)

    logger.info("Processed %d/%d image(s) successfully.", total - len(failed), total)
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
