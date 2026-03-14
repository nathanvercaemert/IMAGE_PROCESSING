"""
Rotate upside-down images back upright with ImageMagick (in-place).

Walks image_dir recursively, reads the matching *.orientation.txt file
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
import os
import subprocess
import sys

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
}


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths under *directory* (recursive)."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def run(cmd: list[str], description: str) -> None:
    """Run a command, printing what it does and aborting on failure."""
    print(f"\n>> {description}")
    print(f"   $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def build_orientation_path(image_path: str, image_root: str, data_root: str) -> str:
    """Map an image path to its orientation text file path under data_root."""
    rel = os.path.relpath(image_path, image_root)
    return os.path.join(data_root, rel + ".orientation.txt")


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
            sys.exit(
                f"ERROR: filename does not start with 'RAW': {image_path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix upside-down images in place using ImageMagick and "
                    "saved 0/180 orientation files.  Replaces RAW prefix "
                    "with ROT and removes the original."
    )
    parser.add_argument("image_dir", help="Directory containing RAW-prefixed images")
    parser.add_argument("data_dir", help="Directory containing orientation files")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"ERROR: image directory not found: {args.image_dir}")

    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: data directory not found: {args.data_dir}")

    images = collect_images(args.image_dir)
    if not images:
        sys.exit(f"ERROR: no image files found in {args.image_dir}")

    verify_raw_prefix(images)

    total = len(images)
    failed: list[tuple[str, str]] = []

    print(f"Found {total} image(s) under '{args.image_dir}'")
    print(f"Processing in place (RAW -> ROT)\n")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        orientation_path = build_orientation_path(
            image_path, args.image_dir, args.data_dir
        )
        rot_path = build_rot_path(image_path)

        print(f"--- [{idx}/{total}] {rel} ---")
        try:
            orientation = read_orientation(orientation_path)

            if orientation == 180:
                run(
                    ["magick", image_path, "-rotate", "180", rot_path],
                    f"Rotating 180 -> '{os.path.basename(rot_path)}'",
                )
                os.remove(image_path)
                action = "rotated 180, removed original"
            else:
                os.rename(image_path, rot_path)
                action = "renamed (no rotation needed)"
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
