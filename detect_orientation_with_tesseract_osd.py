"""
Detect 0/180 image orientation with Tesseract OSD.

For each image under image_dir (recursively), writes a sibling text file
under data_dir that contains either "0" or "180".

Usage:
    python detect_orientation_with_tesseract_osd.py <image_dir> <data_dir>

Requires: tesseract (with osd.traineddata) on PATH
"""

import argparse
import os
import re
import subprocess
import sys

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
}

ORIENTATION_RE = re.compile(r"Orientation in degrees:\s*(\d+)")


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths under *directory* (recursive)."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def detect_orientation(image_path: str) -> int:
    """Run Tesseract OSD on *image_path* and return 0 or 180."""
    cmd = ["tesseract", image_path, "-", "--psm", "0", "-l", "osd"]
    print(f"\n>> Detecting orientation on '{image_path}'")
    print(f"   $ {' '.join(cmd)}")
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


def build_output_path(image_path: str, image_root: str, data_root: str) -> str:
    """Map an image path to its orientation text file path under data_root."""
    rel = os.path.relpath(image_path, image_root)
    return os.path.join(data_root, rel + ".orientation.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect image orientation (0 or 180) with Tesseract OSD."
    )
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("data_dir", help="Directory to store orientation files")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"ERROR: image directory not found: {args.image_dir}")

    images = collect_images(args.image_dir)
    if not images:
        sys.exit(f"ERROR: no image files found in {args.image_dir}")

    os.makedirs(args.data_dir, exist_ok=True)

    total = len(images)
    failed: list[tuple[str, str]] = []

    print(f"Found {total} image(s) under '{args.image_dir}'")
    print(f"Data directory: '{args.data_dir}'\n")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        out_path = build_output_path(image_path, args.image_dir, args.data_dir)

        print(f"--- [{idx}/{total}] {rel} ---")
        try:
            orientation = detect_orientation(image_path)
        except RuntimeError as e:
            print(f"    FAILED: {e}\n", file=sys.stderr)
            failed.append((rel, str(e)))
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{orientation}\n")
        print(f"    OK ({orientation})\n")

    print("=" * 60)
    print(f"Processed {total - len(failed)}/{total} image(s) successfully.")
    if failed:
        print(f"\n{len(failed)} failure(s):")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
