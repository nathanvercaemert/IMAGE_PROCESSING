"""
Determine skew angle for ROT-prefixed images using leptonica.

For each ROT-prefixed image under image_dir (recursively), runs lept_skew
and writes a text file under data_dir containing the skew angle and
confidence.  Every image filename must start with the prefix "ROT";
the script terminates immediately if any does not.

Usage:
    python determine_skew_angle.py <image_dir> <data_dir>

Requires: lept_skew on PATH
"""

import argparse
import json
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


def detect_skew(image_path: str) -> tuple[float, float]:
    """Run lept_skew on *image_path* and return (angle, confidence)."""
    cmd = ["lept_skew", image_path]
    print(f"\n>> Detecting skew on '{image_path}'")
    print(f"   $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
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


def build_output_path(image_path: str, image_root: str, data_root: str) -> str:
    """Map an image path to its skew data file path under data_root."""
    rel = os.path.relpath(image_path, image_root)
    return os.path.join(data_root, rel + ".skew.txt")


def verify_rot_prefix(images: list[str]) -> None:
    """Terminate if any image filename does not start with 'ROT'."""
    for image_path in images:
        filename = os.path.basename(image_path)
        if not filename.startswith("ROT"):
            sys.exit(
                f"ERROR: filename does not start with 'ROT': {image_path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Determine skew angle for ROT-prefixed images using leptonica."
    )
    parser.add_argument("image_dir", help="Directory containing ROT-prefixed images")
    parser.add_argument("data_dir", help="Directory to store skew angle files")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"ERROR: image directory not found: {args.image_dir}")

    images = collect_images(args.image_dir)
    if not images:
        sys.exit(f"ERROR: no image files found in {args.image_dir}")

    verify_rot_prefix(images)

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
            angle, confidence = detect_skew(image_path)
        except RuntimeError as e:
            print(f"    FAILED: {e}\n", file=sys.stderr)
            failed.append((rel, str(e)))
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{angle}\n{confidence}\n")
        print(f"    OK (angle={angle}, confidence={confidence})\n")

    print("=" * 60)
    print(f"Processed {total - len(failed)}/{total} image(s) successfully.")
    if failed:
        print(f"\n{len(failed)} failure(s):")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
