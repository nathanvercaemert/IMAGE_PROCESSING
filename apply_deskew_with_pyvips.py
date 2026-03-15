"""
Apply deskew correction to ROT-prefixed images using pyvips.

Walks image_dir recursively, reads the matching *.skew.txt file from
data_dir (created by determine_skew_angle.py), and rotates each image
by the stored angle using pyvips with bicubic interpolation.  Every
image filename must start with the prefix "ROT"; the script terminates
immediately if any does not.

The deskewed result is saved with the "SKEW" prefix and the original
"ROT" file is removed.  Images whose skew angle is below a minimum
threshold or whose confidence is below a minimum confidence are simply
renamed from "ROT" to "SKEW" (no rotation needed).

WARNING: This script modifies and removes input files.

Usage:
    python apply_deskew_with_pyvips.py <image_dir> <data_dir>

Requires: pyvips (with libvips) installed
"""

import argparse
import os
import sys

import pyvips

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
}

MIN_ANGLE = 0.05
MIN_CONFIDENCE = 2.0


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths under *directory* (recursive)."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def build_skew_data_path(image_path: str, image_root: str, data_root: str) -> str:
    """Map an image path to its skew data file path under data_root."""
    rel = os.path.relpath(image_path, image_root)
    return os.path.join(data_root, rel + ".skew.txt")


def build_skew_path(image_path: str) -> str:
    """Replace the ROT prefix of the filename with SKEW."""
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    return os.path.join(directory, "SKEW" + filename[3:])


def read_skew_data(skew_path: str) -> tuple[float, float]:
    """Read angle and confidence from a skew data file."""
    if not os.path.isfile(skew_path):
        raise FileNotFoundError(f"Missing skew data file: {skew_path}")

    with open(skew_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    if len(lines) < 2:
        raise ValueError(
            f"Skew data file has fewer than 2 lines: {skew_path}"
        )

    try:
        angle = float(lines[0])
        confidence = float(lines[1])
    except ValueError:
        raise ValueError(
            f"Could not parse angle/confidence from {skew_path}: "
            f"{lines[0]!r}, {lines[1]!r}"
        )

    return angle, confidence


def deskew_image(image_path: str, output_path: str, angle: float) -> None:
    """Rotate *image_path* by *angle* degrees and save to *output_path*."""
    print(f"\n>> Deskewing '{image_path}' by {angle:.4f} degrees")

    page = pyvips.Image.new_from_file(image_path, access="sequential")

    bands = page.bands
    if bands >= 3:
        background = [65535] * bands
    else:
        background = [65535]

    interpolator = pyvips.Interpolate.new("bicubic")
    rotated = page.rotate(angle, interpolate=interpolator, background=background)

    xres = page.get("Xres")
    yres = page.get("Yres")
    rotated = rotated.copy(xres=xres, yres=yres)

    rotated.write_to_file(output_path)
    print(f"   Saved '{output_path}'")


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
        description="Apply deskew correction to ROT-prefixed images using "
                    "pyvips and saved skew angle files.  Replaces ROT prefix "
                    "with SKEW and removes the original."
    )
    parser.add_argument("image_dir", help="Directory containing ROT-prefixed images")
    parser.add_argument("data_dir", help="Directory containing skew data files")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"ERROR: image directory not found: {args.image_dir}")

    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: data directory not found: {args.data_dir}")

    images = collect_images(args.image_dir)
    if not images:
        sys.exit(f"ERROR: no image files found in {args.image_dir}")

    verify_rot_prefix(images)

    total = len(images)
    failed: list[tuple[str, str]] = []

    print(f"Found {total} image(s) under '{args.image_dir}'")
    print(f"Processing in place (ROT -> SKEW)\n")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        skew_data_path = build_skew_data_path(
            image_path, args.image_dir, args.data_dir
        )
        skew_path = build_skew_path(image_path)

        print(f"--- [{idx}/{total}] {rel} ---")
        try:
            angle, confidence = read_skew_data(skew_data_path)

            if abs(angle) < MIN_ANGLE or confidence < MIN_CONFIDENCE:
                os.rename(image_path, skew_path)
                action = (
                    f"renamed (angle={angle:.4f}, confidence={confidence:.4f}, "
                    f"below threshold)"
                )
            else:
                deskew_image(image_path, skew_path, angle)
                os.remove(image_path)
                action = f"deskewed {angle:.4f} deg (confidence={confidence:.4f})"
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
