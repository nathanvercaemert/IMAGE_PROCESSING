"""
Batch assign a scanner ICC profile and convert to a working colour space
for every image in an input directory.  Processed images are written to
the output directory; originals are never modified.

Usage:
    python batch_assign_convert_icc_profile.py <input_dir> <output_dir> --scanner scanner.icc --working working.icc

Requires: exiftool, ImageMagick (magick) on PATH
"""

import argparse
import os
import sys

from assign_convert_icc_profile import assign_convert_icc

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

    if not os.path.isdir(args.input_dir):
        sys.exit(f"ERROR: input directory not found: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    images = collect_images(args.input_dir)
    if not images:
        sys.exit(f"ERROR: no image files found in {args.input_dir}")

    total = len(images)
    failed: list[tuple[str, str]] = []

    print(f"Found {total} image(s) in '{args.input_dir}'")
    print(f"Output directory: '{args.output_dir}'\n")

    for idx, input_path in enumerate(images, 1):
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, filename)

        print(f"--- [{idx}/{total}] {filename} ---")
        try:
            assign_convert_icc(
                input_path, output_path, args.scanner, args.working
            )
            print(f"    OK\n")
        except (RuntimeError, FileNotFoundError) as e:
            print(f"    FAILED: {e}\n", file=sys.stderr)
            failed.append((filename, str(e)))

    print("=" * 60)
    print(f"Processed {total - len(failed)}/{total} image(s) successfully.")
    if failed:
        print(f"\n{len(failed)} failure(s):")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
