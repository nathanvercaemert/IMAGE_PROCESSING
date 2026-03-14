"""
Assign a scanner ICC profile to an untagged TIFF, then convert
to working color space (Relative Colorimetric intent + BPC).

Usage:
    python assign_convert_icc_profile.py <input.tif> <output.tif>
    python assign_convert_icc_profile.py <input.tif> <output.tif> --scanner scanner.icc --working working.icc

Requires: exiftool, ImageMagick (magick) on PATH
"""

import argparse
import os
import subprocess
import sys
import tempfile


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


def assign_convert_icc(
    input_file: str,
    output_file: str,
    scanner_profile: str,
    working_profile: str,
) -> None:
    """Assign a scanner ICC profile to an image, then convert to a working
    colour space (Relative Colorimetric + BPC).

    The original input file is never modified.  A temporary intermediate file
    is used for the assign step and cleaned up automatically.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    tmp_dir = os.path.dirname(output_file) or "."
    tmp_fd, tagged_tmp = tempfile.mkstemp(suffix=".tif", dir=tmp_dir)
    os.close(tmp_fd)

    try:
        # 1) Confirm the raw image is untagged
        run(
            ["exiftool", "-s", "-ICC_Profile", input_file],
            f"Checking ICC profile on '{input_file}'",
        )

        # 2) Assign scanner profile -> temporary tagged TIFF
        run(
            ["magick", input_file, "-set", "profile", scanner_profile, tagged_tmp],
            f"Assigning scanner profile '{scanner_profile}'",
        )

        # 3) Convert tagged intermediate to working colour space
        run(
            [
                "magick", tagged_tmp,
                "-intent", "Relative",
                "-black-point-compensation",
                "-profile", working_profile,
                output_file,
            ],
            f"Converting to working profile -> '{output_file}'",
        )

        # 4) Verify final embedded profile and bit depth
        run(
            [
                "exiftool", "-a", "-G1", "-s",
                "-ICC_Profile:ProfileDescription",
                "-BitsPerSample",
                output_file,
            ],
            f"Verifying final output '{output_file}'",
        )
    finally:
        if os.path.exists(tagged_tmp):
            os.remove(tagged_tmp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign scanner ICC profile to a TIFF, then convert to a working colour space."
    )
    parser.add_argument("input", help="Input raw TIFF file")
    parser.add_argument("output", help="Final output TIFF file")
    parser.add_argument("--scanner", default="PATH_TO_SCANNER_ICC",
                        help="Scanner ICC profile to assign")
    parser.add_argument("--working", default="PATH_TO_WORKING_ICC",
                        help="Working-space ICC profile to convert to")
    args = parser.parse_args()

    try:
        assign_convert_icc(args.input, args.output, args.scanner, args.working)
    except (RuntimeError, FileNotFoundError) as e:
        sys.exit(f"ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
