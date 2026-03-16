"""
Verify data integrity between RAW source images and working-directory images.

Compares DPI, color depth, channel count, and colorspace.  Determines TRUE
bit depth by sampling three small subsets of actual pixel data from each
image (top-left, center, bottom-right) via ``magick convert ... txt:`` and
analyzing the channel values present.

If a file claims 16-bit/channel but every sampled pixel value is a multiple
of 257 (the pattern produced when 8-bit data is stored in a 16-bit
container: 0->0, 1->257, 2->514, ... 255->65535), the actual data is 8-bit.
Truly 16-bit data will contain values that are NOT multiples of 257.

Pixel dimensions are NOT compared because downstream steps (deskew, crop)
may legitimately change them.

Usage:
    python verify_data_integrity_small_subset.py <raw_dir> <working_dir> <working_prefix>

    working_prefix is one of: ROT, SKEW, BOUND, CROP

Requires: ImageMagick (magick) on PATH
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

logger = logging.getLogger("verify_data_integrity_small_subset")


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

VALID_PREFIXES = {"ROT", "SKEW", "BOUND", "CROP"}

# How many pixels wide/tall each sample crop is.
SAMPLE_SIZE = 200


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths under *directory* (recursive)."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def identify_image(image_path: str) -> dict:
    """Return image properties via ImageMagick identify."""
    fmt = (
        '{"width":%w,"height":%h,"depth":%z,'
        '"xres":"%x","yres":"%y",'
        '"colorspace":"%[colorspace]",'
        '"type":"%[type]"}'
    )
    result = subprocess.run(
        ["magick", "identify", "-format", fmt, image_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"magick identify failed for {image_path}: {result.stderr.strip()}"
        )

    raw = result.stdout.strip()
    if raw.startswith("{"):
        first = raw.split("}{")[0] + ("}" if "}{" in raw else "")
    else:
        first = raw
    info = json.loads(first)

    for key in ("xres", "yres"):
        val = info[key].split()[0] if info[key] else "0"
        try:
            info[key] = float(val)
        except ValueError:
            info[key] = 0.0

    info["file_size"] = os.path.getsize(image_path)

    return info


def count_channels_from_pixels(image_path: str) -> int:
    """Determine channel count by sampling one pixel as txt: and counting
    comma-separated values in the parenthesized group."""
    result = subprocess.run(
        [
            "magick", image_path,
            "-crop", "1x1+0+0", "+repage",
            "-depth", "16",
            "txt:-",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"channel count probe failed for {image_path}: {result.stderr.strip()}"
        )

    value_re = re.compile(r"\(([^)]+)\)")
    for line in result.stdout.splitlines():
        if line.startswith("#"):
            continue
        m = value_re.search(line)
        if m:
            return len(m.group(1).split(","))

    raise RuntimeError(f"could not parse pixel data from {image_path}")


def sample_crop_offsets(width: int, height: int) -> list[tuple[str, str]]:
    """Return 3 (label, crop-geometry) pairs: top-left, center, bottom-right."""
    s = SAMPLE_SIZE
    # Clamp sample size to image dimensions.
    sw = min(s, width)
    sh = min(s, height)

    cx = max(0, (width - sw) // 2)
    cy = max(0, (height - sh) // 2)
    bx = max(0, width - sw)
    by = max(0, height - sh)

    return [
        ("top-left",     f"{sw}x{sh}+0+0"),
        ("center",       f"{sw}x{sh}+{cx}+{cy}"),
        ("bottom-right", f"{sw}x{sh}+{bx}+{by}"),
    ]


def sample_pixel_depth(
    image_path: str, claimed_depth: int, width: int, height: int,
) -> dict:
    """Sample three crops of pixels and determine the true bit depth.

    Returns a dict:
        claimed_depth:   what the header says (e.g. 16)
        actual_depth:    what the pixel data actually is (8 or 16)
        sample_pixels:   total number of pixels sampled across all 3 crops
        unique_values:   total unique channel values seen
        max_value:       largest channel value seen
        non_257_count:   values that are NOT multiples of 257
        regions:         per-region summaries
        verdict:         human-readable summary
    """
    crops = sample_crop_offsets(width, height)
    value_re = re.compile(r"\(([^)]+)\)")

    all_values: set[int] = set()
    total_pixels = 0
    region_summaries: list[str] = []

    for label, geometry in crops:
        cmd = [
            "magick", image_path,
            "-crop", geometry, "+repage",
            "-depth", "16",
            "txt:-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            region_summaries.append(f"{label}: FAILED")
            continue

        region_values: set[int] = set()
        region_pixels = 0

        for line in result.stdout.splitlines():
            if line.startswith("#"):
                continue
            m = value_re.search(line)
            if not m:
                continue
            region_pixels += 1
            for v in m.group(1).split(","):
                v = v.strip()
                try:
                    region_values.add(int(v))
                except ValueError:
                    try:
                        region_values.add(int(float(v)))
                    except ValueError:
                        pass

        total_pixels += region_pixels
        all_values.update(region_values)

        region_non257 = sum(1 for v in region_values if v % 257 != 0)
        region_max = max(region_values) if region_values else 0
        region_summaries.append(
            f"{label}: {region_pixels}px, "
            f"{len(region_values)} unique, "
            f"{region_non257} non-257, "
            f"max={region_max}"
        )

    if total_pixels == 0:
        return {
            "claimed_depth": claimed_depth,
            "actual_depth": claimed_depth,
            "sample_pixels": 0,
            "unique_values": 0,
            "max_value": 0,
            "non_257_count": 0,
            "regions": region_summaries,
            "verdict": "no pixels sampled",
        }

    max_val = max(all_values) if all_values else 0
    non_257 = sum(1 for v in all_values if v % 257 != 0)

    if claimed_depth <= 8:
        actual = 8
        if max_val > 255:
            verdict = (
                f"ANOMALY: header says {claimed_depth}-bit but max sample "
                f"value is {max_val} (>255)"
            )
        else:
            verdict = (
                f"confirmed 8-bit ({len(all_values)} unique values, "
                f"max={max_val}, {total_pixels}px across 3 regions)"
            )
    elif claimed_depth == 16:
        if non_257 == 0 and max_val <= 65535:
            actual = 8
            verdict = (
                f"DATA IS 8-BIT: header says 16-bit but all {len(all_values)} "
                f"unique values are multiples of 257 (8-bit scaled to 16-bit). "
                f"Max={max_val} ({max_val // 257}/255 in 8-bit). "
                f"{total_pixels}px across 3 regions"
            )
        else:
            actual = 16
            verdict = (
                f"confirmed 16-bit ({len(all_values)} unique values, "
                f"{non_257} not multiples of 257, max={max_val}, "
                f"{total_pixels}px across 3 regions)"
            )
    else:
        actual = claimed_depth
        verdict = (
            f"unusual depth {claimed_depth}, {len(all_values)} unique values, "
            f"{total_pixels}px across 3 regions"
        )

    return {
        "claimed_depth": claimed_depth,
        "actual_depth": actual,
        "sample_pixels": total_pixels,
        "unique_values": len(all_values),
        "max_value": max_val,
        "non_257_count": non_257,
        "regions": region_summaries,
        "verdict": verdict,
    }


def match_working_path(
    raw_path: str, raw_root: str, working_root: str, working_prefix: str,
) -> str | None:
    """Find the working-directory counterpart of a RAW image."""
    rel = os.path.relpath(raw_path, raw_root)
    rel_dir = os.path.dirname(rel)
    filename = os.path.basename(rel)

    if not filename.startswith("RAW"):
        return None

    working_name = working_prefix + filename[3:]
    candidate = os.path.join(working_root, rel_dir, working_name)
    if os.path.isfile(candidate):
        return candidate
    return None


def format_size(nbytes: int) -> str:
    """Human-readable file size."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    else:
        return f"{nbytes / (1024 * 1024):.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify data integrity between RAW and working images.  "
                    "Samples actual pixel data from 3 regions to determine "
                    "true bit depth."
    )
    parser.add_argument("raw_dir", help="Directory containing RAW-prefixed images")
    parser.add_argument("working_dir", help="Directory containing working images")
    parser.add_argument(
        "working_prefix",
        choices=sorted(VALID_PREFIXES),
        help="Prefix of the working images (ROT, SKEW, BOUND, CROP)",
    )
    args = parser.parse_args()

    _configure_logging()

    if not os.path.isdir(args.raw_dir):
        logger.error("raw directory not found: %s", args.raw_dir)
        sys.exit(1)
    if not os.path.isdir(args.working_dir):
        logger.error("working directory not found: %s", args.working_dir)
        sys.exit(1)

    raw_images = collect_images(args.raw_dir)
    if not raw_images:
        logger.error("no image files found in %s", args.raw_dir)
        sys.exit(1)

    raw_images = [p for p in raw_images if os.path.basename(p).startswith("RAW")]
    if not raw_images:
        logger.error("no RAW-prefixed images found in %s", args.raw_dir)
        sys.exit(1)

    total = len(raw_images)
    missing: list[str] = []
    mismatches: list[tuple[str, list[str]]] = []
    ok_count = 0

    logger.info(
        "Comparing %d RAW image(s) against %s-prefixed working images",
        total, args.working_prefix,
    )
    logger.info("RAW dir: %s", args.raw_dir)
    logger.info("Working dir: %s", args.working_dir)
    logger.info(
        "Pixel sample: 3 x %dx%d regions (top-left, center, bottom-right)",
        SAMPLE_SIZE, SAMPLE_SIZE,
    )

    for idx, raw_path in enumerate(raw_images, 1):
        rel = os.path.relpath(raw_path, args.raw_dir)
        working_path = match_working_path(
            raw_path, args.raw_dir, args.working_dir, args.working_prefix
        )

        logger.info("[%d/%d] %s", idx, total, rel)

        if working_path is None:
            logger.info("[%d/%d] %s -- MISSING: no matching %s file", idx, total, rel, args.working_prefix)
            missing.append(rel)
            continue

        try:
            raw_info = identify_image(raw_path)
            work_info = identify_image(working_path)
        except RuntimeError as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            mismatches.append((rel, [str(e)]))
            continue

        # Get true channel count from pixel data
        try:
            raw_channels = count_channels_from_pixels(raw_path)
            work_channels = count_channels_from_pixels(working_path)
        except RuntimeError as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            mismatches.append((rel, [str(e)]))
            continue

        problems: list[str] = []

        # --- DPI ---
        if raw_info["xres"] != work_info["xres"]:
            problems.append(
                f"X-DPI mismatch: RAW={raw_info['xres']:.2f}  "
                f"{args.working_prefix}={work_info['xres']:.2f}"
            )
        if raw_info["yres"] != work_info["yres"]:
            problems.append(
                f"Y-DPI mismatch: RAW={raw_info['yres']:.2f}  "
                f"{args.working_prefix}={work_info['yres']:.2f}"
            )

        # --- Channel count ---
        if raw_channels != work_channels:
            problems.append(
                f"Channel count mismatch: RAW={raw_channels}  "
                f"{args.working_prefix}={work_channels}"
            )

        # --- Colorspace ---
        if raw_info["colorspace"] != work_info["colorspace"]:
            problems.append(
                f"Colorspace mismatch: RAW={raw_info['colorspace']}  "
                f"{args.working_prefix}={work_info['colorspace']}"
            )

        # --- Pixel-level depth analysis (3 regions each) ---
        try:
            raw_depth = sample_pixel_depth(
                raw_path, raw_info["depth"],
                raw_info["width"], raw_info["height"],
            )
            work_depth = sample_pixel_depth(
                working_path, work_info["depth"],
                work_info["width"], work_info["height"],
            )
        except RuntimeError as e:
            problems.append(f"Pixel sampling error: {e}")
            raw_depth = work_depth = None

        if raw_depth and work_depth:
            logger.info("[%d/%d] %s -- RAW pixel depth: %s", idx, total, rel, raw_depth["verdict"])
            for rs in raw_depth["regions"]:
                logger.debug("[%d/%d] %s -- RAW region: %s", idx, total, rel, rs)
            logger.info("[%d/%d] %s -- %s pixel depth: %s", idx, total, rel, args.working_prefix, work_depth["verdict"])
            for rs in work_depth["regions"]:
                logger.debug("[%d/%d] %s -- %s region: %s", idx, total, rel, args.working_prefix, rs)

            # Header depth mismatch
            if raw_info["depth"] != work_info["depth"]:
                problems.append(
                    f"Header depth mismatch: RAW={raw_info['depth']}-bit  "
                    f"{args.working_prefix}={work_info['depth']}-bit"
                )

            # Actual (pixel-verified) depth mismatch
            if raw_depth["actual_depth"] != work_depth["actual_depth"]:
                problems.append(
                    f"Actual pixel depth mismatch: RAW={raw_depth['actual_depth']}-bit  "
                    f"{args.working_prefix}={work_depth['actual_depth']}-bit"
                )

            # Flag 8-bit data in 16-bit container
            for label, depth_result in [
                ("RAW", raw_depth),
                (args.working_prefix, work_depth),
            ]:
                if (
                    depth_result["claimed_depth"] == 16
                    and depth_result["actual_depth"] == 8
                ):
                    problems.append(
                        f"{label}: header claims 16-bit but pixel data is 8-bit "
                        f"(all {depth_result['unique_values']} sampled values "
                        f"across 3 regions are multiples of 257)"
                    )

        if problems:
            mismatches.append((rel, problems))
            for p in problems:
                logger.info("[%d/%d] %s -- MISMATCH: %s", idx, total, rel, p)
        else:
            ok_count += 1
            actual = raw_depth["actual_depth"] if raw_depth else raw_info["depth"]
            total_bits = actual * raw_channels
            logger.info(
                "[%d/%d] %s -- OK actual_depth=%dbit/ch (%dbit total) "
                "channels=%d dpi=%.0fx%.0f RAW=%s %s=%s",
                idx, total, rel, actual, total_bits, raw_channels,
                raw_info["xres"], raw_info["yres"],
                format_size(raw_info["file_size"]),
                args.working_prefix, format_size(work_info["file_size"]),
            )

    # --- Summary ---
    logger.info(
        "RESULTS: %d/%d OK, %d mismatched, %d missing",
        ok_count, total, len(mismatches), len(missing),
    )

    if missing:
        logger.info("MISSING (%d):", len(missing))
        for name in missing:
            logger.info("  - %s", name)

    if mismatches:
        logger.info("MISMATCHES (%d):", len(mismatches))
        for name, problems in mismatches:
            logger.info("  - %s:", name)
            for p in problems:
                logger.info("      %s", p)
        logger.info("NOTE ON 8-BIT vs 16-BIT:")
        logger.info("If pixel data is '8-bit in a 16-bit container', every channel")
        logger.info("value is a multiple of 257 (e.g. 0, 257, 514, ... 65535).")
        logger.info("This means the file stores 16-bit values but carries only 8-bit")
        logger.info("worth of actual color information (256 levels, not 65536).")
        logger.info("IrfanView showing '24-bit' = 8 bits x 3 channels = 24 total.")
        logger.info("True 48-bit = 16 bits x 3 channels with values NOT on 257 boundaries.")

    if mismatches or missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
