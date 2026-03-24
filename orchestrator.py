"""
Orchestrate the full image processing pipeline.

Runs every stage in sequence:
  1. ICC profile assignment + conversion  (raw_dir -> working_dir)
  2. Orientation detection                (working_dir -> data_dir)
  3. Upside-down correction               (working_dir, data_dir)
  4. Skew angle detection                  (working_dir -> data_dir)
  5. Deskew application                    (working_dir, data_dir)
  6. Bounding box detection                (working_dir -> data_dir)
  7. Compound bounding box drawing         (working_dir, data_dir)
  8. Crop to compound bounding box         (working_dir, data_dir)

Takes RAW-prefixed files from raw_dir and produces CROP-prefixed files
in working_dir, with all sidecar data files accumulated in data_dir.

Usage:
    python orchestrator.py <raw_dir> <working_dir> <data_dir>

Requires: all dependencies of the individual pipeline scripts
"""

import argparse
import json
import logging
import os
import sys

import pyvips
from paddleocr import TextDetection

import assign_convert_icc_profile as icc_mod
import batch_assign_convert_icc_profile as batch_icc_mod
import detect_orientation_with_tesseract_osd as orient_mod
import fix_upside_down_with_magick as rotate_mod
import determine_skew_angle as skew_mod
import apply_deskew_with_pyvips as deskew_mod
import detect_bounding_boxes_with_paddleocr as bbox_mod
import draw_compound_bounding_boxes as draw_mod
import crop_compound_bounding_boxes as crop_mod

logger = logging.getLogger("orchestrator")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCANNER_ICC = os.path.join(_SCRIPT_DIR, "Scanner.icc")
WORKING_ICC = os.path.join(_SCRIPT_DIR, "ProPhoto.icc")


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
    logging.getLogger("pyvips").setLevel(logging.WARNING)


def verify_prefix(images: list[str], prefix: str) -> None:
    """Raise if any image filename does not start with *prefix*."""
    for path in images:
        if not os.path.basename(path).startswith(prefix):
            raise RuntimeError(
                f"filename does not start with '{prefix}': {path}"
            )


# ── Stage 1: ICC profile assignment + conversion ────────────────────

def stage_icc(
    raw_dir: str, working_dir: str, scanner: str, working: str,
) -> None:
    images = batch_icc_mod.collect_images(raw_dir)
    if not images:
        raise RuntimeError(f"no image files found in {raw_dir}")

    os.makedirs(working_dir, exist_ok=True)

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) in '%s'", total, raw_dir)
    logger.info("Output directory: '%s'", working_dir)

    for idx, input_path in enumerate(images, 1):
        filename = os.path.basename(input_path)
        output_path = os.path.join(working_dir, filename)

        try:
            icc_mod.assign_convert_icc(
                input_path, output_path, scanner, working,
            )
            logger.debug("[%d/%d] %s -- OK", idx, total, filename)
        except (RuntimeError, FileNotFoundError) as e:
            logger.error("[%d/%d] %s -- %s", idx, total, filename, e)
            failed.append((filename, str(e)))

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  %s: %s", name, err)
        raise RuntimeError(f"ICC stage failed: {len(failed)} error(s)")


# ── Stage 2: Orientation detection ───────────────────────────────────

def stage_orientation_detect(working_dir: str, data_dir: str) -> None:
    images = orient_mod.collect_images(working_dir)
    if not images:
        raise RuntimeError(f"no image files found in {working_dir}")

    os.makedirs(data_dir, exist_ok=True)

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, working_dir)
    logger.info("Data directory: '%s'", data_dir)

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, working_dir)
        out_path = orient_mod.build_output_path(image_path, data_dir)

        try:
            orientation = orient_mod.detect_orientation(image_path)
        except RuntimeError as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{orientation}\n")
        logger.debug("[%d/%d] %s -- OK (%d)", idx, total, rel, orientation)

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  %s: %s", name, err)
        raise RuntimeError(
            f"Orientation detection failed: {len(failed)} error(s)"
        )


# ── Stage 3: Upside-down correction ─────────────────────────────────

def stage_fix_upside_down(working_dir: str, data_dir: str) -> None:
    images = rotate_mod.collect_images(working_dir)
    if not images:
        raise RuntimeError(f"no image files found in {working_dir}")

    verify_prefix(images, "RAW")

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, working_dir)
    logger.info("Processing in place (RAW -> ROT)")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, working_dir)
        orientation_path = rotate_mod.build_orientation_path(
            image_path, data_dir,
        )
        rot_path = rotate_mod.build_rot_path(image_path)

        try:
            orientation = rotate_mod.read_orientation(orientation_path)

            if orientation == 180:
                depth = rotate_mod.get_bit_depth(image_path)
                rotate_mod.run(
                    [
                        "magick", image_path,
                        "-depth", depth,
                        "-rotate", "180",
                        rot_path,
                    ],
                    f"Rotating 180 (depth={depth}) "
                    f"-> '{os.path.basename(rot_path)}'",
                )
                os.remove(image_path)
                action = "rotated 180, removed original"
            else:
                os.rename(image_path, rot_path)
                action = "renamed (no rotation needed)"
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        logger.debug("[%d/%d] %s -- OK (%s)", idx, total, rel, action)

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)
        raise RuntimeError(
            f"Upside-down correction failed: {len(failed)} error(s)"
        )


# ── Stage 4: Skew angle detection ───────────────────────────────────

def stage_skew_detect(working_dir: str, data_dir: str) -> None:
    images = skew_mod.collect_images(working_dir)
    if not images:
        raise RuntimeError(f"no image files found in {working_dir}")

    verify_prefix(images, "ROT")

    os.makedirs(data_dir, exist_ok=True)

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, working_dir)
    logger.info("Data directory: '%s'", data_dir)

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, working_dir)
        out_path = skew_mod.build_output_path(image_path, data_dir)

        try:
            angle, confidence = skew_mod.detect_skew(image_path)
        except RuntimeError as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{angle}\n{confidence}\n")
        logger.debug(
            "[%d/%d] %s -- OK (angle=%.4f, confidence=%.4f)",
            idx, total, rel, angle, confidence,
        )

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)
        raise RuntimeError(
            f"Skew detection failed: {len(failed)} error(s)"
        )


# ── Stage 5: Deskew application ─────────────────────────────────────

def stage_deskew(working_dir: str, data_dir: str) -> None:
    images = deskew_mod.collect_images(working_dir)
    if not images:
        raise RuntimeError(f"no image files found in {working_dir}")

    verify_prefix(images, "ROT")

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, working_dir)
    logger.info("Processing in place (ROT -> SKEW)")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, working_dir)
        skew_data_path = deskew_mod.build_skew_data_path(
            image_path, data_dir,
        )
        skew_path = deskew_mod.build_skew_path(image_path)

        try:
            angle, confidence = deskew_mod.read_skew_data(skew_data_path)

            if (
                abs(angle) < deskew_mod.MIN_ANGLE
                or confidence < deskew_mod.MIN_CONFIDENCE
            ):
                os.rename(image_path, skew_path)
                action = (
                    f"renamed (angle={angle:.4f}, confidence={confidence:.4f},"
                    f" below threshold)"
                )
            else:
                deskew_mod.deskew_image(image_path, skew_path, angle)
                os.remove(image_path)
                action = (
                    f"deskewed {angle:.4f} deg "
                    f"(confidence={confidence:.4f})"
                )
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        logger.debug("[%d/%d] %s -- OK (%s)", idx, total, rel, action)

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)
        raise RuntimeError(f"Deskew failed: {len(failed)} error(s)")


# ── Stage 6: Bounding box detection ─────────────────────────────────

def stage_bbox_detect(working_dir: str, data_dir: str) -> None:
    images = bbox_mod.collect_images(working_dir)
    if not images:
        raise RuntimeError(f"no TIFF files found in {working_dir}")

    verify_prefix(images, "SKEW")

    os.makedirs(data_dir, exist_ok=True)

    detector = TextDetection(device="cpu", thresh=0.3, box_thresh=0.6)

    # PaddlePaddle reconfigures the root logger during init, wiping our
    # handler.  Re-apply logging configuration.
    root = logging.getLogger()
    root.handlers.clear()
    _configure_logging()

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d TIFF image(s) under '%s'", total, working_dir)
    logger.info("Data directory: '%s'", data_dir)

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, working_dir)
        out_path = bbox_mod.build_output_path(image_path, data_dir)

        try:
            boxes = bbox_mod.detect_boxes(detector, image_path)
        except Exception as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, indent=2)
        logger.debug(
            "[%d/%d] %s -- OK (%d box(es))", idx, total, rel, len(boxes),
        )

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)
        raise RuntimeError(
            f"Bounding box detection failed: {len(failed)} error(s)"
        )


# ── Stage 7: Compound bounding box drawing ──────────────────────────

def stage_draw_boxes(working_dir: str, data_dir: str) -> None:
    images = draw_mod.collect_images(working_dir)
    if not images:
        raise RuntimeError(f"no image files found in {working_dir}")

    verify_prefix(images, "SKEW")

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, working_dir)
    logger.info("Processing in place (SKEW -> BOUND)")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, working_dir)
        boxes_data_path = draw_mod.build_boxes_data_path(image_path, data_dir)
        compound_data_path = draw_mod.build_compound_data_path(
            image_path, data_dir,
        )
        bound_path = draw_mod.build_bound_path(image_path)

        try:
            boxes = draw_mod.read_boxes(boxes_data_path)

            if not boxes:
                os.rename(image_path, bound_path)
                action = "renamed (no bounding boxes detected)"
            else:
                crop_left, crop_top, crop_w, crop_h = (
                    draw_mod.draw_compound_box(image_path, bound_path, boxes)
                )
                os.remove(image_path)

                with open(compound_data_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"{crop_left}\n{crop_top}\n{crop_w}\n{crop_h}\n"
                    )

                action = (
                    f"drew compound box around {len(boxes)} region(s), "
                    f"crop={crop_left},{crop_top} {crop_w}x{crop_h}"
                )
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        logger.debug("[%d/%d] %s -- OK (%s)", idx, total, rel, action)

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)
        raise RuntimeError(
            f"Compound box drawing failed: {len(failed)} error(s)"
        )


# ── Stage 8: Crop to compound bounding box ──────────────────────────

def _save_drawing(src: str, dest: str) -> None:
    """Save a low-resolution 24-bit copy of *src* to *dest* for visual checks."""
    image = pyvips.Image.new_from_file(src, access="sequential")
    image = image.shrink(12, 12)
    image = image.cast("uchar", shift=True)
    dpi = 100 / 25.4
    image = image.copy(xres=dpi, yres=dpi)
    image.write_to_file(dest)


def stage_crop(
    working_dir: str, data_dir: str, drawings_dir: str | None = None,
) -> None:
    images = crop_mod.collect_images(working_dir)
    if not images:
        raise RuntimeError(f"no image files found in {working_dir}")

    verify_prefix(images, "BOUND")

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, working_dir)
    logger.info("Processing in place (BOUND -> CROP)")

    if drawings_dir is not None:
        os.makedirs(drawings_dir, exist_ok=True)
        logger.info("Preserving drawings to '%s'", drawings_dir)

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, working_dir)
        compound_data_path = crop_mod.build_compound_data_path(
            image_path, data_dir,
        )
        crop_path = crop_mod.build_crop_path(image_path)

        if drawings_dir is not None:
            draw_name = "DRAW" + os.path.basename(image_path)[5:]
            draw_dest = os.path.join(drawings_dir, draw_name)
            _save_drawing(image_path, draw_dest)

        try:
            if not os.path.isfile(compound_data_path):
                os.rename(image_path, crop_path)
                action = "renamed (no compound data, no crop applied)"
            else:
                left, top, width, height = crop_mod.read_compound_data(
                    compound_data_path,
                )
                crop_mod.crop_image(
                    image_path, crop_path, left, top, width, height,
                )
                os.remove(image_path)
                action = f"cropped to {left},{top} {width}x{height}"
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        logger.debug("[%d/%d] %s -- OK (%s)", idx, total, rel, action)

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)
        raise RuntimeError(f"Crop failed: {len(failed)} error(s)")


# ── Single-file pipeline ─────────────────────────────────────────────

def _validate_single_file(
    filename: str, raw_dir: str, data_dir: str,
) -> tuple[str, str, str, str | None]:
    """Validate inputs for single-file mode.

    Returns (raw_path, orientation_path, skew_path, compound_path).
    compound_path is None when no compound data file exists (meaning
    no text was detected — the image passes through without cropping).

    Raises RuntimeError if any required file is missing.
    """
    if not filename.startswith("RAW"):
        raise RuntimeError(
            f"filename must start with 'RAW': {filename}"
        )

    suffix = filename[3:]

    raw_path = os.path.join(raw_dir, filename)
    if not os.path.isfile(raw_path):
        raise RuntimeError(f"raw file not found: {raw_path}")

    orientation_path = os.path.join(
        data_dir, f"RAW{suffix}.orientation.txt",
    )
    if not os.path.isfile(orientation_path):
        raise RuntimeError(
            f"orientation data not found: {orientation_path}"
        )

    skew_path = os.path.join(data_dir, f"ROT{suffix}.skew.txt")
    if not os.path.isfile(skew_path):
        raise RuntimeError(f"skew data not found: {skew_path}")

    compound_path = os.path.join(
        data_dir, f"SKEW{suffix}.compound.txt",
    )
    if not os.path.isfile(compound_path):
        compound_path = None

    return raw_path, orientation_path, skew_path, compound_path


def run_single_file(
    filename: str,
    raw_dir: str,
    working_dir: str,
    data_dir: str,
    scanner_icc: str,
    working_icc: str,
    drawings_dir: str | None = None,
) -> None:
    """Process a single RAW file through the application stages.

    Detection stages are skipped — all data files must already exist
    in data_dir.
    """
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"data directory not found: {data_dir}")

    raw_path, orientation_path, skew_path, compound_path = (
        _validate_single_file(filename, raw_dir, data_dir)
    )

    os.makedirs(working_dir, exist_ok=True)

    suffix = filename[3:]

    # ── 1. ICC assign + convert ──────────────────────────────────
    working_raw = os.path.join(working_dir, filename)
    logger.info("ICC assign + convert: %s", filename)
    icc_mod.assign_convert_icc(
        raw_path, working_raw, scanner_icc, working_icc,
    )

    # ── 2. Fix upside-down (RAW -> ROT) ─────────────────────────
    rot_name = "ROT" + suffix
    rot_path = os.path.join(working_dir, rot_name)
    orientation = rotate_mod.read_orientation(orientation_path)

    if orientation == 180:
        depth = rotate_mod.get_bit_depth(working_raw)
        rotate_mod.run(
            [
                "magick", working_raw,
                "-depth", depth,
                "-rotate", "180",
                rot_path,
            ],
            f"Rotating 180 (depth={depth}) -> '{rot_name}'",
        )
        os.remove(working_raw)
        logger.info("Rotated 180: %s -> %s", filename, rot_name)
    else:
        os.rename(working_raw, rot_path)
        logger.info("Renamed (no rotation): %s -> %s", filename, rot_name)

    # ── 3. Deskew (ROT -> SKEW) ─────────────────────────────────
    skew_name = "SKEW" + suffix
    skew_image_path = os.path.join(working_dir, skew_name)
    angle, confidence = deskew_mod.read_skew_data(skew_path)

    if (
        abs(angle) < deskew_mod.MIN_ANGLE
        or confidence < deskew_mod.MIN_CONFIDENCE
    ):
        os.rename(rot_path, skew_image_path)
        logger.info(
            "Renamed (angle=%.4f, confidence=%.4f, below threshold): "
            "%s -> %s", angle, confidence, rot_name, skew_name,
        )
    else:
        deskew_mod.deskew_image(rot_path, skew_image_path, angle)
        os.remove(rot_path)
        logger.info(
            "Deskewed %.4f deg (confidence=%.4f): %s -> %s",
            angle, confidence, rot_name, skew_name,
        )

    # ── 4. Draw + crop (SKEW -> BOUND -> CROP) ──────────────────
    bound_name = "BOUND" + suffix
    bound_path = os.path.join(working_dir, bound_name)

    if compound_path is None:
        # No text detected — pass through without drawing or cropping.
        os.rename(skew_image_path, bound_path)
        logger.info(
            "Renamed (no compound data): %s -> %s",
            skew_name, bound_name,
        )
    else:
        left, top, width, height = crop_mod.read_compound_data(
            compound_path,
        )

        image = pyvips.Image.new_from_file(
            skew_image_path, access="sequential",
        )
        img_w, img_h = image.width, image.height
        if left + width > img_w or top + height > img_h:
            raise RuntimeError(
                f"compound coordinates ({left},{top} {width}x{height}) "
                f"exceed image dimensions ({img_w}x{img_h})"
            )

        draw_mod.draw_rect_from_compound(
            image, bound_path, left, top, width, height,
        )
        os.remove(skew_image_path)
        logger.info(
            "Drew compound rect %d,%d %dx%d: %s -> %s",
            left, top, width, height, skew_name, bound_name,
        )

    # Preserve drawing if requested.
    if drawings_dir is not None:
        os.makedirs(drawings_dir, exist_ok=True)
        draw_dest_name = "DRAW" + suffix
        draw_dest = os.path.join(drawings_dir, draw_dest_name)
        _save_drawing(bound_path, draw_dest)
        logger.info("Saved drawing: %s", draw_dest_name)

    # Crop.
    crop_name = "CROP" + suffix
    crop_path = os.path.join(working_dir, crop_name)

    if compound_path is None:
        os.rename(bound_path, crop_path)
        logger.info(
            "Renamed (no crop): %s -> %s", bound_name, crop_name,
        )
    else:
        left, top, width, height = crop_mod.read_compound_data(
            compound_path,
        )
        crop_mod.crop_image(
            bound_path, crop_path, left, top, width, height,
        )
        os.remove(bound_path)
        logger.info(
            "Cropped to %d,%d %dx%d: %s -> %s",
            left, top, width, height, bound_name, crop_name,
        )


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Orchestrate the full image processing pipeline.  "
                    "Takes RAW-prefixed files from raw_dir, processes them "
                    "through all stages, and produces CROP-prefixed files "
                    "in working_dir."
    )
    parser.add_argument(
        "raw_dir",
        help="Directory containing RAW-prefixed source images",
    )
    parser.add_argument(
        "working_dir",
        help="Working directory for in-place processing",
    )
    parser.add_argument(
        "data_dir",
        help="Directory to accumulate sidecar data files",
    )
    parser.add_argument(
        "--preserve-drawings",
        metavar="DRAWINGS_DIR",
        default=None,
        help="Copy BOUND images (with drawn bounding boxes) to "
             "DRAWINGS_DIR before cropping, renaming the BOUND "
             "prefix to DRAW",
    )
    parser.add_argument(
        "--single-file",
        metavar="FILENAME",
        default=None,
        help="Process a single RAW-prefixed file instead of the "
             "full batch.  Detection stages are skipped; all data "
             "files must already exist in data_dir.  The compound "
             "data file defines both the drawn rectangle and the "
             "crop region.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.raw_dir):
        logger.error("raw directory not found: %s", args.raw_dir)
        sys.exit(1)

    for label, path in [("Scanner", SCANNER_ICC), ("Working", WORKING_ICC)]:
        if not os.path.isfile(path):
            logger.error("%s ICC profile not found: %s", label, path)
            sys.exit(1)

    if args.single_file is not None:
        try:
            run_single_file(
                args.single_file,
                args.raw_dir,
                args.working_dir,
                args.data_dir,
                SCANNER_ICC,
                WORKING_ICC,
                args.preserve_drawings,
            )
        except RuntimeError as e:
            logger.error("Single-file pipeline failed: %s", e)
            sys.exit(1)
        logger.info("Single-file pipeline complete: %s", args.single_file)
        return

    stages = [
        (
            "ICC profile assignment",
            lambda: stage_icc(
                args.raw_dir, args.working_dir,
                SCANNER_ICC, WORKING_ICC,
            ),
        ),
        (
            "Orientation detection",
            lambda: stage_orientation_detect(args.working_dir, args.data_dir),
        ),
        (
            "Upside-down correction",
            lambda: stage_fix_upside_down(args.working_dir, args.data_dir),
        ),
        (
            "Skew angle detection",
            lambda: stage_skew_detect(args.working_dir, args.data_dir),
        ),
        (
            "Deskew application",
            lambda: stage_deskew(args.working_dir, args.data_dir),
        ),
        (
            "Bounding box detection",
            lambda: stage_bbox_detect(args.working_dir, args.data_dir),
        ),
        (
            "Compound bounding box drawing",
            lambda: stage_draw_boxes(args.working_dir, args.data_dir),
        ),
        (
            "Crop to compound bounding box",
            lambda: stage_crop(
                args.working_dir, args.data_dir, args.preserve_drawings,
            ),
        ),
    ]

    for stage_name, stage_func in stages:
        logger.info("=" * 60)
        logger.info("STAGE: %s", stage_name)
        logger.info("=" * 60)
        try:
            stage_func()
        except RuntimeError as e:
            logger.error("Pipeline aborted at '%s': %s", stage_name, e)
            sys.exit(1)
        logger.info("STAGE COMPLETE: %s", stage_name)

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
