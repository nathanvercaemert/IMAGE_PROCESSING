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
            logger.info("[%d/%d] %s -- OK", idx, total, filename)
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
        out_path = orient_mod.build_output_path(
            image_path, working_dir, data_dir,
        )

        try:
            orientation = orient_mod.detect_orientation(image_path)
        except RuntimeError as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{orientation}\n")
        logger.info("[%d/%d] %s -- OK (%d)", idx, total, rel, orientation)

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
            image_path, working_dir, data_dir,
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

        logger.info("[%d/%d] %s -- OK (%s)", idx, total, rel, action)

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
        out_path = skew_mod.build_output_path(
            image_path, working_dir, data_dir,
        )

        try:
            angle, confidence = skew_mod.detect_skew(image_path)
        except RuntimeError as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{angle}\n{confidence}\n")
        logger.info(
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
            image_path, working_dir, data_dir,
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

        logger.info("[%d/%d] %s -- OK (%s)", idx, total, rel, action)

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
        out_path = bbox_mod.build_output_path(
            image_path, working_dir, data_dir,
        )

        try:
            boxes = bbox_mod.detect_boxes(detector, image_path)
        except Exception as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, indent=2)
        logger.info(
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
        boxes_data_path = draw_mod.build_boxes_data_path(
            image_path, working_dir, data_dir,
        )
        compound_data_path = draw_mod.build_compound_data_path(
            image_path, working_dir, data_dir,
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

                os.makedirs(
                    os.path.dirname(compound_data_path), exist_ok=True,
                )
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

        logger.info("[%d/%d] %s -- OK (%s)", idx, total, rel, action)

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

def stage_crop(working_dir: str, data_dir: str) -> None:
    images = crop_mod.collect_images(working_dir)
    if not images:
        raise RuntimeError(f"no image files found in {working_dir}")

    verify_prefix(images, "BOUND")

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, working_dir)
    logger.info("Processing in place (BOUND -> CROP)")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, working_dir)
        compound_data_path = crop_mod.build_compound_data_path(
            image_path, working_dir, data_dir,
        )
        crop_path = crop_mod.build_crop_path(image_path)

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

        logger.info("[%d/%d] %s -- OK (%s)", idx, total, rel, action)

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.error("%d failure(s):", len(failed))
        for name, err in failed:
            logger.error("  - %s: %s", name, err)
        raise RuntimeError(f"Crop failed: {len(failed)} error(s)")


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
    args = parser.parse_args()

    if not os.path.isdir(args.raw_dir):
        logger.error("raw directory not found: %s", args.raw_dir)
        sys.exit(1)

    for label, path in [("Scanner", SCANNER_ICC), ("Working", WORKING_ICC)]:
        if not os.path.isfile(path):
            logger.error("%s ICC profile not found: %s", label, path)
            sys.exit(1)

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
            lambda: stage_crop(args.working_dir, args.data_dir),
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
