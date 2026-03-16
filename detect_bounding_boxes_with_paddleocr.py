"""
Detect text bounding boxes for SKEW-prefixed TIFF images using PaddleOCR.

For each SKEW-prefixed .tif/.tiff image under image_dir (recursively),
runs PaddleOCR detection and writes a JSON file under data_dir containing
the bounding box coordinates and scores.  Every image filename must start
with the prefix "SKEW"; the script terminates immediately if any does not.

Usage:
    python detect_bounding_boxes_with_paddleocr.py <image_dir> <data_dir>
    python detect_bounding_boxes_with_paddleocr.py <image_dir> <data_dir> --det_model_dir ./ppocr_det_slim_infer --box_thresh 0.6

Requires: paddleocr (pip install paddleocr), paddlepaddle==3.2.2, pyvips
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pyvips
from paddleocr import TextDetection

TIFF_EXTENSIONS = {".tif", ".tiff"}


def collect_images(directory: str) -> list[str]:
    """Return sorted list of TIFF file paths under *directory* (recursive)."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in TIFF_EXTENSIONS:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def verify_skew_prefix(images: list[str]) -> None:
    """Terminate if any image filename does not start with 'SKEW'."""
    for image_path in images:
        filename = os.path.basename(image_path)
        if not filename.startswith("SKEW"):
            sys.exit(
                f"ERROR: filename does not start with 'SKEW': {image_path}"
            )


def build_output_path(image_path: str, image_root: str, data_root: str) -> str:
    """Map an image path to its bounding box JSON file path under data_root."""
    rel = os.path.relpath(image_path, image_root)
    return os.path.join(data_root, rel + ".boxes.json")


def to_temp_jpg(image_path: str) -> str:
    """Convert a TIFF to a temporary 8-bit RGB JPEG for PaddleOCR."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    img = pyvips.Image.new_from_file(image_path, access="sequential")

    # Ensure 8-bit: shift down if 16-bit
    if img.format == "ushort":
        img = img >> 8
        img = img.cast("uchar")

    # Ensure exactly 3 bands (RGB)
    if img.bands == 1:
        img = img.bandjoin([img, img])
    elif img.bands == 4:
        img = img[:3]

    img.write_to_file(tmp_path)
    return tmp_path


def detect_boxes(detector: TextDetection, image_path: str) -> list[dict]:
    """Run PaddleOCR detection on *image_path* and return bounding boxes."""
    print(f"\n>> Detecting bounding boxes on '{image_path}'")

    ext = os.path.splitext(image_path)[1].lower()
    tmp_path = None
    if ext in TIFF_EXTENSIONS:
        tmp_path = to_temp_jpg(image_path)
        predict_path = tmp_path
    else:
        predict_path = image_path

    try:
        results = detector.predict(predict_path, batch_size=1)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    boxes = []
    for res in results:
        polys = res["dt_polys"]
        scores = res["dt_scores"]
        for poly, score in zip(polys, scores):
            if isinstance(poly, np.ndarray):
                poly = poly.tolist()
            boxes.append({
                "points": [[float(x), float(y)] for x, y in poly],
                "score": float(score),
            })

    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect text bounding boxes for SKEW-prefixed TIFF images "
                    "using PaddleOCR.  Writes JSON bounding box files to "
                    "data_dir."
    )
    parser.add_argument("image_dir", help="Directory containing SKEW-prefixed TIFF images")
    parser.add_argument("data_dir", help="Directory to store bounding box JSON files")
    parser.add_argument("--det_model_dir", default=None,
                        help="Path to PaddleOCR detection model directory")
    parser.add_argument("--thresh", type=float, default=0.3,
                        help="Pixel-level detection confidence threshold (default: 0.3)")
    parser.add_argument("--box_thresh", type=float, default=0.6,
                        help="Box-level detection confidence threshold (default: 0.6)")
    parser.add_argument("--device", default="cpu",
                        help="Inference device: cpu, gpu:0, etc. (default: cpu)")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"ERROR: image directory not found: {args.image_dir}")

    images = collect_images(args.image_dir)
    if not images:
        sys.exit(f"ERROR: no TIFF files found in {args.image_dir}")

    verify_skew_prefix(images)

    os.makedirs(args.data_dir, exist_ok=True)

    det_kwargs: dict = {
        "device": args.device,
        "thresh": args.thresh,
        "box_thresh": args.box_thresh,
    }
    if args.det_model_dir:
        det_kwargs["model_dir"] = args.det_model_dir

    detector = TextDetection(**det_kwargs)

    total = len(images)
    failed: list[tuple[str, str]] = []

    print(f"Found {total} TIFF image(s) under '{args.image_dir}'")
    print(f"Data directory: '{args.data_dir}'\n")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        out_path = build_output_path(image_path, args.image_dir, args.data_dir)

        print(f"--- [{idx}/{total}] {rel} ---")
        try:
            boxes = detect_boxes(detector, image_path)
        except Exception as e:
            print(f"    FAILED: {e}\n", file=sys.stderr)
            failed.append((rel, str(e)))
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, indent=2)
        print(f"    OK ({len(boxes)} box(es))\n")

    print("=" * 60)
    print(f"Processed {total - len(failed)}/{total} image(s) successfully.")
    if failed:
        print(f"\n{len(failed)} failure(s):")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
