"""
Binarize CROP-prefixed images using a trained DP-LinkNet model.

Walks image_dir recursively, processes each CROP-prefixed image through
DP-LinkNet for document binarization, and writes a single-channel binary
mask to output_dir with the "BINARY" prefix.  Every image filename must
start with the prefix "CROP"; the script terminates immediately if any
does not.

Unlike other pipeline steps, this script does NOT modify or remove input
files.  The output directory receives the binarized results.

The inference pipeline matches the DP-LinkNet repository's test-time
behavior: BGR input via OpenCV, repo-specific normalization, 256x256
overlapping tiles, optional 8-view test-time augmentation, and
thresholded sigmoid output.

The weights directory should contain .th files named as
{dataset}_{model}.th (e.g. dibco_dplinknet34.th).  The script selects
the correct file from --dataset and --model.

Usage:
    python binarize_with_dplinknet.py <image_dir> <output_dir> <weights_dir> --dataset dibco
    python binarize_with_dplinknet.py <image_dir> <output_dir> <weights_dir> --dataset dibco --model dplinknet34
    python binarize_with_dplinknet.py <image_dir> <output_dir> <weights_dir> --dataset dibco --no-tta

Requires (Python >= 3.9):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install "opencv-python>=4.5" "numpy>=1.22" pyvips
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pyvips
import torch

from networks import MODEL_REGISTRY

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
}

TILE_SIZE = 256
PADDING_SIZE = 21

TTA_THRESHOLD = 5.0
SINGLE_THRESHOLD = 0.5


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths under *directory* (recursive)."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def verify_crop_prefix(images: list[str]) -> None:
    """Terminate if any image filename does not start with 'CROP'."""
    for image_path in images:
        filename = os.path.basename(image_path)
        if not filename.startswith("CROP"):
            sys.exit(
                f"ERROR: filename does not start with 'CROP': {image_path}"
            )


def build_binary_path(
    image_path: str, image_root: str, output_root: str,
) -> str:
    """Map a CROP image path to its BINARY output path under output_root."""
    rel = os.path.relpath(image_path, image_root)
    rel_dir = os.path.dirname(rel)
    filename = os.path.basename(rel)
    binary_name = "BINARY" + filename[4:]
    binary_rel = os.path.join(rel_dir, binary_name) if rel_dir else binary_name
    return os.path.join(output_root, binary_rel)


def load_model(weights_path: str, model_name: str) -> torch.nn.Module:
    """Instantiate the model, load weights on CPU, and set to eval mode."""
    if model_name not in MODEL_REGISTRY:
        sys.exit(
            f"ERROR: unknown model '{model_name}'. "
            f"Choose from: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
        )

    model = MODEL_REGISTRY[model_name]()

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.removeprefix("module."): v for k, v in state_dict.items()
        }

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(
            f"ERROR: weight loading failed.  This usually means the "
            f"checkpoint was saved from a different architecture than "
            f"'{model_name}'.  Check --model and verify that the layer "
            f"names in BINARIZATION/networks.py match the checkpoint.\n",
            file=sys.stderr,
        )
        raise

    model.eval()
    return model


def read_image_bgr8(path: str) -> np.ndarray:
    """Read an image as 8-bit BGR using OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def preprocess_tile(tile: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC -> float32 NCHW tensor, DP-LinkNet normalization."""
    x = tile.astype(np.float32) / 255.0 * 3.2 - 1.6
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x).unsqueeze(0)


def predict_tile(model: torch.nn.Module, tile: np.ndarray) -> np.ndarray:
    """Run a single forward pass on one tile.  Returns HW float32 map."""
    tensor = preprocess_tile(tile)
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze().cpu().numpy()


def predict_tile_tta(model: torch.nn.Module, tile: np.ndarray) -> np.ndarray:
    """Run 8-view TTA on one tile.  Returns summed HW float32 map."""
    t = tile.transpose(1, 0, 2)

    views = [
        tile,                                         # 0: original
        np.flip(tile, 1).copy(),                      # 1: hflip
        np.flip(tile, 0).copy(),                      # 2: vflip
        np.flip(np.flip(tile, 0), 1).copy(),          # 3: hvflip
        t.copy(),                                     # 4: transpose
        np.flip(t, 1).copy(),                         # 5: transpose + hflip
        np.flip(t, 0).copy(),                         # 6: transpose + vflip
        np.flip(np.flip(t, 0), 1).copy(),             # 7: transpose + hvflip
    ]

    total = np.zeros((tile.shape[0], tile.shape[1]), dtype=np.float32)

    for i, view in enumerate(views):
        pred = predict_tile(model, view)
        if i == 1:
            pred = np.flip(pred, 1).copy()
        elif i == 2:
            pred = np.flip(pred, 0).copy()
        elif i == 3:
            pred = np.flip(np.flip(pred, 0), 1).copy()
        elif i == 4:
            pred = pred.T.copy()
        elif i == 5:
            pred = np.flip(pred, 1).T.copy()
        elif i == 6:
            pred = np.flip(pred, 0).T.copy()
        elif i == 7:
            pred = np.flip(np.flip(pred, 0), 1).T.copy()
        total += pred

    return total


def read_dpi(path: str) -> float:
    """Read the horizontal resolution from an image file via pyvips."""
    img = pyvips.Image.new_from_file(path, access="sequential")
    return img.get("Xres")


def save_bilevel_tiff(
    mask: np.ndarray, output_path: str, xres: float,
) -> None:
    """Save a 0/255 mask as an uncompressed 1-bit TIFF with DPI preserved."""
    h, w = mask.shape
    vimg = pyvips.Image.new_from_memory(mask.data, w, h, 1, "uchar")
    vimg = vimg.copy(xres=xres, yres=xres)
    vimg.write_to_file(
        output_path,
        compression="none",
        bitdepth=1,
    )


def binarize_image(
    image_path: str,
    output_path: str,
    model: torch.nn.Module,
    tta: bool,
    threshold: float,
) -> None:
    """Read an image, tile it, run inference, stitch, threshold, and save."""
    print(f"\n>> Binarizing '{image_path}'")

    xres = read_dpi(image_path)
    img = read_image_bgr8(image_path)
    h, w = img.shape[:2]
    stride = TILE_SIZE - 2 * PADDING_SIZE

    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride

    img_padded = np.pad(
        img,
        ((PADDING_SIZE, PADDING_SIZE + pad_h),
         (PADDING_SIZE, PADDING_SIZE + pad_w),
         (0, 0)),
        mode="reflect",
    )

    n_y = (h + pad_h) // stride
    n_x = (w + pad_w) // stride
    total_tiles = n_y * n_x
    print(f"   Image {w}x{h}, padded to {w + pad_w}x{h + pad_h}, "
          f"{n_x}x{n_y} = {total_tiles} tile(s)")

    output = np.zeros((h + pad_h, w + pad_w), dtype=np.float32)

    tile_idx = 0
    for iy in range(n_y):
        for ix in range(n_x):
            tile_idx += 1
            y = iy * stride
            x = ix * stride
            tile = img_padded[y:y + TILE_SIZE, x:x + TILE_SIZE]

            if tta:
                pred = predict_tile_tta(model, tile)
            else:
                pred = predict_tile(model, tile)

            inner = pred[PADDING_SIZE:PADDING_SIZE + stride,
                         PADDING_SIZE:PADDING_SIZE + stride]
            output[iy * stride:(iy + 1) * stride,
                   ix * stride:(ix + 1) * stride] = inner

    output = output[:h, :w]

    mask = (output > threshold).astype(np.uint8) * 255

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_bilevel_tiff(mask, output_path, xres)
    print(f"   Saved '{output_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binarize CROP-prefixed images using a trained DP-LinkNet "
                    "model.  Writes binary masks with 'BINARY' prefix to "
                    "output_dir.  Input files are not modified."
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing CROP-prefixed images",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write BINARY-prefixed output masks",
    )
    parser.add_argument(
        "weights_dir",
        help="Directory containing .th weight files named "
             "{dataset}_{model}.th",
    )
    parser.add_argument(
        "--dataset", default="dibco",
        help="Dataset name used to select the weight file "
             "(default: dibco)",
    )
    parser.add_argument(
        "--model", default="dplinknet34",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture matching the weights (default: dplinknet34)",
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Disable 8-view test-time augmentation (faster, slightly lower "
             "quality).  Threshold changes from 5.0 to 0.5.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override binarization threshold (default: 5.0 with TTA, "
             "0.5 without TTA)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"ERROR: image directory not found: {args.image_dir}")

    if not os.path.isdir(args.weights_dir):
        sys.exit(f"ERROR: weights directory not found: {args.weights_dir}")

    weights_file = f"{args.dataset}_{args.model}.th"
    weights_path = os.path.join(args.weights_dir, weights_file)
    if not os.path.isfile(weights_path):
        available = [f for f in os.listdir(args.weights_dir) if f.endswith(".th")]
        sys.exit(
            f"ERROR: weight file not found: {weights_path}\n"
            f"Available .th files in '{args.weights_dir}':\n"
            + "\n".join(f"  - {f}" for f in sorted(available))
        )

    images = collect_images(args.image_dir)
    if not images:
        sys.exit(f"ERROR: no image files found in {args.image_dir}")

    verify_crop_prefix(images)

    tta = not args.no_tta
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = TTA_THRESHOLD if tta else SINGLE_THRESHOLD

    print(f"Loading model '{args.model}' from '{weights_path}'")
    model = load_model(weights_path, args.model)

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(images)
    failed: list[tuple[str, str]] = []
    mode_label = "8-view TTA" if tta else "single-view"

    print(f"Found {total} image(s) under '{args.image_dir}'")
    print(f"Output directory: '{args.output_dir}'")
    print(f"Mode: {mode_label}, threshold: {threshold}\n")

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        output_path = build_binary_path(
            image_path, args.image_dir, args.output_dir,
        )

        print(f"--- [{idx}/{total}] {rel} ---")
        try:
            binarize_image(image_path, output_path, model, tta, threshold)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"    FAILED: {e}\n", file=sys.stderr)
            failed.append((rel, str(e)))
            continue

        print(f"    OK\n")

    print("=" * 60)
    print(f"Processed {total - len(failed)}/{total} image(s) successfully.")
    if failed:
        print(f"\n{len(failed)} failure(s):")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
