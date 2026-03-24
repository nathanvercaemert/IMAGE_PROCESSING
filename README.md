# IMAGE_PROCESSING

## Build

Uses cached layers when possible. Fast if only later layers changed.

```bash
docker build -t image-processing .
```

## Rebuild (no cache)

Ignores all cached layers and rebuilds everything from scratch. Use this after pushing code changes to GitHub to ensure the cloned repo inside the image is up to date.

```bash
docker build --no-cache -t image-processing .
```

## Run

Runs the full pipeline. Expects `RAW/` inside the bind-mounted folder. Creates `WORKING/` and `DATA/` there.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing
```

### Preserve drawings

To keep a copy of the bounding-box drawings before they are cropped away, add `--preserve-drawings`. The BOUND images are copied to the given directory with the prefix renamed from `BOUND` to `DRAW`.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing /data/RAW /data/WORKING /data/DATA --preserve-drawings /data/DRAWINGS
```

### Single-file reprocessing

To reprocess a single image using existing data files (e.g. after manually editing orientation, skew, or compound crop coordinates), add `--single-file`. Detection stages are skipped entirely; all data files must already exist in `DATA/`. The compound data file defines both the drawn rectangle and the crop region.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing /data/RAW /data/WORKING /data/DATA --single-file RAW_0001.tif
```

Required data files for a given `RAW_0001.tif`:
- `DATA/RAW_0001.tif.orientation.txt` — orientation (0 or 180)
- `DATA/ROT_0001.tif.skew.txt` — skew angle and confidence
- `DATA/SKEW_0001.tif.compound.txt` — crop coordinates (optional; if absent, no drawing or crop is applied)

Can be combined with `--preserve-drawings`:

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing /data/RAW /data/WORKING /data/DATA --single-file RAW_0001.tif --preserve-drawings /data/DRAWINGS
```

## Cleanup

Remove stopped containers and dangling layers. Keeps the built image.

```bash
docker rm $(docker ps -a -q --filter ancestor=image-processing) 2>/dev/null; docker image prune -f
```

## Individual Scripts

All scripts live in `/opt/image_processing` inside the container. Override the entrypoint to run them directly.

### Stage 1: ICC Profile Assignment + Conversion

Assign scanner ICC profile to every image in `RAW/`, convert to ProPhoto working space, write results to `WORKING/`.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" --entrypoint python image-processing batch_assign_convert_icc_profile.py /data/RAW /data/WORKING --scanner Scanner.icc --working ProPhoto.icc
```

### Stage 2a: Detect Orientation

Detect 0/180 orientation for each image in `WORKING/` using Tesseract OSD. Writes `.orientation.txt` files to `DATA/`.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" --entrypoint python image-processing detect_orientation_with_tesseract_osd.py /data/WORKING /data/DATA
```

### Stage 2b: Fix Upside-Down

Rotate 180-degree images upright using ImageMagick. Renames `RAW*` to `ROT*` in `WORKING/` (rotating where needed).

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" --entrypoint python image-processing fix_upside_down_with_magick.py /data/WORKING /data/DATA
```

### Stage 3a: Determine Skew Angle

Detect skew angle for each ROT-prefixed image in `WORKING/` using Leptonica. Writes `.skew.txt` files to `DATA/`.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" --entrypoint python image-processing determine_skew_angle.py /data/WORKING /data/DATA
```

### Stage 3b: Apply Deskew

Rotate images to correct skew using pyvips. Renames `ROT*` to `SKEW*` in `WORKING/` (deskewing where needed).

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" --entrypoint python image-processing apply_deskew_with_pyvips.py /data/WORKING /data/DATA
```

### Stage 4a: Detect Bounding Boxes

Detect text bounding boxes for each SKEW-prefixed image in `WORKING/` using PaddleOCR. Writes `.boxes.json` files to `DATA/`.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" --entrypoint python image-processing detect_bounding_boxes_with_paddleocr.py /data/WORKING /data/DATA
```

### Stage 4b: Draw Compound Bounding Boxes

Compute and draw a compound bounding rectangle around all detected text regions. Renames `SKEW*` to `BOUND*` in `WORKING/`.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" --entrypoint python image-processing draw_compound_bounding_boxes.py /data/WORKING /data/DATA
```

### Stage 4c: Crop Compound Bounding Boxes

Crop images to their compound bounding box. Renames `BOUND*` to `CROP*` in `WORKING/`.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" --entrypoint python image-processing crop_compound_bounding_boxes.py /data/WORKING /data/DATA
```

> **Note:** When run via the orchestrator with `--preserve-drawings /data/DRAWINGS`, BOUND images are copied to the drawings directory (prefix renamed to `DRAW`) before this crop step executes. The standalone script above does not support this option — it is orchestrator-only.
