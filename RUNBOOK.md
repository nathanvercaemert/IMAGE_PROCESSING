# Runbook

## Standard pipeline

Runs all stages from RAW to CROP. Bounding-box drawings (BOUND images) are discarded after cropping.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing
```

This is equivalent to:

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing /data/RAW /data/WORKING /data/DATA
```

## Pipeline with preserved drawings

Same as above, but before the crop step the BOUND images (which show the drawn bounding rectangles) are copied to a separate directory. The copy renames the filename prefix from `BOUND` to `DRAW`.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing /data/RAW /data/WORKING /data/DATA --preserve-drawings /data/DRAWINGS
```

After the run completes you will find:

| Directory | Contents |
|---|---|
| `/data/WORKING/` | `CROP*` files (final cropped images) |
| `/data/DATA/` | Sidecar data (orientation, skew, boxes, compound) |
| `/data/DRAWINGS/` | `DRAW*` files (images with bounding rectangles drawn, before crop) |
