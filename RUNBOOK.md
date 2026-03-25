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

## Single-file reprocessing

Reprocess one image after manually editing a data file. Requires `--single-file` with exactly one of `--rotate`, `--deskew`, or `--draw`.

### After editing orientation

Edit `DATA/RAW_0001.tif.orientation.txt` (change 0 to 180 or vice versa), then:

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing /data/RAW /data/WORKING /data/DATA --single-file RAW_0001.tif --rotate
```

Skew, bounding boxes, and compound data are re-detected fresh. The existing `.skew.txt`, `.boxes.json`, and `.compound.txt` for this image are overwritten.

### After editing skew

Edit `DATA/ROT_0001.tif.skew.txt` (change the angle or confidence), then:

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing /data/RAW /data/WORKING /data/DATA --single-file RAW_0001.tif --deskew
```

Bounding boxes and compound data are re-detected fresh. The existing `.boxes.json` and `.compound.txt` for this image are overwritten.

### After editing compound crop coordinates

Edit `DATA/SKEW_0001.tif.compound.txt` (change left, top, width, height), then:

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing /data/RAW /data/WORKING /data/DATA --single-file RAW_0001.tif --draw
```

No re-detection. The edited compound data is used directly for both the drawn rectangle and the crop. Bounding box data is ignored.

### With preserved drawings

Add `--preserve-drawings` to any of the above:

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing /data/RAW /data/WORKING /data/DATA --single-file RAW_0001.tif --deskew --preserve-drawings /data/DRAWINGS
```
