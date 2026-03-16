# IMAGE_PROCESSING

## Build

```bash
docker build -t image-processing .
```

## Run

Expects `RAW/` inside the bind-mounted folder. Creates `WORKING/` and `DATA/` there.

```bash
docker run -v "/mnt/c/Users/natha/OneDrive/Desktop/TEST_IMAGE_PROCESSING/DOCKER_TEST:/data" image-processing
```

## Rebuild

Remove old image and build fresh.

```bash
docker rmi image-processing && docker build -t image-processing .
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
