#!/bin/bash
ROOT_DIR="$(dirname "$0")"
DOCKERFILE=$1
MORPH=$2
docker build -f docker/Dockerfile.$DOCKERFILE -t warp-ik-$DOCKERFILE .
docker run --gpus all -it --rm --user="root" \
-v $ROOT_DIR/.env:/root/warp-ik/.env \
-v $ROOT_DIR/output:/root/warp-ik/output \
warp-ik-$DOCKERFILE bash -c "uv run python /root/warp-ik/src/run.py --mutate --morph $MORPH"