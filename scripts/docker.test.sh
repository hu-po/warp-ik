#!/bin/bash
ROOT_DIR="$(dirname "$0")"
DOCKERFILE=$1
docker build -f docker/Dockerfile.$DOCKERFILE -t warp-ik-$DOCKERFILE .
docker run --gpus all -it --rm \
-v $ROOT_DIR/.env:/warp-ik/.env \
-v $ROOT_DIR/src:/warp-ik/src \
-v $ROOT_DIR/data:/warp-ik/data \
-v $ROOT_DIR/output:/warp-ik/output \
warp-ik-$DOCKERFILE bash -c "uv venv && source .venv/bin/activate && uv run python /warp-ik/src/test.py"