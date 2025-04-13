#!/bin/bash
ROOT_DIR="$(dirname "$0")"
DOCKERFILE=$1
MORPH=$2
docker build -f docker/Dockerfile.$DOCKERFILE \
-t warp-ik-$DOCKERFILE \
-e DOCKERFILE=$DOCKERFILE \
-e MORPH=$MORPH \
.
docker run --gpus all -it --rm --user="root" \
-v $ROOT_DIR/.env:/home/ubuntu/warp-ik/.env \
-v $ROOT_DIR/src:/home/ubuntu/warp-ik/src \
-v $ROOT_DIR/data:/home/ubuntu/warp-ik/data \
-v $ROOT_DIR/output:/home/ubuntu/warp-ik/output \
warp-ik-$DOCKERFILE bash -c "uv venv && source .venv/bin/activate && \
uv run python /home/ubuntu/warp-ik/src/test.py && \
uv run python /home/ubuntu/warp-ik/src/ai.py --test && \
uv run python /home/ubuntu/warp-ik/src/device_properties.py"