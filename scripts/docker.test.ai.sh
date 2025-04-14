#!/bin/bash
ROOT_DIR="$(dirname "$0")"
DOCKERFILE=$1

# Build the Docker image
docker build -f docker/Dockerfile.$DOCKERFILE -t warp-ik-$DOCKERFILE .

# Set GPU flag based on dockerfile type
GPU_FLAG=""
if [[ "$DOCKERFILE" != "x86-meerkat" ]]; then
    GPU_FLAG="--gpus all"
fi

# Run the container with conditional GPU flag
docker run $GPU_FLAG -it --rm --user="root" \
-v $ROOT_DIR/output:/root/warp-ik/output \
warp-ik-$DOCKERFILE bash -c "
source /root/warp-ik/.venv/bin/activate && \
source /root/warp-ik/.env && \
python /root/warp-ik/src/test.py && \
python /root/warp-ik/src/ai.py --test"