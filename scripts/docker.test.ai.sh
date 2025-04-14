#!/bin/bash
ROOT_DIR="$(dirname "$0")"
DOCKERFILE=$1
docker build -f docker/Dockerfile.$DOCKERFILE -t warp-ik-$DOCKERFILE .
GPU_FLAG=""
if [[ "$DOCKERFILE" != "x86-meerkat" ]]; then
    GPU_FLAG="--gpus all"
fi
docker run $GPU_FLAG -it --rm --user="root" \
warp-ik-$DOCKERFILE bash -c "
source /root/warp-ik/.venv/bin/activate && \
source /root/warp-ik/.env && \
python /root/warp-ik/src/test.py && \
python /root/warp-ik/src/ai.py --test"