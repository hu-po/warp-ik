#!/bin/bash
ROOT_DIR="$(dirname "$(dirname "$0")")" # the warp-ik directory
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
uv pip freeze && \
bash /root/warp-ik/scripts/specs.sh && \
if [[ \"$DOCKERFILE\" != \"x86-meerkat\" ]]; then
    python /root/warp-ik/src/warp/device_properties.py
fi && \
python /root/warp-ik/src/test.py"