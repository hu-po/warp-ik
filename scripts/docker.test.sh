#!/bin/bash
ROOT_DIR="$(dirname "$0")"
DOCKERFILE=$1
docker build -f docker/Dockerfile.$DOCKERFILE -t warp-ik-$DOCKERFILE .
docker run --gpus all -it --rm --user="root" \
-v $ROOT_DIR/.env:/root/warp-ik/.env \
-v $ROOT_DIR/output:/root/warp-ik/output \
warp-ik-$DOCKERFILE bash -c "
source /root/warp-ik/.venv/bin/activate && \
uv pip freeze && \
python /root/warp-ik/src/test.py && \
python /root/warp-ik/src/ai.py --test && \
python /root/warp-ik/src/warp/device_properties.py"