#!/bin/bash
ROOT_DIR="$(dirname "$0")"
docker build -f docker/Dockerfile.$DOCKERFILE -t warp-ik-$DOCKERFILE .
sudo docker run --gpus all -it --rm \
-v $ROOT_DIR/.env:/warp-ik/.env \
-v $ROOT_DIR/src:/warp-ik/src \
-v $ROOT_DIR/data:/warp-ik/data \
-v $ROOT_DIR/output:/warp-ik/output \
warp-ik-$DOCKERFILE bash -c "cd warp-ik && uv venv && source .venv/bin/activate && uv pip install -e . && uv run python /root/src/evolve.py --device $DEVICE --morph $MORPH --stdout"