#!/bin/bash
DEVICE=$1 # device
DEV_DIR="/home/$USER/dev/warp-ik"
docker build -f docker/Dockerfile.arm -t warp-ik-$DEVICE .
sudo docker run --gpus all -it --rm \
-v $DEV_DIR/.env:/warp-ik/.env \
-v $DEV_DIR/src:/warp-ik/src \
-v $DEV_DIR/data:/warp-ik/data \
-v $DEV_DIR/output:/warp-ik/output \
warp-ik-$DEVICE bash -c "cd warp-ik && uv venv && source .venv/bin/activate && uv pip install -e . && uv run python /root/src/test.py --device $DEVICE --stdout"