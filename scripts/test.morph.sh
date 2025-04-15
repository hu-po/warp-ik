#!/bin/bash
ROOT_DIR="$(dirname "$(dirname "$0")")" # the warp-ik directory

# Check if BACKEND environment variable is set
if [ -z "${BACKEND}" ]; then
    echo "Error: BACKEND environment variable is not set"
    exit 1
fi

MORPH=${1:-template}
docker build -f docker/Dockerfile.$BACKEND -t warp-ik-$BACKEND .
GPU_FLAG=""
if [[ "$BACKEND" != "x86-meerkat" && "$BACKEND" != "arm-rpi" ]]; then
    GPU_FLAG="--gpus all"
fi
docker run $GPU_FLAG -it --rm --user="root" \
-v $ROOT_DIR/output:/root/warp-ik/output \
-v $ROOT_DIR/warp_ik/morphs:/root/warp-ik/warp_ik/morphs \
warp-ik-$BACKEND bash -c "
source /root/warp-ik/.venv/bin/activate && \
source /root/warp-ik/.env && \
python /root/warp-ik/warp_ik/src/morph.py --backend $BACKEND --morph $MORPH --headless" 