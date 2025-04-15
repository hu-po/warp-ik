#!/bin/bash
ROOT_DIR="$(dirname "$(dirname "$0")")" # the warp-ik directory

# Check if BACKEND environment variable is set
if [ -z "${BACKEND}" ]; then
    echo "Error: BACKEND environment variable is not set"
    exit 1
fi

PROTOMORPHS=$1
docker build -f docker/Dockerfile.$BACKEND -t warp-ik-$BACKEND .
docker run --gpus all -it --rm --user="root" \
-v $ROOT_DIR/output:/root/warp-ik/output \
-v $ROOT_DIR/warp_ik/morphs:/root/warp-ik/warp_ik/morphs \
warp-ik-$BACKEND bash -c "
source /root/warp-ik/.venv/bin/activate && \
source /root/warp-ik/.env && \
uv run python /root/warp-ik/warp_ik/src/evolve.py --backend $BACKEND --protomorphs $PROTOMORPHS"