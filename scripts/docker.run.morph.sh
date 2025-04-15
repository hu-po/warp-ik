#!/bin/bash
ROOT_DIR="$(dirname "$(dirname "$0")")" # the warp-ik directory
DOCKERFILE=$1
MORPH=${2:-template}
docker build -f docker/Dockerfile.$DOCKERFILE -t warp-ik-$DOCKERFILE .
docker run --gpus all -it --rm --user="root" \
-v $ROOT_DIR/output:/root/warp-ik/output \
-v $ROOT_DIR/warp_ik/morphs:/root/warp-ik/warp_ik/morphs \
warp-ik-$DOCKERFILE bash -c "
source /root/warp-ik/.venv/bin/activate && \
source /root/warp-ik/.env && \
uv run python /root/warp-ik/warp_ik/src/morph.py --dockerfile $DOCKERFILE --morph $MORPH --track --seed 2"