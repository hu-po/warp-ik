docker run --gpus all -it --rm \
-v /home/oop/dev/evoarc:/evoarc \
-v /home/oop/dev/evoarc/data:/kaggle/input/arc-prize-2024 \
-v /home/oop/dev/evoarc/output:/kaggle/working \
-e COMPUTE_BACKEND="oop" \
-e MORPH=$1 \
-e WANDB_API_KEY=$WANDB_API_KEY \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c "./evoarc/scripts/\$COMPUTE_BACKEND/setup.sh && jupyter nbconvert --to notebook --execute /evoarc/morphs/\$MORPH.ipynb --stdout"