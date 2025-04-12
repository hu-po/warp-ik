sudo docker run --gpus all -it --rm \
-v /home/ubuntu/evoarc:/evoarc \
-v /home/ubuntu/evoarc/data:/kaggle/input/arc-prize-2024 \
-v /home/ubuntu/evoarc/output:/kaggle/working \
-e COMPUTE_BACKEND="big" \
-e MORPH=$1 \
-e WANDB_API_KEY=$WANDB_API_KEY \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c "./evoarc/scripts/\$COMPUTE_BACKEND/setup.sh && WANDB_API_KEY=\$WANDB_API_KEY jupyter nbconvert --to notebook --execute /evoarc/morphs/\$MORPH.ipynb --stdout"