jetson-containers run \
-v /home/ojo/dev/evoarc/:/evoarc \
-v /home/ojo/dev/evoarc/data:/kaggle/input/arc-prize-2024 \
-v /home/ojo/dev/evoarc/output:/kaggle/working \
-e COMPUTE_BACKEND="ojo" \
-e MORPH=$1 \
-e WANDB_API_KEY=$WANDB_API_KEY \
$(autotag jax) \
bash -c "./evoarc/scripts/\$COMPUTE_BACKEND/setup.sh && jupyter nbconvert --to notebook --execute /evoarc/morphs/\$MORPH.ipynb --stdout"