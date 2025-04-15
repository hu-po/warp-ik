# Checking on GH200

make sure to send .env file to the nodes:

```bash
scp -i ~/.ssh/oop.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null .env ubuntu@192.222.50.145:/home/ubuntu/warp-ik/.env
scp .env ojo@192.168.1.96:/home/ojo/dev/warp-ik/.env
scp .env trossen-ai@192.168.1.97:/home/trossen-ai/dev/warp-ik/.env
scp .env rpi1@192.168.1.98:/home/rpi1/dev/warp-ik/.env
```

using a lambda labs `gpu_1x_gh200` node and the `oop` ssh key:

```bash
ssh -i ~/.ssh/oop.pem ubuntu@192.222.50.145
git clone https://github.com/hu-po/warp-ik.git
cd warp-ik
sudo usermod -aG docker ubuntu
newgrp docker
./scripts/test.sh arm-gh200
```

# Running Morphs

testing out all the morphs locally

```bash
./scripts/run.morph.sh x86-3090 template && \
usdview /home/oop/dev/warp-ik/output/template/recording.usd
```

```bash
./scripts/run.morph.sh x86-3090 jacobian_geom_3d && \
usdview /home/oop/dev/warp-ik/output/jacobian_geom_3d/recording.usd
```

```bash
./scripts/run.morph.sh x86-3090 jacobian_geom_6d && \
usdview /home/oop/dev/warp-ik/output/jacobian_geom_6d/recording.usd
```

```bash
./scripts/run.morph.sh x86-3090 jacobian_fd && \
usdview /home/oop/dev/warp-ik/output/jacobian_fd/recording.usd
```

run them all

```bash
./scripts/run.morph.sh x86-3090 template && \
./scripts/run.morph.sh x86-3090 jacobian_geom_3d && \
./scripts/run.morph.sh x86-3090 jacobian_geom_6d && \
./scripts/run.morph.sh x86-3090 jacobian_fd && \
./scripts/run.morph.sh x86-3090 jacobian_dls_6d && \
./scripts/run.morph.sh x86-3090 optim_adam_6d
```

when logging in fresh to each node, set the environment variables:

```bash
export BACKEND="x86-3090"
export BACKEND="arm-agx"
export BACKEND="x86-meerkat"
export BACKEND="arm-gh200"
export BACKEND="arm-rpi"
```

test the backend

```bash
git pull && ./scripts/test.sh
```

test the morphs

```bash
git pull && ./scripts/test.sh && ./scripts/test.morph.sh
```

