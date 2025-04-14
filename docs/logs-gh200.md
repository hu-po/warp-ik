make sure to send .env file to the nodes:

```bash
scp .env -i ~/.ssh/oop ubuntu@FOOO:/home/ubuntu/warp-ik/.env
scp .env ojo@192.168.1.96:/home/ojo/dev/warp-ik/.env
scp .env trossen-ai@192.168.1.97:/home/trossen-ai/dev/warp-ik/.env
```

using a lambda labs `gpu_1x_gh200` node and the `oop` ssh key:

```bash
ssh -i ~/.ssh/oop ubuntu@FOOO
git clone https://github.com/hu-po/warp-ik.git
cd warp-ik
sudo usermod -aG docker ubuntu
newgrp docker
./scripts/docker.test.sh arm-gh200
```

