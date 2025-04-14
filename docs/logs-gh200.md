using a lambda labs `gpu_1x_gh200` node and the `oop` ssh key:

```bash
ssh -i ~/.ssh/oop ubuntu@FOOO
git clone https://github.com/hu-po/warp-ik.git
cd warp-ik
sudo usermod -aG docker ubuntu
newgrp docker
./scripts/docker.test.sh arm-gh200
```

