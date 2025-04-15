# warp-ik 🦾🦾🦾🦾🦾🦾

# `i`nverse `k`inematic solution evolution using nvidia `warp`

uses an llm (e.g. claude,gpt,xai,gemini) to rewrite and improve the code solution to a multi-robot parallel inverse kinematics problem. Iterative rounds to improve solutions, called `morphs`, which are run in docker containers and compete to get the highest score. A basic evolutionary algorithm with mutation and selection improves the morphs over time.

local setup

```bash
git clone https://github.com/hu-po/warp-ik.git && \
cd warp-ik && \
uv venv && \
source .venv/bin/activate && \
uv pip install -e .
```

system information

```bash
./scripts/specs.sh
uv run python src/device_properties.py
```

local test

```bash
uv run python src/test.py
```

build and test the docker container for a specific compute backend `x86-3090`

```bash
export BACKEND=x86-3090
./scripts/test.sh
```

test that ai (openai, anthropic, gemini, xai, replicate) is working:

```bash
./scripts/test.ai.sh
```

test that the morphs work, tune a number of parallel environments `NUM_ENVS` based on your GPU memory:

```bash
export NUM_ENVS=32
./scripts/test.morph.sh
```

run a morph by name, e.g. `jacobian_geom_6d`:

```bash
./scripts/run.morph.sh jacobian_geom_6d
```

start the evolutionary process using protomorphs `jacobian_geom_3d,jacobian_geom_6d`:

```bash
./scripts/run.evolve.sh jacobian_geom_3d,jacobian_geom_6d
```

create a mutation of a morph, check the output in `warp_ik/morphs/` folder:

```bash
./scripts/run.mutate.sh jacobian_geom_3d
```

clean out the output directory:

```bash
./scripts/clean.sh
```

create context file for asking about codebase to an llm:

```bash
./scripts/context.sh
```

use usd viewer to view rendered outputs, requires USD installation [here](https://developer.nvidia.com/usd?sortBy=developer_learning_library%2Fsort%2Ffeatured_in.usd_resources%3Adesc%2Ctitle%3Aasc#section-getting-started)

```bash
sudo apt-get install libxkbcommon-x11-0 libxcb-xinerama0 libxcb-image0 libxcb-shape0 libxcb-render-util0 libxcb-icccm4 libxcb-keysyms1
unzip ~/Downloads/usd.py310.linux-x86_64.usdview.release-0.25.02-ba8aaf1f.zip -d ~/dev/usd
alias usdview="~/dev/usd/scripts/usdview_gui.sh"
usdview /home/oop/dev/warp-ik/output/template.usd
```

Scroll in with mousewheel
Use <space> to play and pause
Use <alt>+<left click> to rotate the camera
Use <ctrl>+<6> for geometry view

to profile a morph, use NVIDIA Nsight Compute, requires installation [here](https://developer.nvidia.com/nsight-systems/get-started)

```bash
chmod +x ~/Downloads/NsightSystems-linux-public-2025.2.1.130-3569061.run
sudo ~/Downloads/NsightSystems-linux-public-2025.2.1.130-3569061.run
/opt/nvidia/nsight-systems/2025.2.1/bin/nsys profile --trace=cuda ./warp_ik/src/morph.py
/opt/nvidia/nsight-systems/2025.2.1/bin/nsys viewer ./report1.nsys-rep
```

```
@misc{hupo2025warpik,
  title={warp-ik: inverse kinematics solution evolution using nvidia warp},
  author={Hugo Ponte},
  year={2025},
  url={https://github.com/hu-po/warp-ik}
}
```
