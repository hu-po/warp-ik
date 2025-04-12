# warp-ik 🦾🦾🦾🦾🦾🦾

# `i`nverse `k`inematic solution evolution using nvidia `warp`

uses an llm (e.g. claude,gpt,xai,gemini) to rewrite and improve the code solution to a multi-robot parallel inverse kinematics problem. Iterative rounds to improve solutions, called `morphs`, which are run in docker containers and compete to get the highest score. A basic evolutionary algorithm with mutation and selection improves the morphs over time.

one-liner

```bash
git clone https://github.com/hu-po/warp-ik-evo.git && \
cd warp-ik-evo && \
uv venv && \
source .venv/bin/activate && \
uv pip install -e . && \
uv run python /root/src/evolve.py --seed 42 --device 0
```

## Usage

run the morph `ik_6d` locally on `x86`:

```bash
./scripts/test.sh x86 ik_6d
```

install python dependencies and run cloth sim creation

```bash
cd warp
uv venv && source .venv/bin/activate
uv pip install warp-lang[extras]
# for arm agx orin cuda11
uv pip install https://github.com/NVIDIA/warp/releases/download/v1.7.0/warp_lang-1.7.0+cu11-py3-none-manylinux2014_aarch64.whl
uv run python cloth.py
uv run python ik.py
```

use usd viewer to view the cloth sim
download binaries from https://developer.nvidia.com/usd?sortBy=developer_learning_library%2Fsort%2Ffeatured_in.usd_resources%3Adesc%2Ctitle%3Aasc#section-getting-started

```bash
sudo apt-get install libxkbcommon-x11-0 libxcb-xinerama0 libxcb-image0 libxcb-shape0 libxcb-render-util0 libxcb-icccm4 libxcb-keysyms1
unzip ~/Downloads/usd.py310.linux-x86_64.usdview.release-0.25.02-ba8aaf1f.zip -d ~/dev/usd
/home/oop/dev/usd/scripts/usdview_gui.sh /home/oop/dev/cu/warp/cloth.usd
/home/oop/dev/usd/scripts/usdview_gui.sh /home/oop/dev/cu/warp/ik.usd
```

Scroll in with mousewheel
Use <space> to play and pause
Use <alt>+<left click> to rotate the camera
Use <ctrl>+<6> for geometry view

# Using a custom URDF

IK example but with trossen widowx arm

```bash
cd ~/dev # this is where I keep my repos
git clone https://github.com/TrossenRobotics/trossen_arm_description.git
# ik with custom URDF requires trimesh
uv pip install trimesh
uv run python ik_trossen.py
/home/oop/dev/usd/scripts/usdview_gui.sh /home/oop/dev/cu/warp/ik_trossen.usd
```


start the evolutionary process using `ik_6d` as the protomorph, creates a morphline:

```bash
python3 evolve.py --seed 42 --device device --protomorphs ik_6d
```

mutate a single morph:

```bash
python3 mutate.py --morph foo
```

clean out the output directory:

```bash
./scripts/clean.sh
```

```
@misc{hupo2025warpik,
  title={warp-ik-evo: inverse kinematics solution evolution},
  author={Hugo Ponte},
  year={2025},
  url={https://github.com/hu-po/warp-ik-evo}
}
```
