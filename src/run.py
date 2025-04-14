import argparse
from dataclasses import dataclass, field
import os
import random
import re
import subprocess
import time
import uuid
import yaml
from typing import List
from enum import Enum, auto
import asyncio
import logging

from ai import inference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class EvolveConfig:
    # Directory settings
    root_dir: str = os.path.abspath(os.path.dirname(__file__))
    output_dir: str = field(default="", init=False)
    morph_dir: str = field(default="", init=False)
    prompt_dir: str = field(default="", init=False)
    mutation_prompts_dir: str = field(default="", init=False)

    # Evolution parameters
    seed: int = 0
    protomorphs: str = "ik_geojac"
    num_rounds: int = 12
    num_morphs: int = 12
    topk_morphs: int = 6
    compute_backend: str = "oop"
    mutate_on_start: bool = False

    # Mutation settings
    enabled_models: List[str] = field(default_factory=lambda: ["gpt", "claude", "gemini", "xapi"])
    mutations: List[str] = field(default_factory=lambda: [
        "open_ended",
        "tune_config",
    ])
    proc_glaze_prompt: float = 0.5
    proc_arcdoc_prompt: float = 0.1

    def __post_init__(self):
        # Initialize dependent paths
        self.output_dir = os.path.join(self.root_dir, "output")
        self.morph_dir = os.path.join(self.root_dir, "morphs")
        self.prompt_dir = os.path.join(self.root_dir, "prompts")
        self.mutation_prompts_dir = os.path.join(self.prompt_dir, "mutations")
        os.makedirs(self.morph_dir, exist_ok=True)

class MorphState(Enum):
    NOT_RUN_YET = auto()
    ALREADY_RAN = auto()
    ERRORED_OUT = auto()

@dataclass(order=True)
class Morph:
    score: float
    name: str
    state: MorphState = MorphState.NOT_RUN_YET

def load_prompt(config: EvolveConfig, prompt_path: str) -> str:
    prompt_filepath = os.path.join(config.prompt_dir, prompt_path)
    with open(prompt_filepath, "r") as f:
        return f.read()

def morph_to_prompt(config: EvolveConfig, morph: Morph) -> str:
    morph_filepath = os.path.join(config.morph_dir, f"{morph.name}.py")
    with open(morph_filepath, "r", encoding="utf-8") as f:
        return f.read()

def reply_to_morph(reply: str, name: str, output_dir: str) -> Morph:
    # remove leading ```python and trailing trailing ```
    reply = re.sub(r'^```python\s*', '', reply, flags=re.MULTILINE)
    reply = re.sub(r'^```\s*', '', reply, flags=re.MULTILINE)
    morph = Morph(0, name)
    morph_filepath = os.path.join(output_dir, f"{name}.py")
    with open(morph_filepath, "w", encoding="utf-8") as f:
        f.write(reply)
    return morph

def mutate(config: EvolveConfig, protomorph: Morph, mutation_prompt_filename: str) -> Morph:
    log.info("🧫 mutating...")
    log.info(f"\t👵 ancestor ~{protomorph.name}~")
    mutation_prompt_filepath = os.path.join(config.mutation_prompts_dir, f"{mutation_prompt_filename}.txt")
    system = load_prompt(config, mutation_prompt_filepath)
    format_prompt_filepath = os.path.join(config.prompt_dir, "format.txt")
    system += f"\n{load_prompt(config, format_prompt_filepath)}"
    if random.random() < config.proc_glaze_prompt:
        log.info("\t\t🍯 adding glazing prompt...")
        glazing_prompt_filepath = os.path.join(config.prompt_dir, "glazing.txt")
        system += f"\n\n{load_prompt(config, glazing_prompt_filepath)}"
    if random.random() < config.proc_arcdoc_prompt:
        log.info("\t\t📋 adding documentation prompt...")
        # https://nvidia.github.io/warp/modules/differentiability.html
        # https://nvidia.github.io/warp/debugging.html
        # https://nvidia.github.io/warp/modules/contribution_guide.html
        # https://nvidia.github.io/warp/configuration.html
        # https://nvidia.github.io/warp/modules/interoperability.html
        system += f"\n<helpful_docs>\n{load_prompt(config, challenge_prompt_filepath)}\n</helpful_docs>"
    prompt = morph_to_prompt(config, protomorph)
    neomorph_name = str(uuid.uuid4())[:6]
    neomorph_output_dir = os.path.join(config.output_dir, neomorph_name)
    os.makedirs(neomorph_output_dir, exist_ok=True)
    neomorph_prompt_filepath = os.path.join(neomorph_output_dir, "prompt.txt")
    with open(neomorph_prompt_filepath, "w") as f:
        f.write(f"SYSTEM:\n{system}\n\nPROMPT:\n{prompt}")
    reply = inference(f"system:\n{system}\n\nprompt:\n{prompt}", models=config.enabled_models)
    neomorph = reply_to_morph(reply, neomorph_name, config.morph_dir)
    log.info(f"\t🥚 welcome ~{neomorph_name}~")
    return neomorph

def evolve(config: EvolveConfig):
    log.info(f"Seed: {config.seed}")
    random.seed(config.seed)

    morphs: List[Morph] = []
    for protomorph in config.protomorphs.split(","):
        if os.path.exists(os.path.join(config.morph_dir, f"{protomorph}.py")):
            morphs.append(Morph(0, protomorph))
    log.info("protomorphs:")
    for morph in morphs:
        log.info(f"\t🧬\t~{morph.name}~")
    session_id = str(uuid.uuid4())[:6]
    leaderboard_dir = os.path.join(config.output_dir, f"session.{session_id}")
    os.makedirs(leaderboard_dir, exist_ok=True)
    for round_num in range(config.num_rounds):
        log.info(f"🥊 round {round_num}")
        log.info("\t mutating until full morphs...")
        protomophs = morphs.copy()
        if config.mutate_on_start:
            morphs = []
        else:
            morphs = random.choices(protomophs, k=config.num_morphs)
        while len(morphs) < config.num_morphs:
            protomorph = random.choice(protomophs)
            neomorph = mutate(config, protomorph, random.choice(config.mutations))
            morphs.append(neomorph)
        log.info("\t morphs:")
        for morph in morphs:
            log.info(f"\t🧬\t~{morph.name}~")
        log.info("\t running morphs...")
        leaderboard = {}
        leaderboard_filepath = os.path.join(leaderboard_dir, f"leaderboard.r{round_num}.yaml")
        for morph in morphs:
            if morph.state == MorphState.ALREADY_RAN:
                log.info(f"\t⏩\tSkipping {morph.name} with score {morph.score}")
                continue
            elif morph.state == MorphState.ERRORED_OUT:
                log.info(f"\t⏩\tSkipping {morph.name} with errors")
                continue
            else:
                log.info(f"\t⏯️\tRunning {morph.name}")
            log.info("killing stale morphs...")
            subprocess.run("docker kill $(docker ps -aq)", shell=True)
            subprocess.run("docker rm $(docker ps -aq)", shell=True)
            time.sleep(2)
            try:
                log.info("running morph...")
                proc = subprocess.Popen(["bash", f"scripts/{config.compute_backend}/run.sh", morph.name])
                proc.wait()
                if proc.returncode != 0:
                    log.error(f"\t❌\tError when running {morph.name}")
                    morph.state = MorphState.ERRORED_OUT
                    continue
                morph_output_dir = os.path.join(config.output_dir, morph.name)
                os.makedirs(morph_output_dir, exist_ok=True)
                morph_output_filepath = os.path.join(morph_output_dir, "results.json")
                with open(morph_output_filepath, "r") as f:
                    morph_output = yaml.safe_load(f)
                score = morph_output["accuracy"]
            except Exception as e:
                log.error(f"\t❌\tError when running {morph.name}: {e}")
                score = 0
                continue
            leaderboard[morph.name] = score
            morph.score = score
            log.info(f"\t🏁\t{morph.name} scored {score}")
            morph.state = MorphState.ALREADY_RAN
        
        # write sorted leaderboard
        leaderboard = {k: v for k, v in sorted(leaderboard.items(), key=lambda item: item[1], reverse=True)}
        with open(leaderboard_filepath, "w") as f:
            yaml.safe_dump(leaderboard, f, default_flow_style=False)

        # ---- elimination ----
        log.info("Elimination:")
        doomed = []
        for i, morph in enumerate(sorted(morphs, key=lambda m: m.score, reverse=True)):
            score = morph.score
            if i < config.topk_morphs:
                log.info(f"\t🏆\t{morph.name} is in the top {config.topk_morphs} with score {score}")
            else:
                log.info(f"\t🗑\t{morph.name} is in the bottom with score {score}")
                doomed.append(morph)

        morphs = [morph for morph in morphs if morph not in doomed]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", type=str, default="gpt-4")
    parser.add_argument("--protomorphs", type=str, default="ik_3d,ik_6d", help="comma separated list of protomorphs to seed evolution")
    parser.add_argument("--num_rounds", type=int, default=12, help="number of rounds to run")
    parser.add_argument("--num_morphs", type=int, default=12, help="number of morphs per round")
    parser.add_argument("--topk_morphs", type=int, default=6, help="number of top morphs to keep each round")
    parser.add_argument("--compute_backend", type=str, default="oop")
    parser.add_argument("--mutate_on_start", action="store_true", help="whether to mutate protomorphs at the start")
    args = parser.parse_args()

    # Create config from args
    config = EvolveConfig(
        seed=args.seed,
        agent=args.agent,
        protomorphs=args.protomorphs,
        num_rounds=args.num_rounds,
        num_morphs=args.num_morphs,
        topk_morphs=args.topk_morphs,
        compute_backend=args.compute_backend,
        mutate_on_start=args.mutate_on_start,
    )
    evolve(config)

