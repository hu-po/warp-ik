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
import logging

from ai import inference
from mutate import mutate
from morph import ActiveMorph, MorphState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class EvolveConfig:
    seed: int = 33
    num_rounds: int = 12 # number of rounds of evolution (mutation + elimination)
    num_morphs: int = 12 # size of the population of morphs (replentished every round)
    topk_morphs: int = 6 # number of top morphs to keep each round
    mutate_on_start: bool = False # whether to mutate protomorphs at the start
    backend: str = os.environ.get("BACKEND") # compute backend variant
    root_dir: str = os.environ.get("WARP_IK_ROOT") # root directory of the warp-ik project
    protomorphs: str = os.environ.get("PROTOMORPHS") # comma separated list of protomorphs to seed evolution
    output_dir: str = f"{root_dir}/output" # output directory for the morphs
    morph_dir: str = f"{root_dir}/warp_ik/morphs" # directory for the morphs


def evolve(config: EvolveConfig):
    log.info(f"Seed: {config.seed}")
    random.seed(config.seed)
    morphs: List[ActiveMorph] = []
    for protomorph in config.protomorphs.split(","):
        if os.path.exists(os.path.join(config.morph_dir, f"{protomorph}.py")):
            morphs.append(MorphState(0, protomorph))
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
            try:
                log.info("running morph...")
                proc = subprocess.Popen(["uv run python", f"{config.root_dir}/morphs/{morph.name}.py", "--morph", morph.name, "--headless"])
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
    parser.add_argument("--seed", type=int, default=EvolveConfig.seed)
    parser.add_argument("--backend", type=str, default=EvolveConfig.backend, help="Override default compute backend variant.")
    parser.add_argument("--protomorphs", type=str, default=EvolveConfig.protomorphs, help="comma separated list of protomorphs to seed evolution")
    parser.add_argument("--num_rounds", type=int, default=EvolveConfig.num_rounds, help="number of rounds to run")
    parser.add_argument("--num_morphs", type=int, default=EvolveConfig.num_morphs, help="number of morphs per round")
    parser.add_argument("--topk_morphs", type=int, default=EvolveConfig.topk_morphs, help="number of top morphs to keep each round")
    parser.add_argument("--mutate_on_start", action="store_true", help="whether to mutate protomorphs at the start")
    args = parser.parse_args()

    # Create config from args
    config = EvolveConfig(
        seed=args.seed,
        backend=args.backend,
        protomorphs=args.protomorphs,
        num_rounds=args.num_rounds,
        num_morphs=args.num_morphs,
        topk_morphs=args.topk_morphs,
        mutate_on_start=args.mutate_on_start,
    )
    evolve(config)

