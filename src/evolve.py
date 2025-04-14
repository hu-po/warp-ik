import argparse
from dataclasses import dataclass
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

import nbformat
from openai import OpenAI
from ai import (
    AI_MODEL_MAP,
    ENABLED_MODELS,
    AI_MAX_TOKENS,
    log,
)

# set up directories
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MORPH_DIR = os.path.join(ROOT_DIR, "morphs")
PROMPT_DIR = os.path.join(ROOT_DIR, "prompts")
MUTATION_PROMPTS_DIR = os.path.join(PROMPT_DIR, "mutations")
os.makedirs(MORPH_DIR, exist_ok=True)

# choose which mutations are active
MUTATIONS: List[str] = [
    "open_ended",
    "tune_config",
]

# mutation prompt modifiers (proc chance in 0 to 1)
PROC_GLAZE_PROMPT: float = 0.5
PROC_ARCDOC_PROMPT: float = 0.1

DEFAULT_MORPHS = ",".join([
    "ik_3d",
    "ik_6d",
])

class MorphState(Enum):
    NOT_RUN_YET = auto()
    ALREADY_RAN = auto()
    ERRORED_OUT = auto()

@dataclass(order=True)
class Morph:
    score: float
    name: str
    state: MorphState = MorphState.NOT_RUN_YET

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent", type=str, default=DEFAULT_AGENT)
parser.add_argument("--protomorphs", type=str, default=DEFAULT_MORPHS, help="comma separated list of protomorphs to seed evolution")
parser.add_argument("--num_rounds", type=int, default=12, help="number of rounds to run")
parser.add_argument("--num_morphs", type=int, default=12, help="number of morphs per round")
parser.add_argument("--topk_morphs", type=int, default=6, help="number of top morphs to keep each round")
parser.add_argument("--compute_backend", type=str, default="oop")
parser.add_argument("--mutate_on_start", action="store_true", help="whether to mutate protomorphs at the start")
args = parser.parse_args()

# Setup and seeding
print(f"Seed: {args.seed}")
random.seed(args.seed)

def load_prompt(prompt_path):
    prompt_filepath = os.path.join(PROMPT_DIR, prompt_path)
    with open(prompt_filepath, "r") as f:
        return f.read()

def morph_to_prompt(morph: Morph) -> str:
    morph_filepath = os.path.join(MORPH_DIR, f"{morph.name}.py")
    with open(morph_filepath, "r", encoding="utf-8") as f:
        return f.read()

def reply_to_morph(reply: str, name:str, output_dir: str) -> Morph:
    # remove leading ```python and trailing trailing ```
    reply = re.sub(r'^```python\s*', '', reply, flags=re.MULTILINE)
    reply = re.sub(r'^```\s*', '', reply, flags=re.MULTILINE)
    morph = Morph(0, name)
    morph_filepath = os.path.join(output_dir, f"{name}.py")
    with open(morph_filepath, "w", encoding="utf-8") as f:
        f.write(reply)
    return morph

def run_agent(system: str, prompt: str, agent: str = DEFAULT_AGENT):
    print(f"\t🧠 calling enabled models: {ENABLED_MODELS}...")
    
    if not ENABLED_MODELS:
        raise ValueError("No AI models are enabled")
    
    # Combine system and user prompts into a single context
    full_prompt = f"SYSTEM:\n{system}\n\nUSER:\n{prompt}"
    
    # Create and run a temporary event loop for async calls
    async def _run_models():
        tasks = []
        ai_models = []
        
        for model_name in ENABLED_MODELS:
            tasks.append(AI_MODEL_MAP[model_name](full_prompt))
            ai_models.append(model_name)
            
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for model_name, response in zip(ai_models, responses):
            if isinstance(response, Exception):
                print(f"\t❌ {model_name} failed: {str(response)}")
                continue
            results[model_name] = response
            
        if not results:
            raise ValueError("All AI models failed to respond")
            
        return next(iter(results.values()))
    
    try:
        # Run async code in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(_run_models())
        loop.close()
        print("\t... completed")
        return response
        
    except Exception as e:
        print(f"\t❌ All models failed: {str(e)}")
        raise


def mutate(protomorph: Morph, mutation_prompt_filename: str) -> Morph:
    print("🧫 mutating...")
    print(f"\t👵 ancestor ~{protomorph.name}~")
    mutation_prompt_filepath = os.path.join(MUTATION_PROMPTS_DIR, f"{mutation_prompt_filename}.txt")
    system = load_prompt(mutation_prompt_filepath)
    format_prompt_filepath = os.path.join(PROMPT_DIR, "format.txt")
    system += f"\n{load_prompt(format_prompt_filepath)}"
    if random.random() < PROC_GLAZE_PROMPT:
        print("\t\t🍯 adding glazing prompt...")
        glazing_prompt_filepath = os.path.join(PROMPT_DIR, "glazing.txt")
        system += f"\n\n{load_prompt(glazing_prompt_filepath)}"
    if random.random() < PROC_ARCDOC_PROMPT:
        print("\t\t📋 adding documentation prompt...")
        # https://nvidia.github.io/warp/modules/differentiability.html
        # https://nvidia.github.io/warp/debugging.html
        # https://nvidia.github.io/warp/modules/contribution_guide.html
        # https://nvidia.github.io/warp/configuration.html
        # https://nvidia.github.io/warp/modules/interoperability.html
        system += f"\n<helpful_docs>\n{load_prompt(challenge_prompt_filepath)}\n</helpful_docs>"
    prompt = morph_to_prompt(protomorph)
    neomorph_name = str(uuid.uuid4())[:6]
    neomorph_output_dir = os.path.join(OUTPUT_DIR, neomorph_name)
    os.makedirs(neomorph_output_dir, exist_ok=True)
    neomorph_prompt_filepath = os.path.join(neomorph_output_dir, "prompt.txt")
    with open(neomorph_prompt_filepath, "w") as f:
        f.write(f"SYSTEM:\n{system}\n\nPROMPT:\n{prompt}")
    reply = run_agent(system, prompt)
    neomorph = reply_to_morph(reply, neomorph_name, MORPH_DIR)
    print(f"\t🥚 welcome ~{neomorph_name}~")
    return neomorph

if __name__ == "__main__":
    morphs: List[Morph] = []
    for protomorph in args.protomorphs.split(","):
        if os.path.exists(os.path.join(MORPH_DIR, f"{protomorph}.py")):
            morphs.append(Morph(0, protomorph))
    print("protomorphs:")
    for morph in morphs:
        print(f"\t🧬\t~{morph.name}~")
    session_id = str(uuid.uuid4())[:6]
    leaderboard_dir = os.path.join(OUTPUT_DIR, f"session.{session_id}")
    os.makedirs(leaderboard_dir, exist_ok=True)
    for round_num in range(args.num_rounds):
        print(f"🥊 round {round_num}")
        print("\t mutating until full morphs...")
        protomophs = morphs.copy()
        if args.mutate_on_start:
            morphs = []
        else:
            morphs = random.choices(protomophs, k=args.num_morphs)
        while len(morphs) < args.num_morphs:
            protomorph = random.choice(protomophs)
            neomorph = mutate(protomorph, random.choice(MUTATIONS))
            morphs.append(neomorph)
        print("\t morphs:")
        for morph in morphs:
            print(f"\t🧬\t~{morph.name}~")
        print("\t running morphs...")
        leaderboard = {}
        leaderboard_filepath = os.path.join(leaderboard_dir, f"leaderboard.r{round_num}.yaml")
        for morph in morphs:
            if morph.state == MorphState.ALREADY_RAN:
                print(f"\t⏩\tSkipping {morph.name} with score {morph.score}")
                continue
            elif morph.state == MorphState.ERRORED_OUT:
                print(f"\t⏩\tSkipping {morph.name} with errors")
                continue
            else:
                print(f"\t⏯️\tRunning {morph.name}")
            print("killing stale morphs...")
            subprocess.run("docker kill $(docker ps -aq)", shell=True)
            subprocess.run("docker rm $(docker ps -aq)", shell=True)
            time.sleep(2)
            try:
                print("running morph...")
                proc = subprocess.Popen(["bash", f"scripts/{args.compute_backend}/run.sh", morph.name])
                proc.wait()
                if proc.returncode != 0:
                    print(f"\t❌\tError when running {morph.name}")
                    morph.state = MorphState.ERRORED_OUT
                    continue
                morph_output_dir = os.path.join(OUTPUT_DIR, morph.name)
                os.makedirs(morph_output_dir, exist_ok=True)
                morph_output_filepath = os.path.join(morph_output_dir, "results.json")
                with open(morph_output_filepath, "r") as f:
                    morph_output = yaml.safe_load(f)
                score = morph_output["accuracy"]
            except Exception as e:
                print(f"\t❌\tError when running {morph.name}: {e}")
                score = 0
                # TODO: run a "bugfix" mutation
                continue
            leaderboard[morph.name] = score
            morph.score = score
            print(f"\t🏁\t{morph.name} scored {score}")
            morph.state = MorphState.ALREADY_RAN
        
        # write sorted leaderboard
        leaderboard = {k: v for k, v in sorted(leaderboard.items(), key=lambda item: item[1], reverse=True)}
        with open(leaderboard_filepath, "w") as f:
            yaml.safe_dump(leaderboard, f, default_flow_style=False)

        # ---- elimination ----
        print("Elimination:")
        doomed = []
        for i, morph in enumerate(sorted(morphs, key=lambda m: m.score, reverse=True)):
            score = morph.score
            if i < args.topk_morphs:
                print(f"\t🏆\t{morph.name} is in the top {args.topk_morphs} with score {score}")
            else:
                print(f"\t🗑\t{morph.name} is in the bottom with score {score}")
                doomed.append(morph)

        morphs = [morph for morph in morphs if morph not in doomed]
