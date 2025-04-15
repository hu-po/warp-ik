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
from morph import ActiveMorph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class MutateConfig:
    seed: int = 33
    morph: str = "test" # unique identifier for the morph
    root_dir: str = os.environ.get("WARP_IK_ROOT") # root directory of the warp-ik project
    morph_dir: str = f"{root_dir}/warp_ik/morphs" # directory for the morphs
    prompts_dir: str = f"{root_dir}/warp_ik/prompts" # directory for the prompts
    enabled_models: List[str] = field(default_factory=lambda: ["gpt", "claude", "gemini", "xapi", "replicate"])
    proc_glaze_prompt: float = 0.5
    proc_helpdocs_prompt: float = 0.1

def load_prompt(config: MutateConfig, prompt_path: str) -> str:
    prompt_filepath = os.path.join(config.prompt_dir, prompt_path)
    with open(prompt_filepath, "r") as f:
        return f.read()

def morph_to_prompt(config: MutateConfig, morph: ActiveMorph) -> str:
    morph_filepath = os.path.join(config.morph_dir, f"{morph.name}.py")
    with open(morph_filepath, "r", encoding="utf-8") as f:
        return f.read()

def reply_to_morph(reply: str, name: str, output_dir: str) -> ActiveMorph:
    # remove leading ```python and trailing trailing ```
    reply = re.sub(r'^```python\s*', '', reply, flags=re.MULTILINE)
    reply = re.sub(r'^```\s*', '', reply, flags=re.MULTILINE)
    morph = ActiveMorph(0, name)
    morph_filepath = os.path.join(output_dir, f"{name}.py")
    with open(morph_filepath, "w", encoding="utf-8") as f:
        f.write(reply)
    return morph

def mutate(config: MutateConfig, protomorph: ActiveMorph) -> ActiveMorph:
    log.info("🧫 mutating...")
    log.info(f"\t👵 ancestor ~{protomorph.name}~")
    mutation_prompt_filepath = os.path.join(config.prompts_dir, "mutate.txt")
    prompt = load_prompt(config, mutation_prompt_filepath)
    random.seed(config.seed)
    if random.random() < config.proc_glaze_prompt:
        log.info("\t\t🍯 adding glazing prompt...")
        glazing_prompt_filepath = os.path.join(config.prompts_dir, "glazing.txt")
        prompt += f"\n\n{load_prompt(config, glazing_prompt_filepath)}"
    # TODO: add spec based on backend
    if random.random() < config.proc_helpdocs_prompt:
        log.info("\t\t📋 adding documentation prompt...")
        # TODO: randomly choose and add one of the following pages of documentation
        # https://nvidia.github.io/warp/modules/differentiability.html
        # https://nvidia.github.io/warp/debugging.html
        # https://nvidia.github.io/warp/modules/contribution_guide.html
        # https://nvidia.github.io/warp/configuration.html
        # https://nvidia.github.io/warp/modules/interoperability.html
        # prompt += f"\n<helpful_docs>\n{load_prompt(config, challenge_prompt_filepath)}\n</helpful_docs>"
    morph_prompt = morph_to_prompt(config, protomorph)
    neomorph_name = str(uuid.uuid4())[:6]
    neomorph_output_dir = os.path.join(config.output_dir, neomorph_name)
    os.makedirs(neomorph_output_dir, exist_ok=True)
    neomorph_prompt_filepath = os.path.join(neomorph_output_dir, "mutate_prompt.txt")
    with open(neomorph_prompt_filepath, "w") as f:
        f.write(f"prompt:\n{prompt}\n\nmorph:\n{morph_prompt}")
    reply = inference(f"{prompt}\n\n<code>\n{morph_prompt}\n</code>", models=config.enabled_models)
    neomorph = reply_to_morph(reply, neomorph_name, config.morph_dir)
    log.info(f"\t🥚 welcome ~{neomorph_name}~")
    return neomorph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--morph", type=str, default=MutateConfig.morph)
    parser.add_argument("--models", type=str, help="Comma-separated list of models to use",
                       default=",".join(MutateConfig.enabled_models))
    args = parser.parse_args()

    config = MutateConfig(
        seed=args.seed,
        morph=args.morph,
        enabled_models=args.models.split(",") if args.models else MutateConfig.enabled_models,
    )
    mutate(config)

