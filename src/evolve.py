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

import nbformat
from openai import OpenAI

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
    "rewrite_model",
    "rewrite_data",
    "rewrite_training",
    "tune_config",
]

# mutation prompt modifiers (proc chance in 0 to 1)
PROC_GLAZE_PROMPT: float = 0.5
PROC_ARXIV_PROMPT: float = 0.8
PROC_ARCDOC_PROMPT: float = 0.1

# agent is used for mutations
DEFAULT_AGENT = "gpt-4o"
DEFAULT_MORPHS = ",".join([
    "cnn.o1",
    "mlp.o1",
    "rnn.sonnet.o1",
    "2eab46",
    "3ec84c",
    "6dd30c",
    "7aa353",
    "7117f0",
    "6825e2",
])

# morph states
NOT_RUN_YET = 0
ALREADY_RAN = -1
ERRORED_OUT = -2

@dataclass(order=True)
class Morph:
    score: float
    name: str
    state: int = NOT_RUN_YET

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

def random_arxiv_abstract(num_terms: int = 2) -> str:
    import arxiv
    query_filepath = os.path.join(PROMPT_DIR, "arxiv_query.txt")
    with open(query_filepath, "r") as f:
        terms = f.read().strip().split(',')
    query = " AND ".join(random.sample(terms, num_terms))
    try:
        search_results = arxiv.Search(
            query=query,
            max_results=12,  # Adjust the number if necessary
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        paper = random.choice(list(search_results.results()))
        return f"This paper from arxiv contains a helpful hint:\n\n{paper.title}\n\n{paper.summary}"
    except Exception:
        return "The Bitter Lesson"

def load_prompt(prompt_path):
    prompt_filepath = os.path.join(PROMPT_DIR, prompt_path)
    with open(prompt_filepath, "r") as f:
        return f.read()

def morph_to_prompt(morph: Morph) -> str:
    morph_nb_filepath = os.path.join(MORPH_DIR, f"{morph.name}.ipynb")
    with open(morph_nb_filepath, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)

def reply_to_morph(reply: str, name:str, output_dir: str) -> Morph:
    # remove leading ```python and trailing trailing ```
    reply = re.sub(r'^```python\s*', '', reply, flags=re.MULTILINE)
    reply = re.sub(r'^```\s*', '', reply, flags=re.MULTILINE)
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source=reply))
    morph = Morph(0, name)
    nb_filepath = os.path.join(output_dir, f"{name}.ipynb")
    with open(nb_filepath, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return morph

def run_agent(system: str, prompt: str, agent: str = DEFAULT_AGENT):
    print(f"\t🧠 calling {agent}...")
    if agent in ["gpt-4o"]: # TODO
        client = OpenAI()
        completion = client.chat.completions.create(
            model=agent,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
        reply = completion.choices[0].message.content
    elif agent in ["sonnet3.5"]:
        pass # TODO
    else:
        raise ValueError(f"Unknown agent: {agent}")
    print("\t... completed")
    return reply

def export(morph: Morph):
    export_prompt_filepath = os.path.join(PROMPT_DIR, "export.txt")
    export_prompt = load_prompt(export_prompt_filepath)
    prompt = morph_to_prompt(morph)
    reply = run_agent(export_prompt, prompt)
    export_dir = os.path.join(OUTPUT_DIR, morph.name)
    os.makedirs(export_dir, exist_ok=True)
    reply_to_morph(reply, f"export.{morph.name}", export_dir)


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
    if random.random() < PROC_ARXIV_PROMPT:
        print("\t\t📚 adding arxiv prompt...")
        system += f"\n\n{random_arxiv_abstract()}"
    if random.random() < PROC_ARCDOC_PROMPT:
        print("\t\t📋 adding challenge prompt...")
        challenge_prompt_filepath = os.path.join(PROMPT_DIR, "challenge.txt")
        system += f"\n\n{load_prompt(challenge_prompt_filepath)}"
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
        if os.path.exists(os.path.join(MORPH_DIR, f"{protomorph}.ipynb")):
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
            if morph.state == ALREADY_RAN:
                print(f"\t⏩\tSkipping {morph.name} with score {morph.score}")
                continue
            elif morph.state == ERRORED_OUT:
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
                    morph.state = ERRORED_OUT
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
            morph.state = ALREADY_RAN
            # export(morph)
        
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
