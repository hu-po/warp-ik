import argparse

from evolve import export, Morph

parser = argparse.ArgumentParser()
parser.add_argument("--morph", help="morph name", default="6825e2")
args = parser.parse_args()

export(Morph(0, args.morph))