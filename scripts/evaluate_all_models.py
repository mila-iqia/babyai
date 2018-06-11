import os
from subprocess import call
import sys
import argparse

import babyai.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes of evaluation (default: 1000)")
args = parser.parse_args()

folder = os.path.join(utils.storage_dir(), "models")
for model in sorted(os.listdir(folder)):
    if model.startswith('.'):
        continue
    env = model.split("_")[0]
    print("> Env: {}".format(env))
    command = ["python evaluate.py --env {} --model {} --episodes {}".format(env, model, args.episodes)] + sys.argv[1:]
    call(" ".join(command), shell=True)