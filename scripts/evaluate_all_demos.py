'''
Script to evaluate all available demos. It assumes all demos (human and agent, except the "valid" ones)
are generated with seed 1
'''
import os
from subprocess import call
import sys

import babyai.utils as utils

folder = os.path.join(utils.storage_dir(), "demos")
for filename in sorted(os.listdir(folder)):
    if filename.endswith(".pkl"):
        env = filename.split("_")[0]
        demos = filename[:-4]  # Remove the .pkl part of the name
        if not env.startswith('BabyAI-') and not env.endswith('-v0'):
            print("> File: {} has to start with the level name (BabyAI-xxx-v0)".format(demos))
            continue
        print("> Env: {} - {}".format(env, demos))
        seed = 0 if demos.endswith('valid') else 1
        command = ["python evaluate.py --env {} --demos {} --seed {}".format(env, demos, seed)] + sys.argv[1:]
        call(" ".join(command), shell=True)