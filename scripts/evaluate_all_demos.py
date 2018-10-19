"""
Script to evaluate all available demos.

Assumes all demos (human and agent, except the "valid" ones)
are generated with seed 1
"""

import os
from subprocess import call
import sys

import babyai.utils as utils

folder = os.path.join(utils.storage_dir(), "demos")
for filename in sorted(os.listdir(folder)):
    if filename.endswith(".pkl") and 'valid' in filename:
        env = 'BabyAI-BossLevel-v0'  # It doesn't really matter. The evaluation only considers the lengths of demos.
        demo = filename[:-4]  # Remove the .pkl part of the name

        print("> Demos: {}".format(demo))

        command = ["python evaluate.py --env {} --demos {} --worst-episodes-to-show 0".format(env, demo)] + sys.argv[1:]
        call(" ".join(command), shell=True)
