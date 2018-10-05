"""
Evaluate all models in a storage directory.

In order to use this script make sure to add baby-ai-game/scripts to the $PATH
environment variable.

Sample usage:
evaluate_all_models.py --episodes 200 --argmax
"""

import os
from subprocess import call
import sys

import babyai.utils as utils
from babyai.levels import level_dict
import re

# List of all levels ordered by length of the level name from longest to shortest
LEVELS = sorted(list(level_dict.keys()), key=len)[::-1]


def get_levels_from_model_name(model):
    levels = []
    # Assume that our model names are separated with _ or -
    model_name_parts = re.split('_|-', model)
    for part in model_name_parts:
        # Assume that each part contains at most one level name.
        # Sorting LEVELS using length of level name is to avoid scenarios like
        # extracting 'GoTo' from the model name 'GoToLocal-model'
        for level in LEVELS:
            if level in part:
                levels.append('BabyAI-{}-v0'.format(level))
                break
    return list(set(levels))


folder = os.path.join(utils.storage_dir(), "models")

for model in sorted(os.listdir(folder)):
    if model.startswith('.'):
        continue
    envs = get_levels_from_model_name(model)
    print("> Envs: {} > Model: {}".format(envs, model))
    for env in envs:
        command = ["evaluate.py --env {} --model {}".format(env, model)] + sys.argv[1:]
        print("Command: {}".format(" ".join(command)))
        call(" ".join(command), shell=True)
