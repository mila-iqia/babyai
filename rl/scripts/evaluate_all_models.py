import os
from subprocess import call
import sys

import utils

folder = os.path.join(utils.storage_dir(), "models")
for model in sorted(os.listdir(folder)):
    env = "BabyAI-{}-v0".format(model.split("_")[0])
    print("> Env: {}".format(env))
    command = ["python -m scripts.evaluate --env {} --model {}".format(env, model)] + sys.argv[1:]
    call(" ".join(command), shell=True)