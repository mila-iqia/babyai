import os
from subprocess import call
import sys

import utils

models_folder = os.path.join(utils.storage_dir(), "models")
for model in os.listdir(models_folder):
    env = "BabyAI-{}-v0".format(model.split("_")[0])
    print("> Env: {}".format(env))
    call(["python -m scripts.evaluate --env {} --model {}".format(env, model)] + sys.argv[1:], shell=True)