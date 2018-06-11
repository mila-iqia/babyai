import os
from subprocess import call
import sys

import babyai.utils as utils

folder = os.path.join(utils.storage_dir(), "demos")
for filename in sorted(os.listdir(folder)):
    if filename.endswith(".pkl"):
        env = filename.split("_")[0]
        demos_maker = filename.split("_")[-1].split('.')[0]
        print("> Env: {} - {}".format(env, demos_maker))
        command = ["python evaluate.py --env {} --demos-origin {}".format(env, demos_maker)] + sys.argv[1:]
        call(" ".join(command), shell=True)