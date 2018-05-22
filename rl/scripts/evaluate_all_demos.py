import os
from subprocess import call
import sys

import utils

folder = os.path.join(utils.storage_dir(), "demos")
for filename in sorted(os.listdir(folder)):
    if filename.endswith(".pkl"):
        env = filename.split("_")[0]
        print("> Env: {}".format(env))
        command = ["python -m scripts.evaluate --env {}".format(env)] + sys.argv[1:]
        call(" ".join(command), shell=True)