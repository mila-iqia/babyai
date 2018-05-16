import subprocess
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, default="UnlockPickup",
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--tune_lr", action="store_true", default=False,
                    help="use ConvNet in the model")
args = parser.parse_args()

if args.tune_lr:
    lrs = [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 5e-3, 1e-2, 5e-2]
else:
    lrs = [7e-4]



components = {
    "GoToRedDoor": [],
    "GoToDoor": ["instr"],
    "GoToObjDoor": ["instr"],
    "LocalAction": ["instr"],
    "UnlockDoor": [],
    "UnlockDoorDist": [],
    # "PickupAbove": [TODO],
    "OpenRedBlueDoors": ["mem"],
    "OpenTwoDoors": ["instr", "mem"],
    "FindObj": ["mem"],
    "FindObjLarge": ["mem"],
    "UnlockPickup": ["instr"],
    "FourObjects":  ["instr", "mem"],
    "LockedRoom": ["mem"]
}

level = args.env
options = components[level]
for seed in range(1, 6):
    for use_cnn in [False, True]:
        for lr in lrs:
            subprocess.Popen("sbatch --account=rpp-bengioy --time=3:0:0 --ntasks=4\
                              --gres=gpu:1 --mem=4G ./train_rl.sh python -m scripts.train_rl\
                              --env BabyAI-{}-v0 --algo ppo {} {} {} --tb --seed {} \
                              --save-interval 10 --lr {}\
                             ".format(level,
                                      "--model-instr" if "instr" in options else "",
                                      "--model-mem" if "mem" in options else "",
                                      "--model-cnn" if use_cnn else "",
                                      seed,
                                      lr
                                     ),
                             shell=True)
            time.sleep(1)