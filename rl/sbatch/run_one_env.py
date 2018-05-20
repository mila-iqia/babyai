import subprocess
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, default="UnlockPickup",
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--tune_lr", action="store_true", default=False,
                    help="tune lr")
parser.add_argument("--use_gpu", action="store_true", default=False,
                    help="use gpu")
args = parser.parse_args()

if args.tune_lr:
    lrs = [1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3]
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
    "LockedRoom": ["mem"],
    "BlockedUnlockPickup": []
}

level = args.env
options = components[level]
for seed in range(1, 6):
    for use_cnn in [False, True]:
        for lr in lrs:
            subprocess.Popen("sbatch --account={}-bengioy --time=0:20:0 --ntasks=4\
                              {} {} ./train_rl.sh python -m scripts.train_rl\
                              --env BabyAI-{}-v0 --algo ppo {} {} {} --tb --seed {} \
                              --save-interval 10 --lr {}\
                             ".format('ref' if args.use_gpu else 'def',
                                      'gres=gpu:1' if args.use_gpu else '',
                                      '--mem=4G' if args.use_gpu else '--mem-per-cpu=4G',
                                      level,
                                      "--model-instr" if "instr" in options else "",
                                      "--model-mem" if "mem" in options else "",
                                      "--model-cnn" if use_cnn else "",
                                      seed,
                                      lr
                                     ),
                             shell=True)
            time.sleep(1)