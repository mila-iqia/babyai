#!/usr/bin/env python3

"""
Script to train the experts.
"""

import argparse
import time
import subprocess

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--tags", default="all",
                    help="comma-separated tags for selecting the environments to train the experts on")
parser.add_argument("--no-slurm", action="store_true", default=False,
                    help="don't use slurm")
args = parser.parse_args()

# Define parameters for training the agents
params = [
    {"tags": ["1RoomS8", "all"],
     "time": "1:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "1RoomS8",
     "arguments": "--env BabyAI-1RoomS8-v0 --no-instr"},
    {"tags": ["1Room-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "1RoomS12,S16,S20",
     "arguments": "--curriculum 1Room --no-instr"},
    {"tags": ["ActionObjDoor", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "ActionObjDoor",
     "arguments": "--env BabyAI-ActionObjDoor-v0"},
    {"tags": ["BlockedUnlockPickup-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "BlockedUnlockPickup",
     "arguments": "--curriculum BlockedUnlockPickup --no-instr"},
    {"tags": ["FindObjS5", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "FindObjS5",
     "arguments": "--env BabyAI-FindObjS5-v0 --no-instr"},
    {"tags": ["FindObj-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "FindObjS6,S7",
     "arguments": "--curriculum FindObj --no-instr"},
    {"tags": ["FourObjsS5-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "FourObjsS5",
     "arguments": "--env BabyAI-FourObjsS5-v0"},
    {"tags": ["FourObjs-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "FourObjsS6,S7",
     "arguments": "--curriculum FourObjs"},
    {"tags": ["GoToObjDoor", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "GoToObjDoor",
     "arguments": "--env BabyAI-GoToObjDoor-v0"},
    {"tags": ["KeyCorridor-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "KeyCorridorS6R3",
     "arguments": "--curriculum KeyCorridor --no-instr"},
    {"tags": ["OpenDoorDebug-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "OpenDoor,Color,Loc",
     "arguments": "--curriculum OpenDoorDebug"},
    {"tags": ["OpenRedBlueDoorsDebug", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "OpenRedBlueDoors",
     "arguments": "--env BabyAI-OpenRedBlueDoorsDebug-v0 --no-instr"},
    {"tags": ["OpenTwoDoorsDebug-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "OpenTwoDoors",
     "arguments": "--curriculum OpenTwoDoorsDebug"},
    {"tags": ["OpenRedDoor", "all"],
     "time": "1:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "OpenRedDoor",
     "arguments": "--env BabyAI-OpenRedDoor-v0 --no-instr"},
    {"tags": ["PickupDistDebug", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "PickupDistDebug",
     "arguments": "--env BabyAI-PickupDistDebug-v0"},
    {"tags": ["Unlock", "all"],
     "time": "2:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "Unlock",
     "arguments": "--env BabyAI-Unlock-v0 --no-instr"},
    {"tags": ["UnlockDist", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "UnlockDist",
     "arguments": "--env BabyAI-UnlockDist-v0 --no-instr"},
    {"tags": ["UnlockPickup", "all"],
     "time": "2:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "UnlockPickup",
     "arguments": "--env BabyAI-UnlockPickup-v0 --no-instr"},
    {"tags": ["UnlockPickupDist-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "UnlockPickupDist",
     "arguments": "--curriculum UnlockPickupDist"},
    {"tags": ["UnlockToUnlock-cur", "all"],
     "time": "12:0:0",
     "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     "model": "UnlockToUnlock",
     "arguments": "--curriculum UnlockToUnlock --no-instr"},
]

# Filter the params given tags
def have_something_in_common(l1, l2):
    return bool(set(l1).intersection(l2))

# Select based on tags passed on the command line
tags = args.tags.split(",")
params = [
    paramSet for paramSet in params
    if have_something_in_common(paramSet["tags"], tags)
]

# Assemble the shell commands to launch all jobs
shell_cmds = []

# Execute the filtered params
for paramSet in params:
    # Compute Canada recommends requesting memory in megabytes
    # https://docs.computecanada.ca/wiki/Running_jobs
    slurm_cmd = "sbatch --account=def-bengioy --time={} --ntasks=1 --mem=6000M".format(paramSet["time"])
    for seed in paramSet["seeds"]:
        model = "baselines/{}/seed{}".format(paramSet["model"], seed)

        shell_cmd = "{} scripts/run_slurm.sh python -m scripts.train_rl {} --frames 50000000 --algo ppo --model {} --seed {} --save-interval 10".format(
            slurm_cmd if not args.no_slurm else "",
            paramSet["arguments"],
            model,
            seed
        )
        shell_cmds.append(shell_cmd)

# Launch all the jobs one by one
# Note: we avoid launch these all at once because this can overload the scheduler
for idx, shell_cmd in enumerate(shell_cmds):

    print('Launching job {}/{}'.format(idx+1, len(shell_cmds)))
    print(shell_cmd)

    subprocess.check_call(shell_cmd, shell=True)
    time.sleep(1)
