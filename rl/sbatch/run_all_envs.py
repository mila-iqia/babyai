import subprocess
import time

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
for seed in range(1, 6):
    for level, options in components.items():
        for use_cnn in [False, True]:
            subprocess.Popen("sbatch --account=rpp-bengioy --time=1:30:0 --ntasks=4 --gres=gpu:1 --mem=4G ./train_rl.sh python -m scripts.train_rl --env BabyAI-{}-v0 --algo ppo {} {} {} --tb --seed {} --save-interval 10".format(
                level,
                "--model-instr" if "instr" in options else "",
                "--model-mem" if "mem" in options else "",
                "--model-cnn" if use_cnn else "",
                seed
                             ), shell=True)
            time.sleep(1)