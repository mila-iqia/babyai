import gym
import babyai.multienv as menv

curriculums = {
    "KeyCorridor": [
        "BabyAI-KeyCorridorS3R1-v0",
        "BabyAI-KeyCorridorS3R2-v0",
        "BabyAI-KeyCorridorS3R3-v0",
        "BabyAI-KeyCorridorS4R3-v0",
        "BabyAI-KeyCorridorS5R3-v0",
        "BabyAI-KeyCorridorS6R3-v0"
    ],
    "BlockedUnlockPickup": [
        "BabyAI-Unlock-v0",
        "BabyAI-UnlockPickup-v0",
        "BabyAI-BlockedUnlockPickup-v0"
    ],
    "FindObj": [
        "BabyAI-FindObjS5-v0",
        "BabyAI-FindObjS6-v0",
        "BabyAI-FindObjS7-v0"
    ],
    "FourObjs": [
        "BabyAI-FourObjsS5-v0",
        "BabyAI-FourObjsS6-v0",
        "BabyAI-FourObjsS7-v0"
    ],
    "UnlockPickupDist": [
        "BabyAI-Unlock-v0",
        "BabyAI-UnlockPickup-v0",
        "BabyAI-UnlockPickupDist-v0"
    ],

    "OpenDoor": [
        "BabyAI-OpenDoorColor-v0",
        "BabyAI-OpenTwoDoors-v0",
        "BabyAI-OpenDoorsOrder-v0"
    ],
    "1Room": [
        "BabyAI-1RoomS8-v0",
        "BabyAI-1RoomS12-v0",
        "BabyAI-1RoomS16-v0",
        "BabyAI-1RoomS20-v0"
    ],
    "OpenDoorDebug": [
        "BabyAI-OpenDoorColorDebug-v0",
        "BabyAI-OpenDoorLocDebug-v0",
        "BabyAI-OpenDoorDebug-v0"
    ],
    "OpenTwoDoorsDebug": [
        "BabyAI-OpenRedBlueDoorsDebug-v0",
        "BabyAI-OpenTwoDoorsDebug-v0"
    ],
    "UnlockToUnlock": [
        "BabyAI-Unlock-v0",
        "BabyAI-UnlockPickup-v0",
        "BabyAI-UnlockToUnlock-v0"
    ],
    "PutNext": [
        "BabyAI-PutNextS6N5-v0",
        "BabyAI-PutNextS7N5-v0",
        "BabyAI-PutNextS8N6-v0"
    ],
    "DoorObj": [
        "BabyAI-OpenDoorColor-v0",
        "BabyAI-GoToObjDoor-v0",
        "BabyAI-ActionObjDoor-v0"
    ]
}

def create_menvs(curriculum, num_procs, seed):
    """Creates X multi-environments with 1 multi-environment head where X
    is given by `num_procs`. Each multi-environment can simulate all the
    environments in the curriculum given."""

    # Define constants explained in https://arxiv.org/abs/1707.00183
    alpha = 0.2
    K = 20
    eps = 0.2

    # Compute the number of environments
    num_envs = len(curriculum)

    # Instantiate the return history for each environment
    return_hists = [menv.ReturnHistory() for _ in range(num_envs)]

    # Instantiate the learning progress computer
    compute_lp = menv.WindowLpComputer(return_hists, alpha, K)

    # Instantiate the distribution creator
    create_dist = menv.GreedyPropDistCreator(eps)

    # Instantiate the distribution computer
    compute_dist = menv.LpDistComputer(return_hists, compute_lp, create_dist)

    # Instantiate the head of the multi-environments
    menv_head = menv.MultiEnvHead(num_procs, num_envs, compute_dist)

    # Instantiate all the multi-environments
    menvs = []
    for i in range(num_procs):
        envs = []
        for j in range(num_envs):
            env = gym.make(curriculum[j])
            env.seed(seed + j)
            envs.append(env)
        menvs.append(menv.MultiEnv(envs, menv_head.remotes[i], seed))
    return menv_head, menvs
