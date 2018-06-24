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
    "UnlockPickupDist": [
        "BabyAI-Unlock-v0",
        "BabyAI-UnlockPickup-v0",
        "BabyAI-UnlockPickupDist-v0"
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