from gym.envs.registration import register

import gym_aigame.envs

register(
    id='AIGame-v0',
    entry_point='gym_aigame.envs:AIGameEnv',
    reward_threshold=1000.0
)
