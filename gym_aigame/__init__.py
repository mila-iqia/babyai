from gym.envs.registration import register

register(
    id='AI-Game-v0',
    entry_point='gym_aigame.envs:AIGameEnv',
    reward_threshold=1000.0,
)
