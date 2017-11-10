import os
import numpy
import gym

from gym.spaces.box import Box

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        #env.seed(seed)

        return env

    return _thunk


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)
