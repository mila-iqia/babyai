from copy import deepcopy
import gym

from .instrs import *
from .instr_gen import gen_instr_seq, gen_surface
from .env_gen import gen_env
from .verifier import InstrSeqVerifier

# - want way to iterate through levels
# - generate environments and instructions given seeds
# - probably want way to sample mission objects for a given level

# TODO: more commenting

class Mission(gym.Wrapper):
    """
    Wrapper for missions, usable as a gym environment.
    """

    def __init__(self, instrs, env):
        self.instrs = instrs

        # Keep a copy of the original env
        self.orig_env = env
        self.env = deepcopy(self.orig_env)

        super().__init__(self.env)

        self.reset()

    def reset(self, **kwargs):
        # Reset the environment by making a copy of the original
        self.env = deepcopy(self.orig_env)

        self.verifier = InstrSeqVerifier(self.env, self.instrs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        done = self.verifier.step()
        reward = 1 if done else 0

        return obs, reward, done, info

class Level:
    """
    Base class for all levels.
    A level, given a random seed, generates missions generated from
    one or more patterns. Levels should produce a family of missions
    of approximately similar difficulty.
    """

    def __init__(self):
        pass

    def gen_mission(self, seed):
        raise NotImplementedError

class Level0(Level):
    """
    Level 0: go to the red door (which in the current room)
    """

    def __init__(self):
        super().__init__()

    def gen_mission(self, seed):
        instrs = [Instr(action="goto", object=Object(type="door", color="red", loc=None, state=None))]
        env = gen_env(instrs, seed)
        return Mission(instrs, env)

# Levels array, indexable by level number
# ie: levels[0] is a Level0 instance
levels = [
    Level0()
]

def test():
    mission = levels[0].gen_mission(0)
    mission.step(0)
    mission.reset()
    mission.step(0)
