import random
from copy import deepcopy

import gym
import gym_minigrid
from gym_minigrid.minigrid import COLOR_NAMES

from .instrs import *
from .instr_gen import gen_instr_seq, gen_surface
from .env_gen import gen_env
from .verifier import InstrSeqVerifier

class Mission(gym.Wrapper):
    """
    Wrapper for missions, usable as a gym environment.
    """

    def __init__(self, seed, instrs, env):
        self.seed = seed

        self.instrs = instrs

        # Keep a copy of the original environment so we can reset it
        self.orig_env = env
        self.env = deepcopy(self.orig_env)

        self.actions = env.actions

        super().__init__(self.env)

        self.reset()

    def reset(self, **kwargs):
        # Reset the environment by making a copy of the original
        self.env = deepcopy(self.orig_env)

        self.verifier = InstrSeqVerifier(self.env, self.instrs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Check if the mission has been completed
        done = self.verifier.step()
        reward = 1 if done else 0

        return obs, reward, done, info

    @property
    def surface(self):
        """Produce an English string for the instructions"""
        return gen_surface(self.instrs, seed=self.seed)

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
        """Generate a mission (instructions and matching environment)"""

        # Create a random-number generator using the seed
        rng = random.Random(seed)

        return self._gen_mission(seed, rng)

    def _gen_mission(self, seed, rng):
        """Derived level classes should implement this method"""
        raise NotImplementedError

class Level0(Level):
    """
    Level 0: go to the red door (which in the current room)
    """

    def __init__(self):
        super().__init__()

    def _gen_mission(self, seed, rng):
        instrs = [Instr(action="goto", object=Object(type="door", color="red", loc=None, state=None))]
        env = gen_env(instrs, seed, max_steps=50)
        return Mission(seed, instrs, env)

class Level1(Level):
    """
    Level 1: go to the door (of a given color, in the current room)
    """

    def __init__(self):
        super().__init__()

    def _gen_mission(self, seed, rng):
        color = rng.choice(COLOR_NAMES)
        instrs = [Instr(action="goto", object=Object(type="door", color=color, loc=None, state=None))]
        env = gen_env(instrs, seed, max_steps=50)
        return Mission(seed, instrs, env)

class Level2(Level):
    """
    Level 2: go to an object or door (of a given type and color, in the current room)
    """

    def __init__(self):
        super().__init__()

    def _gen_mission(self, seed, rng):
        color = rng.choice(COLOR_NAMES)
        type = rng.choice(['door', 'ball', 'key', 'box'])
        instrs = [Instr(action="goto", object=Object(type=type, color=color, loc=None, state=None))]
        env = gen_env(instrs, seed, max_steps=50, distractors=True)
        return Mission(seed, instrs, env)

class Level3(Level):
    """
    Level 3: [pick up an object] or [go to an object or door] (in the current room)
    """

    def __init__(self):
        super().__init__()

    def _gen_mission(self, seed, rng):
        color = rng.choice(COLOR_NAMES)

        if rng.choice([True, False]):
            type = rng.choice(['ball', 'key', 'box'])
            instr = Instr(action="pickup", object=Object(type=type, color=color, loc=None, state=None))
        else:
            type = rng.choice(['door', 'ball', 'key', 'box'])
            instr = Instr(action="goto", object=Object(type=type, color=color, loc=None, state=None))

        env = gen_env(
            [instr],
            seed,
            max_steps=50,
            distractors=True
        )

        return Mission(seed, [instr], env)

# Level list, indexable by level number
# ie: level_list[0] is a Level0 instance
level_list = [
    Level0(),
    Level1(),
    Level2(),
    Level3()
]

def test():
    for idx, level in enumerate(level_list):
        print('Level %d' % idx)
        mission = level.gen_mission(0)
        mission.step(0)
        mission.reset()
        mission.step(0)

        assert isinstance(mission.surface, str)

        # The same seed should always yield the same mission
        m0 = level.gen_mission(0)
        m1 = level.gen_mission(0)
        grid1 = m0.unwrapped.grid
        grid2 = m1.unwrapped.grid
        assert grid1 == grid2
        assert m0.surface == m1.surface
