import random
from copy import deepcopy

import gym

from .roomgrid import RoomGrid
from .instrs import *
from .instr_gen import gen_instr_seq, gen_object, gen_surface
from .verifier import InstrSeqVerifier

class Mission(gym.Wrapper):
    """
    Wrapper for missions, usable as a gym environment.
    """

    def __init__(self, seed, instrs, surface, env):
        self.seed = seed

        self.instrs = instrs

        self.surface = surface

        # Keep a copy of the original environment so we can reset it
        self.orig_env = env
        self.env = deepcopy(self.orig_env)

        self.actions = env.actions

        super().__init__(self.env)

        self.reset()

    def reset(self, **kwargs):
        # Reset the environment by making a copy of the original
        self.env = deepcopy(self.orig_env)

        # Recreate the verifier
        self.verifier = InstrSeqVerifier(self.env, self.instrs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # If we've successfully completed the mission
        if self.verifier.step() is True:
            done = True
            reward = 1

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
        """
        Generate a mission (instructions and matching environment)
        Derived level classes should implement this method
        """
        raise NotImplementedError

class Level0(Level):
    """
    Level 0: go to the red door
    (which in the current room)
    """

    def __init__(self):
        super().__init__()

    def gen_mission(self, seed):
        env = RoomGrid(room_size=7, num_cols=3, max_steps=50, seed=seed)
        obj, pos = env.add_door(1, 1, 3, 'red')

        instrs = [Instr(action="goto", object=Object(obj, pos))]
        surface = gen_surface(instrs, seed, lang_variation=1)

        return Mission(seed, instrs, surface, env)

class Level1(Level):
    """
    Level 1: go to the door
    (of a given color, in the current room)
    """

    def __init__(self):
        super().__init__()

    def gen_mission(self, seed):
        env = RoomGrid(room_size=7, num_cols=3, max_steps=50, seed=seed)
        door, pos = env.add_door(1, 1)
        env.connect_all()

        instrs = [Instr(action="goto", object=Object(door, pos))]
        surface = gen_surface(instrs, seed, lang_variation=1)

        return Mission(seed, instrs, surface, env)

class Level2(Level):
    """
    Level 2: go to an object or door
    (of a given type and color, in the current room)
    """

    def __init__(self):
        super().__init__()

    def gen_mission(self, seed):
        env = RoomGrid(room_size=7, num_cols=3, max_steps=50, seed=seed)
        if env._rand_bool():
            obj, pos = env.add_door(1, 1)
        else:
            obj, pos = env.add_object(1, 1)
        env.connect_all()

        instrs = [Instr(action="goto", object=Object(obj, pos))]
        surface = gen_surface(instrs, seed, lang_variation=2)

        return Mission(seed, instrs, surface, env)

class Level3(Level):
    """
    Level 3:
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self):
        super().__init__()

    def gen_mission(self, seed):
        env = RoomGrid(room_size=7, num_cols=3, max_steps=50, seed=seed)
        if env._rand_bool():
            obj, pos = env.add_door(1, 1)
        else:
            obj, pos = env.add_object(1, 1)
        env.connect_all()
        env.add_distractors()

        if obj.type == 'door':
            action = env._rand_elem(['goto', 'open'])
        else:
            action = env._rand_elem(['goto', 'pickup'])

        instrs = [Instr(action=action, object=Object(obj, pos))]
        surface = gen_surface(instrs, seed)

        return Mission(seed, instrs, surface, env)

class Level4(Level):
    """
    Level 4: fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, distractors=False):
        self.distractors = distractors
        super().__init__()

    def gen_mission(self, seed):
        env = RoomGrid(room_size=7, num_cols=3, max_steps=50, seed=seed)
        door, door_pos = env.add_door(1, 1, locked=True)
        key, key_pos = env.add_object(1, 1, 'key', door.color)
        env.connect_all()
        if self.distractors:
            env.add_distractors()

        instrs = [
            Instr(action="pickup", object=Object(key, key_pos)),
            Instr(action="open", object=Object(door, door_pos))
        ]
        surface = gen_surface(instrs, seed)

        return Mission(seed, instrs, surface, env)

class Level5(Level4):
    """
    Level 5: fetch a key and unlock a door
    (in the current room, with distractors)
    """

    def __init__(self):
        super().__init__(distractors=True)

class Level6(Level):
    """
    Level 6: pick up an object (in another room)
    """

    def __init__(self):
        super().__init__()

    def gen_mission(self, seed):
        env = RoomGrid(room_size=7, num_cols=3, max_steps=50, seed=seed)
        # Add a random object to the top-middle room
        obj, pos = env.add_object(1, 0)
        # Make sure the two rooms are directly connected
        env.add_door(1, 1, 3)
        env.connect_all()
        env.add_distractors()

        instrs = [Instr(action="pickup", object=Object(obj, pos))]
        surface = gen_surface(instrs, seed)

        return Mission(seed, instrs, surface, env)

class Level7(Level):
    """
    Level 7: unlock a door, then pick up an object in another room.
    """

    def __init__(self):
        super().__init__()

    def gen_mission(self, seed):
        env = RoomGrid(room_size=7, num_cols=3, max_steps=50, seed=seed)
        # Add a random object to the top-middle room
        obj, pos = env.add_object(1, 0)
        # Make sure the two rooms are directly connected by a locked door
        door, door_pos = env.add_door(1, 1, 3, locked=True)
        env.add_object(1, 1, 'key', door.color)
        env.connect_all()
        env.add_distractors()

        instrs = [
            Instr(action="open", object=Object(door, door_pos)),
            Instr(action="pickup", object=Object(obj, pos))
        ]
        surface = gen_surface(instrs, seed)

        return Mission(seed, instrs, surface, env)

# Level list, indexable by level number
# ie: level_list[0] is a Level0 instance
level_list = [
    Level0(),
    Level1(),
    Level2(),
    Level3(),
    Level4(),
    Level5(),
    Level6(),
    Level7()
]

def test():
    for idx, level in enumerate(level_list):
        print('Level %d' % idx)

        mission = level.gen_mission(0)
        assert isinstance(mission.surface, str)

        # Run the mission for a few episodes
        rng = random.Random(0)
        num_episodes = 0
        while num_episodes < 20:
            action = rng.randint(0, mission.action_space.n - 1)
            obs, reward, done, info = mission.step(action)
            if done:
                num_episodes += 1
                mission.reset()

        # The same seed should always yield the same mission
        m0 = level.gen_mission(0)
        m1 = level.gen_mission(0)
        grid1 = m0.unwrapped.grid
        grid2 = m1.unwrapped.grid
        assert grid1 == grid2
        assert m0.surface == m1.surface
