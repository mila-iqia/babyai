import random
from copy import deepcopy

import gym

from .roomgrid import RoomGrid
from .instrs import *
from .instr_gen import gen_instr_seq, gen_object, gen_surface
from .verifier import InstrSeqVerifier

class RoomGridLevel(RoomGrid):
    """
    Base for levels based on RoomGrid
    A level, given a random seed, generates missions generated from
    one or more patterns. Levels should produce a family of missions
    of approximately similar difficulty.
    """

    def __init__(self, lang_variation=1, **kwargs):
        self.lang_variation = lang_variation
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.verifier = InstrSeqVerifier(self, self.instrs)

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If we've successfully completed the mission
        if self.verifier.step() is True:
            done = True
            reward = 1

        return obs, reward, done, info

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Generate the mission
        self.gen_mission()

        # Make sure all rooms are reachable
        self.connect_all()

        # Generate the surface form for the instructions
        seed = self._rand_int(0, 0xFFFFFFFF)
        self.surface = gen_surface(self.instrs, seed, self.lang_variation)
        self.mission = self.surface

    def gen_mission(self):
        """
        Generate a mission (instructions and matching environment)
        Derived level classes should implement this method
        """
        raise NotImplementedError

class Level0(RoomGridLevel):
    """
    Level 0: go to the red door
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=2,
            max_steps=50,
            lang_variation=1,
            seed=seed
        )

    def gen_mission(self):
        obj, pos = self.add_door(0, 0, 0, 'red', locked=False)
        self.place_agent(0, 0)
        self.instrs = [Instr(action="goto", object=Object(obj, pos))]

class Level1(RoomGridLevel):
    """
    Level 1: go to the door
    (of a given color, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            max_steps=50,
            lang_variation=1,
            seed=seed
        )

    def gen_mission(self):
        door, pos = self.add_door(1, 1)
        self.place_agent(1, 1)
        self.instrs = [Instr(action="goto", object=Object(door, pos))]

class Level2(RoomGridLevel):
    """
    Level 2: go to an object or door
    (of a given type and color, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            max_steps=50,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        if self._rand_bool():
            obj, pos = self.add_door(1, 1)
        else:
            obj, pos = self.add_object(1, 1)
        self.place_agent(1, 1)

        self.instrs = [Instr(action="goto", object=Object(obj, pos))]

class Level3(RoomGridLevel):
    """
    Level 3:
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            max_steps=50,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        if self._rand_bool():
            obj, pos = self.add_door(1, 1)
        else:
            obj, pos = self.add_object(1, 1)
        self.place_agent(1, 1)

        if obj.type == 'door':
            action = self._rand_elem(['goto', 'open'])
        else:
            action = self._rand_elem(['goto', 'pickup'])

        self.instrs = [Instr(action=action, object=Object(obj, pos))]

class Level4(RoomGridLevel):
    """
    Level 4: fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors
        super().__init__(
            max_steps=50,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        door, door_pos = self.add_door(1, 1, locked=True)
        key, key_pos = self.add_object(1, 1, 'key', door.color)
        if self.distractors:
            self.add_distractors()
        self.place_agent(1, 1)

        self.instrs = [
            Instr(action="pickup", object=Object(key, key_pos)),
            Instr(action="open", object=Object(door, door_pos))
        ]

class Level5(Level4):
    """
    Level 5: fetch a key and unlock a door
    (in the current room, with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)

class Level6(RoomGridLevel):
    """
    Level 6: pick up an object (in the room above)
    """

    def __init__(self, seed=None):
        super().__init__(
            max_steps=50,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the top-middle room
        obj, pos = self.add_object(1, 0)
        # Make sure the two rooms are directly connected
        self.add_door(1, 1, 3, locked=False)
        self.add_distractors()
        self.place_agent(1, 1)

        self.instrs = [Instr(action="pickup", object=Object(obj, pos))]

class Level7(RoomGridLevel):
    """
    Level 7: pick up an object (in a random room)
    This level requires potentially exhaustive exploration
    """

    def __init__(self, room_size=5, max_steps=120, seed=None):
        super().__init__(
            room_size=room_size,
            max_steps=max_steps,
            lang_variation=1,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to a random room
        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)
        obj, pos = self.add_object(i, j)
        self.place_agent(1, 1)

        self.instrs = [Instr(action="pickup", object=Object(obj, pos))]

class Level8(Level7):
    """
    Level 8: the same as level 7, but with larger rooms
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            max_steps=200,
            seed=seed
    )

class Level9(RoomGridLevel):
    """
    Level 9: unlock a door, then pick up an object in another room.
    """

    def __init__(self, seed=None):
        super().__init__(
            max_steps=75,
            lang_variation=1,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the top-middle room
        obj, pos = self.add_object(1, 0)
        # Make sure the two rooms are directly connected by a locked door
        door, door_pos = self.add_door(1, 1, 3, locked=True)
        self.add_object(1, 1, 'key', door.color)
        self.add_distractors()
        self.place_agent(1, 1)

        self.instrs = [
            Instr(action="open", object=Object(door, door_pos)),
            Instr(action="pickup", object=Object(obj, pos))
        ]

# Level list, indexable by level number
# ie: level_list[0] is a Level0 instance
level_list = [
    Level0,
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
    Level6,
    Level7,
    Level8,
    Level9
]

def test():
    for idx, level in enumerate(level_list):
        print('Level %d' % idx)

        # Run the mission for a few episodes
        rng = random.Random(0)
        num_episodes = 0
        for i in range(0, 20):
            mission = level(seed=i)
            assert isinstance(mission.surface, str)

            obs = mission.reset()
            assert obs['mission'] == mission.surface

            while True:
                action = rng.randint(0, mission.action_space.n - 1)
                obs, reward, done, info = mission.step(action)
                if done:
                    obs = mission.reset()
                    break

            num_episodes += 1

        # The same seed should always yield the same mission
        m0 = level(seed=0)
        m1 = level(seed=0)
        grid1 = m0.unwrapped.grid
        grid2 = m1.unwrapped.grid
        assert grid1 == grid2
        assert m0.surface == m1.surface
