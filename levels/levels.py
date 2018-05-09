import random
from collections import OrderedDict
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

        # Generate the surface form for the instructions
        seed = self._rand_int(0, 0xFFFFFFFF)
        self.surface = gen_surface(self.instrs, seed, lang_variation=self.lang_variation)
        self.mission = self.surface

    def gen_mission(self):
        """
        Generate a mission (instructions and matching environment)
        Derived level classes should implement this method
        """
        raise NotImplementedError

class Level_GoToRedDoor(RoomGridLevel):
    """
    Go to the red door
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        room_size = 6

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=4*room_size**2,
            lang_variation=1,
            seed=seed
        )

    def gen_mission(self):
        obj, pos = self.add_door(0, 0, 0, 'red', locked=False)
        self.place_agent(0, 0)
        self.instrs = [Instr(action="goto", object=Object(obj.type, obj.color))]

class Level_GoToDoor(RoomGridLevel):
    """
    Go to the door
    (of a given color, always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        room_size = 6

        super().__init__(
            room_size=room_size,
            max_steps=4*room_size**2,
            lang_variation=1,
            seed=seed
        )

    def gen_mission(self):
        colors = COLOR_NAMES[:]
        objs = []

        for i in range(4):
            color = self._rand_elem(colors)
            colors.remove(color)
            obj, _ = self.add_door(1, 1, door_idx=i, color=color, locked=False)
            objs.append(obj)

        self.place_agent(1, 1)
        self.instrs = [Instr(action="goto", object=Object(objs[0].type, objs[0].color))]

class Level_GoToObjDoor(RoomGridLevel):
    """
    Go to an object or door
    (of a given type and color, in the current room)
    """

    def __init__(self, seed=None):
        room_size = 6

        super().__init__(
            room_size=room_size,
            max_steps=4*room_size**2,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        if self._rand_bool():
            obj, pos = self.add_door(1, 1)
        else:
            obj, pos = self.add_object(1, 1)
        self.add_distractors(2)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = [Instr(action="goto", object=Object(obj.type, obj.color))]

class Level_LocalAction(RoomGridLevel):
    """
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self, seed=None):
        room_size = 6

        super().__init__(
            room_size=room_size,
            max_steps=4*room_size**2,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        if self._rand_bool():
            obj, pos = self.add_door(1, 1, locked=False)
            action = self._rand_elem(['goto', 'open'])
        else:
            obj, pos = self.add_object(1, 1)
            action = self._rand_elem(['goto', 'pickup'])
        self.place_agent(1, 1)

        self.instrs = [Instr(action=action, object=Object(obj.type, obj.color))]

class Level_UnlockDoor(RoomGridLevel):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors

        super().__init__(
            max_steps=50,
            lang_variation=1,
            seed=seed
        )

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        self.add_object(1, 1, 'key', door.color)
        if self.distractors:
            self.add_distractors(2)
        self.place_agent(1, 1)

        self.instrs = [
            Instr(action="open", object=Object(door.type, door.color))
        ]

class Level_UnlockDoorDist(Level_UnlockDoor):
    """
    Fetch a key and unlock a door
    (in the current room, with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)

class Level_PickupAbove(RoomGridLevel):
    """
    Pick up an object (in the room above)
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
        self.connect_all()

        self.instrs = [Instr(action="pickup", object=Object(obj.type, obj.color))]

class Level_OpenRedBlueDoors(RoomGridLevel):
    """
    Open red door, then open blue door
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self, seed=None):
        room_size = 6

        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            lang_variation=1,
            seed=seed
        )

    def gen_mission(self):
        door1, _ = self.add_door(1, 1, 0, color="blue", locked=False)
        door2, _ = self.add_door(1, 1, 2, color="red", locked=False)

        self.place_agent(1, 1)
        self.start_dir = 0

        self.instrs = [
            Instr(action="open", object=Object(door2.type, door2.color)),
            Instr(action="open", object=Object(door1.type, door1.color))
        ]

class Level_OpenTwoDoors(RoomGridLevel):
    """
    Open door X, then open door Y
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self, seed=None):
        room_size = 6

        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        colors = COLOR_NAMES[:]
        objs = []

        for i in range(4):
            color = self._rand_elem(colors)
            colors.remove(color)
            obj, _ = self.add_door(1, 1, door_idx=i, color=color, locked=False)
            objs.append(obj)

        self.place_agent(1, 1)

        self.instrs = [
            Instr(action="open", object=Object(objs[0].type, objs[0].color)),
            Instr(action="open", object=Object(objs[2].type, objs[2].color))
        ]

class Level_FindObj(RoomGridLevel):
    """
    Pick up an object (in a random room)
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
        self.connect_all()

        self.instrs = [Instr(action="pickup", object=Object(obj.type, obj.color))]

class Level_FindObjLarge(Level_FindObj):
    """
    Same as the previous level, but with larger rooms
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            max_steps=200,
            seed=seed
    )

class Level_UnlockPickup(RoomGridLevel):
    """
    Unlock a door, then pick up an object in another room.
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
        self.connect_all()

        self.instrs = [
            Instr(action="open", object=Object(door.type, door.color)),
            Instr(action="pickup", object=Object(obj.type, obj.color))
        ]

class Level_FourObjects(RoomGridLevel):
    """
    Four identical objects in four different rooms. The task is
    to pick up the correct one, distinguished by its location.
    """

    def __init__(self, seed=None):
        super().__init__(
            max_steps=75,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_object(1, 0)
        self.add_object(1, 2, obj.type, obj.color)
        self.add_object(0, 1, obj.type, obj.color)
        self.add_object(2, 1, obj.type, obj.color)

        self.add_distractors()

        # Make sure the start room is directly connected to the
        # four adjacent rooms
        for i in range(0, 4):
            _, _ = self.add_door(1, 1, i, locked=False)

        # The agent starts facing right
        self.place_agent(1, 1, rand_dir=False)

        self.connect_all()

        # Choose a random object to pick up
        loc = self._rand_elem(['left', 'right', 'front', 'behind'])
        rand_obj = Object(obj.type, obj.color, loc)
        self.instrs = [Instr(action="pickup", object=rand_obj)]

class Level_LockedRoom(RoomGridLevel):
    """
    An object is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(self, num_rows=3, room_size=6, seed=None):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=500,
            lang_variation=3,
            seed=seed,
        )

    def gen_mission(self):
        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, 3)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, 3), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, 1)

        # Make sure all rooms are accessible
        self.connect_all()

        self.instrs = [
            Instr(action="open", object=Object(door.type, door.color)),
            Instr(action="pickup", object=Object(obj.type, obj.color))
        ]

# Dictionary of levels, indexed by name, lexically sorted
level_dict = OrderedDict()

# Iterate through global names
for global_name in sorted(list(globals().keys())):
    if not global_name.startswith('Level_'):
        continue

    module_name = __name__
    level_name = global_name.split('Level_')[-1]

    # Add the level to the dictionary
    level_dict[level_name] = globals()[global_name]

    # Register the levels with OpenAI Gym
    level_id = 'BabyAI-%s-v0' % (level_name)
    entry_point = '%s:%s' % (module_name, global_name)
    #print(level_id)
    #print(entry_point)
    gym.envs.registration.register(
        id=level_id,
        entry_point=entry_point,
    )

def test():
    for idx, level_name in enumerate(level_dict.keys()):
        print('Level %s (%d/%d)' % (level_name, idx, len(level_dict)))

        level = level_dict[level_name]

        # Run the mission for a few episodes
        rng = random.Random(0)
        num_episodes = 0
        for i in range(0, 20):
            mission = level(seed=i)
            assert isinstance(mission.surface, str)
            assert len(mission.surface) > 0

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
