import random
from collections import OrderedDict
from copy import deepcopy

import gym

from .roomgrid import RoomGrid, Ball
from .instrs import *
from .instr_gen import gen_instr_seq, gen_object, gen_surface
from .verifier import InstrSeqVerifier, OpenVerifier, PickupVerifier

class RoomGridLevel(RoomGrid):
    """
    Base for levels based on RoomGrid
    A level, given a random seed, generates missions generated from
    one or more patterns. Levels should produce a family of missions
    of approximately similar difficulty.
    """

    def __init__(
        self,
        lang_variation=1,
        room_size=6,
        max_steps=None,
        **kwargs
    ):
        # Default max steps computation
        if max_steps is None:
            max_steps = 4 * (room_size ** 2)

        self.lang_variation = lang_variation
        super().__init__(
            room_size=room_size,
            max_steps=max_steps,
            **kwargs
        )

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
            reward = self._reward()

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

class Level_OpenRedDoor(RoomGridLevel):
    """
    Go to the red door
    (always unlocked, in the current room)
    Note: this level is intentionally meant for debugging and is
    intentionally kept very simple.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=5,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_door(0, 0, 0, 'red', locked=False)
        self.place_agent(0, 0)
        self.instrs = [Instr(action="open", object=Object(obj.type, obj.color))]

class Level_OpenDoor(RoomGridLevel):
    """
    Go to the door
    The door to open is given by its color or by its location.
    (always unlocked, in the current room)
    """

    def __init__(self, select_by=None, seed=None):
        self.select_by = select_by

        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        door_colors = self._rand_subset(COLOR_NAMES, 4)
        objs = []

        for i, color in enumerate(door_colors):
            obj, _ = self.add_door(1, 1, door_idx=i, color=color, locked=False)
            objs.append(obj)
        
        select_by = self.select_by
        if select_by is None:
            select_by = self._rand_elem(["color", "loc"])
        if select_by == "color":
            object = Object(objs[0].type, color=objs[0].color)
        elif select_by == "loc":
            object = Object(objs[0].type, loc=self._rand_elem(LOC_NAMES))

        self.place_agent(1, 1)
        self.instrs = [Instr(action="open", object=object)]
    
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.verifier = InstrSeqVerifier(self, self.instrs)
        # Recreate the open verifier
        self.open_verifier = OpenVerifier(self, Object("door"))

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If we've successfully completed the mission
        if self.verifier.step() is True:
            done = True
            reward = self._reward()
        # If we've opened the wrong door
        elif self.open_verifier.step() is True:
            done = True

        return obs, reward, done, info

class Level_OpenDoorColor(Level_OpenDoor):
    """
    Go to the door
    The door is selected by color.
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            select_by="color",
            seed=seed
        )

class Level_OpenDoorLoc(Level_OpenDoor):
    """
    Go to the door
    The door is selected by location.
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            select_by="loc",
            seed=seed
        )

class Level_GoToObjDoor(RoomGridLevel):
    """
    Go to an object or door
    (of a given type and color, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(num_distractors=5, room_i=1, room_j=1)
        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append((door.type, door.color))
        self.place_agent(1, 1)

        type, color = self._rand_elem(objs)
        self.instrs = [Instr(action="goto", object=Object(type, color))]

class Level_ActionObjDoor(RoomGridLevel):
    """
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(num_distractors=5, room_i=1, room_j=1)
        for _ in range(4):
            door, _ = self.add_door(1, 1, locked=False)
            objs.append((door.type, door.color))

        self.place_agent(1, 1)

        type, color = self._rand_elem(objs)
        if type == door.type:
            action = self._rand_elem(['goto', 'open'])
        else:
            action = self._rand_elem(['goto', 'pickup'])
        self.instrs = [Instr(action=action, object=Object(type, color))]

class Level_Unlock(RoomGridLevel):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors

        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        self.add_object(1, 1, 'key', door.color)
        if self.distractors:
            self.add_distractors(num_distractors=3, room_i=1, room_j=1)
        self.place_agent(1, 1)

        self.instrs = [Instr(action="open", object=Object(door.type))]

class Level_UnlockDist(Level_Unlock):
    """
    Fetch a key and unlock a door
    (in the current room, with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)

class Level_UnlockPickup(RoomGridLevel):
    """
    Unlock a door, then pick up a box in another room
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors

        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)
        if self.distractors:
            self.add_distractors(num_distractors=4)

        self.place_agent(0, 0)

        self.instrs = [Instr(action="pickup", object=Object(obj.type, obj.color))]

class Level_UnlockPickupDist(Level_UnlockPickup):
    """
    Unlock a door, then pick up an object in another room
    (with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)

class Level_BlockedUnlockPickup(RoomGridLevel):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0]-1, pos[1], Ball(color))
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)

        self.place_agent(0, 0)

        self.instrs = [Instr(action="pickup", object=Object(obj.type))]

class Level_UnlockToUnlock(RoomGridLevel):
    """
    Unlock a door A that requires to unlock a door B before
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=30*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        # Add a door of color A connecting left and middle room
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=True)

        # Add a key of color A in the room on the right
        self.add_object(2, 0, kind="key", color=colors[0])

        # Add a door of color B connecting middle and right room
        self.add_door(1, 0, door_idx=0, color=colors[1], locked=True)

        # Add a key of color B in the middle room
        self.add_object(1, 0, kind="key", color=colors[1])

        obj, _ = self.add_object(0, 0, kind="ball")

        self.place_agent(1, 0)

        self.instrs = [Instr(action="pickup", object=Object(obj.type))]

class Level_PickupDist(RoomGridLevel):
    """
    Pick up an object
    The object to pick up is given by its type only, or
    by its color, or by its type and color.
    (in the current room, with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows = 1,
            num_cols = 1,
            room_size=7,
            lang_variation=2,
            seed=seed
        )
    
    def gen_mission(self):
        # Add 5 random objects in the room
        objs = self.add_distractors(5)
        self.place_agent(0, 0)
        type, color = self._rand_elem(objs)

        select_by = self._rand_elem(["type", "color", "both"])
        if select_by == "color":
            type = None
        elif select_by == "type":
            color = None

        self.instrs = [Instr(action="pickup", object=Object(type, color))]

class Level_PickupDistDebug(Level_PickupDist):
    """
    Same as PickupDist but the level stops when any object is picked
    """

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.verifier = InstrSeqVerifier(self, self.instrs)
        # Recreate the pickup verifier
        self.pickup_verifier = PickupVerifier(self, Object())

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If we've successfully completed the mission
        if self.verifier.step() is True:
            done = True
            reward = self._reward()
        # If we've picked up the wrong object
        elif self.pickup_verifier.step() is True:
            done = True

        return obs, reward, done, info

class Level_PickupAbove(RoomGridLevel):
    """
    Pick up an object (in the room above)
    This task requires to use the compass to be solved effectively.
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=8*room_size**2,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the top-middle room
        obj, pos = self.add_object(1, 0)
        # Make sure the two rooms are directly connected
        self.add_door(1, 1, 3, locked=False)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = [Instr(action="pickup", object=Object(obj.type, obj.color))]

class Level_OpenTwoDoors(RoomGridLevel):
    """
    Open door X, then open door Y
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self, first_color=None, second_color=None, seed=None):
        self.first_color = first_color
        self.second_color = second_color

        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        first_color = self.first_color
        if first_color is None:
            first_color = colors[0]
        second_color = self.second_color
        if second_color is None:
            second_color = colors[1]

        door1, _ = self.add_door(1, 1, 2, color=first_color, locked=False)
        door2, _ = self.add_door(1, 1, 0, color=second_color, locked=False)

        self.place_agent(1, 1)

        self.instrs = [
            Instr(action="open", object=Object(door1.type, door1.color)),
            Instr(action="open", object=Object(door2.type, door2.color))
        ]

class Level_OpenTwoDoorsDebug(Level_OpenTwoDoors):
    """
    Same as OpenTwoDoors but the level stops when the second door is opened
    """

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.verifier = InstrSeqVerifier(self, self.instrs)
        # Recreate the open second verifier
        second_color = self.instrs[1].object.color
        self.open_second_verifier = OpenVerifier(self, Object("door", second_color))

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If we've successfully completed the mission
        if self.verifier.step() is True:
            done = True
            reward = self._reward()
        # If we've opened the wrong door
        elif self.open_second_verifier.step() is True:
            done = True

        return obs, reward, done, info

class Level_OpenRedBlueDoors(Level_OpenTwoDoors):
    """
    Open red door, then open blue door
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self, seed=None):
        super().__init__(
            first_color="red",
            second_color="blue",
            seed=seed
        )

class Level_OpenRedBlueDoorsDebug(Level_OpenTwoDoorsDebug):
    """
    Same as OpenRedBlueDoors but the level stops when the blue door is opened
    """

    def __init__(self, seed=None):
        super().__init__(
            first_color="red",
            second_color="blue",
            seed=seed
        )

class Level_FindObjS5(RoomGridLevel):
    """
    Pick up an object (in a random room)
    Rooms have a size of 5
    This level requires potentially exhaustive exploration
    """

    def __init__(self, room_size=5, seed=None):
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to a random room
        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(i, j)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = [Instr(action="pickup", object=Object(obj.type))]

class Level_FindObjS6(Level_FindObjS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 6
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            seed=seed
        )

class Level_FindObjS7(Level_FindObjS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 7
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )

class Level_FourObjsS5(RoomGridLevel):
    """
    Four identical objects in four different rooms. The task is
    to pick up the correct one.
    The object to pick up is given by its location.
    Rooms have a size of 5.
    """

    def __init__(self, room_size=5, seed=None):
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            lang_variation=2,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_object(1, 0)
        self.add_object(1, 2, obj.type, obj.color)
        self.add_object(0, 1, obj.type, obj.color)
        self.add_object(2, 1, obj.type, obj.color)

        # Make sure the start room is directly connected to the
        # four adjacent rooms
        for i in range(0, 4):
            _, _ = self.add_door(1, 1, i, locked=False)

        self.place_agent(1, 1)

        # Choose a random object to pick up
        loc = self._rand_elem(LOC_NAMES)
        rand_obj = Object(obj.type, obj.color, loc)
        self.instrs = [Instr(action="pickup", object=rand_obj)]

class Level_FourObjsS6(Level_FourObjsS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 6
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            seed=seed
        )

class Level_FourObjsS7(Level_FourObjsS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 7
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )

class Level_HiddenKeyCorridor(RoomGridLevel):
    """
    A ball is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(self, num_rows=3, obj_type="ball", room_size=6, seed=None):
        self.obj_type = obj_type

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=30*room_size**2,
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
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, 3), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, 1)

        # Make sure all rooms are accessible
        self.connect_all()

        self.instrs = [Instr(action="pickup", object=Object(obj.type))]

class Level_1RoomS8(RoomGridLevel):
    """
    Pick up the ball
    Rooms have a size of 8
    """

    def __init__(self, room_size=8, seed=None):
        super().__init__(
            room_size=room_size,
            num_rows=1,
            num_cols=1,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_object(0, 0, kind="ball")
        self.place_agent()

        self.instrs = [Instr(action="pickup", object=Object(obj.type))]

class Level_1RoomS12(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 12
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=12,
            seed=seed
        )

class Level_1RoomS16(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 16
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=16,
            seed=seed
        )

class Level_1RoomS20(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 20
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=20,
            seed=seed
        )

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
