import random
from collections import OrderedDict
from copy import deepcopy
import gym
from .roomgrid import RoomGrid
from .verifier import *


class RejectSampling(Exception):
    """
    Exception used for rejection sampling
    """

    pass


class RoomGridLevel(RoomGrid):
    """
    Base for levels based on RoomGrid
    A level, given a random seed, generates missions generated from
    one or more patterns. Levels should produce a family of missions
    of approximately similar difficulty.
    """

    def __init__(
        self,
        room_size=8,
        max_steps=None,
        **kwargs
    ):
        # Default max steps computation
        if max_steps is None:
            max_steps = 4 * (room_size ** 2)

        super().__init__(
            room_size=room_size,
            max_steps=max_steps,
            **kwargs
        )

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.instrs.reset_verifier(self)

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If we've successfully completed the mission
        status = self.instrs.verify(action)

        if status is 'success':
            done = True
            reward = self._reward()
        elif status is 'failure':
            done = True
            reward = 0

        return obs, reward, done, info

    def _gen_grid(self, width, height):
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        while True:
            try:
                super()._gen_grid(width, height)

                # Generate the mission
                self.gen_mission()

            except RecursionError as error:
                print('Timeout during mission generation:', error)
                continue

            except RejectSampling as error:
                #print('Sampling rejected:', error)
                continue

            break

        # Generate the surface form for the instructions
        #seed = self._rand_int(0, 0xFFFFFFFF)
        self.surface = self.instrs.surface(self)
        self.mission = self.surface

    def gen_mission(self):
        """
        Generate a mission (instructions and matching environment)
        Derived level classes should implement this method
        """
        raise NotImplementedError

    @property
    def level_name(self):
        return self.__class__.level_name

    @property
    def gym_id(self):
        return self.__class__.gym_id

    def check_objs_reachable(self, raise_exc=True):
        """
        Check that all objects are reachable from the agent's starting
        position without requiring any other object to be moved
        (without unblocking)
        """

        # Reachable positions
        reachable = set()

        # Work list
        stack = [self.start_pos]

        while len(stack) > 0:
            i, j = stack.pop()

            if i < 0 or i >= self.grid.width or j < 0 or j >= self.grid.height:
                continue

            if (i, j) in reachable:
                continue

            # This position is reachable
            reachable.add((i, j))

            cell = self.grid.get(i, j)

            # If there is something other than a door in this cell, it
            # blocks reachability
            if cell and cell.type is not 'door':
                continue

            # Visit the horizontal and vertical neighbors
            stack.append((i+1, j))
            stack.append((i-1, j))
            stack.append((i, j+1))
            stack.append((i, j-1))

        # Check that all objects are reachable
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                cell = self.grid.get(i, j)

                if not cell or cell.type is 'wall':
                    continue

                if (i, j) not in reachable:
                    if not raise_exc:
                        return False
                    raise RejectSampling('unreachable object at ' + str((i, j)))

        # All objects reachable
        return True


class LevelGen(RoomGridLevel):
    """
    Level generator which attempts to produce every possible sentence in
    the baby language as an instruction.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        locked_room_prob=0.5,
        locations=True,
        unblocking=True,
        action_kinds=['goto', 'pickup', 'open', 'putnext'],
        instr_kinds=['action', 'and', 'seq'],
        debug=False,
        seed=None
    ):
        self.num_dists = num_dists
        self.locked_room_prob = locked_room_prob
        self.locations=locations
        self.unblocking=unblocking
        self.action_kinds = action_kinds
        self.instr_kinds = instr_kinds
        self.debug = debug

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        self.add_distractors(self.num_dists)
        self.place_agent()

        if self._rand_float(0, 1) < self.locked_room_prob:
            self.add_locked_room()

        self.connect_all()

        if not self.unblocking:
            self.check_objs_reachable()

        # Generate random instructions
        self.instrs = self.rand_instr(
            action_kinds=self.action_kinds,
            instr_kinds=self.instr_kinds
        )

    def add_locked_room(self):
        start_room = self.room_from_pos(*self.start_pos)

        # Until we've successfully added a locked room
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            door_idx = self._rand_int(0, 4)
            locked_room = self.get_room(i, j)

            # Don't lock the room the agent starts in
            if locked_room is start_room:
                continue

            # Don't add a locked door in an external wall
            if locked_room.neighbors[door_idx] is None:
                continue

            door, _ = self.add_door(
                i, j,
                door_idx,
                locked=True
            )

            # Done adding locked room
            break

        # Until we find a room to put the key
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            key_room = self.get_room(i, j)

            if key_room is locked_room:
                continue

            self.add_object(i, j, 'key', door.color)
            break

    def rand_obj(self, types=OBJ_TYPES, colors=COLOR_NAMES):
        """
        Generate a random object descriptor
        """

        # Keep trying until we find a matching object
        while True:
            color = self._rand_elem([None, *colors])
            type = self._rand_elem(types)

            loc = None
            if self.locations and self._rand_bool():
                loc = self._rand_elem(LOC_NAMES)

            desc = ObjDesc(type, color, loc)

            objs, poss = desc.find_matching_objs(self)

            if len(objs) > 0:
                break

        return desc

    def rand_instr(
        self,
        action_kinds,
        instr_kinds,
        depth=0
    ):
        """
        Generate random instructions
        """

        kind = self._rand_elem(instr_kinds)

        if kind is 'action':
            action = self._rand_elem(action_kinds)

            if action is 'goto':
                return GoToInstr(self.rand_obj())
            elif action is 'pickup':
                return PickupInstr(self.rand_obj(types=OBJ_TYPES_NOT_DOOR))
            elif action is 'open':
                return OpenInstr(self.rand_obj(types=['door']))
            elif action is 'putnext':
                return PutNextInstr(
                    self.rand_obj(types=OBJ_TYPES_NOT_DOOR),
                    self.rand_obj()
                )

            assert False

        elif kind is 'and':
            instr_a = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action'],
                depth=depth+1
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action'],
                depth=depth+1
            )
            return AndInstr(instr_a, instr_b)

        elif kind is 'seq':
            instr_a = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action', 'and'],
                depth=depth+1
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action', 'and'],
                depth=depth+1
            )

            kind = self._rand_elem(['before', 'after'])

            if kind is 'before':
                return BeforeInstr(instr_a, instr_b)
            elif kind is 'after':
                return AfterInstr(instr_a, instr_b)

            assert False

        assert False


# Dictionary of levels, indexed by name, lexically sorted
level_dict = OrderedDict()


def register_levels(module_name, globals):
    """
    Register OpenAI gym environments for all levels in a file
    """

    # Iterate through global names
    for global_name in sorted(list(globals.keys())):
        if not global_name.startswith('Level_'):
            continue

        level_name = global_name.split('Level_')[-1]
        level_class = globals[global_name]

        # Register the levels with OpenAI Gym
        gym_id = 'BabyAI-%s-v0' % (level_name)
        entry_point = '%s:%s' % (module_name, global_name)
        gym.envs.registration.register(
            id=gym_id,
            entry_point=entry_point,
        )

        # Add the level to the dictionary
        level_dict[level_name] = level_class

        # Store the name and gym id on the level class
        level_class.level_name = level_name
        level_class.gym_id = gym_id


def test():
    for idx, level_name in enumerate(level_dict.keys()):
        print('Level %s (%d/%d)' % (level_name, idx+1, len(level_dict)))

        level = level_dict[level_name]

        # Run the mission for a few episodes
        rng = random.Random(0)
        num_episodes = 0
        for i in range(0, 15):
            mission = level(seed=i)

            # Reduce max_steps because otherwise tests take too long
            mission.max_steps = 200

            # Check that the surface form was generated
            assert isinstance(mission.surface, str)
            assert len(mission.surface) > 0
            obs = mission.reset()
            assert obs['mission'] == mission.surface

            # Check for some known invalid patterns in the surface form
            import re
            surface = mission.surface
            assert not re.match(r".*pick up the [^ ]*door.*", surface), surface

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

    # Check that gym environment names were registered correctly
    gym.make('BabyAI-1RoomS8-v0')
    gym.make('BabyAI-BossLevel-v0')
