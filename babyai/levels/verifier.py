import numpy as np
from enum import Enum
from gym_minigrid.minigrid import COLOR_NAMES, DIR_TO_VEC

# Object types we are allowed to describe in language
OBJ_TYPES = ['box', 'ball', 'key', 'door']

# Object types we are allowed to describe in language
OBJ_TYPES_NOT_DOOR = list(filter(lambda t: t is not 'door', OBJ_TYPES))

# Locations are all relative to the agent's starting position
LOC_NAMES = ['left', 'right', 'front', 'behind']


def dot_product(v1, v2):
    """
    Compute the dot product of the vectors v1 and v2.
    """

    return sum([i*j for i, j in zip(v1, v2)])


def pos_next_to(pos_a, pos_b):
    """
    Test if two positions are next to each other.
    The positions have to line up either horizontally or vertically,
    but positions that are diagonally adjacent are not counted.
    """

    xa, ya = pos_a
    xb, yb = pos_b
    d = abs(xa - xb) + abs(ya - yb)
    return d < 2


class ObjDesc:
    """
    Description of a set of objects in an environment
    """

    def __init__(self, type, color=None, loc=None):
        if type is 'locked_door':
            type = 'door'

        assert type in [None, *OBJ_TYPES], type
        assert color in [None, *COLOR_NAMES], color
        assert loc in [None, *LOC_NAMES], loc

        self.color = color
        self.type = type
        self.loc = loc

        # Set of objects possibly matching the description
        self.obj_set = []

        # Set of initial object positions
        self.obj_poss = []

    def surface(self, env):
        """
        Generate a natural language representation of the object description
        """

        self.find_matching_objs(env)
        assert len(self.obj_set) > 0, "no object matching description"

        if self.type:
            s = str(self.type)
        else:
            s = 'object'

        if self.color:
            s = self.color + ' ' + s

        if self.loc:
            if self.loc == 'front':
                s = s + ' in front of you'
            elif self.loc == 'behind':
                s = s + ' behind you'
            else:
                s = s + ' on your ' + self.loc

        # Singular vs plural
        if len(self.obj_set) > 1:
            s = 'a ' + s
        else:
            s = 'the ' + s

        return s

    def find_matching_objs(self, env):
        """
        Find the set of objects matching the description and their positions
        """

        self.obj_set = []
        self.obj_poss = []

        agent_room = env.room_from_pos(*env.start_pos)

        for i in range(env.grid.width):
            for j in range(env.grid.height):
                cell = env.grid.get(i, j)
                if cell == None:
                    continue

                if cell.type == "locked_door":
                    type = "door"
                else:
                    type = cell.type

                # Check if object's type matches description
                if self.type != None and type != self.type:
                    continue

                # Check if object's color matches description
                if self.color != None and cell.color != self.color:
                    continue

                # Check if object's position matches description
                if self.loc in ["left", "right", "front", "behind"]:
                    # Locations apply only to objects in the same room
                    # the agent starts in
                    if not agent_room.pos_inside(i, j):
                        continue

                    # Direction from the agent to the object
                    v = (i-env.start_pos[0], j-env.start_pos[1])

                    # (d1, d2) is an oriented orthonormal basis
                    d1 = DIR_TO_VEC[env.start_dir]
                    d2 = (-d1[1], d1[0])

                    # Check if object's position matches with location
                    pos_matches = {
                        "left": dot_product(v, d2) < 0,
                        "right": dot_product(v, d2) > 0,
                        "front": dot_product(v, d1) > 0,
                        "behind": dot_product(v, d1) < 0
                    }

                    if not(pos_matches[self.loc]):
                        continue

                self.obj_set.append(cell)
                self.obj_poss.append((i, j))

        return self.obj_set, self.obj_poss


class Instr:
    """
    Base class for all instructions in the baby language
    """

    def __init__(self):
        self.env = None

    def surface(self, env):
        """
        Produce a natural language representation of the instruction
        """

        raise NotImplementedError

    def reset_verifier(self, env):
        """
        Must be called at the beginning of the episode
        """

        self.env = env

    def verify(self, action):
        """
        Verify if the task described by the instruction is incomplete,
        complete with success or failed. The return value is a string,
        one of: 'success', 'failure' or 'continue'.
        """

        raise NotImplementedError


class ActionInstr(Instr):
    """
    Base class for all action instructions (clauses)
    """

    pass


class OpenInstr(ActionInstr):
    def __init__(self, obj_desc, strict=False):
        assert obj_desc.type is 'door'
        self.desc = obj_desc
        self.strict = strict

    def surface(self, env):
        return 'open ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify(self, action):
        for door in self.desc.obj_set:
            if door.is_open:
                return 'success'

        # If in strict mode and the wrong door is opened, failure
        if self.strict:
            if action == self.env.actions.toggle:
                front_cell = self.env.grid.get(*self.env.front_pos)
                if front_cell and front_cell.type is 'door':
                    return 'failure'

        return 'continue'


class GoToInstr(ActionInstr):
    """
    Go next to (and look towards) an object matching a given description
    eg: go to the door
    """

    def __init__(self, obj_desc):
        self.desc = obj_desc

    def surface(self, env):
        return 'go to ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify(self, action):
        # For each object position
        for pos in self.desc.obj_poss:
            # If the agent is next to (and facing) the object
            if np.array_equal(pos, self.env.front_pos):
                return 'success'

        return 'continue'


class PickupInstr(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, obj_desc, strict=False):
        assert obj_desc.type is not 'door'
        self.desc = obj_desc
        self.strict = strict

    def surface(self, env):
        return 'pick up ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify(self, action):
        for obj in self.desc.obj_set:
            if self.env.carrying is obj:
                return 'success'

        # If in strict mode and the wrong door object is picked up, failure
        if self.strict:
            if action == self.env.actions.pickup and self.env.carrying:
                return 'failure'

        return 'continue'


class PutNextInstr(ActionInstr):
    """
    Put an object next to another object
    eg: put the red ball next to the blue key
    """

    def __init__(self, obj_move, obj_fixed, strict=False):
        assert obj_move.type is not 'door'
        self.desc_move = obj_move
        self.desc_fixed = obj_fixed
        self.strict = strict

    def surface(self, env):
        return 'put ' + self.desc_move.surface(env) + ' next to ' + self.desc_fixed.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc_move.find_matching_objs(env)
        self.desc_fixed.find_matching_objs(env)

    def verify(self, action):
        for obj_a in self.desc_move.obj_set:
            pos_a = obj_a.cur_pos

            for pos_b in self.desc_fixed.obj_poss:
                if pos_next_to(pos_a, pos_b):
                    return 'success'

        # In strict mode, picking up the wrong object fails
        if self.strict:
            if action == self.env.actions.pickup and self.env.carrying:
                return 'failure'

        return 'continue'


class SeqInstr(Instr):
    """
    Base class for sequencing instructions (before, after, and)
    """

    def __init__(self, instr_a, instr_b, strict=False):
        assert isinstance(instr_a, ActionInstr) or isinstance(instr_a, AndInstr)
        assert isinstance(instr_b, ActionInstr) or isinstance(instr_b, AndInstr)
        self.instr_a = instr_a
        self.instr_b = instr_b
        self.strict = strict


class BeforeInstr(SeqInstr):
    """
    Sequence two instructions in order:
    eg: go to the red door then pick up the blue ball
    """

    def surface(self, env):
        return self.instr_a.surface(env) + ', then ' + self.instr_b.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.a_done is 'success':
            self.b_done = self.instr_b.verify(action)

            if self.b_done is 'failure':
                return 'failure'

            if self.b_done is 'success':
                return 'success'
        else:
            self.a_done = self.instr_a.verify(action)
            self.b_done = self.instr_b.verify(action)

            if self.a_done is 'failure':
                return 'failure'

            # In strict mode, completing b first means failure
            if self.strict and self.b_done is 'success':
                return 'failure'

        return 'continue'


class AfterInstr(SeqInstr):
    """
    Sequence two instructions in reverse order:
    eg: go to the red door after you pick up the blue ball
    """

    def surface(self, env):
        return self.instr_a.surface(env) + ' after you ' + self.instr_b.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.b_done is 'success':
            self.a_done = self.instr_a.verify(action)

            if self.a_done is 'success':
                return 'success'

            if self.a_done is 'failure':
                return 'failure'
        else:
            self.a_done = self.instr_a.verify(action)
            self.b_done = self.instr_b.verify(action)

            if self.b_done is 'failure':
                return 'failure'

            # In strict mode, completing a first means failure
            if self.strict and self.a_done is 'success':
                return 'failure'

        return 'continue'


class AndInstr(SeqInstr):
    """
    Conjunction of two actions, both can be completed in any other
    eg: go to the red door and pick up the blue ball
    """

    def __init__(self, instr_a, instr_b, strict=False):
        assert isinstance(instr_a, ActionInstr)
        assert isinstance(instr_b, ActionInstr)
        super().__init__(instr_a, instr_b, strict)

    def surface(self, env):
        return self.instr_a.surface(env) + ' and ' + self.instr_b.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.a_done is not 'success':
            self.a_done = self.instr_a.verify(action)

        if self.b_done is not 'success':
            self.b_done = self.instr_b.verify(action)

        if self.a_done is 'success' and self.b_done is 'success':
            return 'success'

        return 'continue'
