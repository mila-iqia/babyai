from abc import ABC, abstractmethod
from collections import namedtuple

State = namedtuple("State", ["dir", "pos", "carry"])

def dot_product(v1, v2):
    """
    Compute the dot product of the vectors v1 and v2.
    """

    return sum([i*j for i, j in zip(v1, v2)])

class Verifier(ABC):
    def __init__(self, env):
        self.env = env
        self.startDirVec = env.get_dir_vec()

    @abstractmethod
    def step(self):
        """
        Update verifier's internal state and returns true
        iff the agent did what he was expected to.
        """

        return

    def _obj_desc_to_poss(self, obj_desc):
        """
        Get the positions of all the objects that match the description.
        """

        poss = []

        for i in range(self.env.grid.width):
            for j in range(self.env.grid.height):
                cell = self.env.grid.get(i, j)
                if cell == None:
                    continue

                if cell.type == "locked_door":
                    type = "door"
                    state = "locked"
                else:
                    type = cell.type
                    state = None

                # Check if object's type matches description                
                if obj_desc.type != None and type != obj_desc.type:
                    continue

                # Check if object's state matches description
                if obj_desc.state != None and state != obj_desc.state:
                    continue

                # Check if object's color matches description
                if obj_desc.color != None and cell.color != obj_desc.color:
                    continue

                # Check if object's position matches description
                if obj_desc.loc in ["left", "right", "front", "behind"]:
                    # Direction from the agent to the object
                    v = (i-self.env.start_pos[0], j-self.env.start_pos[1])

                    # (d1, d2) is an oriented orthonormal basis
                    d1 = self.startDirVec
                    d2 = (-d1[1], d1[0])

                    # Check if object's position matches with location
                    pos_matches = {
                        "left": dot_product(v, d2) < 0,
                        "right": dot_product(v, d2) > 0,
                        "front": dot_product(v, d1) > 0,
                        "behind": dot_product(v, d1) < 0
                    }

                    if not(pos_matches[obj_desc.loc]):
                        continue

                poss.append((i, j))

        return poss

    def _get_in_front_of_pos(self):
        """
        Get the position in front of the agent.
        The agent's state is the 2-tuple (agent_dir, agent_pos).
        """

        pos = self.env.agent_pos
        d = self.env.get_dir_vec()
        pos = (pos[0] + d[0], pos[1] + d[1])

        return pos

class InstrSeqVerifier(Verifier):
    def __init__(self, env, instr):
        super().__init__(env)

        self.instr = instr
        self.instr_index = 0

        self.obj_to_drop = None
        self.intermediary_state = None

        self._load_next_verifier()

    def step(self):
        if self.verifier != None and self.verifier.step():
            self._close_verifier()
            self._load_next_verifier()
        return self.verifier == None

    def _load_next_verifier(self):
        if self.instr_index >= len(self.instr):
            return

        instr = self.instr[self.instr_index]
        self.instr_index += 1

        if instr.action == "open":
            self.verifier = OpenVerifier(self.env, instr.object)
        elif instr.action == "goto":
            self.verifier = GotoVerifier(self.env, instr.object)
        elif instr.action == "pickup":
            self.verifier = PickupVerifier(self.env, instr.object)
        elif instr.action == "drop":
            self.verifier = DropVerifier(self.env, self.obj_to_drop)

        self.verifier.state = self.intermediary_state

    def _close_verifier(self):
        if isinstance(self.verifier, PickupVerifier):
            self.obj_to_drop = self.verifier.state.carry

        self.intermediary_state = self.verifier.state

        self.verifier = None

class InstrVerifier(Verifier):
    def __init__(self, env):
        super().__init__(env)

        self.previous_state = None
        self.state = None

    def step(self):
        """
        Update verifier's internal state and returns true
        iff the agent did what he was expected to.
        """

        self.previous_state = self.state
        self.state = State(
            dir=self.env.agent_dir,
            pos=self.env.agent_pos,
            carry=self.env.carrying
        )

        return self._done()

    @abstractmethod
    def _done(self):
        """
        Check if the agent did what he was expected to.
        """

        return

class GotoVerifier(InstrVerifier):
    def __init__(self, env, obj):
        super().__init__(env)

        self.obj_poss = self._obj_desc_to_poss(obj)
        self.obj_cells = [self.env.grid.get(*pos) for pos in self.obj_poss]

    def _done(self):
        on_cell = self.env.grid.get(*self.state.pos)
        ifo_pos = self._get_in_front_of_pos()
        ifo_cell = self.env.grid.get(*ifo_pos)

        check_on_goal = on_cell != None and on_cell.type == "goal"
        check_goto_goal = check_on_goal and on_cell in self.obj_cells

        check_not_ifo_goal = ifo_cell == None or ifo_cell.type != "goal"
        check_goto_not_goal = check_not_ifo_goal and ifo_cell in self.obj_cells

        return check_goto_goal or check_goto_not_goal

class PickupVerifier(InstrVerifier):
    def __init__(self, env, obj):
        super().__init__(env)

        self.obj_poss = self._obj_desc_to_poss(obj)
        self.obj_cells = [self.env.grid.get(*pos) for pos in self.obj_poss]

    def _done(self):
        check_wasnt_carrying = self.previous_state == None or self.previous_state.carry == None
        check_carrying = self.state.carry in self.obj_cells

        return check_wasnt_carrying and check_carrying

class OpenVerifier(InstrVerifier):
    def __init__(self, env, obj):
        super().__init__(env)

        self.obj_poss = self._obj_desc_to_poss(obj)
        self.obj_cells = [self.env.grid.get(*pos) for pos in self.obj_poss]

    def _done(self):
        ifo_pos = self._get_in_front_of_pos()
        ifo_cell = self.env.grid.get(*ifo_pos)

        check_opened = ifo_cell in self.obj_cells and ifo_cell.is_open

        return check_opened

class DropVerifier(InstrVerifier):
    def __init__(self, env, obj_to_drop):
        super().__init__(env)

        self.obj_to_drop = obj_to_drop

    def _done(self):
        check_was_carrying = self.previous_state != None and self.previous_state.carry == self.obj_to_drop
        check_isnt_carrying = self.state.carry == None

        return check_was_carrying and check_isnt_carrying