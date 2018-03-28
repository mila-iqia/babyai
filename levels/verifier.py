from abc import ABC, abstractmethod
from collections import namedtuple

State = namedtuple("State", ["dir", "pos", "carry"])

def get_in_front_of_pos(env):
    """
    Get the position in front of the agent.
    The agent's state is the 2-tuple (agentDir, agentPos).
    """

    pos = env.agentPos
    u, v = env.getDirVec()
    pos = (pos[0] + u, pos[1] + v)

    return pos

def obj_desc_to_poss(env, obj_desc):
    """
    Get the positions of all the objects that match object's description.
    """

    poss = []

    for i in range(env.grid.width):
        for j in range(env.grid.height):
            cell = env.grid.get(i, j)
            if cell == None:
                continue

            if obj_desc.type == "door" and obj_desc.state == "locked":
                type = "locked_door"
            else:
                type = obj_desc.type

            if cell.type != type:
                continue

            if obj_desc.state in ["closed", "locked"] and cell.isOpen:
                continue

            if obj_desc.color != None and cell.color != obj_desc.color:
                continue

            # TODO: handle positions

            poss.append((i, j))

    return poss

class Verifier(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def step(self):
        """
        Update verifier's internal state and returns true
        iff the agent did what he was expected to.
        """

        return

class InstrSeqVerifier(Verifier):
    def __init__(self, env, instr):
        super().__init__(env)
        self.instr = instr
        self.ainstrIndex = 0

        self.obj_to_drop = None
        self.intermediary_state = None

        self._load_next_verifier()

    def step(self):
        if self.verifier != None and self.verifier.step():
            self._close_verifier()
            self._load_next_verifier()
        return self.verifier == None

    def _load_next_verifier(self):
        if self.ainstrIndex >= len(self.instr):
            return

        ainstr = self.instr[self.ainstrIndex]
        self.ainstrIndex += 1

        if ainstr.action == "open":
            self.verifier = OpenVerifier(self.env, ainstr.object)
        elif ainstr.action == "goto":
            self.verifier = GotoVerifier(self.env, ainstr.object)
        elif ainstr.action == "pickup":
            self.verifier = PickupVerifier(self.env, ainstr.object)
        else:
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
            dir=self.env.agentDir,
            pos=self.env.agentPos,
            carry=self.env.carrying)

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
        self.obj_poss = obj_desc_to_poss(env, obj)
        self.obj_cells = [self.env.grid.get(*pos) for pos in self.obj_poss]

    def _done(self):
        on_cell = self.env.grid.get(*self.state.pos)
        ifo_pos = get_in_front_of_pos(self.env)
        ifo_cell = self.env.grid.get(*ifo_pos)

        check_on_goal = on_cell != None and on_cell.type == "goal"
        check_goto_goal = check_on_goal and on_cell in self.obj_cells

        check_not_ifo_goal = ifo_cell == None or ifo_cell.type != "goal"
        check_goto_not_goal = check_not_ifo_goal and ifo_cell in self.obj_cells

        return check_goto_goal or check_goto_not_goal

class PickupVerifier(InstrVerifier):
    def __init__(self, env, obj):
        super().__init__(env)
        self.obj_poss = obj_desc_to_poss(env, obj)
        self.obj_cells = [self.env.grid.get(*pos) for pos in self.obj_poss]

    def _done(self):
        check_wasnt_carrying = self.previous_state.carry == None
        check_carrying = self.state.carry in self.obj_cells

        return check_wasnt_carrying and check_carrying

class OpenVerifier(InstrVerifier):
    def __init__(self, env, obj):
        super().__init__(env)
        if obj.state == None:
            obj = obj._replace(state="closed")
        self.obj_poss = obj_desc_to_poss(env, obj)
        self.obj_cells = [self.env.grid.get(*pos) for pos in self.obj_poss]

    def _done(self):
        ifo_pos = get_in_front_of_pos(self.env)
        ifo_cell = self.env.grid.get(*ifo_pos)

        check_opened = ifo_cell in self.obj_cells and ifo_cell.isOpen

        return check_opened

class DropVerifier(InstrVerifier):
    def __init__(self, env, obj_to_drop):
        super().__init__(env)
        self.obj_to_drop = obj_to_drop

    def _done(self):
        check_was_carrying = self.previous_state.carry == self.obj_to_drop
        check_isnt_carrying = self.state.carry == None

        return check_was_carrying and check_isnt_carrying
