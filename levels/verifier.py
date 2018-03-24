from abc import ABC, abstractmethod

def get_dir_vec(dir):
    """
    Get the direction vector for the agent, pointing in the direction
    of forward movement.
    """

    # Pointing right
    if dir == 0:
        return (1, 0)
    # Down (positive Y)
    elif dir == 1:
        return (0, 1)
    # Pointing left
    elif dir == 2:
        return (-1, 0)
    # Up (negative Y)
    elif dir == 3:
        return (0, -1)
    else:
        assert False

def get_next_state(env, action, state=None):
    """
    Get agent's state (i.e. direction and position) after `action`.
    The agent's state is the 2-tuple (agentDir, agentPos).
    """

    if state == None:
        state = env.agentDir, env.agentPos
    dir, pos = state

    if action == env.actions.left:
        dir = (dir - 1) % 4
    elif action == env.actions.right:
        dir = (dir + 1) % 4
    elif action == env.actions.forward:
        u, v = get_dir_vec(dir)
        newPos = (pos[0] + u, pos[1] + v)
        targetCell = env.grid.get(newPos[0], newPos[1])
        if targetCell == None or targetCell.canOverlap():
            pos = newPos

    return dir, pos

def get_front_pos(env, state=None):
    """
    Get the position in front of the agent.
    The agent's state is the 2-tuple (agentDir, agentPos).
    """

    if state == None:
        state = env.agentDir, env.agentPos
    dir, pos = state

    u, v = get_dir_vec(dir)
    pos = (pos[0] + u, pos[1] + v)

    return pos

def is_in_front_of(env, pos, state=None):
    """
    Check if the position in front of the agent is `pos`.
    The agent's state is the 2-tuple (agentDir, agentPos).
    """

    return get_front_pos(env, state) == pos

def is_empty_in_front_of(env, state=None):
    """
    Check if agent is in front of an empty cell.
    The agent's state is the 2-tuple (agentDir, agentPos).
    """

    return env.grid.get(*get_front_pos(env, state)) == None

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
            
            if obj_desc.color != None and cell.color != obj_desc.color:
                continue

            # TODO: handle positions

            poss.append((i, j))
    
    return poss

class Verifier(ABC):
    def __init__(self, env):
        self.env = env
        self.actions = self.env.unwrapped.actions
    
    @abstractmethod
    def isSucceedAction(self, action):
        """
        Check if the agent will solve the mission with this action.
        """
        return

class InstrVerifier(Verifier):
    def __init__(self, env, instr):
        super().__init__(env)
        self.instr = instr
        self.ainstrIndex = 0
        self._loadNextVerifier()
    
    def _loadNextVerifier(self):
        if self.ainstrIndex >= len(self.instr):
            self.verifier = None
            return
        
        ainstr = self.instr[self.ainstrIndex]
        self.ainstrIndex += 1

        if ainstr.action == "open":
            self.verifier = OpenVerifier(self.env, ainstr.object)
        elif ainstr.action == "goto":
            self.verifier = GotoVerifier(self.env, ainstr.object)
        elif ainstr.action == "pick":
            self.verifier = PickVerifier(self.env, ainstr.object)
        else:
            self.verifier = DropVerifier(self.env)

    def isSucceedAction(self, action):
        if self.verifier != None and self.verifier.isSucceedAction(action):
            self._loadNextVerifier()
        return self.verifier == None

class AOVerifier(Verifier, ABC):
    """
    Verifier for action-object atomic instructions.
    """

    def __init__(self, env, obj):
        super().__init__(env)
        self.obj_poss = obj_desc_to_poss(env, obj)
    
    @abstractmethod
    def isSucceedActionObject(self, action, obj_pos):
        return
    
    def isSucceedAction(self, action):
        for obj_pos in self.obj_poss:
            if self.isSucceedActionObject(action, obj_pos):
                return True
        return False

class GotoVerifier(AOVerifier):
    def isSucceedActionObject(self, action, obj_pos):
        next_state = get_next_state(self.env, action)
        obj_cell = self.env.grid.get(*obj_pos)

        check_goal = obj_cell.type == "goal"
        check_will_in = next_state[1] == obj_pos
        check_will_be_front = is_in_front_of(self.env, obj_pos, state=next_state)
        check_position = check_goal and check_will_in or not(check_goal) and check_will_be_front

        return check_position

class PickVerifier(AOVerifier):
    def isSucceedActionObject(self, action, obj_pos):
        check_action = action == self.actions.toggle
        check_position = is_in_front_of(self.env, obj_pos)
        check_not_carrying = self.env.carrying == None

        return check_action and check_position and check_not_carrying

class OpenVerifier(AOVerifier):
    def isSucceedActionObject(self, action, obj_pos):
        obj_cell = self.env.grid.get(*obj_pos)

        check_action = action == self.actions.toggle
        check_position = is_in_front_of(self.env, obj_pos)
        check_closed = not(obj_cell.isOpen)
        check_not_locked = obj_cell.type != "locked_door"
        check_carrying = self.env.carrying != None and self.env.carrying.type == 'key' and self.env.carrying.color == obj_cell.color

        return check_action and check_position and check_closed and (check_not_locked or check_carrying)

class DropVerifier(Verifier):
    def __init__(self, env):
        super().__init__(env)

    def isSucceedAction(self, action):
        check_action = action == self.actions.toggle
        check_empty_front = is_empty_in_front_of(self.env)
        check_carrying = self.env.carrying != None

        return check_action and check_empty_front and check_carrying