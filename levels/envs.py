import gym_minigrid
from gym_minigrid.envs import RoomGrid
from gym_minigrid.minigrid import COLOR_NAMES

from .instrs import *

def door_from_loc(loc):
    """
    Get the door index for a given location
    The door indices correspond to: right, down, left, up
    """

    if isinstance(loc, AbsLoc):
        if loc.loc == 'east':
            return 0
        if loc.loc == 'south':
            return 1
        if loc.loc == 'west':
            return 2
        if loc.loc == 'north':
            return 3
        assert False, loc

    if isinstance(loc, RelLoc):
        if loc.loc == 'left':
            return 3
        if loc.loc == 'right':
            return 1
        if loc.loc == 'front':
            return 0
        if loc.loc == 'behind':
            return 2
        assert False, loc

    assert False, loc

def room_from_loc(loc):
    """
    Get the room coordinates for a given location
    """

    if isinstance(loc, AbsLoc):
        if loc.loc == 'north':
            return (1, 0)
        if loc.loc == 'south':
            return (1, 2)
        if loc.loc == 'west':
            return (0, 1)
        if loc.loc == 'east':
            return (2, 1)
        assert False, loc

    if isinstance(loc, RelLoc):
        if loc.loc == 'left':
            return (1, 0)
        if loc.loc == 'right':
            return (1, 2)
        if loc.loc == 'front':
            return (2, 1)
        if loc.loc == 'behind':
            return (0, 1)
        assert False, loc

    # By default, use the central room
    return (1, 1)

def gen_env(instrs, seed):
    """
    Generate an environment from a list of instructions (structured instruction).

    :param seed: seed to be used for the random number generator.
    """

    # Set of objects to be placed
    objs = set()

    # For each instruction
    for instr in instrs:
        # The pick, goto and open actions mean the referenced objects must exist
        if instr.action == 'pick' or instr.action == 'goto' or instr.action == 'open':
            obj = instr.object
            objs.add(obj)

    # Create the environment
    # Note: we must have at least 3x3 rooms to support absolute locations
    env = RoomGrid(room_size=7, num_cols=3)
    env.seed(seed)

    # Assign colors to objects that don't have one
    for obj in objs:
        if obj.color is None:
            objs.remove(obj)
            color = env._randElem(COLOR_NAMES)
            obj = Object(kind=obj.kind, loc=obj.loc, color=color)
            objs.add(obj)

    # For each object to be added
    for obj in objs:
        if obj.kind == 'door':
            room = (1, 1)
            door = door_from_loc(obj.loc)
            env.add_door(*room, door, obj.color)
        else:
            room = room_from_loc(obj.loc)
            env.add_object(*room, obj.kind, obj.color)

    # Make sure that all rooms are reachable by the agent
    env.connect_all()

    return env

def test():
    """
    Test the gen_env function, render the result

    To try this code:
    python3
    >>> import levels.envs
    >>> levels.envs.test()
    """

    import random

    seed = random.randint(0, 0xFFFF)

    instrs = [
        Instr(action="pick", object=Object(color="red", loc=RelLoc('front'), kind="key")),
        #Instr(action="pick", object=Object(color="blue", loc=RelLoc('front'), kind="key")),

        Instr(action="pick", object=Object(color=None, loc=RelLoc('left'), kind="ball")),

        #Instr(action="drop", object=Object(color="red", loc=None, kind="key")),
        #Instr(action="goto", object=Object(color="blue", loc=AbsLoc('north'), kind="ball")),
        #Instr(action="goto", object=Object(color="green", loc=AbsLoc('east'), kind="box")),
        #Instr(action="open", object=Object(color="yellow", loc=AbsLoc('north'), kind="door")),
    ]

    print(instrs)

    env = gen_env(instrs, seed)

    def keyDownCb(keyName):
        if keyName == 'ESCAPE':
            env.gridRender.window.close()
    renderer = env.render('human')
    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        if env.gridRender.window.closed:
            break
