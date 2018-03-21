import gym_minigrid
from gym_minigrid.envs import RoomGrid

from .instrs import *

# Actions to support:
# - goto
# - open
# - pick
# - drop

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

        # If an object is to be dropped, it doesn't place any constraint on
        # the environment, except maybe that the location where we drop it must
        # be reachable
        elif instr.action == 'drop':
            pass

        else:
            assert False, 'unknown action %s' % instr.action

    # Create the environment
    # Note: we must have at least 3x3 rooms to support absolute locations
    env = RoomGrid(room_size=7, num_cols=3)
    env.seed(seed)

    # For each object to be added
    for obj in objs:

        if obj.kind == 'door':
            door = 0
            if isinstance(obj.loc, AbsLoc):
                if obj.loc.loc == 'east':
                    door = 0
                if obj.loc.loc == 'south':
                    door = 1
                if obj.loc.loc == 'west':
                    door = 2
                if obj.loc.loc == 'north':
                    door = 3

            env.add_door(1, 1, door, obj.color)

        else:
            room = (1, 1)
            if isinstance(obj.loc, AbsLoc):
                if obj.loc.loc == 'north':
                    room = (1, 0)
                if obj.loc.loc == 'south':
                    room = (1, 2)
                if obj.loc.loc == 'west':
                    room = (0, 1)
                if obj.loc.loc == 'east':
                    room = (2, 1)

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

    # TODO: test relative and absolute locations

    # TODO: test doors

    # Sample instructions:
    # [Instr(action="pick", object=Object(color="red", loc=None, state=None, kind="key")),
    # Instr(action="pick", object=Object(color="red", loc=RelLoc("left"), state=None, kind="key")),
    # Instr(action="pick", object=Object(color="red", loc=AbsLoc("north"), state=None, kind="key"))]

    instrs = [
        Instr(action="pick", object=Object(color="red", loc=None, kind="key")),
        Instr(action="drop", object=Object(color="red", loc=None, kind="key")),
        Instr(action="goto", object=Object(color="blue", loc=AbsLoc('north'), kind="ball")),
        Instr(action="goto", object=Object(color="green", loc=AbsLoc('east'), kind="box")),

        Instr(action="open", object=Object(color="yellow", loc=AbsLoc('north'), kind="door")),
    ]

    env = gen_env(instrs, 0)

    while True:
        env.render('human')
        if env.gridRender.window.closed:
            break
