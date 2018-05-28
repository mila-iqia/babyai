from collections import namedtuple

from gym_minigrid.minigrid import COLOR_NAMES

ACTION_NAMES = ['goto', 'open', 'pickup', 'drop']
TYPE_NAMES = ['door', 'locked_door', 'box', 'ball', 'key', 'wall']
LOC_NAMES = ['left', 'right', 'front', 'behind']
STATE_NAMES = ['locked']

class Instr:
    def __init__(self, action, object):
        assert action in [*ACTION_NAMES]

        self.action = action
        self.object = object

class Object:
    def __init__(self, type=None, color=None, loc=None, state=None):
        assert type in [None, *TYPE_NAMES]
        assert color in [None, *COLOR_NAMES]
        assert loc in [None, *LOC_NAMES]
        assert state in [None, *STATE_NAMES]

        self.type = type
        self.color = color
        self.loc = loc
        self.state = state

        if type is 'locked_door':
            self.type = 'door'