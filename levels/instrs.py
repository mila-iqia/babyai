from collections import namedtuple

from gym_minigrid.minigrid import COLOR_NAMES

class Instr:
    def __init__(self, action, object):
        # action: goto, open, pickup, drop
        self.action = action
        self.object = object

class Object:
    def __init__(self, type, color, loc=None):
        assert isinstance(type, str)
        assert color in COLOR_NAMES
        assert loc in [None, 'left', 'right', 'front', 'behind']

        self.type = type
        self.color = color

        if type is 'locked_door':
            self.type = 'door'
            self.state = 'locked'
        else:
            self.state = None

        self.loc = loc
