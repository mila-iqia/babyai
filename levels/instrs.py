from collections import namedtuple

class Instr:
    def __init__(self, action, object):
        # action: goto, open, pickup, drop
        self.action = action
        self.object = object

class Object:
    def __init__(self, type, color, loc=None):
        assert isinstance(type, str)
        assert isinstance(color, str)

        self.type = type
        self.color = color

        if type is 'locked_door':
            self.type = 'door'
            self.state = 'locked'
        else:
            self.state = None

        self.loc = loc
