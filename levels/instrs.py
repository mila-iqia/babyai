from collections import namedtuple

class Instr:
    def __init__(self, action, object):
        # action: goto, open, pickup, drop
        self.action = action
        self.object = object

class Object:
    def __init__(self, obj, pos):
        self.type = obj.type
        self.color = obj.color
        self.pos = pos

        if self.type is 'locked_door':
            self.type = 'door'
            self.state = 'locked'
        else:
            self.state = None

        # TODO: eventually, gen_surface should just use
        # the pos variable to describe the object location
        self.loc = None
