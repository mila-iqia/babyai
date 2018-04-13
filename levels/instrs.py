from collections import namedtuple

class Instr:
    def __init__(self, action, object):
        # action: goto, open, pickup, drop
        self.action = action
        self.object = object

class Object:
    def __init__(
        self,
        type,
        color
    ):
        if type is 'locked_door':
            type = 'door'
            locked = True
        else:
            locked = False

        self.type = type
        self.color = color

        self.loc = None

        # Locked flag, applies to doors only
        self.locked = locked

        self.state = 'locked' if locked else None
