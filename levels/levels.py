from .instrs import *
from .instr_gen import gen_instr_seq, gen_surface
from .env_gen import gen_env

# - want way to iterate through levels
# - generate environments and instructions given seeds
# - probably want way to sample mission objects for a given level

# TODO: more commenting

class Level:
    def __init__(self):
        pass

    def gen_mission(seed):
        pass





class Mission:
    def __init__(self, instrs):
        self.instrs = instrs






class Level0(Level):
    def __init__(self):
        super().__init__()




# Levels array, indexable by level number
# ie: levels[0] is a Level0 instance
levels = [
    Level0()
]





def test():
    pass
