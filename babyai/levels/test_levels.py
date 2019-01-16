"""
Regression tests.
"""

import numpy as np

import gym
from .verifier import *
from .levelgen import *
from gym_minigrid.minigrid import *


class Level_TestGoToBlocked(RoomGridLevel):
    """
    Go to a yellow ball that is blocked with a lot of red balls.
    """

    def __init__(self, room_size=8, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.start_pos = np.array([3, 3])
        self.start_dir = 0
        obj = Ball('yellow')
        self.grid.set(1, 1, obj)
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if (i, j) not in [(1 ,1), (3, 3)]:
                    self.grid.set(i, j, Ball('red'))
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))



class Level_TestPutNextToBlocked(RoomGridLevel):
    """
    Pick up a yellow ball and put it next to a blocked blue ball.
    """

    def __init__(self, room_size=8, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.start_pos = np.array([3, 3])
        self.start_dir = 0
        obj1 = Ball('yellow')
        obj2 = Ball('blue')
        self.place_obj(obj1, (4, 4), (1, 1))
        self.place_obj(obj2, (1, 1), (1, 1))
        self.grid.set(1, 2, Ball('red'))
        self.grid.set(2, 1, Ball('red'))
        self.instrs = PutNextInstr(ObjDesc(obj1.type, obj1.color),
                                   ObjDesc(obj2.type, obj2.color))


register_levels(__name__, globals())
