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

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.agent_pos = np.array([3, 3])
        self.agent_dir = 0
        obj = Ball('yellow')
        self.grid.set(1, 1, obj)
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if (i, j) not in [(1 ,1), (3, 3)]:
                    self.place_obj(Ball('red'), (i, j), (1, 1))
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))



class Level_TestPutNextToBlocked(RoomGridLevel):
    """
    Pick up a yellow ball and put it next to a blocked blue ball.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.agent_pos = np.array([3, 3])
        self.agent_dir = 0
        obj1 = Ball('yellow')
        obj2 = Ball('blue')
        self.place_obj(obj1, (4, 4), (1, 1))
        self.place_obj(obj2, (1, 1), (1, 1))
        self.grid.set(1, 2, Ball('red'))
        self.grid.set(2, 1, Ball('red'))
        self.instrs = PutNextInstr(ObjDesc(obj1.type, obj1.color),
                                   ObjDesc(obj2.type, obj2.color))


class Level_TestPutNextToCloseToDoor1(RoomGridLevel):
    """
    The yellow ball must be put near the blue ball.
    But blue ball is right next to a door.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=2,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([3, 3])
        self.agent_dir = 0
        door, pos = self.add_door(0, 0, None, 'red', False)
        self.obj1 = Ball('yellow')
        self.obj2 = Ball('blue')
        self.place_obj(self.obj1, (4, 4), (1, 1))
        self.place_obj(self.obj2, (pos[0], pos[1] + 1), (1, 1))
        self.instrs = BeforeInstr(
            OpenInstr(ObjDesc('door', door.color)),
            PutNextInstr(ObjDesc(self.obj1.type, self.obj1.color),
                         ObjDesc(self.obj2.type, self.obj2.color)))


class Level_TestPutNextToCloseToDoor2(Level_TestPutNextToCloseToDoor1):
    """
    The yellow ball must be put near the blue ball.
    But blue ball is right next to a door.
    """

    def gen_mission(self):
        super().gen_mission()
        self.instrs = PutNextInstr(ObjDesc(self.obj1.type, self.obj1.color),
                                   ObjDesc(self.obj2.type, self.obj2.color))



class Level_TestPutNextToIdentical(RoomGridLevel):
    """
    Test that the agent does not endlessly hesitate between
    two identical objects.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([3, 3])
        self.agent_dir = 0
        self.place_obj(Box('yellow'), (1, 1), (1, 1))
        self.place_obj(Ball('blue'), (4, 4), (1, 1))
        self.place_obj(Ball('red'), (2, 2), (1, 1))
        instr1 = PutNextInstr(ObjDesc('ball', 'blue'),
                              ObjDesc('box', 'yellow'))
        instr2 = PutNextInstr(ObjDesc('box', 'yellow'),
                              ObjDesc('ball', None))
        self.instrs = BeforeInstr(instr1, instr2)


class Level_TestUnblockingLoop(RoomGridLevel):
    """Test that unblocking does not results into an infinite loop."""

    def __init__(self, seed=None):
        super().__init__(
            num_rows=2,
            num_cols=2,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([15, 4])
        self.agent_dir = 2
        door, pos = self.add_door(0, 0, 1, 'red', False)
        door, pos = self.add_door(0, 1, 0, 'red', False)
        door, pos = self.add_door(1, 1, 3, 'blue', False)
        self.place_obj(Box('yellow'), (9, 1), (1, 1))
        self.place_obj(Ball('blue'), (5, 3), (1, 1))
        self.place_obj(Ball('yellow'), (6, 2), (1, 1))
        self.place_obj(Key('blue'), (15, 15), (1, 1))
        put = PutNextInstr(ObjDesc('key', 'blue'), ObjDesc('door', 'blue'))
        goto1 = GoToInstr(ObjDesc('ball', 'yellow'))
        goto2 = GoToInstr(ObjDesc('box', 'yellow'))
        self.instrs = BeforeInstr(put, AndInstr(goto1, goto2))


class Level_TestPutNextCloseToDoor(RoomGridLevel):
    """Test putting next when there is door where the object should be put."""

    def __init__(self, seed=None):
        super().__init__(
            num_rows=2,
            num_cols=2,
            room_size=9,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([5, 10])
        self.agent_dir = 2
        door, pos1 = self.add_door(0, 0, 1, 'red', False)
        door, pos2 = self.add_door(0, 1, 0, 'red', False)
        door, pos3 = self.add_door(1, 1, 3, 'blue', False)
        self.place_obj(Ball('blue'), (pos1[0], pos1[1] - 1), (1, 1))
        self.place_obj(Ball('blue'), (pos1[0], pos1[1] - 2), (1, 1))
        if pos1[0] - 1 >= 1:
            self.place_obj(Box('green'), (pos1[0] - 1, pos1[1] - 1), (1, 1))
        if pos1[0] + 1 < 8:
            self.place_obj(Box('green'), (pos1[0] + 1, pos1[1] - 1), (1, 1))
        self.place_obj(Box('yellow'), (3, 15), (1, 1))
        self.instrs = PutNextInstr(ObjDesc('box', 'yellow'), ObjDesc('ball', 'blue'))


class Level_TestLotsOfBlockers(RoomGridLevel):
    """
    Test that the agent does not endlessly hesitate between
    two identical objects.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=8,
            seed=seed
        )

    def gen_mission(self):
        self.agent_pos = np.array([5, 5])
        self.agent_dir = 0
        self.place_obj(Box('yellow'), (2, 1), (1, 1))
        self.place_obj(Box('yellow'), (2, 2), (1, 1))
        self.place_obj(Box('yellow'), (2, 3), (1, 1))
        self.place_obj(Box('yellow'), (3, 4), (1, 1))
        self.place_obj(Box('yellow'), (2, 6), (1, 1))
        self.place_obj(Box('yellow'), (1, 3), (1, 1))
        self.place_obj(Ball('blue'), (1, 2), (1, 1))
        self.place_obj(Ball('red'), (3, 6), (1, 1))
        self.instrs = PutNextInstr(ObjDesc('ball', 'red'),
                                   ObjDesc('ball', 'blue'))


register_levels(__name__, globals())
