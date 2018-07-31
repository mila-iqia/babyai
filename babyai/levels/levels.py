import gym
from .verifier import *
from .levelgen import *


class Level_GoToObj(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=1)
        obj = objs[0]
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToLocal(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=8, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_PutNextLocal(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, no distractors
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=8, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )

        self.instrs.reset_verifier(self)
        if self.instrs.verify(None) is 'success':
            raise RejectSampling('objs already next to each other')


class Level_GoTo(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToObjMaze(RoomGridLevel):
    """
    Go to an object, the object may be in another room. No distractors.
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=1, all_unique=False)
        self.check_objs_reachable()
        obj = objs[0]
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_Pickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room.
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_UnblockPickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room. The path may
    be blocked by one or more obstructors.
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=20, all_unique=False)

        # Ensure that at least one object is not reachable without unblocking
        # Note: the selected object will still be reachable most of the time
        if self.check_objs_reachable(raise_exc=False):
            raise RejectSampling('all objects reachable')

        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_Open(RoomGridLevel):
    """
    Open a door, which may be in another room
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()

        # Collect a list of all the doors in the environment
        doors = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        doors.append(door)

        door = self._rand_elem(doors)
        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_Unlock(RoomGridLevel):
    """
    Unlock a door.
    Competencies: Maze, Open, Unlock
    No unblocking.
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()

        # Add a locked door to a random room
        id = self._rand_int(0, self.num_rows)
        jd = self._rand_int(0, self.num_cols)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_rows)
            jk = self._rand_int(0, self.num_cols)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, 'key', door.color)
            break

        self.connect_all()
        self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_PutNext(RoomGridLevel):
    """
    Put an object next to another object. Either of these may be in another room.
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )

        self.instrs.reset_verifier(self)
        if self.instrs.verify(None) is 'success':
            raise RejectSampling('objs already next to each other')


class Level_GoToSeq(LevelGen):
    """
    Competencies: Maze, GoTo, Seq
    No locked room.
    No locations.
    No unblocking.
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            action_kinds=['goto'],
            locked_room_prob=0,
            locations=False,
            unblocking=False
        )


class Level_MiniBossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25
        )


class Level_BossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(seed=seed)


# Register the levels in this file
register_levels(__name__, globals())
