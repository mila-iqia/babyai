from gym_minigrid.minigrid import *
from babyai.levels.verifier import *
from babyai.levels.verifier import (ObjDesc, pos_next_to,
                                    GoToInstr, GoNextToInstr, OpenInstr,
                                    PickupInstr, PutNextInstr, BeforeInstr,
                                    AndInstr, AfterInstr)
from babyai.bot import Bot, GoNextToSubgoal


class StackBot(Bot):
    def __init__(self, mission):
        'wrapper which get instruction from bot'
        super(StackBot, self).__init__(mission)

    def is_GoTo(self, subgoal):
        'first subgoal is goto instruction'
        return isinstance(subgoal, GoNextToSubgoal)

    def is_exploratory(self, subgoal):
        'true if subgoal exploratory or goto tuple'
        return isinstance(subgoal.datum, tuple)

    def goTo_instr(self, subgoal):
        'if instruction is goto, get instruction string'
        datum = subgoal.datum
        if subgoal.reason == 'PutNext':
            instr = GoNextToInstr(datum)
        else:
            instr = GoToInstr(datum)
        return instr.surface(self.mission)

    def get_instruction(self):
        'get instruction from first subgoal of bot'
        subgoal = self.stack[-1]
        if self.is_exploratory(subgoal):
            return 'explore'
        if self.is_GoTo(subgoal):
            return self.goTo_instr(subgoal)
