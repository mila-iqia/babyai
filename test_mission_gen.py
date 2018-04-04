#!/usr/bin/env python3

import random
import time
from optparse import OptionParser

from levels import level_list

#from levels.instr_gen import gen_instr_seq, gen_surface
#from levels.env_gen import gen_env
#from levels.instrs import *
#from levels.verifier import InstrSeqVerifier

def test():
    parser = OptionParser()
    parser.add_option(
        "--level-no",
        type="int",
        default=0
    )
    parser.add_option(
        "--seed",
        type="int",
        default=0
    )
    (options, args) = parser.parse_args()

    level = level_list[options.level_no]
    mission = level.gen_mission(options.seed)

    # TODO: if seed is -1, pick random?

    print(mission.instrs)
    print(mission.surface)

    def keyDownCb(keyName):
        if keyName == 'ESCAPE':
            renderer.window.close()

        action = 0
        if keyName == 'LEFT':
            action = mission.actions.left
        elif keyName == 'RIGHT':
            action = mission.actions.right
        elif keyName == 'UP':
            action = mission.actions.forward
        elif keyName == 'SPACE':
            action = mission.actions.toggle
        elif keyName == 'PAGE_UP':
            action = mission.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = mission.actions.drop
        else:
            return

        obs, reward, done, info = mission.step(action)
        print(done)

    renderer = mission.render('human')
    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        time.sleep(0.01)
        mission.render('human')
        if renderer.window.closed:
            break

test()
