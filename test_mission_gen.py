#!/usr/bin/env python3

import random
import time
from optparse import OptionParser

from levels import level_list

from PyQt5.QtWidgets import QApplication
from gym_minigrid.rendering import Window

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
        default=-1
    )
    (options, args) = parser.parse_args()

    rng = random.Random()
    seed = options.seed

    level = level_list[options.level_no]
    mission = None

    app = QApplication([])
    window = Window()

    def reset():
        nonlocal seed
        nonlocal mission

        if options.seed == -1:
            seed = rng.randint(0, 0xFFFFFF)

        mission = level.gen_mission(seed)

        print('seed=%d' % seed)
        print(mission.instrs)
        print(mission.surface)

        mission.reset()
        pixmap = mission.render('pixmap')
        window.setPixmap(pixmap)
        window.setKeyDownCb(keyDownCb)

    def keyDownCb(keyName):
        if keyName == 'ESCAPE':
            window.close()
            return

        if keyName == 'BACKSPACE':
            reset()
            return

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
        print("is done:", done)

        if done == True:
            reset()

    reset()

    while True:
        time.sleep(0.01)
        pixmap = mission.render('pixmap')
        window.setPixmap(pixmap)
        app.processEvents()
        if window.closed:
           break

test()
