#!/usr/bin/env python3

"""
Test the level/mission generation code.

This script allows users to visualize generated missions for a given
level and control the agent manually. The instruction string for the
current mission is printed in the console.
"""

import random
import time
from optparse import OptionParser

from babyai.levels import level_dict

from PyQt5.QtWidgets import QApplication
from gym_minigrid.rendering import Window

def test():
    parser = OptionParser()
    parser.add_option(
        "--level-name",
        default='OpenRedDoor'
    )
    parser.add_option(
        "--seed",
        type="int",
        default=-1
    )
    parser.add_option(
        "--partial-obs",
        action='store_true'
    )
    (options, args) = parser.parse_args()

    rng = random.Random()
    seed = options.seed

    level = level_dict[options.level_name]
    mission = None

    app = QApplication([])
    window = Window()

    def reset():
        nonlocal seed
        nonlocal mission

        if options.seed == -1:
            seed = rng.randint(0, 0xFFFFFF)

        mission = level(seed=seed)

        print('seed=%d' % seed)
        print(mission.surface)

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
        elif keyName == 'RETURN':
            action = mission.actions.done
        else:
            return

        obs, reward, done, info = mission.step(action)

        if done == True:
            if reward > 0:
                print('success')
            else:
                print('failure')
            reset()

    reset()

    while True:
        time.sleep(0.01)

        if options.partial_obs:
            obs = mission.gen_obs()
            pixmap = mission.unwrapped.get_obs_render(obs['image'], 32)
        else:
            pixmap = mission.render('pixmap')

        window.setPixmap(pixmap)
        app.processEvents()
        if window.closed:
           break

test()
