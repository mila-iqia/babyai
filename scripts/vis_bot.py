#!/usr/bin/env python3

import random
import time
from optparse import OptionParser

from babyai.levels import level_dict
from babyai.agents.bot import Bot

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
    parser.add_option(
        "--forget",
        action='store_true'
    )
    (options, args) = parser.parse_args()

    rng = random.Random()
    seed = options.seed

    app = QApplication([])
    window = Window()

    level = level_dict[options.level_name]
    mission = None
    expert = None

    def reset():
        nonlocal seed
        nonlocal mission
        nonlocal expert

        if options.seed == -1:
            seed = rng.randint(0, 0xFFFFFF)

        mission = level(seed=seed)
        expert = Bot(mission, forget=options.forget)

        print('seed=%d' % seed)
        print(mission.surface)

        pixmap = mission.render('pixmap')
        window.setPixmap(pixmap)
        #window.setKeyDownCb(keyDownCb)

    reset()

    while True:
        time.sleep(0.1)

        if options.partial_obs:
            obs = mission.gen_obs()
            pixmap = mission.unwrapped.get_obs_render(obs['image'], 32)
        else:
            pixmap = mission.render('pixmap')

        window.setPixmap(pixmap)
        app.processEvents()

        obs = mission.gen_obs()
        action = expert.step()

        obs, reward, done, info = mission.step(action)

        if done == True:
            if reward > 0:
                print('success')
            else:
                print('failure')
            reset()

        if window.closed:
           break

test()
