#!/usr/bin/env python3

import random
import time
from optparse import OptionParser
from babyai.levels import level_dict
from babyai.agents.bot import Bot

level_list = [
    'OpenRedDoor',
    'GoToLocal',
    'PutNextLocal',

    'GoToObjMaze',
    'GoTo',
    'Open',
    'Pickup',
    'PickupLoc',
    'PutNext',

    'Unlock',
    'GoToImpUnlock',
    #'UnblockPickup',

    #'GoToSeq',
    #'Synth',
    #'SynthLoc',
    #'SynthSeq',

    #'MiniBossLevel',
    #'BossLevel',
]

parser = OptionParser()
#parser.add_option(
#    "--level",
#    default='OpenRedDoor'
#)
parser.add_option(
    "--seed",
    type="int",
    default=1
)
parser.add_option(
    "--num_runs",
    type="int",
    default=500
)
parser.add_option(
    "--verbose",
    action='store_true'
)
(options, args) = parser.parse_args()

for level_name in level_list:

    num_success = 0
    total_reward = 0

    for run_no in range(options.num_runs):
        level = level_dict[level_name]

        mission_seed = options.seed + run_no
        mission = level(seed=mission_seed)
        expert = Bot(mission)

        if options.verbose:
            print('%s/%s: %s, seed=%d' % (run_no+1, options.num_runs, mission.surface, mission_seed))

        while True:
            action = expert.step()
            obs, reward, done, info = mission.step(action)

            total_reward += reward

            if done == True:
                if reward > 0:
                    num_success += 1
                break

    success_rate = 100 * num_success / options.num_runs
    mean_reward = total_reward / options.num_runs

    print('%16s: %.1f%%, r=%.2f' % (level_name, success_rate, mean_reward))
