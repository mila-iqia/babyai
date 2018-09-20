#!/usr/bin/env python3

import random
import time
from optparse import OptionParser
from babyai.levels import level_dict
from babyai.agents.bot import BotAdvisor
import numpy as np

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
    'UnblockPickup',

    'GoToSeq',
    'Synth',
    'SynthLoc',
    'SynthSeq',
    #'BossLevel',
]

parser = OptionParser()
parser.add_option(
    "--level",
    default=None
)
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
    "--random",
    type="float",
    default=0.
)
parser.add_option(
    "--non_optimal_steps",
    type=int,
    default=20
)
parser.add_option(
    "--verbose",
    action='store_true'
)
(options, args) = parser.parse_args()

if options.level:
    level_list = [options.level]

start_time = time.time()

for level_name in level_list:

    num_success = 0
    total_reward = 0

    for run_no in range(options.num_runs):
        level = level_dict[level_name]

        mission_seed = options.seed + run_no
        mission = level(seed=mission_seed)
        expert = BotAdvisor(mission)
        np.random.seed(mission_seed)

        if options.verbose:
            print('%s/%s: %s, seed=%d' % (run_no+1, options.num_runs, mission.surface, mission_seed))
        random_actions = []
        try:
            #print(mission)
            step = 0
            while True:
                action = expert.get_action()
                if step < options.non_optimal_steps and random.random() < options.random:
                    print(expert.stack)
                    action = np.random.randint(0, 6)
                    print('random action {}'.format(action))
                    random_actions.append(action)
                    expert.take_action(action)
                    print(expert.stack)
                    print('\n')
                else:
                    #print(expert.stack)
                    #print('actual action {}'.format(action))
                    expert.take_action(action)
                    #print(expert.stack)
                    #print('\n')
                obs, reward, done, info = mission.step(action)

                total_reward += reward

                if done:
                    if reward > 0:
                        num_success += 1
                    if reward <= 0:
                        print('FAILURE on %s, seed %d' % (level_name, mission_seed))
                    break
                step += 1
        except Exception as e:
            print('FAILURE on %s, seed %d' % (level_name, mission_seed))
            print(e)
        print(random_actions)

    success_rate = 100 * num_success / options.num_runs
    mean_reward = total_reward / options.num_runs

    print('%16s: %.1f%%, r=%.3f' % (level_name, success_rate, mean_reward))

end_time = time.time()
total_time = end_time - start_time
print('total time: %.1fs' % total_time)
