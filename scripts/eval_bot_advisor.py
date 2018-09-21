#!/usr/bin/env python3

import random
import time
from optparse import OptionParser
from babyai.levels import level_dict
from babyai.agents.bot import BotAdvisor, DisappearedBoxError
import numpy as np
from babyai.utils.agent import ModelAgent, RandomAgent

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
    'BossLevel',
]

parser = OptionParser()
parser.add_option(
    "--level",
    default=None
)
parser.add_option(
    "--model",
    default=None
)
parser.add_option(
    "--random_agent_seed",
    type="int",
    default=1,
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
    default=None
)
parser.add_option(
    "--verbose",
    action='store_true'
)
(options, args) = parser.parse_args()

if options.level:
    level_list = [options.level]

if options.model:
    bad_agent = ModelAgent(options.model, obss_preprocessor=None,
                           argmax=True)
else:
    bad_agent = RandomAgent(seed=options.random_agent_seed,
                            env=level_dict[level_list[0]]())

start_time = time.time()

for level_name in level_list:

    num_success = 0
    num_disappeared_boxes = 0
    total_reward = 0

    for run_no in range(options.num_runs):
        level = level_dict[level_name]

        mission_seed = options.seed + run_no
        mission = level(seed=mission_seed)
        non_optimal_steps = options.non_optimal_steps or int(mission.max_steps // 3)
        expert = BotAdvisor(mission)
        np.random.seed(mission_seed)

        if options.verbose:
            print('%s/%s: %s, seed=%d' % (run_no+1, options.num_runs, mission.surface, mission_seed))
        before_optimal_actions = []
        optimal_actions = []
        try:
            step = 0
            while True:
                action = expert.get_action()
                if step < non_optimal_steps:
                    if np.random.rand() < options.random:
                        action = bad_agent.act(mission.gen_obs())['action'].item()
                    before_optimal_actions.append(action)
                    expert.take_action(action)
                else:
                    optimal_actions.append(action)
                    expert.take_action(action)
                obs, reward, done, info = mission.step(action)

                total_reward += reward

                if done:
                    if reward > 0:
                        num_success += 1
                    if reward <= 0:
                        print('FAILURE on %s, seed %d' % (level_name, mission_seed))
                        print(mission)
                        print(before_optimal_actions, optimal_actions)
                    break
                step += 1
        except Exception as e:
            if isinstance(e, DisappearedBoxError):
                num_disappeared_boxes += 1
                print('box FAILURE on %s, seed %d' % (level_name, mission_seed))
            else:
                print('FAILURE on %s, seed %d' % (level_name, mission_seed))
                print(e)
                print(mission)
                # Playing these 2 should get you to the mission snapshot above
                print(before_optimal_actions, optimal_actions)

    success_rate = 100 * num_success / options.num_runs
    disappeared_boxes_rate = 100 * num_disappeared_boxes / options.num_runs
    mean_reward = total_reward / options.num_runs

    print('%16s: %.2f%% (box errors %.2f%%) r=%.3f' % (level_name, success_rate, disappeared_boxes_rate, mean_reward))

end_time = time.time()
total_time = end_time - start_time
print('total time: %.1fs' % total_time)
