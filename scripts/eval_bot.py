#!/usr/bin/env python3

"""
Evaluate the success rate of the bot
This script is used for testing/debugging purposes

Examples of usage:
- Run the bot on the GoTo level 10 times (seeds 9 to 18)
eval_bot.py --level GoTo --num_runs 10 --seed 9
- for all levels, 100 times, run a Random(seed 0) agent for len(episode)/3 steps before running the bot:
eval_bot.py --advise_mode --num_runs 100
- for all levels, 500 times, during the first 10 steps, choose action form a Random(seed 9) agent with proba .9 or
 optimal (from bot) with proba .1, then continue with optimal bot actions:
eval_boy.py --advise_mode --bad_action_proba .8 --non_optimal_steps 10 --random_agent_seed 9

"""

import random
import time
import traceback
from optparse import OptionParser
from babyai.levels import level_dict
from babyai.bot import Bot
from babyai.utils.agent import ModelAgent, RandomAgent
from random import Random


# MissBossLevel is the only level the bot currently can't always handle
level_list = [name for name, level in level_dict.items()
              if (not getattr(level, 'is_bonus', False) and not name == 'MiniBossLevel')]


parser = OptionParser()
parser.add_option(
    "--level",
    default=None
)
parser.add_option(
    "--advise_mode",
    action='store_true',
    default=False,
    help='If specified, a RandomAgent or ModelAgent will act first, then the bot will take over')
parser.add_option(
    "--non_optimal_steps",
    type=int,
    default=None,
    help='Number of non bot steps ModelAgent or RandomAgent takes before letting the bot take over'
)
parser.add_option(
    "--model",
    default=None,
    help='Model to use to act for a few steps before letting the bot take over'
)
parser.add_option(
    "--random_agent_seed",
    type="int",
    default=1,
    help='Seed of the random agent that acts a few steps before letting the bot take over'
)
parser.add_option(
    "--bad_action_proba",
    type="float",
    default=1.,
    help='Probability of performing the non-optimal action when the random/model agent is performing'
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
    "--verbose",
    action='store_true'
)
(options, args) = parser.parse_args()

if options.level:
    level_list = [options.level]

bad_agent = None
if options.advise_mode:
    if options.model:
        bad_agent = ModelAgent(options.model, obss_preprocessor=None,
                               argmax=True)
    else:
        bad_agent = RandomAgent(seed=options.random_agent_seed)

start_time = time.time()

all_good = True

for level_name in level_list:

    num_success = 0
    total_reward = 0
    total_steps = []
    total_bfs = 0
    total_episode_steps = 0
    total_bfs_steps = 0

    for run_no in range(options.num_runs):
        level = level_dict[level_name]

        mission_seed = options.seed + run_no
        mission = level(seed=mission_seed)
        expert = Bot(mission)

        if options.verbose:
            print('%s/%s: %s, seed=%d' % (run_no+1, options.num_runs, mission.surface, mission_seed))

        optimal_actions = []
        before_optimal_actions = []
        non_optimal_steps = options.non_optimal_steps or int(mission.max_steps // 3)
        rng = Random(mission_seed)

        try:
            episode_steps = 0
            last_action = None
            while True:
                action = expert.replan(last_action)
                if options.advise_mode and episode_steps < non_optimal_steps:
                    if rng.random() < options.bad_action_proba:
                        while True:
                            action = bad_agent.act(mission.gen_obs())['action'].item()
                            fwd_pos = mission.agent_pos + mission.dir_vec
                            fwd_cell = mission.grid.get(*fwd_pos)
                            # The current bot can't recover from two kinds of behaviour:
                            # - opening a box (cause it just disappears)
                            # - closing a door (cause its path finding mechanism get confused)
                            opening_box = (action == mission.actions.toggle
                                and fwd_cell and fwd_cell.type == 'box')
                            closing_door = (action == mission.actions.toggle
                                and fwd_cell and fwd_cell.type == 'door' and fwd_cell.is_open)
                            if not opening_box and not closing_door:
                                break
                    before_optimal_actions.append(action)
                else:
                    optimal_actions.append(action)

                obs, reward, done, info = mission.step(action)
                last_action = action

                total_reward += reward
                episode_steps += 1

                if done:
                    total_episode_steps += episode_steps
                    total_bfs_steps += expert.bfs_step_counter
                    total_bfs += expert.bfs_counter
                    if reward > 0:
                        num_success += 1
                        total_steps.append(episode_steps)
                        if options.verbose:
                            print('SUCCESS on seed {}, reward {:.2f}'.format(mission_seed, reward))
                    if reward <= 0:
                        assert episode_steps == mission.max_steps  # Is there another reason for this to happen ?
                        if options.verbose:
                            print('FAILURE on %s, seed %d, reward %.2f' % (level_name, mission_seed, reward))
                    break
        except Exception as e:
            print('FAILURE on %s, seed %d' % (level_name, mission_seed))
            traceback.print_exc()
            # Playing these 2 sets of actions should get you to the mission snapshot above
            print(before_optimal_actions)
            print(optimal_actions)
            print(expert.stack)
            break

    all_good = all_good and (num_success == options.num_runs)

    success_rate = 100 * num_success / options.num_runs
    mean_reward = total_reward / options.num_runs
    mean_steps = sum(total_steps) / options.num_runs

    print('%16s: %.1f%%, r=%.3f, s=%.2f' % (level_name, success_rate, mean_reward, mean_steps))
    # Uncomment the following line to print the number of steps per episode (useful to look for episodes to debug)
    # print({options.seed + num_run: total_steps[num_run] for num_run in range(options.num_runs)})
end_time = time.time()
total_time = end_time - start_time
print('total time: %.1fs' % total_time)
if not all_good:
    raise Exception("some tests failed")
print('total episode_steps:', total_episode_steps)
print('total bfs:', total_bfs)
print('total bfs steps:', total_bfs_steps)
