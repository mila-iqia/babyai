#!/usr/bin/env python3

"""
Count the number of possible instructions for a given level
Can also count instructions for all levels using multiple processes
When counting for all levels, the resulting data is parse-able as CSV
"""

import argparse
import gym
import time
from multiprocessing import Process, Pipe
import babyai

parser = argparse.ArgumentParser()
parser.add_argument("--all-levels", action='store_true')
parser.add_argument("--env", help="name of the environment to be run (REQUIRED)")
args = parser.parse_args()

def count_instrs(env_id, quiet=False):
    env = gym.make(env_id)

    # Set of all instructions encountered
    instr_set = set()

    # Number of instructions sampled when the last new instruction was found
    last_new_at = 0

    # Number of instructions sampled
    num_sampled = 0

    while True:
        env.reset()
        mission = env.mission
        num_sampled += 1

        prev_len = len(instr_set)
        instr_set.add(mission)
        new_len = len(instr_set)

        if new_len > prev_len:
            last_new_at = num_sampled

        if not quiet and num_sampled % 1000 == 0:
            print('current count: %d, sampled: %d' % (new_len, num_sampled))

        # Stop when we increase the number sampled without finding a new instruction
        if num_sampled > 5000 and num_sampled > 2 * last_new_at:
            break

    if not quiet:
        print('num sampled: %d' % num_sampled)
        print('last new at: %d' % last_new_at)
        print('final count: %d' % len(instr_set))

    return len(instr_set)

def count_all_levels():
    import subprocess

    level_dict = babyai.levels.level_dict

    def proc(conn, level_name):
        level = level_dict[level_name]
        gym_id = level.gym_id

        count = count_instrs(gym_id, quiet=True)

        conn.send([level_name, gym_id, count])
        conn.close()

    proc_list = []

    for idx, level_name in enumerate(level_dict.keys()):
        #print('Level %s (%d/%d)' % (level_name, idx+1, len(level_dict)))

        parent_conn, child_conn = Pipe()
        p = Process(target=proc, args=(child_conn, level_name))
        p.start()
        proc_list.append((p, parent_conn))

    rows = []

    # For each process
    for p, pipe in proc_list:
        data = pipe.recv()
        rows.append(data)
        p.join()

    # Print the data in CSV format
    for row in rows:
        level_name, gym_id, count = row
        print("%s, %s, %s" % (level_name, gym_id, count))

assert args.env or args.all_levels
if args.env:
    count_instrs(args.env)
else:
    count_all_levels()
