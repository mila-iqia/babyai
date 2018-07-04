#!/usr/bin/env python3

"""
Count the number of possible instructions for a given level
"""

import argparse
import gym
import time

import babyai

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, help="name of the environment to be run (REQUIRED)")
args = parser.parse_args()

env = gym.make(args.env)

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

    if num_sampled % 1000 == 0:
        print('current count: %d, sampled: %d' % (new_len, num_sampled))

    # Stop when we increase the number sampled without finding a new instruction
    if num_sampled > 5000 and num_sampled > 2 * last_new_at:
        break

print('num sampled: %d' % num_sampled)
print('last new at: %d' % last_new_at)
print('final count: %d' % len(instr_set))
