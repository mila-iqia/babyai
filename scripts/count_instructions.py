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

instr_set = set()

while True:
    env.reset()
    mission = env.mission

    prev_len = len(instr_set)
    instr_set.add(mission)
    new_len = len(instr_set)

    if new_len > prev_len:
        print('current count: %d' % new_len)
