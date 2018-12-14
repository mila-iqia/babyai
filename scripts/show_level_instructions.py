"""
Randomly sample and print out instructions from a level.
"""

import argparse

import babyai
import gym


parser = argparse.ArgumentParser("Show level instructions")
parser.add_argument("--n-episodes", type=int, default=10000,
                    help="Collect instructions from this many episodes")
parser.add_argument("level",
                    help="The level of interest")
args = parser.parse_args()

env = gym.make(args.level)
instructions = set(env.reset()['mission'] for i in range(args.n_episodes))
for instr in sorted(instructions):
    print(instr)
