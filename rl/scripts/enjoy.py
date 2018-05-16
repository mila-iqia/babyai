#!/usr/bin/env python3

import argparse
import gym
import levels
import time
import torch

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

# Define agent

agent = utils.Agent(args.model, env.observation_space, env.action_space)

# Run the agent

obs = env.reset()

while True:
    time.sleep(0.1)
    renderer = env.render("human")
    print("Mission:", obs["mission"])

    action = agent.get_action(obs, deterministic=args.deterministic)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)

    if done:
        obs = env.reset()

    if renderer.window is None:
        break