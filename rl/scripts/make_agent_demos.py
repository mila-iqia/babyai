#!/usr/bin/env python3

import argparse
import gym
import levels
import torch_rl

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to generate demonstrations for (default: 1000)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--save-interval", type=int, default=0,
                    help="interval between demonstrations saving (default: 0, 0 means only at the end)")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

# Define agent

agent = utils.load_agent(args, env)

# Load demonstrations

demos = utils.load_demos(args.env, "agent")
utils.synthesize_demos(demos)

for i in range(1, args.demonstrations+1):
    # Run the expert for one episode

    done = False
    obs = env.reset()
    demo = []

    while not(done):
        action = agent.get_action(obs)
        new_obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        demo.append((obs, action))
        obs = new_obs

    demos.append(demo)

    # Save demonstrations

    if args.save_interval > 0 and i < args.demonstrations and i % args.save_interval == 0:
        utils.save_demos(demos, args.env, "agent")
        utils.synthesize_demos(demos)

# Save demonstrations

utils.save_demos(demos, args.env, "agent")
utils.synthesize_demos(demos)