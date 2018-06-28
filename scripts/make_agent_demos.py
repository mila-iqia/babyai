#!/usr/bin/env python3

import argparse
import gym

import babyai.utils as utils

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
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--save-interval", type=int, default=0,
                    help="interval between demonstrations saving (default: 0, 0 means only at the end)")
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps (default: 0, No filtering)")
parser.add_argument("--valid", action="store_true", default=False,
                    help="generating demonstrations for validation set")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment
env = gym.make(args.env)
env.seed(args.seed)

# Select Origin
origin = "agent" if not args.valid else "agent_valid"
assert not(args.valid) or args.seed == 0

# Define agent

agent = utils.load_agent(args, env)

# Load demonstrations
demos = utils.load_demos(args.env, origin)
utils.synthesize_demos(demos)

while True:
    # Run the expert for one episode

    done = False
    obs = env.reset()
    demo = []

    while not(done):
        action = agent.get_action(obs)
        new_obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        demo.append((obs, action, reward, done))
        obs = new_obs
    if args.filter_steps is not 0:
        if len(demo) <= args.filter_steps and reward != 0:
            demos.append(demo)
    if len(demos) == args.episodes:
        break

    # Save demonstrations

    if args.save_interval > 0 and i < args.episodes and i % args.save_interval == 0:
        utils.save_demos(demos, args.env, origin)
        utils.synthesize_demos(demos)

# Save demonstrations
utils.save_demos(demos, args.env, origin)
utils.synthesize_demos(demos)