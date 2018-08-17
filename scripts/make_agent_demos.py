#!/usr/bin/env python3

import argparse
import gym

import babyai.utils as utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default='BOT',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="path to save demonstrations (based on --model and --origin by default)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to generate demonstrations for (default: 1000)")
parser.add_argument("--valid-episodes", type=int, default=500,
                    help="number of validation episodes to generate demonstrations for (default: 500)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--save-interval", type=int, default=0,
                    help="interval between demonstrations saving (default: 0, 0 means only at the end)")
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps (default: 0, No filtering)")

args = parser.parse_args()

# Set seed for all randomness sources

if args.seed == 0:
    raise ValueError("seed == 0 is reserved for validation purposes")

def generate_demos(n_episodes, valid, seed):
    utils.seed(seed)

    # Generate environment
    env = gym.make(args.env)
    env.seed(seed)

    agent = utils.load_agent(args, env)

    demos_path = utils.get_demos_path(args.demos, args.env, 'agent', valid)

    demos = []

    offset = 0

    while True:
        # Run the expert for one episode

        done = False
        obs = env.reset()
        agent.on_reset()
        demo = []

        while not done:
            action = agent.get_action(obs)
            new_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)

            demo.append((obs, action, reward, done))
            obs = new_obs
        if reward > 0 and (args.filter_steps == 0 or len(demo) <= args.filter_steps):
                demos.append((demo, offset))
        if len(demos) >= n_episodes:
            break

        # Save demonstrations

        if args.save_interval > 0 and len(demos) < n_episodes and len(demos) % args.save_interval == 0:
            utils.save_demos(demos, demos_path)
            # print statistics for the last 100 demonstrations
            utils.synthesize_demos(demos[-100:])
        offset += 1

    # Save demonstrations
    utils.save_demos(demos, demos_path)
    utils.synthesize_demos(demos[-100:])

generate_demos(args.episodes, False, args.seed)
generate_demos(args.valid_episodes, True, 0)
