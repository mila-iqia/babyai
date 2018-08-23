#!/usr/bin/env python3

import argparse
import gym
import logging

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
parser.add_argument("--valid-episodes", type=int, default=512,
                    help="number of validation episodes to generate demonstrations for (default: 512)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--save-interval", type=int, default=10000,
                    help="interval between demonstrations saving (default: 0, 0 means only at the end)")
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps (default: 0, No filtering)")

args = parser.parse_args()
logger = logging.getLogger(__name__)

# Set seed for all randomness sources

if args.seed == 0:
    raise ValueError("seed == 0 is reserved for validation purposes")


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[0]) for demo in demos]
    logger.info('Demo num frames: {}'.format(num_frames_per_episode))


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

        try:
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
            if reward == 0:
                logger.info("failed to accomplish the mission")
        except Exception:
            logger.exception("error while generating demo #{}".format(len(demos)))
            continue

        logger.info("demo #{}".format(len(demos)))

        # Save demonstrations

        if args.save_interval > 0 and len(demos) < n_episodes and len(demos) % args.save_interval == 0:
            logger.info("Saving demos...")
            utils.save_demos(demos, demos_path)
            logger.info("Demos saved")
            # print statistics for the last 100 demonstrations
            print_demo_lengths(demos[-100:])
        offset += 1

    # Save demonstrations
    logger.info("Saving demos...")
    utils.save_demos(demos, demos_path)
    logger.info("Demos saved")
    print_demo_lengths(demos[-100:])

logging.basicConfig(level='INFO', format="%(asctime)s: %(levelname)s: %(message)s")
logger.info(args)
generate_demos(args.episodes, False, args.seed)
generate_demos(args.valid_episodes, True, 0)
