#!/usr/bin/env python3

"""
Train an agent using an intelligent expert.

The procedure starts with a small set of training demonstrations, and
iteratively grows the training set by some percentage. At every step, the new
demos used to grow the training set are demos the agent is currently failing
on. A new model is trained from scratch at every step.

Sample usage:
scripts/train_intelligent_expert.py --env BabyAI-GoToObj-v0 --demos GoToObj-bot-100k --validation-interval 5

Vanilla imitation learning:
GoToObj, 1000 demos for 100 percent success rate
GoToLocal, over 60K demos needed
"""

import os
import csv
import json
import copy
import gym
import time
import datetime
import numpy as np
import sys
import logging
import babyai.utils as utils
from babyai.arguments import ArgumentParser
from babyai.imitation import ImitationLearning
from babyai.evaluate import batch_evaluate, evaluate
from babyai.utils.agent import BotAgent
import babyai.utils as utils
import torch
import blosc
from babyai.utils.agent import DemoAgent

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or demos-origin required)")
parser.add_argument("--demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos required)")
parser.add_argument("--episodes", type=int, default=0,
                    help="number of episodes of demonstrations to use"
                         "(default: 0, meaning all demos)")
parser.add_argument("--start-demos", type=int, default=5000,
                    help="the starting number of demonstrations")
parser.add_argument("--demo-grow-factor", type=float, default=1.2,
                    help="number of demos to add to the training set")
parser.add_argument("--num-eval-demos", type=int, default=1000,
                    help="number of demos used for evaluation while growing the training set")
parser.add_argument("--phases", type=int, default=1000,
                    help="maximum number of phases to train for")
parser.add_argument("--save-interval", type=int, default=1,
                    help="number of epochs between two saves (default: 1, 0 means no saving)")


logger = logging.getLogger(__name__)

check_obss_equality = DemoAgent.check_obss_equality
def evaluate_agent(il_learn, eval_seed, num_eval_demos, return_obss_actions=False):
    """
    Evaluate the agent on some number of episodes and return the seeds for the
    episodes the agent performed the worst on.
    """

    logger.info("Evaluating agent on {} using {} demos".format(il_learn.args.env, num_eval_demos))

    agent = utils.load_agent(il_learn.env, il_learn.args.model)

    agent.model.eval()
    logs = batch_evaluate(
        agent,
        il_learn.args.env,
        episodes=num_eval_demos,
        seed=eval_seed,
        return_obss_actions=return_obss_actions
    )
    agent.model.train()

    success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
    logger.info("success rate: {:.2f}".format(success_rate))

    # Find the seeds for all the failing demos
    fail_seeds = []
    fail_obss = []
    fail_actions = []

    for idx, ret in enumerate(logs["return_per_episode"]):
        if ret <= 0:
            fail_seeds.append(logs["seed_per_episode"][idx])
            if return_obss_actions:
                fail_obss.append(logs["observations_per_episode"][idx])
                fail_actions.append(logs["actions_per_episode"][idx])

    logger.info("{} fails".format(len(fail_seeds)))

    if not return_obss_actions:
        return success_rate, fail_seeds
    else:
        return success_rate, fail_seeds, fail_obss, fail_actions


def generate_demos(env_name, seeds):
    env = gym.make(env_name)
    agent = BotAgent(env)
    demos = []

    for seed in seeds:
        # Run the expert for one episode
        done = False

        env.seed(int(seed))
        obs = env.reset()
        agent.on_reset()

        actions = []
        mission = obs["mission"]
        images = []
        directions = []

        try:
            while not done:
                action = agent.act(obs)['action']
                new_obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)

                actions.append(action)
                images.append(obs['image'])
                directions.append(obs['direction'])

                obs = new_obs

            if reward > 0:
                demos.append((mission, blosc.pack_array(np.array(images)), directions, actions))
            if reward == 0:
                logger.info("failed to accomplish the mission")

        except Exception:
            logger.exception("error while generating demo #{}".format(len(demos)))
            continue

        # logger.info("demo #{}".format(len(demos)))

    return demos


def grow_training_set(il_learn, train_demos, eval_seed, grow_factor, num_eval_demos):
    """
    Grow the training set of demonstrations by some factor
    We specifically generate demos on which the agent fails
    """

    new_train_set_size = int(len(train_demos) * grow_factor)
    num_new_demos = new_train_set_size - len(train_demos)

    logger.info("Generating {} new demos for {}".format(num_new_demos, il_learn.args.env))

    # Add new demos until we rearch the new target size
    while len(train_demos) < new_train_set_size:
        num_new_demos = new_train_set_size - len(train_demos)

        # Evaluate the success rate of the model
        success_rate, fail_seeds = evaluate_agent(il_learn, eval_seed, num_eval_demos)
        eval_seed += num_eval_demos

        if len(fail_seeds) > num_new_demos:
            fail_seeds = fail_seeds[:num_new_demos]

        # Generate demos for the worst performing seeds
        new_demos = generate_demos(il_learn.args.env, fail_seeds)
        train_demos.extend(new_demos)

    return eval_seed


def get_bot_mean(env_name, episodes_to_evaluate_mean, seed):
    logger.info("Evaluating the average number of steps using {} episodes".format(episodes_to_evaluate_mean))
    env = gym.make(env_name)
    env.seed(seed)
    agent = BotAgent(env)
    logs = evaluate(agent, env, episodes_to_evaluate_mean, model_agent=False)
    average_number_of_steps = np.mean(logs["num_frames_per_episode"])
    logger.info("Average number of steps: {}".format(average_number_of_steps))
    return average_number_of_steps


def main(args):
    args.model = args.model or ImitationLearning.default_model_name(args)
    utils.configure_logging(args.model)
    il_learn = ImitationLearning(args)

    # Define logger and Tensorboard writer
    header = (["update", "frames", "FPS", "duration", "entropy", "policy_loss", "train_accuracy"]
              + ["validation_accuracy", "validation_return", "validation_success_rate"])
    writer = None
    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(utils.get_log_dir(args.model))

    # Define csv writer
    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(il_learn.acmodel)

    # Seed at which demo evaluation/generation will begin
    eval_seed = args.seed + len(il_learn.train_demos)

    # Phase at which we start
    cur_phase = 0

    # Try to load the status (if resuming)
    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
            eval_seed = status.get('eval_seed', eval_seed)
            cur_phase = status.get('cur_phase', cur_phase)

    model_name = args.model

    for phase_no in range(cur_phase, args.phases):
        logger.info("Starting phase {} with {} demos, eval_seed={}".format(phase_no, len(il_learn.train_demos), eval_seed))

        # Each phase trains a different model from scratch
        args.model = model_name + ('_phase_%d' % phase_no)
        il_learn = ImitationLearning(args)

        # Train the imitation learning agent
        if len(il_learn.train_demos) > 0:
            train_status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
            il_learn.train(il_learn.train_demos, writer, csv_writer, train_status_path, header)

        # Stopping criterion
        valid_log = il_learn.validate(args.val_episodes)
        success_rate = np.mean([1 if r > 0 else 0 for r in valid_log[0]['return_per_episode']])

        if success_rate >= 0.99:
            logger.info("Reached target success rate with {} demos, stopping".format(len(il_learn.train_demos)))
            break

        eval_seed = grow_training_set(
            il_learn,
            il_learn.train_demos,
            eval_seed,
            args.demo_grow_factor,
            args.num_eval_demos
        )

        # Save the current demo generation seed
        with open(status_path, 'w') as dst:
            status = {
                'eval_seed': eval_seed,
                'cur_phase':phase_no + 1
            }
            json.dump(status, dst)

        # Save the demos
        demos_path = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=False)
        print('saving demos to:', demos_path)
        utils.save_demos(il_learn.train_demos, demos_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
