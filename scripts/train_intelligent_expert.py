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
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import BotAgent
import torch
import blosc


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
parser.add_argument("--finetune", action="store_true", default=False,
                    help="fine-tune the model at every phase instead of retraining")
parser.add_argument("--phases", type=int, default=1000,
                    help="maximum number of phases to train for")


logger = logging.getLogger(__name__)


def evaluate_agent(il_learn, eval_seed, num_eval_demos):
    """
    Evaluate the agent on some number of episodes and return the seeds for the
    episodes the agent performed the worst on.
    """

    logger.info("Evaluating agent on {}".format(il_learn.args.env))

    agent = utils.load_agent(il_learn.env, il_learn.args.model)

    agent.model.eval()
    logs = batch_evaluate(
        agent,
        il_learn.args.env,
        episodes=num_eval_demos,
        seed=eval_seed,
        seed_shift=0
    )
    agent.model.train()

    success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
    logger.info("success rate: {:.2f}".format(success_rate))

    # Find the seeds for all the failing demos
    fail_seeds = []
    for idx, ret in enumerate(logs["return_per_episode"]):
        if ret <= 0:
            fail_seeds.append(logs["seed_per_episode"][idx])

    return success_rate, fail_seeds


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

        #logger.info("demo #{}".format(len(demos)))

    return demos


def grow_training_set(il_learn, train_demos, eval_seed, grow_factor, num_eval_demos):
    """
    Grow the training set of demonstrations by some factor
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

    # Get the status path
    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(il_learn.acmodel)

    train_demos = []

    # Generate the initial set of training demos
    train_demos += generate_demos(args.env, range(args.seed, args.seed + args.start_demos))

    # Seed at which evaluation will begin
    eval_seed = args.seed + args.start_demos

    model_name = args.model

    for phase_no in range(0, args.phases):
        logger.info("Starting phase {} with {} demos".format(phase_no, len(train_demos)))

        if not args.finetune:
            # Create a new model to be trained from scratch
            logging.info("Creating new model to be trained from scratch")
            args.model = model_name + ('_phase_%d' % phase_no)
            il_learn = ImitationLearning(args)

        # Train the imitation learning agent
        il_learn.train(train_demos, writer, csv_writer, status_path, header, reset_status=True)

        # Stopping criterion
        valid_log = il_learn.validate(args.val_episodes)
        success_rate = np.mean([1 if r > 0 else 0 for r in valid_log[0]['return_per_episode']])

        if success_rate >= 0.99:
            logger.info("Reached target success rate with {} demos, stopping".format(len(train_demos)))
            break

        eval_seed = grow_training_set(il_learn, train_demos, eval_seed, args.demo_grow_factor, args.num_eval_demos)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
