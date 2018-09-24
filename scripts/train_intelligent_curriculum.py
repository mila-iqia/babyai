#!/usr/bin/env python3

"""
Train an agent using an intelligent expert

scripts/train_intelligent_expert.py --env BabyAI-GoToObj-v0 --demos GoToObj-bot-100k --validation-interval 5

Vanilla imitation learning:
GoToObj, 1000 demos for 100 percent success rate
GoToLocal, over 60K demos needed
"""

import os
import argparse
import csv
import copy
import gym
import time
import datetime
import numpy as np
import sys
import logging
import babyai.utils as utils
from babyai.algos.imitation import ImitationLearning
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import BotAgent
import torch
import blosc

from .train_intelligent_expert import evaluate_agent, generate_demos, grow_training_set


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or demos-origin required)")
parser.add_argument("--demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos required)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: ENV_ORIGIN_il)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--episodes", type=int, default=0,
                    help="number of episodes of demonstrations to use"
                         "(default: 0, meaning all demos)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--csv", action="store_true", default=False,
                    help="log in a csv file")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate (default: 1e-4)")
parser.add_argument("--entropy-coef", type=float, default=0.0,
                    help="entropy term coefficient")
parser.add_argument("--recurrence", type=int, default=20,
                    help="number of timesteps gradient is backpropagated (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam optimizer epsilon (default: 1e-5)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (In case of memory, the batch size is the number of demos, otherwise, it is the number of frames)(default: 10)")
parser.add_argument("--no-instr", action="store_true", default=False,
                    help="don't use instructions in the model")
parser.add_argument("--instr-arch", default="gru",
                    help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
parser.add_argument("--arch", default='expert_filmcnn',
                    help="image embedding architecture, possible values: cnn1, cnn2, filmcnn (default: cnn1)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--validation-interval", type=int, default=1,
                    help="number of epochs between two validation checks (default: 1)")
parser.add_argument("--val-episodes", type=int, default=500,
                    help="number of episodes used to evaluate the agent, and to evaluate validation accuracy")
parser.add_argument("--val-seed", type=int, default=0,
                    help="seed for environment used for validation (default: 0)")
parser.add_argument("--image-dim", type=int, default=128,
                    help="dimensionality of the image embedding")
parser.add_argument("--instr-dim", type=int, default=128,
                    help="dimensionality of the image embedding")
parser.add_argument("--memory-dim", type=int, default=128,
                    help="dimensionality of the memory LSTM")

# TODO: adjust values. These are only small for testing purposes.
parser.add_argument("--patience", type=int, default=10,
                    help="patience for early stopping")
parser.add_argument("--start-demos", type=int, default=5000,
                    help="the starting number of demonstrations")
parser.add_argument("--demo-grow-factor", type=float, default=1.2,
                    help="number of demos to add to the training set")

logger = logging.getLogger(__name__)

env_names = [
    'BabyAI-GoToLocalS8N2-v0',
    'BabyAI-GoToLocalS8N3-v0',
    'BabyAI-GoToLocalS8N4-v0',
    'BabyAI-GoToLocalS8N5-v0',
    'BabyAI-GoToLocalS8N6-v0',
    'BabyAI-GoToLocalS8N7-v0',
    'BabyAI-GoToLocal-v0',
]

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

    # Ignore the training demos that were loaded, we will generate new ones
    train_demos = []

    demo_seed = args.seed

    for env_name in env_names:
        il_learn.args.env = env_name
        il_learn.env = gym.make(env_name)

        logger.info("Generating initial demos for {}".format(env_name))

        # FIXME: sample only failing episodes here too?
        # Generate the initial set of training demos
        train_demos += generate_demos(env_name, range(demo_seed, demo_seed + args.start_demos))
        demo_seed += args.start_demos

        while True:
            logger.info("Training on {} with {} demos".format(env_name, len(train_demos)))

            # Train the imitation learning agent
            il_learn.train(train_demos, writer, csv_writer, status_path, header, reset_status=True)

            valid_log = il_learn.validate(args.val_episodes)
            success_rate = np.mean([1 if r > 0 else 0 for r in valid_log['return_per_episode']])
            logger.info("success rate: {}".format(success_rate))

            # Stop at 95% for intermediate steps, 99% for the last step
            if success_rate >= 0.99 or (success_rate >= 0.95 and env_name != env_names[-1]):
                logger.info("Reached target success rate for curriculum step with {} demos".format(len(train_demos)))
                break

            demo_seed = grow_training_set(il_learn, train_demos, args.demo_grow_factor, demo_seed)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
