#!/usr/bin/env python3

"""
Script to train agent through imitation learning using demonstrations.
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
import torch


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
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
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
parser.add_argument("--patience", type=int, default=100,
                    help="patience for early stopping (default: 100)")
parser.add_argument("--val-seed", type=int, default=0,
                    help="seed for environment used for validation (default: 0)")
parser.add_argument("--image-dim", type=int, default=128,
                    help="dimensionality of the image embedding")
parser.add_argument("--memory-dim", type=int, default=128,
                    help="dimensionality of the memory LSTM")


def main(args):
    logger = utils.get_logger(args.model)

    il_learn = ImitationLearning(args)

    # Define logger and Tensorboard writer
    header = (["update", "frames", "FPS", "duration", "entropy", "policy_loss", "train_accuracy"]
              + ["validation_accuracy", "validation_return", "validation_success_rate"])
    writer = None
    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(utils.get_log_dir(il_learn.model_name))

    # Define csv writer
    csv_writer = None
    if args.csv:
        csv_path = os.path.join(utils.get_log_dir(il_learn.model_name), 'log.csv')
        first_created = not os.path.exists(csv_path)
        # we don't buffer data going in the csv log, cause we assume
        # that one update will take much longer that one write to the log
        csv_writer = csv.writer(open(csv_path, 'a', 1))
        if first_created:
            csv_writer.writerow(header)

    # Get the status path
    status_path = os.path.join(utils.get_log_dir(il_learn.model_name), 'status.json')

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(il_learn.acmodel)

    if not args.no_mem:
        il_learn.train(il_learn.train_demos, writer, csv_writer, status_path, header)
    else:
        il_learn.train(il_learn.flat_train_demos, writer, csv_writer, status_path, header)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
