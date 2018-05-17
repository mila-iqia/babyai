#!/usr/bin/env python3
# TODO the whole file has to be changed - it's an old version right now

import argparse
import gym
import gym_minigrid
import levels
import time
import datetime
import numpy as np
import sys
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torch_rl
import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the pre-trained model (default: ENV_il)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--demonstrations", type=int, default=100,
                    help="number of demonstrations to use (default: 100)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="interval between log display (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="interval between model sactionaving (default: 0, 0 means no saving)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate (default: 7e-4)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam optimizer epsilon (default: 1e-5)")
parser.add_argument("--epochs", type=int, default=10,
                    help="number of epochs (default: 10)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (default: 256, 0 means all)")
args = parser.parse_args()

# Set numpy and pytorch seeds

torch_ac.seed(args.seed)

# Generate environments

env = gym.make(args.env)

# Load demonstrations

demos = utils.load_demos(args.env)[:args.demonstrations]
flat_demos = []
for demo in demos:
    flat_demos += demo

# Define model name

model_name = args.model or args.env + "_" + "il"

# Define obss preprocessor

obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)

# Define actor-critic model

acmodel = utils.load_model(obss_preprocessor.obs_space, env.action_space, model_name)
if torch_ac.gpu_available:
    acmodel.cuda()

# Define optimizer

optimizer = torch.optim.Adam(acmodel.parameters(), args.lr, eps=args.optim_eps)

# Define loss

loss = torch.nn.CrossEntropyLoss()

# Define logger and log command and model

suffix = datetime.datetime.now().strftime("%y%m%d%H%M%S")
logger = utils.Logger(model_name + "_" + suffix)
logger.log(" ".join(sys.argv), to_print=False)
logger.log(acmodel)

# Train model

total_start_time = time.time()

batch_size = len(flat_demos) if args.batch_size == 0 else args.batch_size

for i in range(1, args.epochs + 1):
    # Update parameters

    random.shuffle(flat_demos)

    obs, action = zip(*flat_demos)
    action = torch.from_numpy(np.array(action)).unsqueeze(1)
    if torch_ac.gpu_available:
        action = action.cuda()

    log_entropies = []
    log_action_losses = []

    for j in range(0, len(flat_demos), batch_size):
        b_obs = obs[j:j + batch_size]
        b_action = action[j:j + batch_size]

        # Compute loss

        preprocessed_obs = obss_preprocessor(b_obs, volatile=False, use_gpu=torch_ac.gpu_available)
        rdist = acmodel.get_rdist(preprocessed_obs)

        log_dist = F.log_softmax(rdist, dim=1)
        dist = F.softmax(rdist, dim=1)
        entropy = -(log_dist * dist).sum(dim=1).mean()

        action_loss = -log_dist.gather(1, Variable(b_action)).mean()

        loss = action_loss - args.entropy_coef * entropy

        # Update actor-critic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log some values

        log_entropies.append(entropy.data[0])
        log_action_losses.append(action_loss.data[0])

    # Print logs

    if i % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)

        logger.log(
            "U {} | D {} | H {:.3f} | aL {: .3f}"
                .format(i, datetime.timedelta(seconds=total_ellapsed_time),
                        np.mean(log_entropies), np.mean(log_action_losses)))

    # Save model

    if args.save_interval > 0 and i % args.save_interval == 0:
        if torch_ac.gpu_available:
            acmodel.cpu()
        utils.save_model(acmodel, model_name)
        if torch_ac.gpu_available:
            acmodel.cuda()