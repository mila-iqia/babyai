#!/usr/bin/env python3
# TODO add support for recurrent policy

import argparse
import gym
from babyai import levels
import time
import datetime
import numpy
import sys
import torch
import torch.nn.functional as F
import torch_rl

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: ENV_ORIGIN_il)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of demonstrations to use (default: 100)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate (default: 7e-4)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam optimizer epsilon (default: 1e-5)")
parser.add_argument("--epochs", type=int, default=10,
                    help="number of epochs (default: 10)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (default: 256)")
parser.add_argument("--model-instr", action="store_true", default=False,
                    help="use instructions in the model")
parser.add_argument("--model-mem", action="store_true", default=False,
                    help="use memory in the model")
parser.add_argument("--arch", default='cnn1',
                    help="image embedding architecture")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)

# Load demonstrations

demos = utils.load_demos(args.env, args.demos_origin)[:args.episodes]
flat_demos = []
for demo in demos:
    flat_demos += demo

# Define model name

default_model_name = "{}_{}_il".format(args.env, args.demos_origin)
model_name = args.model or default_model_name
print("The model is saved in {}".format(model_name))

# Define obss preprocessor

obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)

# Define actor-critic model

acmodel = utils.load_model(obss_preprocessor.obs_space, env.action_space, model_name)
if torch.cuda.is_available():
    acmodel.cuda()

# Define optimizer

optimizer = torch.optim.Adam(acmodel.parameters(), args.lr, eps=args.optim_eps)

# Define logger and Tensorboard writer

logger = utils.Logger(model_name)
if args.tb:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(utils.get_log_path(model_name, ext=False))

# Log command, availability of CUDA, and model

logger.log(args, to_print=False)
logger.log("CUDA is {}available".format('' if torch.cuda.is_available() else 'not '))
logger.log(acmodel)

# Prepare experiences

obs, action, reward, done = zip(*flat_demos)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exps = torch_rl.DictList({
    "obs": obss_preprocessor(obs, device=device),
    "action": torch.tensor(action, device=device),
    "reward": torch.tensor(reward, device=device),
    "mask": 1 - torch.tensor(done, device=device, dtype=torch.float)
})

# Train model

total_start_time = time.time()

batch_size = len(flat_demos) if args.batch_size == 0 else args.batch_size

for i in range(1, args.epochs + 1):
    # Update parameters

    numpy.random.shuffle(flat_demos)

    log_entropies = []
    log_policy_losses = []

    update_start_time = time.time()

    for j in range(0, len(flat_demos), batch_size):
        b = exps[j:j + batch_size]

        # Compute loss

        dist, _ = acmodel(b.obs)

        entropy = dist.entropy().mean()

        policy_loss = -dist.log_prob(b.action).mean()

        loss = policy_loss - args.entropy_coef * entropy

        # Update actor-critic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log some values

        log_entropies.append(entropy.data[0])
        log_policy_losses.append(policy_loss.data[0])
    
    update_end_time = time.time()

    # Print logs

    if i % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = len(flat_demos)/(update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)

        log_entropy = numpy.mean(log_entropies)
        log_policy_loss = numpy.mean(log_policy_losses)

        logger.log(
            "U {} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f}"
                .format(i, fps, duration,
                        log_entropy, log_policy_loss))
    
        if args.tb:
            writer.add_scalar("FPS", fps, i)
            writer.add_scalar("duration", total_ellapsed_time, i)
            writer.add_scalar("entropy", log_entropy, i)
            writer.add_scalar("policy_loss", log_policy_loss, i)

    # Save obss preprocessor vocabulary and model

    if args.save_interval > 0 and i % args.save_interval == 0:
        obss_preprocessor.vocab.save()

        if torch.cuda.is_available():
            acmodel.cpu()
        utils.save_model(acmodel, model_name)
        if torch.cuda.is_available():
            acmodel.cuda()