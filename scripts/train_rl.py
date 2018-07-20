#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import argparse
import gym
import time
import datetime
import torch
import torch_rl
import numpy as np
import subprocess

import babyai
import babyai.utils as utils
from babyai.model import ACModel
from babyai.levels import curriculums, create_menvs
from babyai.evaluate import evaluate
from babyai.utils.agent import ModelAgent

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use: a2c | ppo (default: ppo)")
parser.add_argument("--env", default=None,
                    help="name of the environment to train on (REQUIRED or --curriculum REQUIRED)")
parser.add_argument("--curriculum", default=None,
                    help="name of the curriculum to train on (REQUIRED or --env REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: ENV_ALGO_TIME)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate (default: 7e-4)")
parser.add_argument("--gae-tau", type=float, default=0.95,
                    help="tau coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--no-instr", action="store_true", default=False,
                    help="don't use instructions in the model")
parser.add_argument("--instr-arch", default="gru",
                    help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
parser.add_argument("--arch", default='cnn1',
                    help="image embedding architecture")
parser.add_argument("--test-seed", type=int, default=0,
                    help="random seed for testing (default: 0)")
parser.add_argument("--test-episodes", type=int, default=200,
                    help="Number of episodes to use for testing (default: 200)")

args = parser.parse_args()

assert args.env is not None or args.curriculum is not None, "--env or --curriculum must be specified."

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environments

if args.env is not None:
    envs = []
    for i in range(args.procs):
        env = gym.make(args.env)
        env.seed(args.seed + i)
        envs.append(env)
else:
    curriculum = curriculums[args.curriculum]
    menv_head, envs = create_menvs(curriculum, args.procs, args.seed)

# Define model name

suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
instr = args.instr_arch if args.instr_arch else "noinstr"
mem = "mem" if not args.no_mem else "nomem"
default_model_name = "{}_{}_{}_{}_{}_seed{}_{}".format(args.env or args.curriculum,
                                                       args.algo,
                                                       args.arch,
                                                       instr,
                                                       mem,
                                                       args.seed,
                                                       suffix)
model_name = args.model or default_model_name

# Define obss preprocessor
if 'emb' in args.arch:
    obss_preprocessor = utils.IntObssPreprocessor(model_name, envs[0].observation_space)
else:
    obss_preprocessor = utils.ObssPreprocessor(model_name, envs[0].observation_space)

# Define actor-critic model

acmodel = utils.load_model(model_name, raise_not_found=False)
if acmodel is None:
    acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                      not args.no_instr, args.instr_arch, not args.no_mem, args.arch)
    utils.save_model(acmodel, model_name)
if torch.cuda.is_available():
    acmodel.cuda()


# Define actor-critic algo

if args.algo == "a2c":
    algo = torch_rl.A2CAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_tau,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, obss_preprocessor, utils.reshape_reward)
elif args.algo == "ppo":
    algo = torch_rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_tau,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, obss_preprocessor,
                            utils.reshape_reward)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# Define logger and Tensorboard writer

logger = utils.get_logger(model_name)
if args.tb:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(utils.get_log_dir(model_name))

# Log code state, command, availability of CUDA and model

babyai_code = list(babyai.__path__)[0]
try:
    last_commit = subprocess.check_output(
        'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
    logger.info('LAST COMMIT INFO:')
    logger.info(last_commit)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
try:
    diff = subprocess.check_output(
        'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
    if diff:
        logger.info('GIT DIFF:')
        logger.info(diff)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
logger.info('COMMAND LINE ARGS:')
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
logger.info(acmodel)

# Train model

num_frames = 0
total_start_time = time.time()
i = 0
best_mean_return = 0
test_env_name = args.env if args.env is not None else curriculum[-1]
test_env = gym.make(test_env_name)
while num_frames < args.frames:
    # Update parameters

    update_start_time = time.time()
    logs = algo.update_parameters()
    if args.curriculum is not None:
        menv_head.update_dist()
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    i += 1

    # Print logs

    if i % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:x̄σmM {: .2f} {: .2f} {: .2f} {: .2f} | F:x̄σmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {: .3f} | vL {:.3f}"
            .format(i, num_frames, fps, duration,
                    *rreturn_per_episode.values(),
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"]))
        if args.tb:
            writer.add_scalar("frames", num_frames, i)
            writer.add_scalar("FPS", fps, i)
            writer.add_scalar("duration", total_ellapsed_time, i)
            for key, value in return_per_episode.items():
                writer.add_scalar("return_" + key, value, num_frames)
            for key, value in rreturn_per_episode.items():
                writer.add_scalar("rreturn_" + key, value, num_frames)
            for key, value in num_frames_per_episode.items():
                writer.add_scalar("num_frames_" + key, value, i)
            writer.add_scalar("entropy", logs["entropy"], num_frames)
            writer.add_scalar("value", logs["value"], num_frames)
            writer.add_scalar("policy_loss", logs["policy_loss"], num_frames)
            writer.add_scalar("value_loss", logs["value_loss"], num_frames)

            if args.curriculum is not None:
                for env_id, env_key in enumerate(curriculum):
                    writer.add_scalar("proba/{}".format(env_key),
                                      menv_head.dist[env_id], num_frames)
                    if env_id in menv_head.synthesized_returns.keys():
                        writer.add_scalar("return/{}".format(env_key),
                                          menv_head.synthesized_returns[env_id], num_frames)

    # Save obss preprocessor vocabulary and model

    if args.save_interval > 0 and i % args.save_interval == 0:
        obss_preprocessor.vocab.save()
        if torch.cuda.is_available():
            acmodel.cpu()

        # Testing the model before saving
        test_env.seed(args.test_seed)
        agent = ModelAgent(model_name, obss_preprocessor, argmax=True)
        agent.model = acmodel
        logs = evaluate(agent, test_env, args.test_episodes)
        mean_return = np.mean(logs["return_per_episode"])
        if mean_return > best_mean_return:
            best_mean_return = mean_return
            utils.save_model(acmodel, model_name)



        logger.info("Model is saved.")
        if torch.cuda.is_available():
            acmodel.cuda()
