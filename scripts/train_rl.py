#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import logging
import csv
import json
import argparse
import gym
import time
import datetime
import torch
import numpy as np
import subprocess

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.model import ACModel
from babyai.levels import curriculums, create_menvs
from babyai.levels.supervised_losses import wrap_env
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--env", default=None,
                    help="name of the environment to train on (REQUIRED or --curriculum REQUIRED)")
parser.add_argument("--curriculum", default=None,
                    help="name of the curriculum to train on (REQUIRED or --env REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: ENV_ALGO_TIME)")
parser.add_argument("--pretrained-model", default=None,
                    help='If you\'re using a pre-trained model and want the fine-tuned one to have a new name')
parser.add_argument("--seed", type=int, default=1,
                    help="random seed; if 0, a random random seed will be used  (default: 1)")
parser.add_argument("--task-id-seed", action='store_true',
                    help="use the task id within a Slurm job array as the seed")
parser.add_argument("--procs", type=int, default=64,
                    help="number of processes (default: 64)")
parser.add_argument("--frames", type=int, default=int(5e8),
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=10,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=1000,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--frames-per-proc", type=int, default=40,
                    help="number of frames per process before update (default: 40)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate (default: 1e-4)")
parser.add_argument("--beta1", type=float, default=0.9,
                    help="beta1 for Adam (default: 0.9)")
parser.add_argument("--beta2", type=float, default=0.999,
                    help="beta2 for Adam (default: 0.999)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--recurrence", type=int, default=20,
                    help="number of timesteps gradient is backpropagated (default: 20)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=1280,
                    help="batch size for PPO (default: 1280)")
parser.add_argument("--image-dim", type=int, default=128,
                    help="dimensionality of the image embedding")
parser.add_argument("--memory-dim", type=int, default=128,
                    help="dimensionality of the memory LSTM")
parser.add_argument("--instr-dim", type=int, default=128,
                    help="dimensionality of the memory LSTM")
parser.add_argument("--no-instr", action="store_true", default=False,
                    help="don't use instructions in the model")
parser.add_argument("--instr-arch", default="gru",
                    help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
parser.add_argument("--arch", default='expert_filmcnn',
                    help="image embedding architecture")
parser.add_argument("--aux-loss", nargs='*', default=[],
                    help="List of extra information that the environment yields at each step"
                         "The agent tries to learn that using a supervised loss. If not specified, no info is used"
                         "Possible infos: seen_state, see_door, see_obj, in_front_of_what, visit_proportion, "
                         "obj_in_instr, bot_action")
parser.add_argument("--aux-loss-coef", nargs='*', type=float, default=[],
                    help="Coefficients for the auxiliary supervised loss. There should be as many as extra infos"
                         "If not specified, they will all be set to 1")
parser.add_argument("--test-seed", type=int, default=0,
                    help="random seed for testing (default: 0)")
parser.add_argument("--test-episodes", type=int, default=200,
                    help="Number of episodes to use for testing (default: 200)")

args = parser.parse_args()

assert args.env is not None or args.curriculum is not None, "--env or --curriculum must be specified."

if len(args.aux_loss_coef) == 0:
    args.aux_loss_coef = [1.] * len(args.aux_loss)
assert len(args.aux_loss) == len(args.aux_loss_coef)

# Set seed for all randomness sources

if args.seed == 0:
    args.seed = np.random.randint(10000)
if args.task_id_seed:
    args.seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print('set seed to {}'.format(args.seed))

utils.seed(args.seed)

# Generate environments

if args.env is not None:
    envs = []
    for i in range(args.procs):
        env = gym.make(args.env)
        if args.aux_loss:
            env = wrap_env(env, args.aux_loss)
        env.seed(100 * args.seed + i)
        envs.append(env)
else:
    curriculum = curriculums[args.curriculum]
    menv_head, envs = create_menvs(curriculum, args.procs, args.seed)

# Define model name

suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
instr = args.instr_arch if args.instr_arch else "noinstr"
mem = "mem" if not args.no_mem else "nomem"
model_name_parts = {
    'env': args.env or args.curriculum,
    'algo': args.algo,
    'arch': args.arch,
    'instr': instr,
    'mem': mem,
    'seed': args.seed,
    'info': '',
    'coef': '',
    'suffix': suffix}
if len(args.aux_loss) > 0:
    model_name_parts['info'] = '_' + ''.join([info[0].upper() for info in args.aux_loss])
    model_name_parts['coef'] = '_' + '-'.join(map(str, args.aux_loss_coef))
default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)
if args.pretrained_model:
    default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
args.model = args.model.format(**model_name_parts) if args.model else default_model_name

utils.configure_logging(args.model)
logger = logging.getLogger(__name__)

# Define obss preprocessor
if 'emb' in args.arch:
    obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
else:
    obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)

# Define actor-critic model
acmodel = utils.load_model(args.model, raise_not_found=False)
if acmodel is None:
    if args.pretrained_model:
        acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
    else:
        acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                          args.image_dim, args.memory_dim, args.instr_dim,
                          not args.no_instr, args.instr_arch, not args.no_mem, args.arch, args.aux_loss)

obss_preprocessor.vocab.save()
utils.save_model(acmodel, args.model)

if torch.cuda.is_available():
    acmodel.cuda()
if len(args.aux_loss) > 0:
    acmodel.add_extra_heads_if_necessary(args.aux_loss)

# Define actor-critic algo

reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
if args.algo == "ppo":
    algo = babyai.rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                             args.gae_lambda,
                             args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                             args.optim_eps, args.clip_eps, args.epochs, args.batch_size, obss_preprocessor,
                             reshape_reward, args.aux_loss, args.aux_loss_coef)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
# Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
# the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

utils.seed(args.seed)

# Restore training status

status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {'i': 0,
              'num_episodes': 0,
              'num_frames': 0}

# Define logger and Tensorboard writer and CSV writer

header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
if args.aux_loss:
    header += ["supervised_loss", "supervised_accuracy", "supervised_L2_loss", "supervised_prevalence"]
if args.curriculum is not None:
    for env_key in curriculum:
        header.append("proba/{}".format(env_key))
        header.append("return/{}".format(env_key))
if args.tb:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(utils.get_log_dir(args.model))
csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
first_created = not os.path.exists(csv_path)
# we don't buffer data going in the csv log, cause we assume
# that one update will take much longer that one write to the log
csv_writer = csv.writer(open(csv_path, 'a', 1))
if first_created:
    csv_writer.writerow(header)

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

total_start_time = time.time()
best_success_rate = 0
test_env_name = args.env if args.env is not None else curriculum[-1]
while status['num_frames'] < args.frames:
    # Update parameters

    update_start_time = time.time()
    logs = algo.update_parameters()
    if args.curriculum is not None:
        menv_head.update_dist()
    update_end_time = time.time()

    status['num_frames'] += logs["num_frames"]
    status['num_episodes'] += logs['episodes_done']
    status['i'] += 1

    # Print logs

    if status['i'] % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        data = [status['i'], status['num_episodes'], status['num_frames'],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                success_per_episode['mean'],
                *num_frames_per_episode.values(),
                logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"]]

        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")
        if args.aux_loss:
            data += [logs["supervised_loss"], logs["supervised_accuracy"], logs["supervised_L2_loss"],
                     logs["supervised_prevalence"]]
            format_str += "sL {: .3f} | sA {:.3f} | sL2 {: .3f} | sP {: .3f} | "
        if args.curriculum is not None:
            for env_id, _ in enumerate(curriculum):
                data.append(menv_head.dist[env_id])
                data.append(menv_head.synthesized_returns.get(env_id, np.NaN))
                format_str += "pr{} {:.2f} | ".format(env_id, menv_head.dist[env_id])
                format_str += "R{} {:.2f} | ".format(env_id, menv_head.synthesized_returns.get(env_id, np.NaN))

        logger.info(format_str.format(*data))
        if args.tb:
            assert len(header) == len(data)
            for key, value in zip(header, data):
                writer.add_scalar(key, float(value), status['num_frames'])

            # TODO: CSV logging for curriculum
            if args.curriculum is not None:
                for env_id, env_key in enumerate(curriculum):
                    writer.add_scalar("proba/{}".format(env_key),
                                      menv_head.dist[env_id], status['num_frames'])
                    if env_id in menv_head.synthesized_returns.keys():
                        writer.add_scalar("return/{}".format(env_key),
                                          menv_head.synthesized_returns[env_id], status['num_frames'])
        csv_writer.writerow(data)

    # Save obss preprocessor vocabulary and model

    if args.save_interval > 0 and status['i'] % args.save_interval == 0:
        obss_preprocessor.vocab.save()
        with open(status_path, 'w') as dst:
            json.dump(status, dst)
            utils.save_model(acmodel, args.model)

        # Testing the model before saving
        agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
        agent.model = acmodel
        agent.model.eval()
        logs = batch_evaluate(agent, test_env_name, args.test_seed, args.test_episodes)
        agent.model.train()
        mean_return = np.mean(logs["return_per_episode"])
        success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            utils.save_model(acmodel, args.model + '_best')
            logger.info("Return {: .2f}; best model is saved".format(mean_return))
        else:
            logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))
