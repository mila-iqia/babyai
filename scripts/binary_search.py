#!/usr/bin/env python3

"""
Script to search the minimum number of demonstrations required to achieve a particular level of maximum performance bar (within a given range)
in a binary search manner.
python -m scripts.binary_search --env BabyAI-LevelName-v0 --demos-origin agent --val-episodes 100 --arch cnn1 --min-demo 10 --max-demo 1000 --batch-size <must be smaller than the min-demo>
"""

import numpy as np
import argparse
import csv
import os
from babyai.evaluate import evaluate
import babyai.utils as utils
from babyai.algos.imitation import ImitationLearning
import gym
import babyai
import torch
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED of demos-origin required)")
parser.add_argument("--demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos required)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate (default: 1e-4)")
parser.add_argument("--entropy-coef", type=float, default=0.2,
                    help="entropy term coefficient (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam optimizer epsilon (default: 1e-5)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (In case of memory, the BS is # demos, otherwise, it is # frames)(default: 256)")
parser.add_argument("--no-instr", action="store_true", default=False,
                    help="don't use instructions in the model")
parser.add_argument("--instr-arch", default="gru",
                    help="arch to encode instructions, possible values: gru, conv, bow (default: gru)")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
parser.add_argument("--arch", default='expert_filmcnn',
                    help="image embedding architecture, possible values: cnn1, cnn2, filmcnn (default: expert_filmcnn)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--validation-interval", type=int, default=20,
                    help="number of epochs between two validation checks (default: 20)")
parser.add_argument("--val-episodes", type=int, default=500,
                    help="number of episodes used for validation (default: 500)")
parser.add_argument("--patience", type=int, default=3,
                    help="patience for early stopping (default: 3)")
parser.add_argument("--val-seed", type=int, default=0,
                    help="seed for environment used for validation (default: 0)")
parser.add_argument("--min-demo", type=int, default=50,
                    help="the minimum number of demonstrations to start searching (default: 50)")
parser.add_argument("--max-demo", type=int, default=3000,
                    help="the maximum number of demonstrations to start searching (default: 3000)")
parser.add_argument("--epsilon", type=int, default=0.02,
                    help="tolerable difference between mean rewards")
parser.add_argument("--test-seed", type=int, default=6,
                    help="seed used for testing a model (default: 6)")
parser.add_argument("--test-episodes", type=int, default=1000,
                    help="number of episodes used for testing (default: 1000)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent while evaluating")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs of the sub-IL tasks(default: 1)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log the sub IL tasks into Tensorboard")
parser.add_argument("--csv", action="store_true", default=False,
                    help="log the sub IL tasks in a csv file. The current BS task is logged by default.")
parser.add_argument("--image-dim", type=int, default=128,
                    help="dimensionality of the image embedding")
parser.add_argument("--memory-dim", type=int, default=128,
                    help="dimensionality of the memory LSTM")

args = parser.parse_args()
seeds = [1, 2, 3, 4, 5]

result_dir = os.path.join(utils.storage_dir(), "binary_search_results")
if not (os.path.isdir(result_dir)):
    os.makedirs(result_dir)

csv_path = os.path.join(result_dir, "{}_binary_search.csv").format(args.env)
first_created = not os.path.exists(csv_path)
writer = csv.writer(open(csv_path, 'a', 1))
if first_created:
    writer.writerow(["num_demos", "seed", "model_name", "mean_return_per_episode"])

# Define one logger for everything
logger = utils.get_logger('{}_BinarySearch_{}_{}_{}_{}_{}'.format(args.env,
                                                                  args.arch,
                                                                  args.instr_arch if args.instr_arch else "noinstr",
                                                                  "mem" if not args.no_mem else "nomem",
                                                                  args.min_demo,
                                                                  args.max_demo))

# Log command, availability of CUDA
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))


def run(num_demos, logger, first_run=False):
    results = []
    for seed in seeds:
        # Define model name. No need to add the date suffix to allow binary search to continue training without issues.
        # This assumes that the source of the demos doesn't change from one run to the other, as there is no way
        # to differentiate between different demo sources given their logs.
        instr = args.instr_arch if args.instr_arch else "noinstr"
        mem = "mem" if not args.no_mem else "nomem"
        model_name_parts = {
            'env': args.env,
            'arch': args.arch,
            'instr': instr,
            'mem': mem,
            'seed': seed,
            'num_demos': num_demos}
        args.model = "{env}_IL_{arch}_{instr}_{mem}_seed{seed}_demos{num_demos}".format(**model_name_parts)

        args.episodes = num_demos
        args.seed = seed

        il_learn = ImitationLearning(args)

        # Define logger and Tensorboard writer for sub-IL tasks
        header = (["update", "frames", "FPS", "duration", "entropy", "policy_loss", "train_accuracy"]
                  + ["validation_accuracy", "validation_return", "validation_success_rate"])
        tb_writer = None
        if args.tb:
            from tensorboardX import SummaryWriter
            tb_writer = SummaryWriter(utils.get_log_dir(il_learn.model_name))

        # Define csv writer for sub-IL tasks
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

        # Log model if first time
        if first_run and seed == seeds[0]:
            logger.info(il_learn.acmodel)

        # Log the current step of the loop
        logger.info("\n----\n{} demonstrations. Seed {}.\n".format(num_demos, seed))

        if not args.no_mem:
            il_learn.train(il_learn.train_demos, logger, tb_writer, csv_writer, status_path, header)
        else:
            il_learn.train(il_learn.flat_train_demos, logger, tb_writer, csv_writer, status_path, header)
        logger.info('Training finished. Evaluating the model on {} episodes now.'.format(args.test_episodes))
        env = gym.make(args.env)
        utils.seed(args.test_seed)
        agent = utils.load_agent(args, env)
        env.seed(args.test_seed)

        logs = evaluate(agent, env, args.test_episodes)

        results.append(np.mean(logs["return_per_episode"]))
        logger.info('The mean return per episode is {}'.format(np.mean(logs["return_per_episode"])))
        writer.writerow([num_demos, seed, args.model, str(np.mean(logs["return_per_episode"]))])

    return np.mean(results)


min_demo = args.min_demo
max_demo = args.max_demo

return_min = run(min_demo, logger, first_run=True)
return_max = run(max_demo, logger)

max_performance_bar = return_max

while True:
    assert return_max >= return_min

    if (max_performance_bar - return_min) <= args.epsilon:
        print("Minimum No. of Samples Required = %d" % min_demo)
        break

    if np.log2(max_demo / min_demo) <= 0.5:
        print("Minimum No. of Samples Required = %d" % max_demo)
        print("Ratio : %.3f" % np.log2(max_demo / min_demo))
        break

    mid_demo = (min_demo + max_demo) // 2

    return_mid = run(mid_demo, logger)

    assert return_min <= return_mid <= return_max

    if (max_performance_bar - return_mid) >= args.epsilon:
        min_demo = mid_demo
        return_min = return_mid
    else:
        max_demo = mid_demo
        return_max = return_mid

    max_performance_bar = max(max_performance_bar, return_max)
