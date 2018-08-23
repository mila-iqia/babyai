#!/usr/bin/env python3

"""
Uses Teacher Student Curriculum for imitation learning of multiple environments
([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))
"""

import numpy as np
import argparse
import csv
import copy
import datetime
import time
from collections import defaultdict
from babyai.algos.imitation import ImitationLearning
import babyai.utils as utils
import torch
import torch.nn.functional as F
from babyai.multienv.lp_computer import *
from babyai.multienv.dist_creator import *
from babyai.multienv.dist_computer import *
from babyai.multienv.return_history import *
from babyai.batchsampler import BatchSampler

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate (default: 7e-4)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--recurrence", type=int, default=20,
                    help="number of timesteps gradient is backpropagated (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam optimizer epsilon (default: 1e-5)")
parser.add_argument("--batch-size", type=int, default=50,
                    help="batch size (In case of memory, the batch size is the number of demos, otherwise, it is the number of frames)(default: 50)")
parser.add_argument("--no-instr", action="store_true", default=False,
                    help="don't use instructions in the model")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
parser.add_argument("--arch", default='expert_filmcnn',
                    help="image embedding architecture")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--validation-interval", type=int, default=20,
                    help="number of epochs between two validation checks (default: 20)")
parser.add_argument("--patience", type=int, default=3,
                    help="patience for early stopping (default: 3)")
parser.add_argument("--val-seed", type=int, default=0,
                    help="seed for environment used for validation (default: 0)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: ENV_ORIGIN_il)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--val-episodes", type=int, default=1000,
                    help="number of episodes used for validation (default: 1000)")
parser.add_argument("--eval-episodes", type=int, default=50,
                    help="number of episodes used for evaluation for reassigning task distribution (default: 50)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--instr-arch", default="gru",
                    help="arch to encode instructions, possible values: gru, conv, bow (default: gru)")
parser.add_argument("--dist-cp", default="Lp",
                    help="name of the distribution computer (default: Lp)")
parser.add_argument("--lp-cp", default="Online",
                    help="name of the learning progress computer (default: Online), Window, AbsWindow, Online, AbsOnline")
parser.add_argument("--dist-cr", default="GreedyAmax",
                    help="name of the distribution creator (default: GreedyAmax), ClippedProp, Boltzmann, GreedyAmax")
parser.add_argument("--dist-alpha", type=float, default=0.1,
                    help="learning rate for TS learning progress computers (default: 0.2)")
parser.add_argument("--dist-K", type=int, default=10,
                    help="window size for some learning progress computers (default: 10)")
parser.add_argument("--dist-eps", type=float, default=0.1,
                    help="exploration coefficient for some distribution creators (default: 0.1)")
parser.add_argument("--dist-tau", type=float, default=4e-4,
                    help="temperature for Boltzmann distribution creator (default: 4e-4)")
parser.add_argument("--pot-coeff", type=float, default=0.1,
                    help="potential term coefficient in energy (default: 0.1)")
parser.add_argument("--batchSampler-seed", type=int, default=0,
                    help="seed for batchSampler used for sampling batches (default: 10)")
parser.add_argument("--return-interval", type=int, default=10,
                    help="number of batches to collect the returns to update the task distribution (default: 10)")
parser.add_argument("--num-procs", type=int, default=None,
                    help="number of processes to use to collect returns (default: None)")



def main(args, graphs):
    num_envs = len(graphs)

    return_hists = [ReturnHistory() for _ in range(num_envs)]
    compute_lp = {
        "Online": OnlineLpComputer(return_hists, args.dist_alpha),
        "Window": WindowLpComputer(return_hists, args.dist_alpha, args.dist_K),
        "Linreg": LinregLpComputer(return_hists, args.dist_K),
        "None": None
    }[args.lp_cp]

    # Instantiate the distribution creator
    create_dist = {
        "GreedyAmax": GreedyAmaxDistCreator(args.dist_eps),
        "GreedyProp": GreedyPropDistCreator(args.dist_eps),
        "Boltzmann": BoltzmannDistCreator(args.dist_tau),
        "None": None
    }[args.dist_cr]

    # Instantiate the distribution computer
    compute_dist = {
        "Lp": LpDistComputer(return_hists, compute_lp, create_dist),
        "None": None
    }[args.dist_cp]

    args.env = graphs

    il_learn = ImitationLearning(args)
    utils.save_model(il_learn.acmodel, il_learn.model_name)

    # Define logger
    logger = utils.get_logger(il_learn.model_name)

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(il_learn.acmodel)

    if args.no_mem:
        sampler = BatchSampler(il_learn.flat_train_demos, args.batch_size, args.batchSampler_seed, no_mem=True)
    else:
        sampler = BatchSampler(il_learn.train_demos, args.batch_size, args.batchSampler_seed, no_mem=False)

    total_start_time = time.time()

    best_mean_return, patience = 0, 0
    current_num_evaluate, total_len, current_number_batch, fps = 0, 1, 0, []

    # Log dictionary
    log = {"entropy": [],"value_loss": [],"policy_loss": [],"accuracy" : []}

    writer = None

    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(utils.get_log_dir(il_learn.model_name))

    while True:
        current_batch, should_evaluate = sampler.sample()

        update_start_time = time.time()

        if not(args.no_mem):
            _log = il_learn.run_epoch_recurrence_one_batch(current_batch, is_training=True)
            total_len += sum([len(item) for item in current_batch])
        else:
            _log = il_learn.run_epoch_norecur_one_batch(current_batch, is_training=True)
            total_len += len(current_batch)

        update_end_time = time.time()
        fps.append(total_len/(update_end_time - update_start_time))

        current_number_batch += 1

        # Evaluating the performance, and recalculating task distribution
        if current_number_batch % args.return_interval == 0:
            current_returns = il_learn.collect_returns()
            if compute_dist is not None:
                current_dist_task = compute_dist(current_returns)
                sampler.setDist(current_dist_task)

        log["entropy"].append(_log["entropy"])
        log["policy_loss"].append(_log["policy_loss"])
        log["accuracy"].append(_log["accuracy"])

        if should_evaluate:
            current_num_evaluate += 1

            for key in log:
                log[key] = np.mean(log[key])

            total_ellapsed_time = int(time.time() - total_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)

            logger.info(
                "U {} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f}"
                    .format(current_num_evaluate, np.mean(fps), duration,
                            log["entropy"], log["policy_loss"], log["accuracy"]))

            val_logs = []

            if not(args.no_mem):
                for val_demos in il_learn.val_demos:
                    val_logs += [il_learn.run_epoch_recurrence(val_demos)]
            else:
                for flat_val_demos in il_learn.flat_val_demos:
                    val_logs += [il_learn.run_epoch_norecur(flat_val_demos)]

            val_log_all_task = defaultdict(list)
            for val_log in val_logs:
                for key in val_log:
                    val_log[key] = np.mean(val_log[key])
                    val_log_all_task[key].append(val_log[key])

            val_log_envs, prob_log_envs = {}, {}
            for item in range(num_envs):
                val_log_envs[graphs[item][0]] = val_log_all_task["accuracy"][item]
                prob_log_envs[graphs[item][0]] = sampler.dist_task[item]


            for key in val_log_all_task:
                val_log_all_task[key] = np.mean(val_log_all_task[key])

            if args.tb:
                writer.add_scalar("FPS", np.mean(fps), current_num_evaluate)
                writer.add_scalar("duration", total_ellapsed_time, current_num_evaluate)
                writer.add_scalar("entropy", log["entropy"], current_num_evaluate)
                writer.add_scalar("policy_loss", log["policy_loss"], current_num_evaluate)
                writer.add_scalar("accuracy", log["accuracy"], current_num_evaluate)
                for key in val_log_envs:
                    writer.add_scalar("{}_{}".format(key,"val_accuracy"), val_log_envs[key], current_num_evaluate)
                    writer.add_scalar("{}_{}".format(key,"prob"), prob_log_envs[key], current_num_evaluate)

            if current_num_evaluate % args.validation_interval == 0:
                if torch.cuda.is_available():
                    il_learn.acmodel.cpu()
                mean_return = il_learn.validate(episodes = args.val_episodes, validating=True)
                if args.tb:
                    for item in range(num_envs):
                        writer.add_scalar("{}_{}".format(graphs[item][0],"return"), mean_return[item], current_num_evaluate)

                mean_return = np.mean(list(mean_return.values()))
                print("Mean Validation Return %.3f" % mean_return)

                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    patience = 0
                    # Saving the model
                    print("Saving best model")
                    il_learn.obss_preprocessor.vocab.save()
                    if torch.cuda.is_available():
                        il_learn.acmodel.cpu()
                    utils.save_model(il_learn.acmodel, il_learn.model_name)
                    if torch.cuda.is_available():
                        il_learn.acmodel.cuda()
                else:
                    print("Losing Patience")
                    patience += 1
                    if patience > args.patience:
                        break
                    if torch.cuda.is_available():
                        il_learn.acmodel.cuda()

            fps = []
            total_len = 0
            il_learn.scheduler.step()
            log = {"entropy": [],"value_loss": [],"policy_loss": [],"accuracy" : []}

if __name__ == "__main__":
    args = parser.parse_args()
    graphs = [
        ("BabyAI-GoToObj-v0", 'GoToObj-bot-100k.pkl','bot', 200),
        ("BabyAI-GoToLocal-v0", 'GoToLocal-bot-100k.pkl','bot', 500)
        ]
    main(args, graphs)