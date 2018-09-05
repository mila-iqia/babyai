#!/usr/bin/env python3

"""
Uses Teacher Student Curriculum for imitation learning of multiple environments
([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))

Usage:
python3 -m scripts.curriculum_learner --curriculum CurriculumName

Example:
python3 -m scripts.curriculum_learner --curriculum PutNext-PutNextLocal-100000
"""

import numpy as np
import argparse
import csv
import copy
import datetime
import os
import time
import json
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
from babyai.levels import imitation_curriculums
import logging


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate (default: 7e-4)")
parser.add_argument("--curriculum", required=True,
                    help="the curriculum to train using imitation learning")
parser.add_argument("--entropy-coef", type=float, default=0.0,
                    help="entropy term coefficient (default: 0.0)")
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
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
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
parser.add_argument("--dist-cr", default="GreedyProp",
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
parser.add_argument("--image-dim", type=int, default=128,
                    help="dimensionality of the image embedding")
parser.add_argument("--memory-dim", type=int, default=128,
                    help="dimensionality of the memory LSTM")
parser.add_argument("--instr-dim", type=int, default=128,
                    help="dimensionality of the instruction embedder")
parser.add_argument("--csv", action="store_true", default=False,
                    help="log in a csv file")



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
    args.model = args.model or ImitationLearning.default_model_name(args)

    # Define logger
    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    il_learn = ImitationLearning(args)
    utils.save_model(il_learn.acmodel, il_learn.model_name)


    header = (["update", "frames", "FPS", "duration", "entropy", "policy_loss", "train_accuracy"]
              + ["validation_accuracy", "validation_return", "validation_success_rate"])
    if args.csv:

        csv_path = os.path.join(utils.get_log_dir(il_learn.model_name), 'log.csv')
        first_created = not os.path.exists(csv_path)
        # we don't buffer data going in the csv log, cause we assume
        # that one update will take much longer that one write to the log
        csv_writer = csv.writer(open(csv_path, 'a', 1))
        if first_created:
            csv_writer.writerow(header)

    status_path = os.path.join(utils.get_log_dir(il_learn.model_name), 'status.json')
    status = {'i': 0,
              'num_frames': 0,
              'patience' : 0}
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
    # Define logger and Tensorboard writer

    for env in graphs:
        header.append("proba/{}".format(env[0]))
        header.append("return/{}".format(env[0]))

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
    log = {"entropy": [],"policy_loss": [],"accuracy" : []}

    writer = None

    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(utils.get_log_dir(il_learn.model_name))

    while True:
        if status['patience'] > args.patience:
            break

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
            status['i'] += 1
            current_num_evaluate += 1
            for key in log:
                log[key] = np.mean(log[key])


            total_ellapsed_time = int(time.time() - total_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            status['num_frames'] += total_len

            if status['i'] % args.log_interval == 0:
                train_data = [status['i'], status['num_frames'], np.mean(fps), total_ellapsed_time,
                                  log["entropy"], log["policy_loss"], log["accuracy"]]

                logger.info(
                    "U {} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f}"
                        .format(current_num_evaluate, np.mean(fps), duration,
                                log["entropy"], log["policy_loss"], log["accuracy"]))
                if status['i'] % args.validation_interval != 0:
                    validation_data = ['']*len([key for key in header if 'valid' in key])
                    proba_data = ['']*len([key for key in header if 'proba/' in key])
                    return_data = ['']*len([key for key in header if 'return/' in key])
                    assert len(header) == len(train_data + validation_data + proba_data + return_data)
                    if args.tb:
                        for key, value in zip(header, train_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    if args.csv:
                        csv_writer.writerow(train_data + validation_data + proba_data + return_data)
                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)

            if status['i'] % args.validation_interval == 0:

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


                logs = il_learn.validate(episodes = args.val_episodes)
                mean_return = [np.mean(log['return_per_episode']) for log in logs]
                success_rate = [np.mean([1 if r > 0 else 0 for r in log['return_per_episode']]) for log in logs]

                mean_return_all_task = np.mean(list(mean_return))
                success_rate_all_task = np.mean(list(success_rate))
                validation_data = [val_log_all_task['accuracy'], mean_return_all_task, success_rate_all_task]
                rest_data = []
                for index in range(len(graphs)):
                    rest_data.append(prob_log_envs[graphs[index][0]])
                    rest_data.append(mean_return[index])
                logger.info("Validation: A {: .3f} | R {: .3f} | S {: .3f}".format(*validation_data))

                assert len(header) == len(train_data + validation_data + rest_data)
                if args.tb:
                    for key, value in zip(header, train_data + validation_data + rest_data):
                        writer.add_scalar(key, float(value), status['num_frames'])
                if args.csv:
                    csv_writer.writerow(train_data + validation_data + rest_data)


                if mean_return_all_task > best_mean_return:
                    best_mean_return = mean_return_all_task
                    status['patience'] = 0
                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)
                    # Saving the model
                    logger.info("Saving best model")
                    il_learn.obss_preprocessor.vocab.save()
                    if torch.cuda.is_available():
                        il_learn.acmodel.cpu()
                    utils.save_model(il_learn.acmodel, il_learn.model_name)
                    if torch.cuda.is_available():
                        il_learn.acmodel.cuda()
                else:
                    logger.info("Losing Patience")
                    status['patience'] += 1
                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)
                    if torch.cuda.is_available():
                        il_learn.acmodel.cuda()

            fps = []
            total_len = 0
            il_learn.scheduler.step()
            log = {"entropy": [],"policy_loss": [],"accuracy" : []}

if __name__ == "__main__":
    args = parser.parse_args()
    graphs = imitation_curriculums[args.curriculum]
    main(args, graphs)
