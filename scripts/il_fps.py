#!/usr/bin/env python3
import argparse
import pandas
import os
import json
import re
import numpy as np

from babyai import plotting
from babyai.plotting import model_num_samples


parser = argparse.ArgumentParser("Analyze data efficiency of imitation learning")
parser.add_argument('--path', default='.')
parser.add_argument("--regex", default='.*')
parser.add_argument("--patience", default=2, type=int)
parser.add_argument("--window", default=10, type=int)
parser.add_argument("--limit", default="frames")
args = parser.parse_args()



def best_within_normal_time_mutilated(df, regex, patience, limit='epochs', window=1, normal_time=None, summary_path=None):
    """
    Compute the best success rate that is achieved in all runs within the normal time.

    The normal time is defined as `patience * T`, where `T` is the time it takes for the run
    with the most demonstrations to converge. `window` is the size of the sliding window that is
    used for smoothing.

    Returns a dataframe with the best success rate for the runs that match `regex`.

    """
    models = [model for model in df['model'].unique() if re.match(regex, model)]
    num_samples = [model_num_samples(model) for model in models]
    print(len(num_samples))
    # sort models according to the number of samples
    models, num_samples = zip(*sorted(list(zip(models, num_samples)), key=lambda tupl: tupl[1]))

    maxes = []
    for model, num in zip(models, num_samples):
        df_model = df[df['model'] == model]
        fps = df_model['FPS']
        # success_rate = df_model['validation_success_rate'].rolling(window, center=True).mean()
        # print(success_rate.tolist())
        # success_rate = success_rate[np.logical_not(np.isnan(success_rate))]
        maxes.append(max(fps))# success_rate))
    return np.array(maxes)





# levels = ['GoTo']
# archs = ['expert_filmcnn', 'expert_filmcnn_endpool_res_not_conv_bow']
#
# all_results = []
# for level in levels:
#     arch_results = []
#     for arch in archs:
#         results = ''
#         path = f'../beluga/GoTo/{arch}/'
#         print(path)
#
#         df_logs = pandas.concat(plotting.load_logs(path), sort=True)
#
#         maxes = best_within_normal_time_mutilated(
#             df_logs, args.regex,
#             patience=args.patience, window=args.window, limit=args.limit,
#             summary_path=None)
#         results += f"{np.mean(maxes)}\t"
#         arch_results.append(results)
#     all_results.append(arch_results)
#
# demos = '\t'
# for sample in samples:
#     demos += f'\t{sample}K'
#
# print(demos)
# for level, ar in zip(levels, all_results):
#     for arch, ar in zip(archs, ar):
#         print(f'{level}\t{arch}\t' + ar)



levels = ['GoToRedBallGrey', 'GoToRedBall', 'GoToLocal', 'PutNextLocal', 'PickupLoc']
archs = ['expert_filmcnn', 'expert_filmcnn_endpool_res', 'expert_filmcnn_endpool_res_not_conv_bow']
# samples = [5, 10, 25, 50, 100, 250, 500]
samples = [5, 10, 50, 100, 500]

# all_results = []
# for level in levels:
#     arch_results = []
#     for sample in samples:
#         results = ''
#         for arch in archs:
#             path = f'../beluga/il/{level}/{arch}/{sample}000/'
#             print(path)
#
#             df_logs = pandas.concat(plotting.load_logs(path), sort=True)
#             # print(df_logs['model'].unique().tolist())
#             models = [model for model in df_logs['model'].unique() if re.match(args.regex, model)]
#
#             maxes = best_within_normal_time_mutilated(
#                 df_logs, args.regex,
#                 patience=args.patience, window=args.window, limit=args.limit,
#                 summary_path=None)
#             results += f"{maxes.mean()}\t{maxes.std()}\t"
#             # results += f" & {100.*maxes.mean():.1f} $\pm$ {100.*maxes.std():.1f}"
#         arch_results.append(results)
#     all_results.append(arch_results)



for arch in archs:
    fps = []
    for level in levels:
        path = f'../beluga/il/{level}/{arch}/'
        df_logs = pandas.concat(plotting.load_logs(path), sort=True)
        fps.append(df_logs)

    fps = pandas.concat(fps, sort=True)
    maxes = best_within_normal_time_mutilated(
        fps, args.regex,
        patience=args.patience, window=args.window, limit=args.limit,
        summary_path=None)
    print(f'{maxes.mean()}\t{maxes.std()}')
