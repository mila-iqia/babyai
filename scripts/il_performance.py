#!/usr/bin/env python3
import argparse
import pandas
import os
import json
import re
import numpy as np

from babyai import plotting


parser = argparse.ArgumentParser("Analyze data efficiency of imitation learning")
parser.add_argument('--path', default='.')
parser.add_argument("--regex", default='.*')
parser.add_argument("--patience", default=2, type=int)
parser.add_argument("--window", default=10, type=int)
parser.add_argument("--limit", default="frames")
args = parser.parse_args()



def best_in_training(df, regex, patience, limit='epochs', window=1, normal_time=None, summary_path=None):
    """
    Return the best success rate

    First smooth the success rate with a sliding window of size 'window'.
    Return an array with best result for each runs that matches `regex`.
    """
    models = [model for model in df['model'].unique() if re.match(regex, model)]
    num_samples = range(len(models))
    # sort models according to the number of samples
    models, num_samples = zip(*sorted(list(zip(models, num_samples)), key=lambda tupl: tupl[1]))

    maxes = []
    for model, num in zip(models, num_samples):
        df_model = df[df['model'] == model]
        success_rate = df_model['validation_success_rate'].rolling(window, center=True).mean()
        success_rate = success_rate[np.logical_not(np.isnan(success_rate))]
        maxes.append(max(success_rate))
    return np.array(maxes)



levels = ['GoToRedBallGrey', 'GoToRedBall', 'GoToLocal', 'PutNextLocal', 'PickupLoc']#, 'GoTo']
archs = ['expert_filmcnn', 'expert_filmcnn_endpool_res', 'expert_filmcnn_endpool_res_not_conv_bow']
samples = [5, 10, 50, 100, 500]
# samples = [10, 100]

all_results = []
for level in levels:
    arch_results = []
    for sample in samples:
        results = ''
        for arch in archs:
            path = f'../beluga/il/{level}/{arch}/{sample}000/'
            print(path)

            df_logs = pandas.concat(plotting.load_logs(path), sort=True)
            maxes = best_in_training(
                df_logs, args.regex,
                patience=args.patience, window=args.window, limit=args.limit,
                summary_path=None)
            results += f"{maxes.mean()}\t{maxes.std()}\t"
            # results += f" & {100.*maxes.mean():.1f} $\pm$ {100.*maxes.std():.1f}"
        arch_results.append(results)
    all_results.append(arch_results)



demos = '\t'
for sample in samples:
    demos += f'\t{sample}K'
print(demos)

for level, ar in zip(levels, all_results):
    for sample, ar in zip(samples, ar):
        print('&' + ar + ' \\\ ')
        # print(f'{level}\t{arch}\t' + ar)
