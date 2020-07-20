#!/usr/bin/env python3
import os
import re
import babyai.plotting as bp
import pandas
import argparse
import json
import numpy as np
from scipy import stats


parser = argparse.ArgumentParser("Analyze data efficiency of reinforcement learning")
parser.add_argument("--path", default='.',
    help="path to model logs")
parser.add_argument("--regex", default='.*',
    help="filter out some logs")
parser.add_argument("--other", default=None,
    help="path to model logs for ttest comparison")
parser.add_argument("--other_regex", default='.*',
    help="filter out some logs from comparison")
parser.add_argument("--window", type=int, default=100,
    help="size of sliding window average, 10 for GoToRedBallGrey, 100 otherwise")
args = parser.parse_args()


def dataeff(df_model, window):
    smoothed_sr = df_model['success_rate'].rolling(window, center=True).mean()
    if smoothed_sr.max() < 0.99:
        print('not done, success rate is only {}% so far'.format(100 * smoothed_sr.max()))
        return int(1e9)
    return df_model[smoothed_sr >= 0.99].iloc[0].episodes


def get_data(path, regex):
    print(path)
    print(regex)
    df = pandas.concat(bp.load_logs(path), sort=True)
    fps = bp.get_fps(df)
    models = df['model'].unique()
    models = [model for model in df['model'].unique() if re.match(regex, model)]

    data = []
    for model in models:
        x = df[df['model'] == model]
        eff = float(dataeff(x, args.window))
        print(model, eff)
        if eff != 1e9:
            data.append(eff)
    return np.array(data), fps



if args.other is not None:
    print("is this architecture better")

Z = 2.576
data, fps = get_data(args.path, args.regex)
result = {'samples': len(data), 'mean': data.mean(), 'std': data.std(),
          'min': data.mean() - Z * data.std(), 'max': data.mean() + Z * data.std(),
          'fps_mean': fps.mean(), 'fps_std': fps.std()}
print(result)

if args.other is not None:
    print("\nthan this one")
    data_ttest, fps = get_data(args.other, args.other_regex)
    result = {'samples': len(data_ttest),
        'mean': data_ttest.mean(), 'std': data_ttest.std(),
        'min': data_ttest.mean() - Z * data_ttest.std(),
        'max': data_ttest.mean() + Z * data_ttest.std(),
        'fps_mean': fps.mean(), 'fps_std': fps.std()}
    print(result)
    ttest = stats.ttest_ind(data, data_ttest, equal_var=False)
    print(f"\n{ttest}")
