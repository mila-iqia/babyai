#!/usr/bin/env python3
import argparse
import pandas
import os
import json
import re
import numpy as np
from scipy import stats

from babyai import plotting as bp


parser = argparse.ArgumentParser("Analyze performance of imitation learning")
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


def get_data(path, regex):
    df = pandas.concat(bp.load_logs(path), sort=True)
    fps = bp.get_fps(df)
    models = df['model'].unique()
    models = [model for model in df['model'].unique() if re.match(regex, model)]

    maxes = []
    for model in models:
        df_model = df[df['model'] == model]
        success_rate = df_model['validation_success_rate']
        success_rate = success_rate.rolling(args.window, center=True).mean()
        success_rate = max(success_rate[np.logical_not(np.isnan(success_rate))])
        print(model, success_rate)
        maxes.append(success_rate)
    return np.array(maxes), fps



if args.other is not None:
    print("is this architecture better")
print(args.regex)
maxes, fps = get_data(args.path, args.regex)
result = {'samples': len(maxes), 'mean': maxes.mean(), 'std': maxes.std(),
          'fps_mean': fps.mean(), 'fps_std': fps.std()}
print(result)

if args.other is not None:
    print("\nthan this one")
    maxes_ttest, fps = get_data(args.other, args.other_regex)
    result = {'samples': len(maxes_ttest),
        'mean': maxes_ttest.mean(), 'std': maxes_ttest.std(),
        'fps_mean': fps.mean(), 'fps_std': fps.std()}
    print(result)
    ttest = stats.ttest_ind(maxes, maxes_ttest, equal_var=False)
    print(f"\n{ttest}")
