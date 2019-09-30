#!/usr/bin/env python3

import argparse
import pandas
import os
import json

from babyai import plotting


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='.')
parser.add_argument("--regex", default='.*')
parser.add_argument("--patience", default=2, type=int)
parser.add_argument("--window", default=1, type=int)
parser.add_argument("--limit", default="frames")
parser.add_argument("report", default=None)
args = parser.parse_args()

if args.report:
    if os.path.exists(args.report):
        raise ValueError("report directory already exists")
    os.mkdir(args.report)
    summary_path = os.path.join(args.report, 'summary.csv')
    figure_path = os.path.join(args.report, 'visualization.png')
    result_path = os.path.join(args.report, 'result.json')
else:
    summary_path = None
    figure_path = None
    result_path = None

df_logs = pandas.concat(plotting.load_logs(args.path), sort=True)
df_success_rate = plotting.min_num_samples(
    df_logs, args.regex,
    patience=args.patience, window=args.window, limit=args.limit,
    summary_path=summary_path)
result = plotting.estimate_sample_efficiency(
    df_success_rate, visualize=True, figure_path=figure_path)
with open(result_path, 'w') as dst:
    json.dump(result, dst)
