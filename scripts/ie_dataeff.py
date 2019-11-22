#!/usr/bin/env python3

import os
import argparse
import pandas
import json
import re

from babyai import plotting


parser = argparse.ArgumentParser("Analyze data efficiency of interactive imitation learning")
parser.add_argument("--path", default='.')
parser.add_argument("--regex", default='.*')
parser.add_argument("--window", default=1, type=int)
parser.add_argument("--start-demos", type=float, default=1000000 / 1024)
parser.add_argument("--grow-factor", type=float, default=2 ** 0.25)
parser.add_argument("report", default=None)
args = parser.parse_args()

def summarize_results(master_df, regex):
    data = {
        'num_samples': [],
        'success_rate': []
    }
    models = [model for model in master_df['model'].unique()
              if re.match(regex, model)]
    for model in models:
        print(model)
        df = master_df[master_df['model'] == model]
        solved = False
        for phase, df_phase in df.groupby('phase'):
            phase_max = df_phase['validation_success_rate'].rolling(args.window).mean().max()
            num_demos = int(args.start_demos * args.grow_factor ** phase)
            data['num_samples'].append(num_demos)
            data['success_rate'].append(phase_max)
    return pandas.DataFrame(data)

if os.path.exists(args.report):
    raise ValueError("report directory already exists")
os.mkdir(args.report)
summary_path = os.path.join(args.report, 'summary.csv')
figure_path = os.path.join(args.report, 'visualization.png')
result_path = os.path.join(args.report, 'result.json')

df = pandas.concat(plotting.load_logs(args.path, multiphase=True), sort=True)
summary_df = summarize_results(df, args.regex)
result = plotting.estimate_sample_efficiency(
    summary_df, visualize=True, figure_path=figure_path)

summary_df.to_csv(summary_path)
with open(result_path, 'w') as dst:
    json.dump(result, dst)
