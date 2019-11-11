#!/usr/bin/env python3

import os
import re
import babyai.plotting as bp
import pandas
import argparse
import json

parser = argparse.ArgumentParser("Analyze data efficiency of reinforcement learning")
parser.add_argument("--path", default='.')
parser.add_argument("--regex", default='.*')
parser.add_argument("--window", type=int, default=100)
parser.add_argument("report")
args = parser.parse_args()

def dataeff(df_model, window):
  smoothed_sr = df_model['success_rate'].rolling(window, center=True).mean()
  if smoothed_sr.max() < 0.99:
    print('not done, success rate is only {}% so far'.format(100 * smoothed_sr.max()))
    return int(1e9)
  return df_model[smoothed_sr >= 0.99].iloc[0].episodes

if os.path.exists(args.report):
    raise ValueError("report directory already exists")
os.mkdir(args.report)

df = pandas.concat(bp.load_logs(args.path), sort=True)
models = df['model'].unique()

print(args.regex)
models = [model for model in df['model'].unique() if re.match(args.regex, model)]

data = []
for model in models:
    x = df[df['model'] == model]
    eff = float(dataeff(x, args.window))
    print(model, eff)
    data.append(eff)

result = {'min': min(data), 'max': max(data)}
with open(os.path.join(args.report, 'result.json'), 'w') as dst:
    json.dump(result, dst)
