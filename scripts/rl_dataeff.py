#!/usr/bin/env python3

import os
import re
import babyai.plotting as bp
import pandas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--regex", default='.*')
parser.add_argument("--window", type=int, default=100)
args = parser.parse_args()

def dataeff(df_model, window):
  smoothed_sr = df_model['success_rate'].rolling(window, center=True).mean()
  if smoothed_sr.max() < 0.99:
    print('not done, success rate is only {}% so far'.format(100 * smoothed_sr.max()))
    return int(1e9)
  return df_model[smoothed_sr >= 0.99].iloc[0].episodes

df = pandas.concat(bp.load_logs(args.path), sort=True)
models = df['model'].unique()

print(args.regex)
models = [model for model in df['model'].unique() if re.match(args.regex, model)]

data = []
for model in models:
    x = df[df['model'] == model]
    print(model,  dataeff(x, args.window))
