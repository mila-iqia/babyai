#!/usr/bin/env python3

import argparse
import pandas

from babyai import plotting


parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--regex", default='.*')
parser.add_argument("--patience", default=2, type=int)
parser.add_argument("--window", default=1, type=int)
parser.add_argument("--limit", default="frames")
parser.add_argument("--start-demos", type=float, default=1000000 / 1024)
parser.add_argument("--grow-factor", type=float, default=2 ** 0.25)
args = parser.parse_args()

def analyze_results(master_df):
    for model in sorted(master_df['model'].unique()):
        print(model)
        df = master_df[master_df['model'] == model]
        solved = False
        for phase, df_phase in df.groupby('phase'):
            phase_max = df_phase['validation_success_rate'].rolling(args.window).mean().max()
            output = [phase, int(args.start_demos * args.grow_factor ** phase), phase_max]
            if phase_max > 0.99 and not solved:
                solved = True
                output.append('(!)')
            print(*output)

df = pandas.concat(plotting.load_logs(args.path, multiphase=True), sort=True)
analyze_results(df)
