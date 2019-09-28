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
parser.add_argument("--summary_path", default='summary.csv')
args = parser.parse_args()

df = pandas.concat(plotting.load_logs(args.path), sort=True)
plotting.min_num_samples(
    df, args.regex,
    patience=args.patience, window=args.window, limit=args.limit,
    summary_path=args.summary_path)
