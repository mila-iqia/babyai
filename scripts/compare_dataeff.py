#!/usr/bin/env python3

import argparse
import json
import os
from scipy import stats

from babyai import plotting


parser = argparse.ArgumentParser("Compare data efficiency of two approaches")
parser.add_argument("report1", default=None)
parser.add_argument("report2", default=None)
args = parser.parse_args()

r1 = json.load(open(os.path.join(args.report1, 'result.json')))
r2 = json.load(open(os.path.join(args.report2, 'result.json')))
diff_std = (r1['std_log2'] ** 2 + r2['std_log2'] ** 2) ** 0.5
p_less = stats.norm.cdf(0, r2['mean_log2'] - r1['mean_log2'], diff_std)
print('less samples required with {} probability'.format(p_less))
print('more samples required with {} probability'.format(1 - p_less))
