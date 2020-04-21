#!/usr/bin/env python3

import os
import re
import babyai.plotting as bp
import pandas
import argparse
import json
import numpy
from scipy import stats

# dictionary of level to window used
levels = {
    'GoToRedBallGrey': 10,
    'GoToRedBall': 100,
    'GoToLocal': 100,
    'PickupLoc': 100,
    'PutNextLocal': 100,
    'GoTo': 100,
}

model1 = 'expert_filmcnn'
model2 = 'expert_filmcnn_endpool'
model1 = 'expert_filmcnn_endpool'
model2 = 'expert_filmcnn_endpool_res/lr-0.0001'

model1 = 'expert_filmcnn_endpool_res/lr-0.0001'
model2 = 'expert_filmcnn_endpool_res_not_conv_bow/lr-0.0001'
# model1 = 'expert_filmcnn_endpool_res/lr-0.0001'
# model2 = 'expert_filmcnn_endpool_res_pixels/lr-0.0001'
# model1 = 'expert_filmcnn_endpool_res_not_conv_bow/lr-0.0001'
# model2 = 'expert_filmcnn_endpool_res_pixels/lr-0.0001'

model1 = 'expert_filmcnn_endpool_res/lr-0.00005'
model2 = 'expert_filmcnn_endpool_res_not_conv_bow/lr-0.00005'
model1 = 'expert_filmcnn_endpool_res/lr-0.00005'
model2 = 'expert_filmcnn_endpool_res_pixels/lr-0.00005'
model1 = 'expert_filmcnn_endpool_res_not_conv_bow/lr-0.00005'
model2 = 'expert_filmcnn_endpool_res_pixels/lr-0.00005'

# model1 = 'expert_filmcnn_endpool_res_not_conv_bow/lr-0.0001'
# model2 = 'expert_filmcnn_endpool_res_bow/lr-0.0001'
# model1 = 'expert_filmcnn_endpool_res_not_conv_bow/lr-0.00005'
# model2 = 'expert_filmcnn_endpool_res_bow/lr-0.00005'


path1 = [f'../beluga/logs/rl/{level}/{model1}' for level in levels]
path2 = [f'../beluga/logs/rl/{level}/{model2}' for level in levels]
windows = [levels[i] for i in levels]


def dataeff(df_model, window):
    smoothed_sr = df_model['success_rate'].rolling(window, center=True).mean()
    if smoothed_sr.max() < 0.99:
        # print('not done, success rate is only {}% so far'.format(100 * smoothed_sr.max()))
        return int(1e9)
    return df_model[smoothed_sr >= 0.99].iloc[0].episodes

def get_dataeff(path, window):
    print(path)
    df = pandas.concat(bp.load_logs(path), sort=True)
    models = df['model'].unique()
    data = []
    for model in models:
        x = df[df['model'] == model]
        eff = float(dataeff(x, window))
        # print(model, eff)
        if eff != 1e9:
            data.append(eff)
    return numpy.array(data)

data1 = [get_dataeff(path, window) for path, window in zip(path1, windows)]
data2 = [get_dataeff(path, window) for path, window in zip(path2, windows)]

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
ttests = [stats.ttest_ind(d1, d2, equal_var=False) for d1, d2 in zip(data1, data2)]
for ttest in ttests:
    print(f'{ttest.statistic}\t{ttest.pvalue}')
