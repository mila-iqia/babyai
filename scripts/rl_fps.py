#!/usr/bin/env python3

import os
import re
import babyai.plotting as bp
import pandas
import argparse
import json
import numpy
from scipy import stats

levels = [
    'GoToRedBallGrey',
    'GoToRedBall',
    'GoToLocal',
    'PickupLoc',
    'PutNextLocal',
    # 'GoTo'
]

models = [
    'expert_filmcnn',
    'expert_filmcnn_endpool',
    'expert_filmcnn_endpool_res/lr-0.0001',
    'expert_filmcnn_endpool_res_not_conv_bow/lr-0.0001',
    'expert_filmcnn_endpool_res_pixels/lr-0.0001',
    'expert_filmcnn_endpool_res_bow/lr-0.0001'
]

models = [
    'expert_filmcnn_endpool_res/lr-0.00005',
    'expert_filmcnn_endpool_res_not_conv_bow/lr-0.00005',
    'expert_filmcnn_endpool_res_pixels/lr-0.00005',
    'expert_filmcnn_endpool_res_bow/lr-0.00005'
]


def dataeff(df_model, window):
    smoothed_sr = df_model['success_rate'].rolling(window, center=True).mean()
    if smoothed_sr.max() < 0.99:
        # print('not done, success rate is only {}% so far'.format(100 * smoothed_sr.max()))
        return int(1e9)
    return df_model[smoothed_sr >= 0.99].iloc[0].episodes

def get_fps(path):
    df = pandas.concat(bp.load_logs(path), sort=True)
    data = df['FPS']
    data = data.tolist()
    return data

# # per level average
# for level in levels:
#     row = ''
#     for model in models:
#         path = f'../beluga/logs/rl/{level}/{model}'
#         data = numpy.array(get_fps(path))
#         row += f'\t{data.mean()}\t{data.std()}'
#     print(row)

levels = [
    'GoToRedBallGrey',
    'GoToRedBall',
    'GoToLocal',
    'PickupLoc',
    'PutNextLocal',
    'GoTo'
]

models = [
    'expert_filmcnn',
    'expert_filmcnn_endpool',
    'expert_filmcnn_endpool_res',
    'expert_filmcnn_endpool_res_not_conv_bow',
    'expert_filmcnn_endpool_res_pixels',
    'expert_filmcnn_endpool_res_bow'
]

row = ''
for model in models:
    data = []
    for level in levels:
        data += get_fps(f'../beluga/logs/rl/{level}/{model}')
    data = numpy.array(data)
    row += f'\t{data.mean()}\t{data.std()}'
print(row)
