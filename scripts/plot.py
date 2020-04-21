#!/usr/bin/env python
from babyai.plotting import *

storage = '/home/dyth/Downloads/repositories/beluga/logs/copy/logs/rl/GoTo/expert_filmcnn/1e-4/4'
dfs = load_logs(storage)
df = pandas.concat(dfs, sort=True)
window = 20
regex = re.compile('.*')
plot_all_runs(df, regex, window=window, quantity='success_rate')
# plot_all_runs(df, regex, window=window, quantity='validation_accuracy')
# plot_all_runs(df, regex, window=window, quantity='validation_success_rate')

# storage = '/home/dyth/Downloads/repositories/beluga/logs/il'
# dfs = load_logs(storage)
# df = pandas.concat(dfs, sort=True)
# window = 500
# regex = [
#     re.compile('.*GoToRedBallGrey.*'),
#     re.compile('.*GoToRedBall.*'),
#     re.compile('.*GoToLocal.*'),
#     # re.compile('.*PickupLoc.*'),
#     re.compile('.*PutNextLocal.*')
# ]
# plot_average(df, regex, window=window, y_value='validation_success_rate')

# storage = '/home/dyth/Downloads/repositories/beluga/logs2/logs/il/PickupLoc'
# dfs = load_logs(storage)
# df = pandas.concat(dfs, sort=True)
# window = 20
# regex = re.compile('.*')
# print(df['validation_success_rate'].tolist())
# plot_all_runs(df, regex, window=window, quantity='validation_success_rate')


# storage = '/home/dyth/Downloads/repositories/beluga/logs2/logs/PutNextLocal'
# dfs = load_logs(storage)
# df = pandas.concat(dfs, sort=True)
# window = 20
# regex = re.compile('.*pixels.*')
# # plot_all_runs(df, regex, window=window, quantity='train_accuracy')
# # plot_all_runs(df, regex, window=window, quantity='validation_accuracy')
# # plot_all_runs(df, regex, window=window, quantity='validation_success_rate')
# plot_all_runs(df, regex, window=window, quantity='success_rate')
# regex = [
#     re.compile('.*0.00001.*'), re.compile('.*0.00005.*'),
#     re.compile('.*0.0001.*'), re.compile('.*0.0005.*'),
#     re.compile('.*0.001.*'), re.compile('.*0.005.*'),
#     re.compile('.*0.01.*'), re.compile('.*0.05.*')
# ]
# plot_average(df, regex, window=window, y_value='success_rate')
#
# storage = '/home/dyth/Downloads/repositories/beluga/logs2/logs/PutNextLocal/expert_filmcnn_endpool_res_pixels'
# dfs = load_logs(storage)
# df = pandas.concat(dfs, sort=True)
# window = 20
# regex = [re.compile('.*long.*'), re.compile('.*long.*')]
# plot_average(df, regex, window=window, y_value='success_rate')


# storage = '/home/dyth/Downloads/repositories/logs/BabyAI-GoToLocal-v0_IL_expert_filmcnn_gru_seed100_20-02-17-10-35-11'
# storage = '/home/dyth/Downloads/repositories/logs/BabyAI-GoToLocal-v0_IL_expert_filmcnn_gru_seed100_20-02-17-11-38-37'
# dfs = load_logs(storage)
# df = pandas.concat(dfs, sort=True)
# window = 1
# regex = re.compile('.*')
# plot_all_runs(df, regex, window=window, quantity='train_accuracy')
# plot_all_runs(df, regex, window=window, quantity='validation_accuracy')
# plot_all_runs(df, regex, window=window, quantity='validation_success_rate')

pyplot.show()
