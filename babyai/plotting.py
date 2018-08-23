"""Loading and plotting data from CSV logs.

Schematic example of usage

- load all `log.csv` files that can be found by recursing a root directory:
  `dfs = load_logs($BABYAI_STORAGE)`
- concatenate them in the master dataframe
  `df = pandas.concat(dfs, sort=True)`
- plot average performance for groups of runs using `plot_average(df, ...)`
- plot performance for each run in a group using `plot_all_runs(df, ...)`

Note:
- you can choose what to plot
- groups are defined by regular expressions over full paths to .csv files.
  For example, if your model is called "model1" and you trained it with multiple seeds,
  you can filter all the respective runs with the regular expression ".*model1.*"
- you may want to load your logs from multiple storage directories
  before concatening them into a master dataframe

"""

import os
import re
from matplotlib import pyplot
import pandas


def load_log(dir_):
    """Loads log from a directory and adds it to a list of dataframes."""
    df = pandas.read_csv(os.path.join(dir_, 'log.csv'),
                         error_bad_lines=False,
                         warn_bad_lines=True)
    if not len(df):
        print("empty df at {}".format(dir_))
        return
    df['model'] = dir_
    return df


def load_logs(root):
    dfs = []
    for root, dirs, files in os.walk(root):
        for file_ in files:
            if file_ == 'log.csv':
                dfs.append(load_log(root))
    return dfs


def plot_average(df, regexps, quantity='return_mean', window=1, agg='mean'):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(15, 5))

    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                     for regex in regexps]

    all_values = []
    for regex, models in zip(regexps, model_groups):
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model['frames'].max()
                               for _, df_model in df_re.groupby('model')]
        median_progress = sorted(num_frames_per_model)[(len(num_frames_per_model) - 1) // 2]
        df_re = df_re[df_re['frames'] <= median_progress]

        # smooth
        parts = []
        for _, df_model in df_re.groupby('model'):
            df_model = df_model.copy()
            df_model.loc[:, quantity] = df_model[quantity].rolling(window).mean()
            parts.append(df_model)
        df_re = pandas.concat(parts)

        df_agg = df_re.groupby(['frames']).agg([agg])
        values = df_agg[quantity][agg]
        pyplot.plot(df_agg.index, values, label=regex)
        print(regex, median_progress)
        all_values.append(values)

    pyplot.legend()


def plot_all_runs(df, regex, quantity='return_mean', window=1, color=None):
    """Plot a group of runs defined by a regex."""
    pyplot.figure(figsize=(15, 5))

    kwargs = {}
    if color:
        kwargs['color'] = color
    unique_models = df['model'].unique()
    models = [m for m in unique_models if re.match(regex, m)]
    df_re = df[df['model'].isin(models)]
    for model, df_model in df_re.groupby('model'):
        values = df_model[quantity]
        values = values.rolling(window).mean()
        pyplot.plot(df_model['frames'],
                    values,
                    label=model,
                    **kwargs)
        print(model, df_model['frames'].max())

    pyplot.legend()
