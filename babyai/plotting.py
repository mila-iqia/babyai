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
import numpy as np
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

def load_multiphase_log(dir_):
    df = load_log(dir_)
    phases = []
    cur_phase = 0
    prev_upd = 0
    for i in range(len(df)):
        upd = df.iloc[i]['update']
        if upd < prev_upd:
            cur_phase += 1
        phases.append(cur_phase)
        prev_upd = upd
    df['phase'] = phases
    return df

def load_logs(root, multiphase=False):
    dfs = []
    for root, dirs, files in os.walk(root, followlinks=True):
        for file_ in files:
            if file_ == 'log.csv':
                dfs.append(load_multiphase_log(root) if multiphase else load_log(root))
    return dfs


def plot_average_impl(df, regexps, y_value='return_mean', window=1, agg='mean',
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])

    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                     for regex in regexps]

    for regex, models in zip(regexps, model_groups):
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model[x_value].max()
                               for _, df_model in df_re.groupby('model')]
        median_progress = sorted(num_frames_per_model)[(len(num_frames_per_model) - 1) // 2]
        mean_duration = np.mean([
            df_model['duration'].max() for _, df_model in df_re.groupby('model')])
        df_re = df_re[df_re[x_value] <= median_progress]

        # smooth
        parts = []
        for _, df_model in df_re.groupby('model'):
            df_model = df_model.copy()
            df_model.loc[:, y_value] = df_model[y_value].rolling(window).mean()
            parts.append(df_model)
        df_re = pandas.concat(parts)

        df_agg = df_re.groupby([x_value]).agg([agg])
        values = df_agg[y_value][agg]
        pyplot.plot(df_agg.index, values, label=regex)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])


def plot_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(15, 5))
    plot_average_impl(*args, **kwargs)
    pyplot.legend()


def plot_all_runs(df, regex, quantity='return_mean', x_axis='frames', window=1, color=None):
    """Plot a group of runs defined by a regex."""
    pyplot.figure(figsize=(15, 5))

    df = df.dropna(subset=[quantity])

    unique_models = df['model'].unique()
    models = [m for m in unique_models if re.match(regex, m)]
    df_re = df[df['model'].isin(models)]
    for model, df_model in df_re.groupby('model'):
        values = df_model[quantity]
        values = values.rolling(window, center=True).mean()

        kwargs = {}
        if color:
            kwargs['color'] = color(model)
        pyplot.plot(df_model[x_axis],
                    values,
                    label=model,
                    **kwargs)
        print(model, df_model[x_axis].max())

    pyplot.legend()


def model_num_samples(model):
    # the number of samples is mangled in the name
    return int(re.findall('_([0-9]+)', model)[0])


def min_num_samples(df, regex, patience, limit='epochs', window=1, normal_time=None, summary_path='summary.csv'):
    print()
    print(regex)
    models = [model for model in df['model'].unique() if re.match(regex, model)]
    num_samples = [model_num_samples(model) for model in models]
    # sort models according to the number of samples
    models, num_samples = zip(*sorted(list(zip(models, num_samples)), key=lambda tupl: tupl[1]))

    # choose normal time
    max_samples = max(num_samples)
    limits = []
    for model, num in zip(models, num_samples):
        if num == max_samples:
            df_model = df[df['model'] == model]
            success_rate = df_model['validation_success_rate'].rolling(window, center=True).mean()
            if np.isnan(success_rate.max()) or success_rate.max() < 0.99:
                print('{} has not solved the level yet'.format(model))
                return
            first_solved = (success_rate > 0.99).to_numpy().nonzero()[0][0]
            row = df_model.iloc[first_solved]
            print("the model with {} samples first solved after {} epochs ({} seconds, {} frames)".format(
                max_samples, row['update'], row['duration'], row['frames']))
            limits.append(patience * row[limit] + 1)
    if not normal_time:
        normal_time = np.mean(limits)
        print('using {} as normal time'.format(normal_time))

    summary_data = []

    # check how many examples is required to succeed within normal time
    min_samples_required = None
    need_more_time = False
    for model, num in zip(models, num_samples):
        df_model = df[df['model'] == model]
        success_rate = df_model['validation_success_rate'].rolling(window, center=True).mean()
        max_within_normal_time = success_rate[df_model[limit] < normal_time].max()
        if max_within_normal_time > 0.99:
            min_samples_required = min(num, min_samples_required
                                       if min_samples_required
                                       else int(1e9))
        if df_model[limit].max() < normal_time:
            need_more_time = True
        print("{: <100} {: <5.4g} {: <5.4g} {: <5.3g} {:.3g}".format(
            model.split('/')[-1],
            max_within_normal_time * 100,
            success_rate.max() * 100,
            df_model[limit].max() / normal_time,
            df_model['duration'].max() / 86400))
        summary_data.append((num, max_within_normal_time))

    summary_df = pandas.DataFrame(summary_data, columns=('num_samples', 'success_rate'))
    summary_df.to_csv(summary_path)

    print("min samples required is {}".format(min_samples_required))
    if min(num_samples) == min_samples_required:
        print('should be run with less samples!')
    if need_more_time:
        print('should be run for more time!')
    return min_samples_required, normal_time
