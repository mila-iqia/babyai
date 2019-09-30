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
import scipy
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.linalg import cholesky, cho_solve, solve_triangular


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


def min_num_samples(df, regex, patience, limit='epochs', window=1, normal_time=None, summary_path=None):
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
    print("{: <100} {}\t{}\t{}\t{}".format(
        'model_name', 'sr_nt', 'sr', 'dur_nt', 'dur_days'))
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
        print("{: <100} {: <5.4g}\t{: <5.4g}\t{: <5.3g}\t{:.3g}".format(
            model.split('/')[-1],
            max_within_normal_time * 100,
            success_rate.max() * 100,
            df_model[limit].max() / normal_time,
            df_model['duration'].max() / 86400))
        summary_data.append((num, max_within_normal_time))

    summary_df = pandas.DataFrame(summary_data, columns=('num_samples', 'success_rate'))
    if summary_path:
        summary_df.to_csv(summary_path)

    if min(num_samples) == min_samples_required:
        print('should be run with less samples!')
    if need_more_time:
        print('should be run for more time!')
    return summary_df, normal_time


def estimate_sample_efficiency(df, visualize=False, figure_path=None):
    f, axes = pyplot.subplots(1, 2, figsize=(15, 5))

    print("{} datapoints".format(len(df)))
    x = np.log2(df['num_samples'].values)
    y = (df['success_rate'] - 0.99).values * 100
    print("min x: {}, max x: {}, min y: {}, max y: {}".format(x.min(), x.max(), y.min(), y.max()))

    if (y < 0).sum() < 5:
        print("ATTENTION: you have less than 5 datapoints below the threshold.")
        print("Consider running experiments with less data.")
    if (y > 0).sum() < 5:
        print("ATTENTION: you have less than 5 datapoints above the threshold.")
        print("Consider running experiments with more data.")

    kernel = 1.0 * RBF() + WhiteKernel(noise_level_bounds=(1e-10, 1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False).fit(x[:, None], y)
    print("Kernel:", gp.kernel_)
    print("Marginal likelihood:", gp.log_marginal_likelihood_value_)

    grid = np.arange(x[0], x[-1], 0.02)
    y_grid_mean, y_grid_cov = gp.predict(grid[:, None], return_cov=True)
    noise_level = gp.kernel_.k2.noise_level
    f_grid_cov = y_grid_cov - np.diag(np.ones_like(y_grid_cov[0]) * noise_level)

    if visualize:
        axis = axes[0]
        axis.plot(x, y, 'o')
        axis.plot(grid, y_grid_mean)
        axis.set_xlabel('log2(N)')
        axis.set_ylabel('accuracy minus 99%')
        axis.set_title('Data Points & Posterior')
        axis.fill_between(grid, y_grid_mean - np.sqrt(np.diag(y_grid_cov)),
                         y_grid_mean + np.sqrt(np.diag(y_grid_cov)),
                         alpha=0.2, color='k')
        axis.fill_between(grid, y_grid_mean -np.sqrt(np.diag(f_grid_cov)),
                 y_grid_mean + np.sqrt(np.diag(f_grid_cov)),
                 alpha=0.2, color='g')
        axis.hlines(0, x[0], x[-1])

    probs = []
    total_p = 0.
    print("Estimating N_min using a grid of {} points".format(len(grid)))
    for j in range(len(grid)):
        if j and j % 10 == 0:
            print('{} points done'.format(j))
            print(" ".join(["{:.3g}".format(p) for p in probs[-10:]]))
        mu = y_grid_mean[:j + 1].copy()
        mu[j] *= -1
        sigma = f_grid_cov[:j + 1, :j + 1].copy()
        sigma[j, :j] *= -1
        sigma[:j, j] *= -1
        sigma[np.diag_indices_from(sigma)] += 1e-6
        p = stats.multivariate_normal.cdf(np.zeros_like(mu), mu, sigma, abseps=1e-3, releps=1e-3)
        probs.append(p)
        total_p += p
        if total_p.sum() > 0.999:
            print('the rest is unlikely')
            break
    probs = np.array(probs)
    if (probs.sum() - 1) > 0.01:
        raise ValueError("oops, probabilities don't sum to one")
    else:
        # probs should sum to 1, but there is always a bit of error
        probs = probs / probs.sum()

    cut_grid = grid[:len(probs)]
    mean_n_min = (probs * cut_grid).sum()
    mean_n_min_squared = (probs * cut_grid ** 2).sum()
    std_n_min = (mean_n_min_squared - mean_n_min ** 2) ** 0.5
    print('N_min mean and std: ({}, {})'.format(mean_n_min, std_n_min))

    left, right = (mean_n_min - 3 * std_n_min, mean_n_min + 3 * std_n_min)
    print("Confidence interval (log):", left, mean_n_min, right)
    print("Confidence interval:", 2 ** left, 2 ** mean_n_min, 2 ** right)

    if visualize:
        axis = axes[1]
        axis.plot(x, y, 'o')
        axis.set_xlabel('log2(N)')
        axis.set_ylabel('accuracy minus 99%')
        axis.hlines(0, x[0], x[-1])
        axis.vlines(left, min(y), max(y), color='r')
        axis.vlines(mean_n_min, min(y), max(y), color='k')
        axis.vlines(right, min(y), max(y), color='r')
        axis.set_title('Data points & Conf. interval for min. number of samples')

    pyplot.tight_layout()
    if figure_path:
        pyplot.savefig(figure_path)
    return {'mean_log2': mean_n_min, 'std_log2': std_n_min,
            'min': 2 ** left, 'max': 2 ** right}
