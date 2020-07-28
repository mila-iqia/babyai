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


def get_fps(df):
    data = df['FPS']
    data = data.tolist()
    return np.array(data)


def best_within_normal_time(df, regex, patience, limit='epochs', window=1, normal_time=None, summary_path=None):
    """
    Compute the best success rate that is achieved in all runs within the normal time.

    The normal time is defined as `patience * T`, where `T` is the time it takes for the run
    with the most demonstrations to converge. `window` is the size of the sliding window that is
    used for smoothing.

    Returns a dataframe with the best success rate for the runs that match `regex`.

    """
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
                raise ValueError('{} has not solved the level yet, only at {} so far'.format(
                    model, success_rate.max()))
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
        print("{: <50} {: <5.4g}\t{: <5.4g}\t{: <5.3g}\t{:.3g}".format(
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
        raise ValueError('should be run with less samples!')
    if need_more_time:
        raise ValueError('should be run for more time!')
    return summary_df, normal_time


def estimate_sample_efficiency(df, visualize=False, figure_path=None):
    """
    Estimate sample efficiency and its uncertainty using Gaussian Process.

    This function interpolates between data points given in `df` using a Gaussian Process.
    It returns a 99% interval based on the GP predictions.

    """
    f, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    # preprocess the data
    print("{} datapoints".format(len(df)))
    x = np.log2(df['num_samples'].values)
    y = df['success_rate']
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices].values

    success_threshold = 0.99
    min_datapoints = 5
    almost_threshold = 0.95

    if (y > success_threshold).sum() < min_datapoints:
        raise ValueError(f"You have less than {min_datapoints} datapoints above the threshold.\n"
                         "Consider running experiments with more examples.")
    if ((y > almost_threshold) & (y < success_threshold)).sum() < min_datapoints:
        raise ValueError(f"You have less than {min_datapoints} datapoints"
              " for which the threshold is almost crossed.\n"
              "Consider running experiments with less examples.")
    # try to throw away the extra points with low performance
    # the model is not suitable for handling those
    while True:
        if ((y[1:] > success_threshold).sum() >= min_datapoints
                and ((y[1:] > almost_threshold) & (y[1:] < success_threshold)).sum()
                        >= min_datapoints):
            print('throwing away x={}, y={}'.format(x[0], y[0]))
            x = x[1:]
            y = y[1:]
        else:
            break

    print("min x: {}, max x: {}, min y: {}, max y: {}".format(x.min(), x.max(), y.min(), y.max()))
    y = (y - success_threshold) * 100

    # fit an RBF GP
    kernel = 1.0 * RBF() + WhiteKernel(noise_level_bounds=(1e-10, 10))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False).fit(x[:, None], y)
    print("Kernel:", gp.kernel_)
    print("Marginal likelihood:", gp.log_marginal_likelihood_value_)

    # compute the success rate posterior
    grid_step = 0.02
    grid = np.arange(x[0], x[-1], grid_step)
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

    # compute the N_min posterior
    probs = []
    total_p = 0.
    print("Estimating N_min using a grid of {} points".format(len(grid)))
    for j in range(len(grid)):
        mu = y_grid_mean[:j + 1].copy()
        mu[j] *= -1
        sigma = f_grid_cov[:j + 1, :j + 1].copy()
        sigma[j, :j] *= -1
        sigma[:j, j] *= -1
        sigma[np.diag_indices_from(sigma)] += 1e-6
        # the probability that the first time the success rate crosses the threshold
        # will be between grid[j - 1] and grid[j]
        p = stats.multivariate_normal.cdf(np.zeros_like(mu), mu, sigma, abseps=1e-3, releps=1e-3)
        probs.append(p)
        total_p += p

        can_stop = total_p.sum() > 0.999
        if j and (can_stop or j % 10 == 0):
            print('{} points done'.format(j))
            print(" ".join(["{:.3g}".format(p) for p in probs[-10:]]))
        if can_stop:
            print('the rest is unlikely')
            break
    probs = np.array(probs)
    if (probs.sum() - 1) > 0.01:
        raise ValueError("oops, probabilities don't sum to one")
    else:
        # probs should sum to 1, but there is always a bit of error
        probs = probs / probs.sum()

    first_prob = (probs > 1e-10).nonzero()[0][0]
    subgrid = grid[first_prob:len(probs)]
    subprobs = probs[first_prob:]
    mean_n_min = (subprobs * subgrid).sum()
    mean_n_min_squared = (subprobs * subgrid ** 2).sum()
    std_n_min = (mean_n_min_squared - mean_n_min ** 2) ** 0.5
    if visualize:
        # visualize the N_min posterior density
        # visualize the non-Gaussianity of N_min posterior density
        axis = axes[2]
        axis.plot(subgrid, subprobs)
        axis.plot(subgrid, stats.norm.pdf(subgrid, mean_n_min, std_n_min) * grid_step)

    # compute the credible interval
    cdf = np.cumsum(probs)
    left = grid[(cdf > 0.01).nonzero()[0][0]]
    right = grid[(cdf > 0.99).nonzero()[0][0]]
    print("99% credible interval for N_min:", 2 ** left,  2 ** right)

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
    return {'min': 2 ** left, 'max': 2 ** right,
            'mean_log2': mean_n_min, 'std_log2': std_n_min}
