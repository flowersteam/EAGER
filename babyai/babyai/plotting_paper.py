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
                         on_bad_lines='warn')
    if not len(df):
        print("empty df at {}".format(dir_))
        return
    df['model'] = dir_
    return df


def load_logs(root):
    dfs = []
    for root, dirs, files in os.walk(root, followlinks=True):
        for file_ in files:
            if file_ == 'log.csv':
                dfs.append(load_log(root))
    return dfs


def plot_average_impl(df, regexps, labels, limits, y_value='return_mean', window=1000, agg='mean',
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df[df.frames < limits]
    # df = df[df.frames < 120000040]
    # df[(df.frames > 70000000) & (df.return_mean < 0.5)] = None

    df = df.dropna(subset=[y_value])

    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                    for regex in regexps]

    for regex, models, label in zip(regexps, model_groups, labels):
        print("regex: {}".format(regex))
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model[x_value].max()
                                for _, df_model in df_re.groupby('model')]
        for _, df_model in df_re.groupby('model'):
            print(df_model[x_value].max())
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
        df_agg = df_re.groupby([x_value]).mean()
        values = df_agg[y_value]
        std = df_re.groupby([x_value]).std()[y_value]
        df_max = values + std
        df_min = values - std

        pyplot.plot(df_agg.index, values, label='{} SE: {}'.format(label, round(values.sum()/len(values), 3)))
        # pyplot.plot(df_agg.index, values, label=label)
        pyplot.fill_between(df_agg.index, df_max, df_min, alpha=0.5)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])
        print("{} sample efficiency: {}".format(label, values.sum()/len(values)))


def plot_variance_impl(df, regexps, y_value='return_mean', window=1000, agg='mean',
                       x_value='frames'):
    """Plot variance over groups of runs  defined by regular expressions."""
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
        df_agg = df_re.groupby([x_value]).var()
        values = df_agg[y_value]
        print("{}: {}".format(regex, values.max()))
        pyplot.plot(df_agg.index, values, label=regex)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])


def plot_SR_impl(df, regexps, y_value='success_rate', window=100, agg='mean',
                 x_value='frames'):
    """Plot success rate QA over groups of runs  defined by regular expressions."""
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
        df_agg = df_re.groupby([x_value]).mean()
        df_max = df_re.groupby([x_value]).max()[y_value]
        df_min = df_re.groupby([x_value]).min()[y_value]
        values = df_agg[y_value]
        pyplot.plot(df_agg.index, values, label=regex)
        pyplot.fill_between(df_agg.index, df_max, df_min, alpha=0.5)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])


def plot_SR_QA_impl(df, regexps, labels, y_value='success_rate_QA_mean', window=1000, agg='mean',
                    x_value='frames'):
    """Plot success rate QA over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])
    df = df[df.frames < 80000040]
    # df = df[df.frames < 120000040]
    # df[(df.frames > 70000000) & (df.return_mean < 0.5)] = None
    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                    for regex in regexps]

    for regex, models, label in zip(regexps, model_groups, labels):
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
        df_agg = df_re.groupby([x_value]).mean()
        df_max = df_re.groupby([x_value]).max()[y_value]
        df_min = df_re.groupby([x_value]).min()[y_value]
        values = df_agg[y_value]
        pyplot.plot(df_agg.index, values, label=label)
        pyplot.fill_between(df_agg.index, df_max, df_min, alpha=0.5)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])


def plot_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(7.5, 5))
    plot_average_impl(*args, **kwargs)
    pyplot.legend(handlelength=0.5, handleheight=0.5)
    pyplot.xlabel("Frames")
    pyplot.title("Average Reward")
    pyplot.show()


def plot_variance(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(15, 5))
    plot_variance_impl(*args, **kwargs)
    pyplot.legend()
    pyplot.title("Variance")
    pyplot.show()


def plot_SR(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(15, 5))
    plot_SR_impl(*args, **kwargs)
    pyplot.legend()
    pyplot.title("Success Rate")
    pyplot.show()


def plot_SR_QA(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(7.5, 5))
    plot_SR_QA_impl(*args, **kwargs)
    pyplot.legend(handlelength=0.5, handleheight=0.5)
    pyplot.xlabel("Frames")
    pyplot.title("Success Rate QA")
    pyplot.show()


def plot_all_runs(df, regex, quantity='return_mean', x_axis='frames', window=100, color=None):
    """Plot a group of runs defined by a regex."""
    pyplot.figure(figsize=(15, 5))

    df = df.dropna(subset=[quantity])

    kwargs = {}
    if color:
        kwargs['color'] = color
    unique_models = df['model'].unique()
    models = [m for m in unique_models if re.match(regex, m)]
    df_re = df[df['model'].isin(models)]
    for model, df_model in df_re.groupby('model'):
        values = df_model[quantity]
        values = values.rolling(window).mean()
        pyplot.plot(df_model[x_axis],
                    values,
                    label=model,
                    **kwargs)
        print(model, df_model[x_axis].max())

    pyplot.legend()
    pyplot.show()


dfs = load_logs('storage')
df = pandas.concat(dfs, sort=True)


regexs = ['.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10.*',
          '.*PNL-GTL-RS-Online-025-D(?!-sa.*)',
          '.*PNL-RIDE-reward_scale_20-lambda_05.*']

labels = ['EAGER', 'ELLA   ', 'RIDE   ']
# plot_average(df, regexs, labels)

regexs = ['.*paral_PNM-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10.*',
          '.*PNM-GTM-RS-Online-025-D.*',
          '.*PNM-RIDE-reward_scale_20-lambda_05.*']

labels = ['EAGER', 'ELLA   ', 'RIDE   ']
# plot_average(df, regexs, labels)

regexs = ['.*paral_UNLM-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10.*',
          '.*UNLM-PUM-RS-Online-025-D.*',
          '.*UNLM-RIDE-reward_scale_20-lambda_05.*']

labels = ['EAGER', 'ELLA   ', 'RIDE   ']
# plot_average(df, regexs, labels)

regexs = ['.*paral_SEQ-adjusted-train_env-multienv3_no_answer-lambda_026-model-0_10.*',
          '.*SEQ-GTM-RS-Online-05-D.*',
          '.*SEQ-RIDE-reward_scale_20-lambda_05.*']

labels = ['EAGER', 'ELLA   ', 'RIDE   ']
# plot_average(df, regexs, labels)

regexs = ['.*PNL-adjusted-train_env-PNL-lambda_24-model-0_6.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10.*',
          '.*PNL-simple-train_env-PNL-lambda_24-model-0_6.*',
          '.*PNL-simple-train_env-PNL_no_answer-lambda_24-model-2_10.*',
          '.*PNL-GTL-RS-Online-025-D.*']

labels = ['Adjust', 'Adjust + no_answer', 'Simple', 'Simple + no_answer', 'ELLA']

# plot_average(df, regexs, labels)

regexs = ['.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_9.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_3.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_2.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_0.*']

labels = ['SR: 0.73', 'SR: 0.66', 'SR: 0.56', 'SR: 0.41', 'SR: 0.25']

# plot_average(df, regexs, labels)

regexs = ['.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10.*',
          '.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_9.*',
          '.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_3.*',
          '.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_2.*',
          '.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_0.*']

labels = ['SR: 0.73', 'SR: 0.66', 'SR: 0.56', 'SR: 0.41', 'SR: 0.25']

plot_SR_QA(df, regexs, labels)

regexs = ['.*paral_PNM-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10.*',
          '.*paral_PNM-adjusted-train_env-multienv2_no_answer-lambda_16-model-0_7.*']

labels = ['Wide Distribution', 'Narrow Distribution']
limits = 150000040
# plot_average(df, regexs, labels, limits)

regexs = ['.*paral_SEQ-adjusted-train_env-multienv3_no_answer-lambda_026-model-0_10.*',
          '.*paral_SEQ-adjusted-train_env-multienv2_no_answer-lambda_026-model-0_7.*']

limits = 95000040
labels = ['Wide Distribution', 'Narrow Distribution']
p# lot_average(df, regexs, labels, limits)