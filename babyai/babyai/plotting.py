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


def plot_average_impl(df, regexps, y_value='return_mean', window=1000, agg='mean',
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])

    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                    for regex in regexps]

    for regex, models in zip(regexps, model_groups):
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
        df_max = df_re.groupby([x_value]).max()[y_value]
        df_min = df_re.groupby([x_value]).min()[y_value]
        values = df_agg[y_value]

        pyplot.plot(df_agg.index, values, label=regex)
        pyplot.fill_between(df_agg.index, df_max, df_min, alpha=0.5)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])


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


def plot_SR_QA_impl(df, regexps, y_value='success_rate_QA_mean', window=1000, agg='mean',
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


def plot_bonus_impl(df, regexps, y_value='reshaped_return_bonus_mean', window=100, agg='mean',
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


def plot_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(15, 5))
    plot_average_impl(*args, **kwargs)
    pyplot.legend()
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
    pyplot.figure(figsize=(15, 5))
    plot_SR_QA_impl(*args, **kwargs)
    pyplot.legend()
    pyplot.title("Success Rate QA")
    pyplot.show()


def plot_bonus(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(15, 5))
    plot_bonus_impl(*args, **kwargs)
    pyplot.legend()
    pyplot.title("Success Rate")
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
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_9.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_3.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_2.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_0.*']

# plot_bonus(df, regexs)
"""plot_average(df, regexs)
plot_variance(df, regexs)
for regex in regexs:
    plot_all_runs(df, regex)"""

regexs = ['.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10.*',
          '.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_9.*',
          '.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_3.*',
          '.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_2.*',
          '.*paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_0.*']
# plot_SR_QA(df, regexs)


regexs = ['.*PNLsa-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10.*']
# plot_average(df, regexs)

regexs = ['.*PNMsa-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10.*',
          '.*PNM-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10.*']
# plot_average(df, regexs)

regexs = ['.*PNL-adjusted_biased-0-train_env-PNL_no_answer_biased_debiased_QA-0.*',
          '.*PNL-adjusted_biased-1-train_env-PNL_no_answer_biased_debiased_QA-0.*',
          '.*PNL-adjusted_biased-0-train_env-PNL_no_answer_biased_debiased_QA-1.*',
          '.*PNL-adjusted_biased-1-train_env-PNL_no_answer_biased_debiased_QA-1.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10.*']
# plot_average(df, regexs)
"""plot_variance(df, regexs)
plot_SR_QA(df, ['.*PNL-adjusted_biased-0-train_env-PNL_no_answer_biased_debiased_QA-0.*',
                '.*PNL-adjusted_biased-1-train_env-PNL_no_answer_biased_debiased_QA-0.*',
                '.*PNL-adjusted_biased-0-train_env-PNL_no_answer_biased_debiased_QA-1.*',
                '.*PNL-adjusted_biased-1-train_env-PNL_no_answer_biased_debiased_QA-1.*'])
for regex in regexs:
    plot_all_runs(df, regex)"""

regexs = ['.*POP-adjusted-train_env-multienv2_no_answer-lambda_28-model-0_7.*',
          '.*POP-adjusted-train_env-multienv3_no_answer-lambda_38-model-0_10.*',
          '.*POP-GTM-RS-Online-025.*',
          '.*POP-RIDE-reward_scale_20-lambda_05.*']
# plot_average(df, regexs)
# plot_variance(df, regexs)
"""plot_SR_QA(df, ['.*POP-adjusted-train_env-multienv2_no_answer-lambda_28-model-0_7.*',
                '.*POP-adjusted-train_env-multienv3_no_answer-lambda_38-model-0_10.*'])
for regex in regexs:
    plot_all_runs(df, regex)"""

regexs = ['.*paral_SEQ-adjusted-train_env-multienv2_no_answer-lambda_026-model-0_7.*',
          '.*paral_SEQ-adjusted-train_env-multienv3_no_answer-lambda_026-model-0_10.*',
          '.*SEQ-GTM-RS-Online-05-D.*',
          '.*SEQ-RIDE-reward_scale_20-lambda_05.*']

plot_average(df, regexs)
# plot_variance(df, regexs)
plot_SR_QA(df, ['.*paral_SEQ-adjusted-train_env-multienv2_no_answer-lambda_026-model-0_7.*',
                '.*paral_SEQ-adjusted-train_env-multienv3_no_answer-lambda_026-model-0_10.*'])
for regex in regexs:
    plot_all_runs(df, regex)

regexs = ['.*PNL-adjusted-train_env-PNL-lambda_24-model-0_6.*',
          '.*PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10.*',
          '.*PNL-simple-train_env-PNL-lambda_24-model-0_6.*',
          '.*PNL-simple-train_env-PNL_no_answer-lambda_24-model-2_10.*',
          '.*PNL-RIDE-reward_scale_20-lambda_05.*',
          '.*PNL-GTL-RS-Online-025-D.*']
"""plot_average(df, regexs)
plot_variance(df, regexs)

for regex in regexs:
    plot_all_runs(df, regex)"""

regexs = ['.*paral_PNM-adjusted-train_env-multienv2_no_answer-lambda_16-model-0_7.*',
          '.*paral_PNM-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10.*',
          '.*PNM-RIDE-reward_scale_20-lambda_05.*',
          '.*PNM-GTM-RS-Online-025-D.*']
# plot_average(df, regexs)
"""plot_variance(df, regexs)
plot_SR_QA(df, ['.*paral_PNM-adjusted-train_env-multienv2_no_answer-lambda_16-model-0_7.*',
                '.*paral_PNM-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10.*'])
for regex in regexs:
    plot_all_runs(df, regex)"""

regexs = ['.*paral_PNLA-adjusted-train_env-multienv2_no_answer-lambda_043-model-0_7.*',
          '.*paral_PNLA-adjusted-train_env-multienv3_no_answer-lambda_043-model-0_10.*',
          '.*PNLA-GTM-RS-Online-0068-D.*',
          '.*PNLA-RIDE-reward_scale_20-lambda_05.*']
plot_average(df, regexs)
# plot_variance(df, regexs)
plot_SR_QA(df, ['.*paral_PNLA-adjusted-train_env-multienv2_no_answer-lambda_043-model-0_7.*',
                '.*paral_PNLA-adjusted-train_env-multienv3_no_answer-lambda_043-model-0_10.*'])
for regex in regexs:
    plot_all_runs(df, regex)

regexs = ['.*paral_UNLM-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10.*',
          '.*UNLM-PUM-RS-Online-025-D.*',
          '.*UNLM-RIDE-reward_scale_20-lambda_05.*']
plot_average(df, regexs)
"""plot_SR(df, ['.*paral_UNLM-adjusted-train_env-multienv3_no_answer-lambda_48-model-0_10.*',
             '.*paral_UNLM-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10.*'])
plot_SR_QA(df, ['.*paral_UNLM-adjusted-train_env-multienv3_no_answer-lambda_48-model-0_10.*',
                '.*paral_UNLM-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10.*'])
for regex in regexs:
    plot_all_runs(df, regex)"""
