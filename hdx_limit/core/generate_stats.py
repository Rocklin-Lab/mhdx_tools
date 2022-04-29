import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob as glob
import argparse

sys.path.append(os.getcwd() + '/workflow/scripts/hdx_limit/')
from hdx_limit.core.io import limit_read

def generate_df_rank(l, cols):
    # Create dataframe from list (l) and column names (cols)
    # Compute score per ic and subrank classification
    # Return dataframe (df)
    df = pd.DataFrame(l, columns=cols)
    df.drop(['ID'], axis=1)
    df['total_score'] = df.drop(labels=['ID', 'mono_or_multi','n_tps'], axis=1).sum(axis=1)
    df['score_per_ic'] = df['total_score'] / (df['n_tps'] - 1)
    df.sort_values(by='score_per_ic', inplace=True, ignore_index=True)
    df['subrank'] = 0
    for poi in set(['_'.join(rt_group.split('_')[:-1]) for rt_group in list(df.ID)]):
        subrank = 0
        for name in df.loc[df.ID.str.contains(poi)].ID:
            df.loc[df.ID == name, 'subrank'] = subrank
            subrank += 1
    return df

def plot_distribution(df, key, label, position):
    # Inputs: df: dataframe, key: column name, label (x label string), position (position in the plot)
    sns.histplot(df[key], ax=position, kde=True)
    aux = position.twinx()
    sns.ecdfplot(df[key], ax=aux, color='red', lw=1, ls='-')
    aux.grid()
    position.set_xlabel(label, weight='bold')
    position.set_ylabel('Count', weight='bold')
    aux.set_ylabel('Proportion', weight='bold', color='red')
    aux.tick_params(axis='y', colors='red')


if __name__ == '__main__':
    # Get paths of pkl dataframes generated with ajf_plot
    dfs = glob.glob(os.getcwd() + '/resources/10_ic_time_series/*/*/*df.pkl')

    cols = ['ID', 'mono_or_multi', 'RT', 'winner_intensity_frac', 'coverage_atc', 'coverage_prefiltered',
            'complexity_atc', 'complexity_prefiltered']

    # Generate stats dataframe
    # Compute the following:
    # 'winner_intensity_frac': intensity of winners relative to parent tensors
    # 'coverage_atc': average number of charge states  similar to winner ICs considering all timepoints clusters (ATC)
    # 'coverage_prefiltered': average number of charge states similar to winner ICs considering prefiltered ICs
    # 'complexity_atc': average number of ICs per timepoint before filtering out ICs
    # 'complexity_prefiltered': average number of ICs per timepoint after filtering out ICs
    l = []
    for df in dfs:
        df_tmp = pd.read_pickle(df)
        name = '_'.join(df.split('/')[-1].split('_')[:-2])
        rt = float(df.split('/')[-1].split('_')[:-2][-1])
        mono_or_multi = df.split('/')[-1].split('_')[-1].replace('.df.pkl', '')
        winner_intensity_frac = df_tmp[df_tmp['winner'] == 1]['ic_auc'].sum() / df_tmp[df_tmp['winner'] == 1][
            'tensor_auc'].sum()
        coverage_atc = len(
            df_tmp[(df_tmp['prefiltered'] == 0) & (df_tmp['winner'] == 0) & (df_tmp['ic_winner_corr'] >= 0.95)]) / len(
            np.unique(df_tmp['tp_idx'].values))
        coverage_prefiltered = len(
            df_tmp[(df_tmp['prefiltered'] == 1) & (df_tmp['winner'] == 0) & (df_tmp['ic_winner_corr'] >= 0.95)]) / len(
            np.unique(df_tmp['tp_idx'].values))
        complexity_atc = len(df_tmp[(df_tmp['prefiltered'] == 0) & (df_tmp['winner'] == 0)]) / len(
            np.unique(df_tmp['tp_idx'].values))
        complexity_prefiltered = len(df_tmp[(df_tmp['prefiltered'] == 1) & (df_tmp['winner'] == 0)]) / len(
            np.unique(df_tmp['tp_idx'].values))
        l.append([name, mono_or_multi, rt, winner_intensity_frac, coverage_atc, coverage_prefiltered, complexity_atc,
                  complexity_prefiltered])

    df_stats = pd.DataFrame(l, columns=cols)

    # Get path with cpickle.zlib score files
    winners_scores = glob.glob(os.getcwd() + '/resources/10_ic_time_series/*/*/*scores*')

    # Generate dataframes with score information and ranks for monobody and multibody paths
    l_mono = []
    l_multi = []
    for winner_score in winners_scores:
        if not os.stat(winner_score).st_size == 0:
            f = limit_read(winner_score)
            ID = '_'.join(winner_score.split('/')[-1].split('_')[:-3])
            mono_or_multi = winner_score.split('/')[-2:-1][0]
            n_tps = len(limit_read(winner_score.replace('_scores', '')))
            if mono_or_multi == 'monobody':
                l_mono.append(([ID] + [mono_or_multi] + [f[key][0] * f[key][1] for key in f.keys()] + [n_tps]))
                cols_mono = ['ID', 'mono_or_multi'] + list(f.keys()) + ['n_tps']
            else:
                l_multi.append(([ID] + [mono_or_multi] + [f[key][0] * f[key][1] for key in f.keys()] + [n_tps]))
                cols_multi = ['ID', 'mono_or_multi'] + list(f.keys()) + ['n_tps']

    df_mono_rank = generate_df_rank(l_mono, cols=cols_mono)
    df_multi_rank = generate_df_rank(l_multi, cols=cols_multi)

    # Merge dataframes based on ID and monobody or multibody score terms
    # Only keeps IDs present in both dataframes
    df_mono_complete = pd.merge(df_mono_rank, df_stats, on=['ID', 'mono_or_multi'], how='inner')
    df_multi_complete = pd.merge(df_multi_rank, df_stats, on=['ID', 'mono_or_multi'], how='inner')

    # Save dataframes
    df_mono_complete.to_csv(os.getcwd() + '/results/plots/ic_time_series/all_scores_all_prots_monobody.csv')
    df_multi_complete.to_csv(os.getcwd() + '/results/plots/ic_time_series/all_scores_all_prots_multibody.csv')

    # Generate distribution plots
    sns.set_context('talk')

    fig, ax = plt.subplots(3, 4, figsize=(22, 10), dpi=300)

    plot_distribution(df_mono_complete, 'score_per_ic', 'score per ic', ax[0][0])
    plot_distribution(df_mono_complete, 'winner_intensity_frac', 'winner intensity fraction', ax[0][1])
    plot_distribution(df_multi_complete, 'score_per_ic', 'score per ic', ax[0][2])
    plot_distribution(df_multi_complete, 'winner_intensity_frac', 'winner intensity fraction', ax[0][3])
    plot_distribution(df_mono_complete, 'coverage_atc', 'coverage atc', ax[1][0])
    plot_distribution(df_mono_complete, 'coverage_prefiltered', 'coverage prefiltered', ax[1][1])
    plot_distribution(df_multi_complete, 'coverage_atc', 'coverage atc', ax[1][2])
    plot_distribution(df_multi_complete, 'coverage_prefiltered', 'coverage prefiltered', ax[1][3])
    plot_distribution(df_mono_complete, 'complexity_atc', 'complexity atc', ax[2][0])
    plot_distribution(df_mono_complete, 'complexity_prefiltered', 'complexity prefiltered', ax[2][1])
    plot_distribution(df_multi_complete, 'complexity_atc', 'complexity atc', ax[2][2])
    plot_distribution(df_multi_complete, 'complexity_prefiltered', 'complexity prefiltered', ax[2][3])
    plt.tight_layout()

    plt.savefig(os.getcwd() + '/results/plots/ic_time_series/stats.pdf', dpi=300, format='pdf')
