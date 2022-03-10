import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import seaborn as sns
import numpy as np
from sklearn.cluster import DBSCAN
import glob as glob
import yaml
import math
import argparse

from hdx_limit.core.io import limit_read


def create_df_and_clusterize(prefiltered_ics, winner, tps, cluster_radius=0.75, output=None):
    '''
    Create and returns dataframe from prefiltered_ics and winner ics
    ic: ic object
    tp_idx: timepoint index
    com: center of mass (not needed)
    rt: retention time
    dt: drift time
    charge: charge
    auc: area under curve, i.e., XIC
    winner: bool if belongs to winner ics set
    maxint: max intensity amoung ics of same charge state
    '''
    df = pd.DataFrame(columns=['ic', 'tp_idx', 'com', 'rt', 'dt', 'charge', 'auc', 'winner', 'maxint', 'tensor_auc',
                               'factor_auc', 'ic_auc'])
    cols = df.columns
    for tp in prefiltered_ics:
        for ic in tp:
            tp_idx = tps.index(ic.timepoint_idx)
            com = ic.baseline_integrated_mz_com
            rt = ic.retention_labels[0] + (ic.retention_labels[1] - ic.retention_labels[0])*ic.rt_com
            dt = ic.drift_labels[0] + (ic.drift_labels[1] - ic.drift_labels[0])*ic.dt_coms
            charge = ic.charge_states[0]
            auc = ic.auc[0]
            win = 0
            maxint = max(ic.baseline_integrated_mz)
            df_tmp = pd.Series({cols[0]:ic,
                                cols[1]:tp_idx,
                                cols[2]:com,
                                cols[3]:rt,
                                cols[4]:dt,
                                cols[5]: charge,
                               cols[6]: auc,
                               cols[7]: win,
                               cols[8]: maxint,
                               cols[9]: ic.tensor_auc,
                               cols[10]: ic.factor_auc,
                               cols[11]: ic.ic_auc})
            df = df.append(df_tmp, ignore_index=True)
    for ic in winner:
        tp_idx = tps.index(ic.timepoint_idx)
        com = ic.baseline_integrated_mz_com
        rt = ic.retention_labels[0] + (ic.retention_labels[1] - ic.retention_labels[0])*ic.rt_com
        dt = ic.drift_labels[0] + (ic.drift_labels[1] - ic.drift_labels[0])*ic.dt_coms
        charge = ic.charge_states[0]
        auc = ic.auc[0]
        win = 1
        maxint = max(ic.baseline_integrated_mz)
        df_tmp = pd.Series({cols[0]:ic,
                            cols[1]:tp_idx,
                            cols[2]:com,
                            cols[3]:rt,
                            cols[4]:dt,
                            cols[5]: charge,
                           cols[6]: auc,
                           cols[7]: win,
                           cols[8]: maxint,
                           cols[9]: ic.tensor_auc,
                           cols[10]: ic.factor_auc,
                           cols[11]: ic.ic_auc})
        df = df.append(df_tmp, ignore_index=True)

    # Normlize auc relative to max intensity of ics with same charge
    df['auc_size'] = 0
    for i, line in df.iterrows():
        df.loc[i, 'auc_size'] = np.log2(df.loc[i]['auc']) #/ df[df['charge'] == df.loc[i]['charge']]['auc'].min()

    # z-score dt
    df['dt_norm'] = 0
    charge_states = sorted(np.unique(df.charge.values))
    for charge in charge_states:
        avg = df[df['charge'] == charge]['dt'].mean()
        std = df[df['charge'] == charge]['dt'].std()
        if not math.isnan(std):
            df.loc[df['charge'] == charge, 'dt_norm'] = (df[df['charge'] == charge]['dt'] - avg) / std
        else:
            df.loc[df['charge'] == charge, 'dt_norm'] = 0

    # Create a correction RT based on time wrapped retention labels
    for i, line in df.iterrows():
        df.loc[i, 'rt_corr'] = line['rt'] - line['ic'].retention_labels[int(len(line['ic'].retention_labels)/2)]
    #z-score rt
    df['rt_norm'] = (df['rt_corr'] - df['rt_corr'].mean())/df['rt_corr'].std()

    # Clusterize based on rt and dt
    db = DBSCAN(eps=cluster_radius)
    db.fit(df[['rt_norm', 'dt_norm']])
    clusters = db.fit_predict(df[['rt_norm', 'dt_norm']])
    df['clusters'] = clusters

    if output != None:
        #Plot rt/dt scatter plot coloring dots according to charge or cluster id
        fig, ax = plt.subplots(1,3, figsize=(21,5), dpi=200)
        sns.scatterplot(x=df['dt'], y=df['rt_corr'], hue=df['charge'], palette='bright', ax=ax[0], s=5*(df['auc_size']),
                        alpha=0.7)
        sns.scatterplot(x=df['dt_norm'], y=df['rt_norm'], hue=df['clusters'], palette='bright', ax=ax[1],
                        s=5*(df['auc_size']), alpha=0.7)
        sns.scatterplot(x=df['dt'], y=df['rt_corr'], hue=df['clusters'], palette='bright', ax=ax[2],
                        s=5*(df['auc_size']), alpha=0.7)
        ax[0].set_xlabel('DT')
        ax[0].set_ylabel('RT')
        ax[1].set_xlabel('DT z-score')
        ax[1].set_ylabel('RT z-score')
        ax[2].set_xlabel('DT')
        ax[2].set_ylabel('RT')
        plt.tight_layout()
        plt.savefig(output, dpi=200)

    return df

def ajf_plot(df, winner, tps, output_path):

    pal = sns.color_palette('bright')
    n_cols = 3*len(set(df.charge)) + 4
    min_clust = min(df['clusters'])

    fig = plt.figure(figsize=(2.*n_cols, 40))

    gs = gridspec.GridSpec(2,1, height_ratios=[1, 15], wspace=0., hspace=0.05)

    gs0 = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs[0], wspace=2, hspace=0)

    # Clean left top corner
    ax_clean = fig.add_subplot(gs0[:5])
    ax_clean.axis('off')

    charge_states = sorted(np.unique(df.charge.values))
    # Define top RT/DT scatter plots
    ax_scatter = {}
    for i, charge in enumerate(charge_states):
        ax_scatter[i] = fig.add_subplot(gs0[3*i+4:3*i+7])
        sns.scatterplot(data=df[df['charge'] == charge], x='dt', y='rt_corr', hue=df['clusters']-min_clust,
                        palette='bright', ax=ax_scatter[i], s=5*(df[df['charge'] == charge]['auc_size']), alpha=0.7)
        ax_scatter[i].set_ylim(-0.4, 0.4)
        ax_scatter[i].set_xlim(df[df['charge'] == charge]['dt'].min()-0.05, df[df['charge'] == charge]['dt'].max()+0.05)
        ax_scatter[i].tick_params(axis="y", labelsize=10)
        ax_scatter[i].tick_params(axis="x", labelsize=10)
        ax_scatter[i].set_xlabel('DT', fontsize=10)
        ax_scatter[i].set_ylabel('RT', fontsize=10)
        ax_scatter[i].grid()
        ax_scatter[i].legend('', frameon=False)
    # Label winners and undeuterated ics on scatter plots
    for ic in winner:
        tp_idx = tps.index(ic.timepoint_idx)
        charge = ic.charge_states[0]
        rt = df[df['ic'] == ic]['rt_corr']
        dt = df[df['ic'] == ic]['dt']
        ax_scatter[charge_states.index(int(charge))].text(dt, rt, str(tp_idx), fontsize=8)
    for i, line in df[df['tp_idx'] == 0].iterrows():
        ax_scatter[charge_states.index(int(line['charge']))].text(float(line['dt']), float(line['rt_corr']), 'x',
                                                                  fontsize=10, color='black',)

    # Add legend for cluster information
    legend_elements = [Circle(1, label='cluster %i'%i,
                             facecolor=pal[i-min_clust]) for i in sorted(set(df['clusters'].values))]
    ax_clean.legend(handles=legend_elements, prop={'size': 12}, loc='right',
                   bbox_to_anchor=(0.8, 0.5), bbox_transform=ax_clean.transAxes, borderpad=0.02, columnspacing=0.4,
                    handletextpad=0.3, frameon=False)

    gs1 = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs[1], wspace=0.3, hspace=0.0)

    # Plot winner 1st column
    ax_win = fig.add_subplot(gs1[:2])
    for ic in winner:
        tp_idx = tps.index(ic.timepoint_idx)
        color_index = int(df[(df['ic'] == ic) & (df['winner'] == 1)]['clusters'] + 1)
        ax_win.plot(ic.baseline_integrated_mz/max(ic.baseline_integrated_mz)-tp_idx, c=pal[color_index])
        ax_win.text(0.02, 0.8-tp_idx,'tp_idx=%i'%int(tp_idx), horizontalalignment='left', verticalalignment='center',
                    fontsize=12)
        ax_win.text(0.02, 0.55-tp_idx,'tp=%is'%int(tps[tp_idx]), horizontalalignment='left', verticalalignment='center',
                    fontsize=12)
        ax_win.text(len(ic.baseline_integrated_mz), 0.8-tp_idx,'charge=%i+'%int(ic.charge_states[0]),
                    horizontalalignment='right', verticalalignment='center', fontsize=12)
        if tp_idx == 0:
            ax_win.text(0.02, 1.005,'winner path',
             horizontalalignment='left',
             verticalalignment='center',
             transform = ax_win.transAxes,
                           fontsize=12)
    ax_win.set_ylim(-len(tps)+0.95, 1.05)
    ax_win.set_yticks([])
    ax_win.set_xticks(np.arange(0, len(ic.baseline_integrated_mz)+1, 10))

    #Plot alternatives charge states all together 2nd column
    ax_alt = fig.add_subplot(gs1[2:4])
    for i, line in df.iterrows():
        ax_alt.plot((line['ic'].baseline_integrated_mz/max(line['ic'].baseline_integrated_mz))*(np.log2(line['auc'])/np.log2(df['auc'].max())) - int(line['tp_idx']),
                    c=pal[charge_states.index(int(line['charge']))])
    ax_alt.set_ylim(-len(tps)+0.95, 1.05)
    ax_alt.set_yticks([])
    ax_alt.set_xticks(np.arange(0, len(ic.baseline_integrated_mz)+1, 10))

    # Add legend with charge state information
    legend_elements = [Circle(1, label='%i+'%charge_states[i],
                             facecolor=pal[i]) for i in range(len(charge_states))]
    ax_alt.legend(handles=legend_elements, prop={'size': 12}, loc='upper center', ncol=len(charge_states),
                   bbox_to_anchor=(0.5, 1.01), bbox_transform=ax_alt.transAxes, borderpad=0.02, columnspacing=0.4,
                  handletextpad=0.1, frameon=False)


    # Plot ics per charge state and their rt/dt scatter distributions
    charge_states = sorted(np.unique(df.charge.values))

    # A. Define dict with ics and dt/rt scatter plot defined as a block
    ax_charge_states = {}
    # B. Define dict with ics within a block
    ax_charge_states_ics = {}
    # C. Define dict with dt/rt scatter plot within a block
    ax_charge_states_scatter = {}

    # Define grids per charge state and plot ics (A and B)
    for i, charge in enumerate(charge_states):
        ax_charge_states[i] = gridspec.GridSpecFromSubplotSpec(len(tps), 3, subplot_spec=gs1[3*i+4:3*i+7], wspace=0.05,
                                                               hspace=0.1)
        ax_charge_states_ics[i] = fig.add_subplot(ax_charge_states[i][:,:2])
        for _, line in df[df['charge'] == charge].iterrows():
            if line['winner'] == 0:
                ax_charge_states_ics[i].plot((line['ic'].baseline_integrated_mz/max(line['ic'].baseline_integrated_mz))*(np.log2(line['auc'])/np.log2(df[(df['charge'] == line['charge'])]['auc'].max())) - int(line['tp_idx']),
                    c=pal[int(line['clusters'])-min_clust])
            else:
                ax_charge_states_ics[i].plot((line['ic'].baseline_integrated_mz/max(line['ic'].baseline_integrated_mz))*(np.log2(line['auc'])/np.log2(df[(df['charge'] == line['charge'])]['auc'].max())) - int(line['tp_idx']),
                    c=pal[int(line['clusters'])-min_clust], lw=4)
        set_tps = set(df[df['charge'] == charge]['tp_idx'])
        for tp in set_tps:
            factor_tensor_frac = sum(set(df[(df['charge'] == charge) & (df['tp_idx'] == tp) & (df['winner'] == 0)]['factor_auc']))/df[(df['charge'] == charge) & (df['tp_idx'] == tp)]['tensor_auc'].values[0]
            ics_tensor_frac = sum(df[(df['charge'] == charge) & (df['tp_idx'] == tp) & (df['winner'] == 0)]['ic_auc'])/df[(df['charge'] == charge) & (df['tp_idx'] == tp)]['tensor_auc'].values[0]
            ax_charge_states_ics[i].text(len(ic.baseline_integrated_mz), 0.3-tp,
                                         'factors|tensor=%.2f'%(factor_tensor_frac), horizontalalignment='right',
                                         verticalalignment='center', fontsize=10)
            ax_charge_states_ics[i].text(len(ic.baseline_integrated_mz), 0.1-tp,
                                         'ics|tensor=%.2f'%(ics_tensor_frac), horizontalalignment='right',
                                         verticalalignment='center', fontsize=10)
        ax_charge_states_ics[i].grid()
        ax_charge_states_ics[i].text(len(ic.baseline_integrated_mz), 0.9,'charge=%i+'%int(charge),
                                     horizontalalignment='right', verticalalignment='center', fontsize=12)
        ax_charge_states_ics[i].text(len(ic.baseline_integrated_mz), 0.7,
                                     'max_auc=%.1e'%df[df['charge'] == charge]['auc'].max(),
                                     horizontalalignment='right', verticalalignment='center', fontsize=12)
        ax_charge_states_ics[i].text(len(ic.baseline_integrated_mz), 0.5,  'idotp=%.3f'%df[(df['charge'] == charge) & (df['tp_idx'] == 0)]['ic'].values[0].undeut_ground_dot_product,
                                     horizontalalignment='right', verticalalignment='center', fontsize=12)
        ax_charge_states_ics[i].set_ylim(-len(tps)+0.95, 1.05)
        ax_charge_states_ics[i].set_yticks([])
        ax_charge_states_ics[i].set_xticks(np.arange(0, len(ic.baseline_integrated_mz)+1, 10))


    # Plot rt/dt scatter plots
    for i, charge in enumerate(charge_states):
        for j in range(len(tps)):
            ax_charge_states_scatter[i+j] = fig.add_subplot(ax_charge_states[i][j,2])
            sns.scatterplot(data=df[(df['charge'] == charge) & (df['tp_idx'] == j)], x='dt', y='rt_corr',
                                                 hue=df['clusters']-min_clust, palette='bright',
                            s=5*(df[(df['charge'] == charge) & (df['tp_idx'] == j)]['auc_size']), alpha=0.7,
                            ax=ax_charge_states_scatter[i+j])
            ax_charge_states_scatter[i+j].text(float(df[(df['charge'] == charge) & (df['tp_idx'] == 0) & (df['winner'] == 0)].sort_values(by='idotp')['dt'].values),
                                               float(df[(df['charge'] == charge) & (df['tp_idx'] == 0) & (df['winner'] == 0)]['rt_corr'].values),
                                                    'x', fontsize=10, color='black', ha='center', va='center')
            ax_charge_states_scatter[i+j].set(xlabel=None,ylabel=None)
            ax_charge_states_scatter[i+j].set_yticks([])
            ax_charge_states_scatter[i+j].set_xticks([])
            ax_charge_states_scatter[i+j].legend('', frameon=False)
            ax_charge_states_scatter[i+j].set_ylim(-0.4, 0.4)
            ax_charge_states_scatter[i+j].set_xlim(df[df['charge'] == charge]['dt'].min()-0.05, df[df['charge'] == charge]['dt'].max()+0.05)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close('all')


def plot_ajf_(configfile, atc, winner, output_path):

    df = create_df_and_clusterize(atc, winner, tps=configfile['timepoints'])
    ajf_plot(df, winner=winner, tps=configfile['timepoints'], output_path=output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--configfile",
        help=
        "Configfile path. /config/config.yaml"
    )
    parser.add_argument(
        "-a",
        "--atc",
        help=
        "All timepoint clusters (or all prefiltered ics)"
    )
    parser.add_argument(
        "-w",
        "--winner",
        help=
        "Winner path"
    )
    parser.add_argument(
        "-o",
        "--output",
        help=
        "Output path"
    )

    args = parser.parse_args()


    configfile = yaml.load(open(args.configfile, "rb").read(), Loader=yaml.Loader)
    atc = limit_read(args.atc)
    winner = limit_read(args.winner)

    ajf_plot(configfile=configfile,
             atc=atc,
             winner=winner,
             output_path=args.output)
