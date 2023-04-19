# import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import yaml
import math
import argparse
from pathlib import Path

# sys.path.append(os.getcwd() + "/workflow/scripts/hdx_limit/")
from hdx_limit.core.io import limit_read


def get_attributes_from_ic(tps, ic):

    tp = ic.timepoint_idx
    tp_idx = tps.index(tp)
    com = ic.baseline_integrated_mz_com
    rt = ic.retention_labels[0] + (ic.retention_labels[1] - ic.retention_labels[0]) * ic.rt_com
    rt_tensor_center = ic.retention_labels[len(ic.retention_labels) // 2]
    dt = ic.drift_labels[0] + (ic.drift_labels[1] - ic.drift_labels[0]) * ic.dt_coms
    charge = ic.charge_states[0]
    auc = ic.auc[0]
    maxint = max(ic.baseline_integrated_mz)

    return [
        ic, tp, tp_idx, com, rt, rt_tensor_center, dt, charge, auc, maxint, ic.tensor_auc, ic.factor_auc, ic.ic_auc
    ]


def create_df_and_clusterize(configfile, atc, prefiltered_ics, winner, output_plot_path=None, output_df_path=None):
    """
    Create and returns dataframe from atc, prefiltered_ics and winner ics
    ic: ic object
    tp: timepoint
    com: center of mass (not needed)
    rt: retention time
    dt: drift time
    charge: charge
    auc: area under curve, i.e., XIC
    winner: bool if belongs to winner ics set
    maxint: max intensity among ics of same charge state
    """

    tps = configfile["timepoints"]

    tmp = []
    for ics in atc:
        for ic in ics:
            tmp.append(get_attributes_from_ic(tps, ic) + [0, 0])
    if prefiltered_ics is not None:
        for ics in prefiltered_ics:
            for ic in ics:
                tmp.append(get_attributes_from_ic(tps, ic) + [1, 0])
    if winner is not None:
        for ic in winner:
            tmp.append(get_attributes_from_ic(tps, ic) + [0, 1])

    cols = ["ic", "tp", "tp_idx", "com", "rt", "rt_tensor_center", "dt", "charge", "auc", "maxint", "tensor_auc",
            "factor_auc", "ic_auc", "prefiltered", "winner"]

    df = pd.DataFrame(tmp, columns=cols)

    # Remove lines with NAN values. This is pretty rare!
    df.dropna(inplace=True)

    # Replace ics based on large AUC extrapolations by median value
    # And small AUCs # TODO Check why auc are so large or why Rts suffer from deviation: gaussian extrapolation failing?
    df["auc"] = np.where(df["auc"] > 1e10, np.percentile(df["auc"], 95), df["auc"])
    df["auc"] = np.where(df["auc"] < 1e1, 1e1, df["auc"])
    # # Replace unreasonable RT
    # df["rt"] = np.where((df["rt"] - np.median(df["rt"])) > 0.5, np.median(df["rt"])+0.5, df["rt"])

    # Compute dot product between winner ic and all other ics from that timepoint
    df["ic_winner_corr"] = -1
    for i, line in df[(df["winner"] == 1)].iterrows():
        df.loc[df["tp_idx"] == line["tp_idx"], "ic_winner_corr"] = [round(np.linalg.norm(
            np.dot(line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz),
                   row["ic"].baseline_integrated_mz / max(row["ic"].baseline_integrated_mz))
        ) / np.linalg.norm(line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) / np.linalg.norm(
            row["ic"].baseline_integrated_mz / max(row["ic"].baseline_integrated_mz)), 3
                                                                          )
                                                                    for i, row in

                                                                    df[df["tp_idx"] == line["tp_idx"]].iterrows()]

    # Normalize auc relative to max intensity of ics with same charge
    df["auc_size"] = np.log2(df["auc"])

    # z-score dt
    df["dt_norm"] = 0
    charge_states = sorted(np.unique(df.charge.values))
    for charge in charge_states:
        avg = df[df["charge"] == charge]["dt"].mean()
        std = df[df["charge"] == charge]["dt"].std()
        if not math.isnan(std):
            df.loc[df["charge"] == charge, "dt_norm"] = (df[df["charge"] == charge]["dt"] - avg) / std

    # Create a correction RT based on time wrapped retention labels
    df["rt_corr"] = df["rt"] - df["rt_tensor_center"]
    # Limit offset on the rt dimension
    df["rt_corr"] = np.where(df["rt_corr"] > 0.45, 0.45, df["rt_corr"])
    df["rt_corr"] = np.where(df["rt_corr"] < -0.45, -0.45, df["rt_corr"])
    # z-score rt
    df["rt_norm"] = (df["rt_corr"] - df["rt_corr"].mean()) / df["rt_corr"].std()

    # Clusterize based on rt and dt
    n = df[(df["prefiltered"] == 0) & (df["tp_idx"] != 0)].groupby(by="tp_idx", sort=True).count().max()[0]
    if n > 9:
        n = 9
    kmeans = KMeans(n_clusters=n,
                    n_init="auto")
    df["kmeans_clusters"] = kmeans.fit_predict(df[["rt_norm", "dt_norm"]])

    if output_plot_path is not None:
        # Plot rt/dt scatter plot coloring dots according to charge or cluster id
        fig, ax = plt.subplots(1, 3, figsize=(21, 5), dpi=200)
        sns.scatterplot(x=df["dt"], y=df["rt_corr"], hue=df["charge"], palette="bright", ax=ax[0],
                        s=5 * (df["auc_size"]),
                        alpha=0.7)
        sns.scatterplot(x=df["dt_norm"], y=df["rt_norm"], hue=df["kmeans_clusters"], palette="bright", ax=ax[1],
                        s=5 * (df["auc_size"]), alpha=0.7)
        sns.scatterplot(x=df["dt"], y=df["rt_corr"], hue=df["kmeans_clusters"], palette="bright", ax=ax[2],
                        s=5 * (df["auc_size"]), alpha=0.7)
        ax[0].set_xlabel("DT")
        ax[0].set_ylabel("RT")
        ax[1].set_xlabel("DT z-score")
        ax[1].set_ylabel("RT z-score")
        ax[2].set_xlabel("DT")
        ax[2].set_ylabel("RT")
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=200)
        plt.close()

    if output_df_path is not None:

        df.to_json(output_df_path)

    return df


def ajf_plot(df, configfile, output_plot_path=None, dpi=300):

    if len(df.query("winner == 1")) > 0:
        winner = df.query("winner == 1")["ic"].values
    else:
        winner = None

    tps = configfile["timepoints"]

    ic_winner_corr_cutoff = 0.95
    pal = sns.color_palette("bright")
    n_cols = 6 * len(set(df.charge)) + 6
    min_clust = min(df["kmeans_clusters"])

    x_max = len(df.iloc[0]["ic"].baseline_integrated_mz)

    fig = plt.figure(figsize=(2. * n_cols, 2.5 + 1.2 * len(tps)))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 15], wspace=0., hspace=0.05)

    gs0 = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs[0], wspace=2, hspace=0)

    # Clean left top corner
    ax_clean = fig.add_subplot(gs0[:7])
    ax_clean.axis("off")

    # Add information if files are present
    if len(df) == 0:
        ax_clean.text(0, 0.9, "ATC is NONE", fontsize=12, weight="bold", ha="left")
    if len(df[df["prefiltered"] == 1]) > 0:
        prefiltered_ics = True
    else:
        prefiltered_ics = None
        ax_clean.text(0, 0.7, "PREFILTERED is NONE", fontsize=12, weight="bold", ha="left")
    if winner is None:
        ax_clean.text(0, 0.5, "WINNER is NONE", fontsize=12, weight="bold", ha="left")

    charge_states = sorted(np.unique(df.charge.values))
    # Define top RT/DT scatter plots for ATC
    ax_scatter_atc = {}
    for i, charge in enumerate(charge_states):
        ax_scatter_atc[i] = fig.add_subplot(gs0[6 * i + 6:6 * i + 9])
        sns.scatterplot(data=df[(df["charge"] == charge) & (df["prefiltered"] == 0)], x="dt", y="rt_corr",
                        hue=df["kmeans_clusters"] - min_clust,
                        palette="bright", ax=ax_scatter_atc[i],
                        s=5 * (df[(df["charge"] == charge) & (df["prefiltered"] == 0)]["auc_size"]), alpha=0.7)
        ax_scatter_atc[i].set_ylim(-0.4, 0.4)
        ax_scatter_atc[i].set_xlim(df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[0],
                                   df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[-1])
        ax_scatter_atc[i].tick_params(axis="y", labelsize=10)
        ax_scatter_atc[i].tick_params(axis="x", labelsize=10)
        ax_scatter_atc[i].set_xlabel("DT", fontsize=10)
        ax_scatter_atc[i].set_ylabel("RT", fontsize=10)
        ax_scatter_atc[i].grid()
        ax_scatter_atc[i].legend("", frameon=False)
        for i, line in df[(df["tp_idx"] == 0) & (df["charge"] == charge) & (df["winner"] == 0)
                          & (df["prefiltered"] == 0)].iterrows():
            if line["ic"].idotp >= 0.98:
                ax_scatter_atc[charge_states.index(int(line["charge"]))].text(float(line["dt"]), float(line["rt_corr"]),
                                                                              "x", fontsize=10, color="black", )
    if winner is not None:
        # Label winners and undeuterated ics on scatter plots
        for ic in winner:
            tp_idx = tps.index(ic.timepoint_idx)
            charge = ic.charge_states[0]
            rt = df[df["ic"] == ic]["rt_corr"].values[0]
            dt = df[df["ic"] == ic]["dt"].values[0]
            ax_scatter_atc[charge_states.index(int(charge))].text(dt, rt, str(tp_idx), fontsize=8)

    if prefiltered_ics is not None:
        # Define top RT/DT scatter plots for PREFILTERED ICS
        ax_scatter_prefiltered = {}
        for i, charge in enumerate(charge_states):
            ax_scatter_prefiltered[i] = fig.add_subplot(gs0[6 * i + 9:6 * i + 12])
            sns.scatterplot(data=df[(df["charge"] == charge) & (df["prefiltered"] == 1)], x="dt", y="rt_corr",
                            hue=df["kmeans_clusters"] - min_clust,
                            palette="bright", ax=ax_scatter_prefiltered[i],
                            s=5 * (df[(df["charge"] == charge) & (df["prefiltered"] == 1)]["auc_size"]), alpha=0.7)
            ax_scatter_prefiltered[i].set_ylim(-0.4, 0.4)
            ax_scatter_prefiltered[i].set_xlim(df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[0],
                                               df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[-1])
            ax_scatter_prefiltered[i].tick_params(axis="y", labelsize=10)
            ax_scatter_prefiltered[i].tick_params(axis="x", labelsize=10)
            ax_scatter_prefiltered[i].set_xlabel("DT", fontsize=10)
            ax_scatter_prefiltered[i].set_ylabel("RT", fontsize=10)
            ax_scatter_prefiltered[i].grid()
            ax_scatter_prefiltered[i].legend("", frameon=False)
            for i, line in df[(df["tp_idx"] == 0) & (df["charge"] == charge) & (df["winner"] == 0)
                          & (df["prefiltered"] == 1)].iterrows():
                ax_scatter_prefiltered[charge_states.index(int(line["charge"]))].text(float(line["dt"]),
                                                                                  float(line["rt_corr"]), "x",
                                                                                  fontsize=10, color="black", )
        if winner is not None:
            # Label winners and undeuterated ics on scatter plots
            for ic in winner:
                tp_idx = tps.index(ic.timepoint_idx)
                charge = ic.charge_states[0]
                rt = df[df["ic"] == ic]["rt_corr"].values[0]
                dt = df[df["ic"] == ic]["dt"].values[0]
                ax_scatter_prefiltered[charge_states.index(int(charge))].text(dt, rt, str(tp_idx), fontsize=8)


    # Add legend for cluster information
    legend_elements = [Circle(1, label="cluster %i" % i,
                              facecolor=pal[i - min_clust]) for i in sorted(set(df["kmeans_clusters"].values))]
    ax_clean.legend(handles=legend_elements, prop={"size": 12}, loc="right",
                    bbox_to_anchor=(0.87, 0.5), bbox_transform=ax_clean.transAxes, borderpad=0.02, columnspacing=0.4,
                    handletextpad=0.3, frameon=False)

    gs1 = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs[1], wspace=0.3, hspace=0.0)

    # Plot winner 1st column
    ax_win = fig.add_subplot(gs1[:2])
    if winner is not None:
        for ic in winner:
            tp_idx = tps.index(ic.timepoint_idx)
            color_index = int(df[(df["ic"] == ic) & (df["winner"] == 1)]["kmeans_clusters"] - min_clust)
            ax_win.plot(ic.baseline_integrated_mz / max(ic.baseline_integrated_mz) - tp_idx, c=pal[color_index])
            ax_win.text(0.02, 0.8 - tp_idx, "tp_idx=%i" % int(tp_idx), horizontalalignment="left",
                        verticalalignment="center",
                        fontsize=12)
            ax_win.text(0.02, 0.55 - tp_idx, "tp=%is" % int(tps[tp_idx]), horizontalalignment="left",
                        verticalalignment="center",
                        fontsize=12)
            ax_win.text(x_max, 0.8 - tp_idx, "charge=%i+" % int(ic.charge_states[0]),
                        horizontalalignment="right", verticalalignment="center", fontsize=12)
            if tp_idx == 0:
                ax_win.text(0.02, 1.001, "winner path",
                            horizontalalignment="left",
                            verticalalignment="baseline",
                            transform=ax_win.transAxes,
                            fontsize=14, weight="bold")
        ax_win.set_ylim(-len(tps) + 0.95, 1.05)
        ax_win.set_yticks([])
        ax_win.set_xticks(np.arange(0, x_max + 1, 10))
    else:
        ax_win.set_yticks([])
        ax_win.set_xticks([])

    # Plot alternatives charge states all together 2nd column ATC
    ax_alt_atc = fig.add_subplot(gs1[2:4])
    for i, line in df[df["prefiltered"] == 0].iterrows():
        ax_alt_atc.plot((line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                np.log2(line["auc"]) / np.log2(df["auc"].max())) - int(line["tp_idx"]),
                        c=pal[charge_states.index(int(line["charge"]))])
    if prefiltered_ics is not None:
        for i, line in df[(df["prefiltered"] == 1) & (df["tp_idx"] == 0)].iterrows():
            ax_alt_atc.plot((line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                    np.log2(line["auc"]) / np.log2(df["auc"].max())) - int(line["tp_idx"]),
                            c=pal[charge_states.index(int(line["charge"]))])
    ax_alt_atc.set_ylim(-len(tps) + 0.95, 1.05)
    ax_alt_atc.set_yticks([])
    ax_alt_atc.set_xticks(np.arange(0, x_max + 1, 10))
    ax_alt_atc.text(0.5, 1.01, "All ATC", transform=ax_alt_atc.transAxes, ha="center", weight="bold", fontsize=14)

    # Plot alternatives charge states all together 3rd column PREFILTERED
    ax_alt_prefiltered = fig.add_subplot(gs1[4:6])
    if prefiltered_ics is not None:
        for i, line in df[df["prefiltered"] == 1].iterrows():
            ax_alt_prefiltered.plot((line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                    np.log2(line["auc"]) / np.log2(df["auc"].max())) - int(line["tp_idx"]),
                                    c=pal[charge_states.index(int(line["charge"]))])
        ax_alt_prefiltered.set_ylim(-len(tps) + 0.95, 1.05)
        ax_alt_prefiltered.set_yticks([])
        ax_alt_prefiltered.set_xticks(np.arange(0, x_max + 1, 10))
        ax_alt_prefiltered.text(0.5, 1.01, "All PREFILTERED", transform=ax_alt_prefiltered.transAxes, ha="center",
                                weight="bold", fontsize=14)

        # Legend elements for alternative ics, 2nd and 3rd columns
        legend_elements = [Circle(1, label="%i+" % charge_states[i],
                                  facecolor=pal[i]) for i in range(len(charge_states))]
        ax_alt_atc.legend(handles=legend_elements, prop={"size": 12}, loc="upper center", ncol=len(charge_states),
                          bbox_to_anchor=(0.5, 1.01), bbox_transform=ax_alt_atc.transAxes, borderpad=0.02,
                          columnspacing=0.4,
                          handletextpad=0.1, frameon=False)
        ax_alt_prefiltered.legend(handles=legend_elements, prop={"size": 12}, loc="upper center",
                                  ncol=len(charge_states),
                                  bbox_to_anchor=(0.5, 1.01), bbox_transform=ax_alt_prefiltered.transAxes,
                                  borderpad=0.02,
                                  columnspacing=0.4,
                                  handletextpad=0.1, frameon=False)

    # Plot ics per charge state and their rt/dt scatter distributions
    charge_states = sorted(np.unique(df.charge.values))

    # A. Define dict with ics and dt/rt scatter plot defined as a block
    ax_charge_states_atc = {}
    ax_charge_states_prefiltered = {}
    # B. Define dict with ics within a block
    ax_charge_states_ics_atc = {}
    ax_charge_states_ics_prefiltered = {}
    # C. Define dict with dt/rt scatter plot within a block
    ax_charge_states_scatter_atc = {}
    ax_charge_states_scatter_prefiltered = {}

    # Define grids per charge state and plot ics (A and B) ATC
    for i, charge in enumerate(charge_states):
        ax_charge_states_atc[i] = gridspec.GridSpecFromSubplotSpec(len(tps), 3, subplot_spec=gs1[6 * i + 6:6 * i + 9],
                                                                   wspace=0.05,
                                                                   hspace=0.1)
        ax_charge_states_ics_atc[i] = fig.add_subplot(ax_charge_states_atc[i][:, :2])
        for _, line in df[(df["charge"] == charge) & (df["prefiltered"] == 0)].iterrows():
            if line["ic_winner_corr"] > ic_winner_corr_cutoff:
                ax_charge_states_ics_atc[i].plot(
                    (line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                            np.log2(line["auc"]) / np.log2(df[(df["charge"] == line["charge"])]["auc"].max())) - int(
                        line["tp_idx"]),
                    c=pal[int(line["kmeans_clusters"]) - min_clust], lw=4)
            else:
                ax_charge_states_ics_atc[i].plot(
                    (line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                            np.log2(line["auc"]) / np.log2(df[(df["charge"] == line["charge"])]["auc"].max())) - int(
                        line["tp_idx"]),
                    c=pal[int(line["kmeans_clusters"]) - min_clust])
        if prefiltered_ics is not None:
            for _, line in df[(df["charge"] == charge) & (df["prefiltered"] == 1) & (df["tp_idx"] == 0)].iterrows():
                ax_charge_states_ics_atc[i].plot(
                    (line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                            np.log2(line["auc"]) / np.log2(df[(df["charge"] == line["charge"])]["auc"].max())) - int(
                        line["tp_idx"]),
                    c=pal[int(line["kmeans_clusters"]) - min_clust])
        if winner is not None:
            for _, line in df[(df["charge"] == charge) & (df["winner"] == 1)].iterrows():
                ax_charge_states_ics_atc[i].plot(
                    (line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                            np.log2(line["auc"]) / np.log2(df[(df["charge"] == line["charge"])]["auc"].max())) - int(
                        line["tp_idx"]),
                    c=pal[int(line["kmeans_clusters"]) - min_clust], lw=4)
        set_tps = set(df[(df["charge"] == charge) & (df["prefiltered"] == 0)]["tp_idx"])
        for tp in set_tps:
            if tp != 0:
                factor_tensor_frac = sum(set(df[(df["charge"] == charge) & (df["tp_idx"] == tp) & (
                        df["winner"] == 0) & (df["prefiltered"] == 0)]["factor_auc"])) / \
                                     df[(df["charge"] == charge) & (df["tp_idx"] == tp) & (df["prefiltered"] == 0)][
                                         "tensor_auc"].values[0]
                ics_tensor_frac = sum(df[(df["charge"] == charge) & (df["tp_idx"] == tp) & (df["winner"] == 0) & (
                        df["prefiltered"] == 0)]["ic_auc"]) / \
                                  df[(df["charge"] == charge) & (df["tp_idx"] == tp) & (df["prefiltered"] == 0)][
                                      "tensor_auc"].values[0]
                ax_charge_states_ics_atc[i].text(x_max, 0.8 - tp,
                                                 "f|t=%.2f" % (factor_tensor_frac), horizontalalignment="right",
                                                 verticalalignment="center", fontsize=10)
                ax_charge_states_ics_atc[i].text(x_max, 0.6 - tp,
                                                 "i|t=%.2f" % (ics_tensor_frac), horizontalalignment="right",
                                                 verticalalignment="center", fontsize=10)
        ax_charge_states_ics_atc[i].grid()
        ax_charge_states_ics_atc[i].text(0, 1.1, "ATC charge=%i+" % int(charge),
                                         horizontalalignment="left", verticalalignment="baseline", fontsize=14,
                                         weight="bold")
        ax_charge_states_ics_atc[i].text(x_max, 0.9,
                                         "max_auc=%.1e" % df[df["charge"] == charge]["auc"].max(),
                                         horizontalalignment="right", verticalalignment="center", fontsize=12)
        if len(df[(df["charge"] == charge) & (df["tp_idx"] == 0) & (df["prefiltered"] == 1)]) > 0:
            ax_charge_states_ics_atc[i].text(x_max, 0.7,
                                             "idotp=%.3f" % df[(df["charge"] == charge) & (df["tp_idx"] == 0) & (
                                                         df["prefiltered"] == 1)]["ic"].values[
                                                 0].idotp,
                                             horizontalalignment="right", verticalalignment="center", fontsize=12)
        ax_charge_states_ics_atc[i].set_ylim(-len(tps) + 0.95, 1.05)
        ax_charge_states_ics_atc[i].set_yticks([])
        ax_charge_states_ics_atc[i].set_xticks(np.arange(0, x_max + 1, 10))

    # Define grids per charge state and plot ics (A and B) PREFILTERED
    if prefiltered_ics is not None:
        for i, charge in enumerate(charge_states):
            ax_charge_states_prefiltered[i] = gridspec.GridSpecFromSubplotSpec(len(tps), 3,
                                                                               subplot_spec=gs1[6 * i + 9:6 * i + 12],
                                                                               wspace=0.05,
                                                                               hspace=0.1)
            ax_charge_states_ics_prefiltered[i] = fig.add_subplot(ax_charge_states_prefiltered[i][:, :2])
            for _, line in df[(df["charge"] == charge) & (df["prefiltered"] == 1)].iterrows():
                if line["ic_winner_corr"] > ic_winner_corr_cutoff:
                    ax_charge_states_ics_prefiltered[i].plot(
                        (line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                                np.log2(line["auc"]) / np.log2(df[(df["charge"] == line["charge"])]["auc"].max())) - int(
                            line["tp_idx"]),
                        c=pal[int(line["kmeans_clusters"]) - min_clust], lw=4)
                else:
                    ax_charge_states_ics_prefiltered[i].plot(
                        (line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                                np.log2(line["auc"]) / np.log2(
                            df[(df["charge"] == line["charge"])]["auc"].max())) - int(
                            line["tp_idx"]),
                        c=pal[int(line["kmeans_clusters"]) - min_clust])
            if winner is not None:
                for _, line in df[(df["charge"] == charge) & (df["winner"] == 1)].iterrows():
                    ax_charge_states_ics_prefiltered[i].plot(
                        (line["ic"].baseline_integrated_mz / max(line["ic"].baseline_integrated_mz)) * (
                                np.log2(line["auc"]) / np.log2(
                            df[(df["charge"] == line["charge"])]["auc"].max())) - int(
                            line["tp_idx"]),
                        c=pal[int(line["kmeans_clusters"]) - min_clust], lw=4)
            set_tps = set(df[(df["charge"] == charge) & (df["prefiltered"] == 1)]["tp_idx"])
            for tp in set_tps:
                if tp != 0:
                    factor_tensor_frac = sum(set(df[(df["charge"] == charge) & (df["tp_idx"] == tp) & (
                            df["winner"] == 0) & (df["prefiltered"] == 1)]["factor_auc"])) / \
                                         df[(df["charge"] == charge) & (df["tp_idx"] == tp) & (df["prefiltered"] == 1)][
                                             "tensor_auc"].values[0]
                    ics_tensor_frac = sum(df[(df["charge"] == charge) & (df["tp_idx"] == tp) & (df["winner"] == 0) & (
                            df["prefiltered"] == 1)]["ic_auc"]) / \
                                      df[(df["charge"] == charge) & (df["tp_idx"] == tp) & (df["prefiltered"] == 1)][
                                          "tensor_auc"].values[0]
                    ax_charge_states_ics_prefiltered[i].text(x_max, 0.8 - tp,
                                                             "f|t=%.2f" % (factor_tensor_frac),
                                                             horizontalalignment="right",
                                                             verticalalignment="center", fontsize=10)
                    ax_charge_states_ics_prefiltered[i].text(x_max, 0.6 - tp,
                                                             "i|t=%.2f" % (ics_tensor_frac),
                                                             horizontalalignment="right",
                                                             verticalalignment="center", fontsize=10)
            ax_charge_states_ics_prefiltered[i].grid()
            ax_charge_states_ics_prefiltered[i].text(0, 1.1, "PREFILTERED charge=%i+" % int(charge),
                                                     horizontalalignment="left", verticalalignment="baseline",
                                                     fontsize=14,
                                                     weight="bold")
            ax_charge_states_ics_prefiltered[i].text(x_max, 0.9,
                                                     "max_auc=%.1e" % df[df["charge"] == charge]["auc"].max(),
                                                     horizontalalignment="right", verticalalignment="center",
                                                     fontsize=12)
            if len(df[(df["charge"] == charge) & (df["tp_idx"] == 0) & (df["prefiltered"] == 1)]) > 0:
                ax_charge_states_ics_prefiltered[i].text(x_max, 0.7, "idotp=%.3f" %
                                                         df[(df["charge"] == charge) & (df["tp_idx"] == 0) & (
                                                                     df["prefiltered"] == 1)]["ic"].values[
                                                             0].idotp,
                                                         horizontalalignment="right", verticalalignment="center",
                                                         fontsize=12)
            ax_charge_states_ics_prefiltered[i].set_ylim(-len(tps) + 0.95, 1.05)
            ax_charge_states_ics_prefiltered[i].set_yticks([])
            ax_charge_states_ics_prefiltered[i].set_xticks(np.arange(0, x_max + 1, 10))

    # Plot rt/dt scatter plots ATC
    for i, charge in enumerate(charge_states):
        for j in range(len(tps)):
            ax_charge_states_scatter_atc[i + j] = fig.add_subplot(ax_charge_states_atc[i][j, 2])
            if prefiltered_ics is not None and j == 0:
                sns.scatterplot(data=df[(df["charge"] == charge) & (df["tp_idx"] == j) & (df["prefiltered"] == 1)],
                                x="dt",
                                y="rt_corr",
                                hue=df["kmeans_clusters"] - min_clust, palette="bright",
                                s=5 * (
                                    df[(df["charge"] == charge) & (df["tp_idx"] == j) & (df["prefiltered"] == 1)][
                                        "auc_size"]),
                                alpha=0.7,
                                ax=ax_charge_states_scatter_atc[i + j])
            for _, line in df[(df["charge"] == charge) & (df["tp_idx"] == 0) & (df["prefiltered"] == 0)].iterrows():
                if line["ic"].idotp >= 0.98:
                    ax_charge_states_scatter_atc[i + j].text(float(line["dt"]), float(line["rt_corr"]),
                                                             "x", fontsize=10, color="black", ha="center", va="center")
            sns.scatterplot(data=df[(df["charge"] == charge) & (df["tp_idx"] == j) & (df["prefiltered"] == 0)], x="dt",
                            y="rt_corr",
                            hue=df["kmeans_clusters"] - min_clust, palette="bright",
                            s=5 * (
                                df[(df["charge"] == charge) & (df["tp_idx"] == j) & (df["prefiltered"] == 0)][
                                    "auc_size"]),
                            alpha=0.7,
                            ax=ax_charge_states_scatter_atc[i + j])
            ax_charge_states_scatter_atc[i + j].set(xlabel=None, ylabel=None)
            ax_charge_states_scatter_atc[i + j].set_yticks([])
            ax_charge_states_scatter_atc[i + j].set_xticks([])
            ax_charge_states_scatter_atc[i + j].axvline(
                (df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[0] +
                 df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[-1]) / 2, lw=0.5,
                alpha=0.5, color="black")
            ax_charge_states_scatter_atc[i + j].axhline(0, lw=0.5, alpha=0.5, c="black")
            ax_charge_states_scatter_atc[i + j].legend("", frameon=False)
            ax_charge_states_scatter_atc[i + j].set_ylim(-0.4, 0.4)
            ax_charge_states_scatter_atc[i + j].set_xlim(df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[0],
                                                         df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[-1])

            if len(df[(df["winner"] == 1) & (df["charge"] == charge) & (df["tp_idx"] == j)]) > 0:
                for key, spine in ax_charge_states_scatter_atc[i + j].spines.items():
                    spine.set_linewidth(3)

    if prefiltered_ics is not None:
        # Plot rt/dt scatter plots PREFILTERED
        for i, charge in enumerate(charge_states):
            for j in range(len(tps)):
                ax_charge_states_scatter_prefiltered[i + j] = fig.add_subplot(ax_charge_states_prefiltered[i][j, 2])
                sns.scatterplot(data=df[(df["charge"] == charge) & (df["tp_idx"] == j) & (df["prefiltered"] == 1)],
                                x="dt",
                                y="rt_corr",
                                hue=df["kmeans_clusters"] - min_clust, palette="bright",
                                s=5 * (
                                    df[(df["charge"] == charge) & (df["tp_idx"] == j) & (df["prefiltered"] == 1)][
                                        "auc_size"]),
                                alpha=0.7,
                                ax=ax_charge_states_scatter_prefiltered[i + j])
                if len(df[(df["charge"] == charge) & (df["tp_idx"] == 0) & (df["winner"] == 0) & (
                        df["prefiltered"] == 1)]) == 1:
                    ax_charge_states_scatter_prefiltered[i + j].text(float(
                        df[(df["charge"] == charge) & (df["tp_idx"] == 0) & (df["winner"] == 0) & (
                                    df["prefiltered"] == 1)][
                            "dt"].values), float(df[(df["charge"] == charge) & (df["tp_idx"] == 0) & (
                                df["winner"] == 0) & (df["prefiltered"] == 1)][
                                                     "rt_corr"].values),
                        "x", fontsize=10, color="black", ha="center", va="center")
                ax_charge_states_scatter_prefiltered[i + j].set(xlabel=None, ylabel=None)
                ax_charge_states_scatter_prefiltered[i + j].set_yticks([])
                ax_charge_states_scatter_prefiltered[i + j].set_xticks([])
                ax_charge_states_scatter_prefiltered[i + j].axvline(
                    (df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[0] +
                     df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[-1]) / 2, lw=0.5,
                    alpha=0.5, color="black")
                ax_charge_states_scatter_prefiltered[i + j].axhline(0, lw=0.5, alpha=0.5, c="black")
                ax_charge_states_scatter_prefiltered[i + j].legend("", frameon=False)
                ax_charge_states_scatter_prefiltered[i + j].set_ylim(-0.4, 0.4)
                ax_charge_states_scatter_prefiltered[i + j].set_xlim(
                    df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[0],
                    df[df["charge"] == charge]["ic"].tolist()[0].drift_labels[-1])

                if len(df[(df["winner"] == 1) & (df["charge"] == charge) & (df["tp_idx"] == j)]) > 0:
                    for key, spine in ax_charge_states_scatter_prefiltered[i + j].spines.items():
                        spine.set_linewidth(3)

    if output_plot_path is not None:
        plt.savefig(output_plot_path, dpi=dpi, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()


def plot_ajf_(configfile, atc, prefiltered_ics, winner, output_plot_path=None, output_df_path=None):

    if output_plot_path is not None:
        if (not os.path.isdir(os.path.dirname(output_plot_path))) and (len(os.path.dirname(output_plot_path)) != 0):
            os.makedirs(os.path.dirname(output_plot_path))

    if (atc is None) or (len(atc) < 3):
        print(f"Not enough ICs. Producing empty file: {output_plot_path}")
        Path(output_plot_path).touch()
        return 0

    df = create_df_and_clusterize(configfile,
                                  atc,
                                  prefiltered_ics,
                                  winner,
                                  output_df_path=output_df_path)

    ajf_plot(df, configfile, output_plot_path=output_plot_path)


if __name__ == "__main__":

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
        "-f",
        "--prefiltered_ics",
        help=
        "Prefiltered ics"
    )
    parser.add_argument(
        "-w",
        "--winner",
        help=
        "Winner path"
    )
    parser.add_argument(
        "-o",
        "--output_plot_path",
        help=
        "Output ajf plot path"
    )
    parser.add_argument(
        "-d",
        "--output_df_path",
        help=
        "df output path",
        default=None
    )

    args = parser.parse_args()

    configfile = yaml.load(open(args.configfile, "rb").read(), Loader=yaml.Loader)

    if not os.stat(args.atc).st_size == 0:
        atc = limit_read(args.atc)
    else:
        atc = None
        print("ATC is NONE")
    if not os.stat(args.prefiltered_ics).st_size == 0:
        prefiltered_ics = limit_read(args.prefiltered_ics)
    else:
        prefiltered_ics = None
        print("PREFILTERED IS NONE")
    if not os.stat(args.winner).st_size == 0:
        winner = limit_read(args.winner)
    else:
        winner = None
        print("WINNER IS NONE")

    plot_ajf_(configfile=configfile,
              atc=atc,
              prefiltered_ics=prefiltered_ics,
              winner=winner,
              output_plot_path=args.output_plot_path,
              output_df_path=args.output_df_path)


