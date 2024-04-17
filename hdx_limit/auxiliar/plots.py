import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import scipy as sp
import itertools


def rt_correlation_plot(intermediates,
                        output_path=None,
                        dpi=200):

    if len(intermediates) > 6:
        intermediates = intermediates[:6]

    fs = sorted(intermediates)

    runs = {}
    for i, f in enumerate(fs):
        runs[i] = pd.read_csv(f)

    unique_names = set(runs[0].name)
    for i in runs.keys():
        if i != 0:
            unique_names = set(unique_names).intersection(runs[i].name)

    df_rt = {}
    for i in runs.keys():
        df_rt[i] = []

    for name in unique_names:
        for i in runs.keys():
            df_rt[i].append(runs[i][runs[i].name == name]["RT"].mean())

    combinations = [subset for subset in itertools.combinations(runs.keys(), 2)]

    sns.set_context("talk")

    fig, ax = plt.subplots(len(combinations), 2, figsize=(10, 3.5 * len(combinations)), dpi=dpi,
                           constrained_layout=True)

    if len(ax.shape) == 1:
      ax = [ax]
                          
    for i in range(len(combinations)):
        ax[i][0].scatter(df_rt[combinations[i][0]], df_rt[combinations[i][1]],
                         alpha=0.5, s=50, color="blue", edgecolors="black", lw=0.7)

        r, p = sp.stats.pearsonr(x=df_rt[combinations[i][0]], y=df_rt[combinations[i][1]])
        ax[i][0].text(.01, .95, f"pearson_r={r:.2f}", transform=ax[i][0].transAxes, va="top")

        sns.kdeplot(data=np.array(df_rt[combinations[i][0]]) - np.array(df_rt[combinations[i][1]]), ax=ax[i][1])

        ax[i][0].plot([i for i in range(30)], [i for i in range(30)], "--r", lw=1)
        ax[i][1].axvline(0, ls="--", color="red")
        ax[i][0].set_xlabel(f"RT_{combinations[i][0]}/ min")
        ax[i][0].set_ylabel(f"RT_{combinations[i][1]}/ min")
        ax[i][1].set_xlabel(r"$\Delta$RT/ min")
        ax[i][1].set_xlim(-2, 2)

    if output_path is not None:
        plt.savefig(output_path, format="pdf", dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    plt.close("all")


def rt_distribution_plot(configfile,
                         intermediates,
                         output_path=None,
                         dpi=200):

    runs = {}
    if len(intermediates) == 1:
        runs[0] = pd.read_csv(intermediates[0])
        titles = [configfile[0][0]]
    else:
        fs = sorted(intermediates)
        for i, f in enumerate(fs):
            runs[i] = pd.read_csv(f)
        titles = [f.split("/")[-1].strip("_intermediate.csv") for f in fs]

    sns.set_context("talk", font_scale=1)

    fig, ax = plt.subplots(len(runs), 1, figsize=(5, 3 * len(runs)), dpi=dpi, constrained_layout=True)

    if len(runs) == 1:
        plot_hist_kde(array=runs[0]["RT"], ax=ax, xlabel="RT / min", title=titles[0], binwidth=0.5)
    else:
        for i in range(len(runs)):
            plot_hist_kde(array=runs[i]["RT"], ax=ax[i], xlabel="RT / min", title=titles[i], binwidth=0.5)

    if output_path is not None:
        plt.savefig(output_path, format="pdf", dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    plt.close("all")


def plot_rtdt_recenter(df,
                       output_folder=None,
                       dpi=150):
    # Create folder to save pdf files
    if output_folder is not None:
        os.makedirs("results/plots/tensor-recenter/")

    # Scatter plot for each protein
    for name in list(set(df["name"].values)):

        n_charges = len(set(df[df["name"] == name]["charge"].values))

        fig, ax = plt.subplots(1, n_charges + 1, dpi=200, figsize=(3 * (n_charges + 1), 2.5))

        sns.scatterplot(data=df[df["name"] == name], x="dt", y="rt", palette="bright", hue="charge", size="auc_log",
                        ax=ax[0])
        ax[0].set_ylim(df[df["name"] == name]["RT_weighted_avg"].mean() - 0.4,
                       df[df["name"] == name]["RT_weighted_avg"].mean() + 0.4)
        ax[0].set_xlim(min(df[df["name"] == name]["DT_weighted_avg"].min(),
                           df[df["name"] == name]["dt"].min()) - 0.5,
                       max(df[df["name"] == name]["DT_weighted_avg"].max(),
                           df[df["name"] == name]["dt"].max()) + 0.5)
        ax[0].axhline(df[df["name"] == name]["RT_weighted_avg"].mean(), color="black", alpha=0.5, lw=0.5)
        ax[0].text(0, 1.01, "%s" % name, transform=ax[0].transAxes, fontsize=8)
        ax[0].set_xlabel("DT")
        ax[0].set_ylabel("RT")

        h, l = ax[0].get_legend_handles_labels()
        ax[0].legend(h[1:n_charges + 1], l[1:n_charges + 1], fontsize=10, loc=2,
                     bbox_to_anchor=(0.0, 0.15), bbox_transform=ax[0].transAxes, borderpad=0.02,
                     columnspacing=0.,
                     handletextpad=0.0, frameon=False, ncol=n_charges, prop={"size": 6})

        for j, charge in enumerate(sorted(list(set(df[df["name"] == name]["charge"].values)))):
            sns.scatterplot(data=df[(df["name"] == name) & (df["charge"] == charge)], x="dt", y="rt", palette="bright",
                            hue="file_index", size="auc_log", ax=ax[j + 1])
            ax[j + 1].text(0, 1.01, "charge=%i+" % charge, transform=ax[j + 1].transAxes, fontsize=8)
            ax[j + 1].set_ylim(df[df["name"] == name]["RT_weighted_avg"].mean() - 0.4,
                               df[df["name"] == name]["RT_weighted_avg"].mean() + 0.4)
            ax[j + 1].set_xlim(df[(df["name"] == name) & (df["charge"] == charge)]["DT_weighted_avg"].mean() * 0.94,
                               df[(df["name"] == name) & (df["charge"] == charge)]["DT_weighted_avg"].mean() * 1.06)
            ax[j + 1].axhline(df[df["name"] == name]["RT_weighted_avg"].mean(), color="black", alpha=0.5, lw=0.5)
            ax[j + 1].axvline(df[(df["name"] == name) & (df["charge"] == charge)]["DT_weighted_avg"].mean(),
                              color="black", alpha=0.5, lw=0.5)

            # Plot horizontal and vertical lines corresponding to initial RT and DT centers used to extract tensors
            retention_label_center = \
                df[(df["name"] == name) & (df["charge"] == charge)]["ic"].values[0].retention_labels[
                    len(df[(df["name"] == name) & (df["charge"] == charge)]["ic"].values[0].retention_labels) // 2]
            ax[j + 1].axhline(retention_label_center, color="red", alpha=0.5, lw=0.5)
            drift_label_center = df[(df["name"] == name) & (df["charge"] == charge)]["ic"].values[0].drift_labels[
                len(df[(df["name"] == name) & (df["charge"] == charge)]["ic"].values[0].drift_labels) // 2]
            ax[j + 1].axvline(drift_label_center, color="red", alpha=0.5, lw=0.5)

            ax[0].scatter(df[(df["name"] == name) & (df["charge"] == charge)]["DT_weighted_avg"].mean(),
                          (sum(df[(df["name"] == name) & (df["charge"] == charge)]["rt"] *
                               df[(df["name"] == name) & (df["charge"] == charge)]["auc"])) /
                          sum(df[(df["name"] == name) & (df["charge"] == charge)]["auc"]), marker="x", color="black",
                          s=20)
            ax[j + 1].scatter(df[(df["name"] == name) & (df["charge"] == charge)]["DT_weighted_avg"].mean(),
                              (sum(df[(df["name"] == name) & (df["charge"] == charge)]["rt"] *
                                   df[(df["name"] == name) & (df["charge"] == charge)]["auc"])) /
                              sum(df[(df["name"] == name) & (df["charge"] == charge)]["auc"]), marker="x",
                              color="black", s=20)

            n_files = len(set(df[(df["name"] == name) & (df["charge"] == charge)]["file_index"].values))
            h, l = ax[j + 1].get_legend_handles_labels()
            ax[j + 1].legend(h[1:n_files + 1], l[1:n_files + 1], fontsize=10, loc=2,
                             bbox_to_anchor=(0.0, 0.15), bbox_transform=ax[j + 1].transAxes, borderpad=0.02,
                             columnspacing=0.,
                             handletextpad=0.0, frameon=False, ncol=n_files, prop={"size": 6})
            ax[j + 1].set_xlabel("DT")
            ax[j + 1].set_ylabel("RT")

        name_recentered = "_".join(name.split("_")[:-1]) + "_" + str(
            round(df[(df["name"] == name)]["RT_weighted_avg"].values[0], 2))

        plt.tight_layout()

        if output_folder is None:
            plt.show()
        else:
            plt.savefig(output_folder + "/" + name_recentered + ".pdf", format="pdf", dpi=dpi)

        plt.close("all")


def plot_deviations(df,
                    configfile,
                    output_path=None,
                    dpi=200):
    if len(df.query("n_UN > 1")) == 0:
        print(f"Only one file present... generating empty: {output_path}")
        Path(output_path).touch()
        return 0

    sns.set_context("talk")

    fig, ax = plt.subplots(4, 2, figsize=(10, 12), dpi=200)

    sns.histplot(df["n_UN"].values, ax=ax[0][0], discrete=True)
    sns.histplot(df["n_UN"].values, ax=ax[0][0], kde=True)
    ax[0][0].set_xlabel("n_UN")

    sns.histplot(df["n_signals"].values, ax=ax[0][1], discrete=True)
    sns.histplot(df["n_signals"].values, ax=ax[0][1], kde=True)
    ax[0][1].set_xlabel("n_signals")

    sns.histplot(df["im_mono"].values * configfile["dt_max"] / 200 - df["DT_weighted_avg"].values, ax=ax[1][0])
    sns.histplot(df["im_mono"].values * configfile["dt_max"] / 200 - df["DT_weighted_avg"].values, ax=ax[1][0],
                 kde=True)
    ax[1][0].set_xlabel("DT error")

    sns.histplot(df["RT"].values - df["RT_weighted_avg"].values, ax=ax[1][1])
    sns.histplot(df["RT"].values - df["RT_weighted_avg"].values, ax=ax[1][1], kde=True)
    ax[1][1].set_xlabel("RT error")

    sns.histplot(df["dt_weighted_std"].values, ax=ax[2][0])
    sns.histplot(df["dt_weighted_std"].values, ax=ax[2][0], kde=True)
    ax[2][0].set_xlabel("DT_weighted_std")

    sns.histplot(df["dt_std"].values, ax=ax[2][1])
    sns.histplot(df["dt_std"].values, ax=ax[2][1], kde=True)
    ax[2][1].set_xlabel("DT_std")

    sns.histplot(df["rt_weighted_std"].values, ax=ax[3][0])
    sns.histplot(df["rt_weighted_std"].values, ax=ax[3][0], kde=True)
    ax[3][0].set_xlabel("RT_weighted_std")

    sns.histplot(df["rt_std"].values, ax=ax[3][1], bins=100)
    ax[3][1].set_xlabel("RT_std")

    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, format="pdf", dpi=dpi, bbox_inches="tight")

    plt.close("all")


def get_data_benchmark(fs, key):
    fs_key = [i for i in fs if key in i]

    l = []
    for f in fs_key:
        l.append(pd.read_csv(f, sep="\t")[["s", "max_rss"]].values.tolist()[0])

    return np.array(l)


def plot_hist_kde(array, ax, xlabel=None, title=None, binwidth=None):

    sns.histplot(array, color="blue", ax=ax, binwidth=binwidth)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if title is not None:
        ax.text(0, 1.06, title, transform=ax.transAxes, size=14)

    if len(array) > 1:
        ax_tmp = ax.twinx()

        ax_tmp.tick_params(axis="y", colors="red")
        ax_tmp.yaxis.label.set_color("red")

        sns.kdeplot(array, color="red", cumulative=True, ax=ax_tmp, warn_singular=False)


def generate_benchmark_stats_plot(benchmark_folder,
                                  output_path,
                                  dpi=300):
    fs = glob.glob(f"{benchmark_folder}/*txt") + glob.glob(f"{benchmark_folder}/10*/*txt")

    keys = set([i.split("/")[-1].split(".")[0] for i in fs if "10_generate_tensor_ics" not in i])
    keys = list(keys) + ["10_generate_tensor_ics"]
    keys = sorted(keys, key=lambda x: int(x.split("_")[0]))

    sns.set_context("talk")

    fig, ax = plt.subplots(len(keys) + 1, 2, figsize=(10, len(keys) * 3), dpi=dpi, constrained_layout=True)

    ax[0][0].axis("off")
    ax[0][1].axis("off")

    d = {}

    for idx, key in enumerate(keys):
        array = get_data_benchmark(fs, key)

        d[key] = np.sum(array[:, 0])

        plot_hist_kde(array[:, 0] / 60, ax[idx + 1][0], xlabel="time / min", title=key)

        plot_hist_kde(array[:, 1], ax[idx + 1][1], xlabel="physical mem / Mb")

    x_coord, y_coord = 0, 0.95
    total = 0
    for key, value in d.items():
        total += value
        ax[0][0].text(x_coord, y_coord, f"{key} : {value / 3600:.2f} hours", transform=ax[0][0].transAxes, fontsize=11)
        y_coord -= 0.13

    ax[0][0].text(x_coord, y_coord, f"total : {total / 3600:.2f} hours", transform=ax[0][0].transAxes, fontsize=11)

    if output_path is not None:
        plt.savefig(output_path, format="pdf", dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    plt.close("all")
