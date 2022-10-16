import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


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
        sys.exit()

    sns.set_context("talk")

    fig, ax = plt.subplots(4, 2, figsize=(10, 12), dpi=200)

    sns.histplot(df["n_UN"].values, ax=ax[0][0], discrete=True)
    sns.histplot(df["n_UN"].values, ax=ax[0][0], kde=True)
    ax[0][0].set_xlabel("n_UN")

    sns.histplot(df["n_signals"].values, ax=ax[0][1], discrete=True)
    sns.histplot(df["n_signals"].values, ax=ax[0][1], kde=True)
    ax[0][1].set_xlabel("n_signals")

    sns.histplot(df["im_mono"].values * configfile["dt_max"] / 200 - df["DT_weighted_avg"].values, ax=ax[1][0])
    sns.histplot(df["im_mono"].values * configfile["dt_max"] / 200 - df["DT_weighted_avg"].values, ax=ax[1][0], kde=True)
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