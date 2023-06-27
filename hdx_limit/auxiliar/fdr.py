import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def fdr_sequences(df, level=2):

    """
    Estimates target decoy fdr based on unique sequences identified.
    Expects initial set to have TWICE the number of expected True Positive Targets
    If no identification is found, returns zero
    """

    decoy_hits = math.ceil(len(set(df[df["name"].str.contains("decoy")]["sequence"].values)) / level)
    hits = len(set(df[~df["name"].str.contains("decoy")]["sequence"].values))

    if hits != 0:

        return 100 * decoy_hits / hits

    else:

        return 0


def fdr_signals(df):

    """
    Estimates target decoy fdr based on all signals identified.
    Expects initial set to have TWICE the number of expected True Positive Targets
    If no identification is found, returns zero
    """

    decoy_hits = math.ceil(len(df[df["name"].str.contains("decoy")]) / 2)
    hits = len(df[~df["name"].str.contains("decoy")])

    if hits != 0:

        return 100 * decoy_hits / hits

    else:

        return 0


def plot_fdr_scatter(data,
                     x,
                     y1,
                     y2,
                     x_label,
                     y_label_1,
                     y_label_2,
                     ax):

    """
    Generate FDR scatter plot
    x: independent variable
    y1: key for either number of sequences or number of signals
    y2: FDR
    x_label: label for x axis
    y_label_1: label for left y axis
    y_label_2: label for right y axis
    ax: plot position
    """

    sns.scatterplot(data=data, x=x, y=y1, color="Blue", edgecolor="Black", ax=ax)

    ax.set_ylabel(y_label_1, color="Blue")
    ax.tick_params(axis="y", colors="Blue")

    ax.set_xlabel(x_label)

    axtwin = ax.twinx()

    sns.scatterplot(data=data, x=x, y=y2, color="Orange", edgecolor="Black", ax=axtwin)

    axtwin.set_ylabel(y_label_2, color="Orange")
    axtwin.tick_params(axis="y", colors="Orange")


def plot_fdr_hist(df,
                  x,
                  ax,
                  label=None,
                  discrete=True,
                  xmin=None,
                  xmax=None,
                  step=None):
    """
    Generate histograms for variable of interest.
    x: independent variable
    label: x label
    discrete: if true return bin=1 histogram
    xmin: min x value
    xmax: max x value
    step: x ticks interval
    ax: plot position
    """

    sns.histplot(data=df, x=x,
                 hue="decoy", multiple="dodge", discrete=discrete, palette="bright", ax=ax, legend=False)

    if (xmin is None) and (xmax is None):

        ax.set_xticks(np.arange(int(min(df[x])), int(max(df[x])) + 1, step))

    else:
        ax.set_xticks(np.arange(xmin, xmax, step))
    #         ax.set_xlim(xmin, xmax)

    if label is not None:
        ax.set_xlabel(label)


def get_fdr_df(df,
               key,
               min_value,
               max_value,
               step=None,
               greater=True):
    """
    generate fdr dataframe
    df: dataframe
    key: which independent variable will be evaluated
    min_value: min value evaluated
    max_value: max value evaluated
    step: evaluation interval
    grater: Keep entries greater than value evaluated?

    returns dataframe with FDR for unique sequences and all signals
    """


    l = []

    if step is None:
        step = (max_value - min_value) / 10

    for value in np.arange(min_value, max_value, step):

        if greater:
            df_tmp = df[abs(df[key]) >= value]
        else:
            df_tmp = df[abs(df[key]) <= value]

        l.append([
            value,
            fdr_sequences(df_tmp),
            len(set(df_tmp[df_tmp["name"].str.contains("decoy")]["sequence"].values)),
            len(set(df_tmp[~df_tmp["name"].str.contains("decoy")]["sequence"].values)),
            fdr_signals(df_tmp),
            len(df_tmp[df_tmp["name"].str.contains("decoy")]["sequence"].values),
            len(df_tmp[~df_tmp["name"].str.contains("decoy")]["sequence"].values),
            len(df_tmp)
        ])

    df_FDR = pd.DataFrame(l, columns=[key, "FDR_sequences", "n_FP_sequences", "n_TP_sequences", "FDR_signals",
                                      "n_FP_signals", "n_TP_signals", "n_signals"])

    return df_FDR


def plot_fdr_stats(df,
                   output_path=None,
                   dpi=200):

    sns.set_context("talk")

    fig, ax = plt.subplots(5, 4, figsize=(20, 15), dpi=dpi)

    keys = ["idotp", "n_charges", "n_UN", "log2_ab_cluster_total", "abs_ppm"]
    labels = ["idotp", "# charge states", "# undeuterated files", "log2 intensity", "abs ppm error"]

    for i, (key, label) in enumerate(zip(keys, labels)):

        if key == "idotp":
            discrete = False
            step_FDR = 0.005
            step_hist = 0.01
            greater = True
            xmin = round(min(df[key]), 2)
            xmax = round(max(df[key]), 2)
        elif key == "n_charges":
            discrete = True
            step_FDR = 1
            step_hist = 1
            greater = True
            xmin = min(df[key])
            xmax = max(df[key]) + 1
        elif key == "n_UN":
            discrete = True
            step_FDR = 1
            step_hist = 1
            greater = True
            xmin = min(df[key])
            xmax = max(df[key]) + 1
        elif key == "abs_ppm":
            discrete = True
            greater = False
            step_FDR = 2
            step_hist = 2
            xmin = 0
            xmax = int(max(df[key])) + 2
        elif key == "log2_ab_cluster_total":
            discrete = True
            step_FDR = 2
            step_hist = 5
            greater = True
            xmin = min(df[key])
            xmax = max(df[key])
        else:
            discrete = True
            greater = True
            step_FDR = 1
            step_hist = 2
            xmin = min(df[key])
            xmax = max(df[key])

        df_FDR = get_fdr_df(df=df,
                            key=key,
                            min_value=xmin,
                            max_value=xmax,
                            step=step_FDR, greater=greater)

        plot_fdr_hist(df=df.sort_values(by=key, ascending=False).drop_duplicates(subset="sequence"),
                      x=key,
                      ax=ax[i][0],
                      label=label,
                      xmin=xmin,
                      xmax=xmax,
                      step=step_hist,
                      discrete=discrete)

        plot_fdr_scatter(data=df_FDR, x=key, y1="n_TP_sequences", y2="FDR_sequences", x_label=label,
                         y_label_1="# sequences", y_label_2="% FDR", ax=ax[i][1])

        plot_fdr_hist(df=df,
                      x=key,
                      ax=ax[i][2],
                      label=label,
                      xmin=xmin,
                      xmax=xmax,
                      step=step_hist,
                      discrete=discrete)

        plot_fdr_scatter(data=df_FDR, x=key, y1="n_TP_signals", y2="FDR_signals", x_label=label, y_label_1="# signals",
                         y_label_2="% FDR", ax=ax[i][3])

    ax[0][0].text(0, 1.05, "Protein level", transform=ax[0][0].transAxes, weight="bold")

    ax[0][2].text(0, 1.05, "Signal level", transform=ax[0][2].transAxes, weight="bold")

    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, format="pdf", dpi=dpi, bbox_inches="tight")

    plt.close("all")
