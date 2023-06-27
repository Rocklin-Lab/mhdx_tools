import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import zlib
import _pickle as cpickle


def convert_factor_mz_data_to_cont(mz_data, mz_labels, bins_per_isotope_peak):
    """Description of function.

    Args:
        arg_name (type): Description of input variable.

    Returns:
        out_name (type): Description of any returned objects.

    """
    padded_mz_labels = []
    padded_factor_mz = []
    mz_label_spacing = mz_labels[1] - mz_labels[0]

    for i in range(0, len(mz_labels), bins_per_isotope_peak):
        padded_mz_labels.append(mz_labels[i] - mz_label_spacing)
        padded_mz_labels += list(mz_labels[i:i + bins_per_isotope_peak])
        padded_mz_labels.append(padded_mz_labels[-1] + mz_label_spacing)

        padded_factor_mz.append(0)
        padded_factor_mz += list(mz_data[i:i + bins_per_isotope_peak])
        padded_factor_mz.append(0)

    return padded_mz_labels, padded_factor_mz


def plot_mz_data(fig, gs, row_num, col_num, mz_label, mz_data, plot_label, idotp=None):
    """Creates a subplot in a given figure at set row/column position, plots the m/Z of a passed IC or Factor.

    Args:
        fig (matplotlib.figure): The figure object where m/Z will be plotted.
        gs (gridspec.GridSpec): GridSpec object determining number of rows and columns.
        row_num (int): Dictates the row of the plot being made on the figure.
        col_num (int): Dictates the column of the plot being made on the figure.
        mz_label (list of floats): Labels connecting m/Z values to bins.
        mz_data (list of floats): Binned intensities of m/Z signal.
        plot_label (str): Title of the subplot being made.

    Returns:
        None

    """
    mz_sum = np.round(np.sum(mz_data), 2)

    ax = fig.add_subplot(gs[row_num, col_num])
    plt.plot(mz_label, mz_data, linewidth=0.5, label=plot_label)
    plt.text(0.75, 1.2, s="sum_int=%.2f" % mz_sum, fontsize=10, transform=ax.transAxes)
    if idotp is not None:
        plt.text(0, 1.2, s="idotp=%.3f" % idotp, fontsize=10, transform=ax.transAxes)
    ax.set_yticks([])
    ax.tick_params(length=3, pad=3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.legend(loc="best", fontsize="small")


def plot_ics(list_of_ics_from_a_factor):
    """Creates a plot showing a Factor"s m/Z and the m/Z of all ICs made from that Factor.

    Args:
        list_of_ics_from_a_factor(list of IsotopeCluster objects): List of ICs made from a Factor.

    Returns:
        None

    """
    num_of_mz_plots = len(list_of_ics_from_a_factor) + 1
    num_columns = 3
    num_rows = 0

    for num in range(num_of_mz_plots):
        if num % num_columns == 0:
            num_rows += 1

    pad_factor_mz_label, pad_factor_mz = convert_factor_mz_data_to_cont(
        mz_data=list_of_ics_from_a_factor[0].factor_mz_data,
        mz_labels=list_of_ics_from_a_factor[0].mz_labels,
        bins_per_isotope_peak=list_of_ics_from_a_factor[0].bins_per_isotope_peak)

    fig = plt.figure(figsize=(15, num_rows * 1.6))
    gs = gridspec.GridSpec(ncols=num_columns, nrows=num_rows, figure=fig)
    n_rows = 0
    n_cols = 0

    # Plots Factor m/Z.
    plot_mz_data(fig=fig,
                 gs=gs,
                 row_num=n_rows,
                 col_num=n_cols,
                 mz_label=pad_factor_mz_label,
                 mz_data=pad_factor_mz,
                 plot_label="factor_mz")


    # Plots m/Z for all ICs from Factor.
    n_cols = 1

    for num, ic in enumerate(list_of_ics_from_a_factor):

        ic_mz_label, ic_mz_ = convert_factor_mz_data_to_cont(mz_data=ic.cluster_mz_data,
                                                             mz_labels=ic.mz_labels,
                                                             bins_per_isotope_peak=ic.bins_per_isotope_peak)

        plot_mz_data(fig=fig,
                     gs=gs,
                     row_num=n_rows,
                     col_num=n_cols,
                     mz_label=ic_mz_label,
                     mz_data=ic_mz_,
                     idotp=ic.idotp,
                     plot_label=str(num))

        if (n_cols+1) % num_columns == 0:
            n_rows += 1
            n_cols = 0
        else:
            n_cols += 1

    title = "Factor num %s" % list_of_ics_from_a_factor[0].factor_idx

    plt.suptitle(title)
    plt.tight_layout()


def plot_ics_from_ic_list(list_of_ics, output_path):
    """Creates a .pdf plot of all ICs resulting from factorization of a DataTensor.

    Args:
        list_of_ics (list of IsotopeCluster objects): A list of all ICs resulting from a factorization.
        output_path (str): A path/to/output.pdf.

    Returns:
        None

    """

    # TODO: Do this with new_list = sorted(list_of_ics, key = lambda ic: ic.factor_idx)? Or something similar.
    # Sort IC lists by parent factor index.
    factor_indices = [x.factor_idx for x in list_of_ics]
    unique_factor_idx = np.unique(factor_indices)
    sorted_list_ics = [[] for _ in range(len(unique_factor_idx))]


    for ind, factor_idx in enumerate(unique_factor_idx):
        for ics in list_of_ics:
            if ics.factor_idx == factor_idx:
                sorted_list_ics[ind].append(ics)

    with PdfPages(output_path) as pdf:
        for ind, ic_list in enumerate(sorted_list_ics):
            plot_ics(ic_list)
            pdf.savefig()
            plt.close()


def plot_ics_from_icfile(ic_fpath, output_path):

    list_of_ics = cpickle.loads(zlib.decompress(open(ic_fpath, "rb").read()))
    plot_ics_from_ic_list(list_of_ics=list_of_ics,
                          output_path=output_path)


if __name__ == "__main__":

    pass
