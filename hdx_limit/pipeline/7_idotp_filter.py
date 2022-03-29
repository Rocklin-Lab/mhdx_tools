"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import sys
import glob
import argparse
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from hdx_limit.core.io import limit_read
mpl.use("Agg")


def generate_dataframe_ics(configfile,
                           all_ics_inputs,
                           idotp_cutoff=0.99):
    # Create dictionary containing all ics passing idotp_cutoff
    protein_ics = {}
    for f in all_ics_inputs:
        if f.split('/')[-2:-1][0] not in protein_ics:
            ics = [ic for ic in limit_read(f) if ic.undeut_ground_dot_product >= idotp_cutoff]
            if len(ics) > 0:
                protein_ics[f.split('/')[-2:-1][0]] = [ics]
        else:
            ics = [ic for ic in limit_read(f) if ic.undeut_ground_dot_product >= idotp_cutoff]
            if len(ics) > 0:
                protein_ics[f.split('/')[-2:-1][0]].append(ics)

    # Flat list of lists of ics (all charge states will be one single list
    for key in protein_ics:
        protein_ics[key] = [i for sublist in protein_ics[key] for i in sublist]

    # Extract values for dt, rt, auc, charge and file index from each IC and store in a dataframe
    data = []
    for key in protein_ics:
        for ic in protein_ics[key]:
            rt = ic.retention_labels[0] + (ic.retention_labels[1] - ic.retention_labels[0]) * ic.rt_com
            dt = ic.drift_labels[0] + (ic.drift_labels[1] - ic.drift_labels[0]) * ic.dt_coms
            auc = ic.ic_auc_with_gauss_extrapol
            charge = ic.charge_states[0]
            file_index = configfile[0].index([i for i in configfile[0] if '_'.join(
                ic.info_tuple[0].split('/')[-1].split('.')[-5:-4][0].split('_')[-4:]) in i][0])
            idotp = ic.undeut_ground_dot_product

            data.append([key, ic, rt, dt, auc, charge, file_index, idotp])

    df = pd.DataFrame(data, columns=['name', 'ic', 'rt', 'dt', 'auc', 'charge', 'file_index', 'idotp'])
    df['auc_log'] = 2 * np.log10(df['auc'])

    # Find DT weighted average
    for name, charge in set([(n, c) for (n, c) in df[['name', 'charge']].values]):
        # Remove outliers
        percentile25, percentile75 = df[(df['name'] == name) & (df['charge'] == charge)]['dt'].quantile(0.25), \
                                     df[(df['name'] == name) & (df['charge'] == charge)]['dt'].quantile(0.75)
        iqr = percentile75 - percentile25
        lb, ub = percentile25 - 1.5 * iqr, percentile75 + 1.5 * iqr
        if len(df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                        df['dt'] <= 13)]) > 0:
            df.loc[(df['name'] == name) & (df['charge'] == charge), 'DT_weighted_avg'] = sum(
                df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.)]['dt'] *
                df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.)]['auc']) / sum(
                df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.)]['auc'])
            # How many signals do we see? How many undeuterated files generated passing ICs?
            df.loc[df['name'] == name, 'n_signals'] = len(
                df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.)])
            df.loc[df['name'] == name, 'n_UN'] = len(
                set(df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.)]['file_index'].values))
            if len(df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                    df['dt'] <= 13)]) > 1:
                # DT standard deviation
                df.loc[(df['name'] == name) & (df['charge'] == charge), 'dt_std'] = df[(df['name'] == name) & (
                        df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (df['dt'] <= 13.)]['dt'].std()
                # DT weighted standard deviation
                values = df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.)]['dt']
                weights = df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.)]['auc']
                avg_value = df[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.)]['DT_weighted_avg']
                df.loc[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.), 'dt_weighted_std'] = np.sqrt(
                    (weights * (values - avg_value) ** 2) / sum(weights) * (len(values) - 1))
            else:
                df.loc[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.), 'dt_std'] = 0
                df.loc[(df['name'] == name) & (df['charge'] == charge) & (df['dt'] >= lb) & (df['dt'] <= ub) & (
                            df['dt'] <= 13.), 'dt_weighted_std'] = 0
        else:
            df.loc[(df['name'] == name) & (df['charge'] == charge), 'DT_weighted_avg'] = -1

    # Find RT weighted average
    for name in set(df['name'].values):
        # Remove outliers
        percentile25, percentile75 = df[(df['name'] == name)]['rt'].quantile(0.25), \
                                     df[(df['name'] == name)]['rt'].quantile(0.75)
        iqr = percentile75 - percentile25
        lb, ub = percentile25 - 1.5 * iqr, percentile75 + 1.5 * iqr
        df.loc[df['name'] == name, 'RT_weighted_avg'] = sum(
            df[(df['name'] == name) & (df['rt'] >= lb) & (df['rt'] <= ub)]['rt'] * df[(df['name'] == name)
                                & (df['rt'] >= lb) & (df['rt'] <= ub)]['auc']) / sum(df[(df['name'] == name)
                                & (df['rt'] >= lb) & (df['rt'] <= ub)]['auc'])
        if len(df.loc[df['name'] == name, 'RT_weighted_avg']) > 1:
            # DT standard deviation
            df.loc[df['name'] == name, 'dt_std'] = df[(df['name'] == name) & (df['rt'] >= lb) & (df['rt'] <= ub)]['rt'].std()
            # DT weighted standard deviation
            values = df[(df['name'] == name) & (df['rt'] >= lb) & (df['rt'] <= ub)]['rt']
            weights = df[(df['name'] == name) & (df['rt'] >= lb) & (df['rt'] <= ub)]['auc']
            avg_value = df[(df['name'] == name) & (df['rt'] >= lb) & (df['rt'] <= ub)]['RT_weighted_avg']
            df.loc[(df['name'] == name) & (df['rt'] >= lb) & (df['rt'] <= ub),
                   'rt_weighted_std'] = np.sqrt((weights * (values - avg_value) ** 2) / sum(weights) * (len(values) - 1)
                                                )
        else:
            df.loc[(df['name'] == name) & (df['rt'] >= lb) & (df['rt'] <= ub), 'rt_std'] = 0
            df.loc[(df['name'] == name) & (df['rt'] >= lb) & (df['rt'] <= ub),
                   'rt_weighted_std'] = 0


        # Compute DT weighted avg in bin dimension (this value should be used to extract tensors for consistency with
    # 5_extract_timepoint_tensor code
    df['DT_weighted_avg_bins'] = df['DT_weighted_avg'] * 200.0 / 13.781163434903

    # Scatter plot for each protein
    # Create folder to save pdf files
    if not os.path.isdir('results/plots/tensor-recenter/'):
        os.makedirs('results/plots/tensor-recenter/')

    for name in list(set(df['name'].values)):

        n_charges = len(set(df[df['name'] == name]['charge'].values))

        fig, ax = plt.subplots(1, n_charges + 1, dpi=200, figsize=(3 * (n_charges + 1), 2.5))

        sns.scatterplot(data=df[df['name'] == name], x='dt', y='rt', palette='bright', hue='charge', size='auc_log',
                        ax=ax[0])
        ax[0].set_ylim(df[df['name'] == name]['RT_weighted_avg'].mean() - 0.4,
                       df[df['name'] == name]['RT_weighted_avg'].mean() + 0.4)
        ax[0].set_xlim(min(df[df['name'] == name]['DT_weighted_avg'].min(),
                           df[df['name'] == name]['dt'].min()) - 0.5,
                       max(df[df['name'] == name]['DT_weighted_avg'].max(),
                           df[df['name'] == name]['dt'].max()) + 0.5)
        ax[0].axhline(df[df['name'] == name]['RT_weighted_avg'].mean(), color='black', alpha=0.5, lw=0.5)
        ax[0].text(0, 1.01, '%s' % name, transform=ax[0].transAxes, fontsize=8)
        ax[0].set_xlabel('DT')
        ax[0].set_ylabel('RT')

        h, l = ax[0].get_legend_handles_labels()
        ax[0].legend(h[1:n_charges + 1], l[1:n_charges + 1], fontsize=10, loc=2,
                     bbox_to_anchor=(0.0, 0.15), bbox_transform=ax[0].transAxes, borderpad=0.02,
                     columnspacing=0.,
                     handletextpad=0.0, frameon=False, ncol=n_charges, prop={'size': 6})

        for j, charge in enumerate(sorted(list(set(df[df['name'] == name]['charge'].values)))):
            sns.scatterplot(data=df[(df['name'] == name) & (df['charge'] == charge)], x='dt', y='rt', palette='bright',
                            hue='file_index', size='auc_log', ax=ax[j + 1])
            ax[j + 1].text(0, 1.01, 'charge=%i+' % charge, transform=ax[j + 1].transAxes, fontsize=8)
            ax[j + 1].set_ylim(df[df['name'] == name]['RT_weighted_avg'].mean() - 0.4,
                               df[df['name'] == name]['RT_weighted_avg'].mean() + 0.4)
            ax[j + 1].set_xlim(df[(df['name'] == name) & (df['charge'] == charge)]['DT_weighted_avg'].mean() * 0.94,
                               df[(df['name'] == name) & (df['charge'] == charge)]['DT_weighted_avg'].mean() * 1.06)
            ax[j + 1].axhline(df[df['name'] == name]['RT_weighted_avg'].mean(), color='black', alpha=0.5, lw=0.5)
            ax[j + 1].axvline(df[(df['name'] == name) & (df['charge'] == charge)]['DT_weighted_avg'].mean(),
                              color='black', alpha=0.5, lw=0.5)

            # Plot horizontal and vertical lines corresponding to initial RT and DT centers used to extract tensors
            retention_label_center = \
            df[(df['name'] == name) & (df['charge'] == charge)]['ic'].values[0].retention_labels[
                len(df[(df['name'] == name) & (df['charge'] == charge)]['ic'].values[0].retention_labels) // 2]
            ax[j + 1].axhline(retention_label_center, color='red', alpha=0.5, lw=0.5)
            drift_label_center = df[(df['name'] == name) & (df['charge'] == charge)]['ic'].values[0].drift_labels[
                len(df[(df['name'] == name) & (df['charge'] == charge)]['ic'].values[0].drift_labels) // 2]
            ax[j + 1].axvline(drift_label_center, color='red', alpha=0.5, lw=0.5)

            ax[0].scatter(df[(df['name'] == name) & (df['charge'] == charge)]['DT_weighted_avg'].mean(),
                          (sum(df[(df['name'] == name) & (df['charge'] == charge)]['rt'] *
                               df[(df['name'] == name) & (df['charge'] == charge)]['auc'])) /
                          sum(df[(df['name'] == name) & (df['charge'] == charge)]['auc']), marker='x', color='black',
                          s=20)
            ax[j + 1].scatter(df[(df['name'] == name) & (df['charge'] == charge)]['DT_weighted_avg'].mean(),
                              (sum(df[(df['name'] == name) & (df['charge'] == charge)]['rt'] *
                                   df[(df['name'] == name) & (df['charge'] == charge)]['auc'])) /
                              sum(df[(df['name'] == name) & (df['charge'] == charge)]['auc']), marker='x',
                              color='black', s=20)

            n_files = len(set(df[(df['name'] == name) & (df['charge'] == charge)]['file_index'].values))
            h, l = ax[j + 1].get_legend_handles_labels()
            ax[j + 1].legend(h[1:n_files + 1], l[1:n_files + 1], fontsize=10, loc=2,
                             bbox_to_anchor=(0.0, 0.15), bbox_transform=ax[j + 1].transAxes, borderpad=0.02,
                             columnspacing=0.,
                             handletextpad=0.0, frameon=False, ncol=n_files, prop={'size': 6})
            ax[j + 1].set_xlabel('DT')
            ax[j + 1].set_ylabel('RT')

        name_recentered = '_'.join(name.split('_')[:-1]) + '_' + str(
            round(df[(df['name'] == name)]['RT_weighted_avg'].values[0], 5))

        plt.tight_layout()
        plt.savefig('results/plots/tensor-recenter/' + name_recentered + '.pdf', format='pdf', dpi=200)
        plt.close('all')

    return df


def plot_deviations(df):

    sns.set_context('talk')

    fig, ax = plt.subplots(4, 2, figsize=(10, 12), dpi=200)

    sns.histplot(df['n_UN'].values, ax=ax[0][0])
    sns.histplot(df['n_UN'].values, ax=ax[0][0], kde=True)

    sns.histplot(df['n_signals'].values, ax=ax[0][1])
    sns.histplot(df['n_signals'].values, ax=ax[0][1], kde=True)

    sns.histplot(df['im_mono'].values * 13.781163434903 / 200 - df['DT_weighted_avg'].values, ax=ax[1][0])
    sns.histplot(df['im_mono'].values * 13.781163434903 / 200 - df['DT_weighted_avg'].values, ax=ax[1][0], kde=True)
    ax[1][0].set_xlabel('DT error')

    sns.histplot(df['RT'].values - df['RT_weighted_avg'].values, ax=ax[1][1])
    sns.histplot(df['RT'].values - df['RT_weighted_avg'].values, ax=ax[1][1], kde=True)
    ax[1][1].set_xlabel('RT error')

    sns.histplot(df['dt_weighted_std'].values, ax=ax[2][0])
    sns.histplot(df['dt_weighted_std'].values, ax=ax[2][0], kde=True)

    sns.histplot(df['dt_std'].values, ax=ax[2][1])
    sns.histplot(df['dt_std'].values, ax=ax[2][1], kde=True)

    sns.histplot(df['rt_weighted_std'].values, ax=ax[3][0])
    sns.histplot(df['rt_weighted_std'].values, ax=ax[3][0], kde=True)

    sns.histplot(df['rt_std'].values, ax=ax[3][1], bins=100)

    fig.savefig('results/plots/deviations_UN.pdf', format='pdf', dpi=200)
    plt.close('all')


def main(configfile,
         library_info_path,
         all_idotp_inputs,
         all_ics_inputs,
         library_info_out_path=None,
         plot_out_path=None,
         return_flag=False,
         idotp_cutoff=0.99):
    """Reads all library_info index idotp_check.csv files and returns or saves a list of indices with idotp >= idotp_cutoff.

    Args:
        configfile (dict): configfile dict
        library_info_path (str): path/to/library_info.json
        all_idotp_inputs (list of strings): list of all input idotp json filepaths
        all_ics_inputs (list of strings): list of all input IsotopeCluster-list filepaths
        indices_out_path (str): path/to/filter_passing_indices.csv
        library_info_out_path (str): path/to/checked_library_info.json
        plot_out_path (str): path/to/file.png for idotp_distribution plot
        return_flag (bool): option to return a dictionary of outputs in python context
        idotp_cutoff (float): inclusive lower-bound on idotp [0,1] to be considered for evaluation, default=0.95

    Returns:
        out_df (Pandas DataFrame): Dataframe with all information from library_info and idotp_check for all idotp filter passing rows.

    """
    library_info = pd.read_json(library_info_path)

    df = generate_dataframe_ics(configfile=configfile,
                                all_ics_inputs=all_ics_inputs,
                                idotp_cutoff=idotp_cutoff)

    cols_idotp = ['idotp', 'integrated_mz_width', 'mz_centers', 'theor_mz_dist']
    cols_ics_recenter = ['RT_weighted_avg', 'DT_weighted_avg_bins', 'DT_weighted_avg', 'rt_std', 'dt_std',
                         'rt_weighted_std', 'dt_weighted_std', 'n_signals', 'n_UN']

    out_df = pd.DataFrame(columns=list(library_info.columns) + cols_idotp + cols_ics_recenter + ['name_recentered'])

    for i, (name, charge) in enumerate(set([(i, j) for (i, j) in df[['name', 'charge']].values])):
        open_idotp_f = pd.read_json([i for i in all_idotp_inputs if '%s' % (name + '_charge' + str(charge)) in i][0])
        my_row = library_info.loc[(library_info["name"] == name) & (library_info["charge"] == charge)].copy()
        my_row[cols_idotp] = open_idotp_f[cols_idotp].values
        my_row[cols_ics_recenter] = \
        df[(df['name'] == name) & (df['charge'] == charge)].sort_values(by='idotp', ascending=False)[
            cols_ics_recenter].values[0]
        my_row['name_recentered'] = '_'.join(name.split('_')[:-1]) + '_' + str(
            round(my_row['RT_weighted_avg'].values[0], 5))
        if not my_row['DT_weighted_avg'].values[0] < 0.1:
            out_df = pd.concat([out_df, my_row], ignore_index=True)

    if library_info_out_path is not None:
        out_df.drop_duplicates(subset=['name_recentered', 'charge'], ignore_index=True, inplace=True)
        out_df.to_json(library_info_out_path)

    if plot_out_path is not None:
        idotps = []
        for f in all_idotp_inputs:
            idotps.append(pd.read_json(f)['idotp'].values[0])
        sns.displot(idotps)
        plt.axvline(idotp_cutoff, 0, 1)
        plt.savefig(plot_out_path)
        plt.close('all')

    # Plot deviation plots. Add this to a proper output in the snakemake scope later
    df = pd.read_json(library_info_out_path)
    plot_deviations(df)

    if return_flag:
        return out_df


if __name__ == "__main__":
    # If the snakemake global object is present, save expected arguments from snakemake to be passed to main().
    if "snakemake" in globals():
        configfile_path = snakemake.input[0]
        library_info_path = snakemake.input[1]
        all_idotp_inputs = [item for item in snakemake.input if item.endswith("idotp_check.json")]
        all_ics_inputs = [item for item in snakemake.input if item.endswith(".cpickle.zlib")]

        library_info_out_path = snakemake.output[0]
        plot_out_path = snakemake.output[1]

        configfile = yaml.load(open(configfile_path, 'rb'), Loader=yaml.Loader)

        main(configfile=configfile,
             library_info_path=library_info_path,
             all_idotp_inputs=all_idotp_inputs,
             all_ics_inputs=all_ics_inputs,
             library_info_out_path=library_info_out_path,
             plot_out_path=plot_out_path)
    else:
        # CLI context, set expected arguments with argparse module.
        parser = argparse.ArgumentParser(
            description=
            "Reads all rt-group idotp csvs and returns or saves a list of indices with idotp >= idotp_cutoff."
        )
        parser.add_argument("configfile_path", help="path/to/config.conf")
        parser.add_argument("library_info_path", help="path/to/library_info.json")
        parser.add_argument("-i",
                            "--all_idotp_inputs",
                            help="list of all idotp check .json outputs to be read")
        parser.add_argument("-a",
                            "--all_ics_inputs",
                            help="list of all ics cpickle.zlib outputs to be read")
        parser.add_argument("-d",
                            "--input_dir_path",
                            help="path/to/dir/ containing idotp_check.csv files")
        parser.add_argument("-l",
                            "--library_info_out_path",
                            help="path/to/checked_library_info.json")
        parser.add_argument("--p",
                            "--plot_out_path",
                            help="path/to/idotp_distribution.png")
        parser.add_argument(
            "-c",
            "--idotp_cutoff",
            type=float,
            default=0.99,
            help=
            "lower limit on dot-product between theoretical integrated m/z of POI and int. m/z of observed signal in question. Float in range [0,1], default 0.95 "
        )
        args = parser.parse_args()

        if args.all_idotp_inputs is None and args.input_dir_path is None:
            parser.print_help()
            sys.exit()

        if args.all_idotp_inputs is None and args.input_dir_path is not None:
            args.all_idotp_inputs = sorted(
                list(glob.glob(args.input_dir_path + "*idotp_check.csv")))

        all_idotp_inputs = args.all_idotp_inputs.split(' ')

        configfile = yaml.load(open(args.configfile_path, 'rb'), Loader=yaml.Loader)

        main(configfile=args.configfile,
             library_info_path=args.library_info_path,
             all_idotp_inputs=args.all_idotp_inputs,
             all_ics_inputs=args.all_ics_inputs,
             library_info_out_path = args.library_info_out_path,
             plot_out_path=args.plot_out_path,
             idotp_cutoff=args.idotp_cutoff)
