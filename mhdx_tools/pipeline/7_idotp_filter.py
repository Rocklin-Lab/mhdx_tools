import sys
import glob
import argparse
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import numpy as np
import yaml
import matplotlib.pyplot as plt
from mhdx_tools.auxiliar.plots import plot_rtdt_recenter, plot_deviations
from mhdx_tools.auxiliar.filters import generate_dataframe_ics, remove_duplicates_from_df
from mhdx_tools.auxiliar.fdr import plot_fdr_stats
from mhdx_tools.auxiliar import qvalue_estimator

mpl.use("Agg")


def main(configfile,
         library_info_path,
         all_idotp_inputs,
         all_ics_inputs,
         library_info_out_path=None,
         idotp_plot_out_path=None,
         UN_deviations_plot_output_path=None,
         fdr_plot_output_path=None,
         return_flag=False):
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
                                all_ics_inputs=all_ics_inputs)

    # Generate RT/DT recentering plots
    if configfile["plot_rtdt_recenter"]:
        plot_rtdt_recenter(df,
                           output_folder='results/plots/tensor_rtdt_recenter/')

    # Create new dataframe with recentered names
    cols_idotp = ['idotp', 'integrated_mz_width', 'mz_centers', 'theor_mz_dist']
    cols_ics_recenter = ['RT_weighted_avg', 'DT_weighted_avg_bins', 'DT_weighted_avg', 'rt_std', 'dt_std',
                         'rt_weighted_std', 'dt_weighted_std', 'rt_gaussian_rmse', 'dt_gaussian_rmse',
                         'n_signals', 'n_UN']
    cols = list(library_info.columns) + cols_idotp + cols_ics_recenter + ['name_tmp']

    l = []
    for i, (name, charge) in enumerate(set([(i, j) for (i, j) in df[['name', 'charge']].values])):
        open_idotp_f = pd.read_json([i for i in all_idotp_inputs if '%s' % (name + '_charge' + str(charge)) in i][0])
        my_row = library_info.loc[(library_info["name_rt-group"] == name) & (library_info["charge"] == charge)].copy()
        my_row[cols_idotp] = open_idotp_f[cols_idotp].values
        my_row[cols_ics_recenter] = \
            df[(df['name'] == name) & (df['charge'] == charge)].sort_values(by=['idotp', 'rt_gaussian_rmse',
                                                                                'dt_gaussian_rmse'],
                                                                            ascending=[False, True, True])[
                cols_ics_recenter].values[0]
        my_row['name_tmp'] = '_'.join(name.split('_')[:-1]) + '_' + str(
            round(my_row['RT_weighted_avg'].values[0], 2))
        if not my_row['DT_weighted_avg'].values[0] < 0.1:
            # l.append(my_row.values)
            l.append(my_row.values.tolist()[0])
    out_df = pd.DataFrame(l, columns=cols)

    for name in set(out_df['name_tmp']):
        out_df.loc[out_df['name_tmp'] == name, 'n_charges'] = len(out_df[out_df['name_tmp'] == name])
    out_df['decoy'] = out_df['name_tmp'].str.contains('decoy')
    out_df['log2_ab_cluster_total'] = np.log2(out_df['ab_cluster_total'].values.astype('float32'))

    # This step is critical for the next step to work properly. name will be preserved without rt-group information
    # name-rt-group will be redefined if use_rtdt_recenter is True
    if configfile["use_rtdt_recenter"]:
        out_df.rename(columns={"name_rt-group": "name_rt-group_before_recentering", "name_tmp": "name_rt-group"}, inplace=True)
    else:
        out_df.rename(columns={"name_tmp": "name_rt-group_after_recentering"}, inplace=True)

    if fdr_plot_output_path is not None:
        plot_fdr_stats(out_df,
                       output_path=fdr_plot_output_path)

    # Remove duplicates based on RT and DT proximity
    if configfile["remove_duplicates"]:
        out_df = remove_duplicates_from_df(out_df,
                                           rt_threshold=configfile["rt_threshold"],
                                           dt_threshold=configfile["dt_threshold"])


    # Implement qvalue_estimator filter here
    model_dir = configfile['mhdx_tools_dir'] + "/models/qvalue_estimator/"
    prob_threshold_value = configfile['prob_threshold_value']
    qvalue_estimator_output = qvalue_estimator.apply_model_to_new_data(out_df,
                                                              model_dir=model_dir,
                                                              prob_threshold_value=prob_threshold_value)

    out_df_updated = out_df.merge(qvalue_estimator_output[['name', 'logreg_prob']], on='name', how='inner')

    # Filter based on prob_threshold_value
    out_df_updated = out_df_updated.query(f"logreg_prob > {prob_threshold_value}").reset_index(drop=True)


    if library_info_out_path is not None:
        out_df_updated.to_json(library_info_out_path)

    if idotp_plot_out_path is not None:
        idotps = []
        for f in all_idotp_inputs:
            idotps.append(pd.read_json(f)['idotp'].values[0])
        sns.displot(idotps)
        plt.axvline(configfile["idotp_cutoff"], 0, 1)
        plt.savefig(idotp_plot_out_path, format='png', dpi=150)
        plt.close('all')

    # Plot deviation plots. Add this to a proper output in the snakemake scope later
    if UN_deviations_plot_output_path is not None:
        plot_deviations(out_df_updated,
                        configfile=configfile,
                        output_path=UN_deviations_plot_output_path)

    if return_flag:
        return out_df_updated


if __name__ == "__main__":
    # If the snakemake global object is present, save expected arguments from snakemake to be passed to main().
    if "snakemake" in globals():
        configfile_path = snakemake.input[0]
        library_info_path = snakemake.input[1]
        all_idotp_inputs = [item for item in snakemake.input if item.endswith("idotp_check.json")]
        all_ics_inputs = [item for item in snakemake.input if item.endswith(".cpickle.zlib")]

        library_info_out_path = snakemake.output[0]
        idotp_plot_out_path = snakemake.output[1]
        UN_deviations_plot_output_path = snakemake.output[2]
        fdr_plot_output_path = snakemake.output[3]

        configfile = yaml.load(open(configfile_path, 'rb'), Loader=yaml.Loader)

        main(configfile=configfile,
             library_info_path=library_info_path,
             all_idotp_inputs=all_idotp_inputs,
             all_ics_inputs=all_ics_inputs,
             library_info_out_path=library_info_out_path,
             idotp_plot_out_path=idotp_plot_out_path,
             UN_deviations_plot_output_path=UN_deviations_plot_output_path,
             fdr_plot_output_path=fdr_plot_output_path)
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
                            "--idotp_plot_out_path",
                            help="path/to/idotp_distribution.png")
        parser.add_argument("-u",
                            "--UN_deviations_plot_output_path",
                            help="path/to/undeuterated_deviations")
        parser.add_argument("-f",
                            "--fdr_plot_output_path",
                            help="path/to/fdr_plot")
        parser.add_argument(
            "-c",
            "--idotp_cutoff",
            type=float,
            default=0.99,
            help=
            "lower limit on dot-product between theoretical integrated m/z of POI and int. m/z of observed signal in question. Float in range [0,1], default 0.95 "
        )
        parser.add_argument("--r",
                            "--remove_duplicates",
                            default=False,
                            help="Remove duplicates from checked_library_info")
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
             library_info_out_path=args.library_info_out_path,
             idotp_plot_out_path=args.idotp_plot_out_path,
             fdr_plot_output_path=args.fdr_plot_output_path,
             UN_deviations_plot_output_path=args.UN_deviations_plot_output_path
             )