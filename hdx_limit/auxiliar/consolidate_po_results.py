import os, yaml, argparse, glob
import numpy as np
import pandas as pd
from hdx_limit.core.io import limit_read


def load_data(file_path):
    """
    Attempts to load data from JSON, falls back to pickle if JSON fails.
    Returns the DataFrame and file format loaded.
    """
    try:
        df = pd.read_json(file_path)
        format_used = "json"
    except ValueError:
        df = pd.read_pickle(file_path)
        format_used = "pickle"
    return df, format_used

def generate_extensive_dataframe_po_scores(fs, library, ph):

    print("Extracting PO scores...")

    l, d = [], {}

    for i, f in enumerate(fs):

        if i % 100 == 0:
            print(f"Processed {i} files... {len(fs) - i} files to go")

        if not os.stat(f).st_size == 0:

            name_rt_group = f.split("/")[-3]

            d = limit_read(f)

            scores = [d[key][0] * d[key][1] for key in d]

            total_score = np.sum(scores)

            item = [name_rt_group, library, ph] + scores + [total_score]

            l.append(item)

    print("PO scores Processing Completed!")

    return pd.DataFrame(l, columns=["name_rt-group", "library", "pH"] + ["PO_" + key for key in d.keys()] + [
        "PO_total_score"])


def extract_tensor_stats_info(fs, library, ph):
    """

    Extract winner_frac_intensity, coverage and complexity

    """

    print("Extracting tensors features...")

    l = []

    n_files = len(fs)

    for i, f in enumerate(fs):

        if i % 100 == 0:
            print(f"Processing {i}/{n_files} ...")

        if not os.stat(f).st_size == 0:

            name_rt_group = f.split("/")[-3]

            try:

                df_tmp = pd.read_json(f)
                peak_error_d = {charge:df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0]["ic"]["baseline_peak_error"] for charge in sorted(set(df_tmp.query("prefiltered == 1 & tp_idx == 0")["charge"].values))}
                idotp_d = {charge:df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0]["ic"]["idotp"] for charge in sorted(set(df_tmp.query("prefiltered == 1 & tp_idx == 0")["charge"].values))}
                ic_auc_d = {charge:df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0]["ic"]["ic_auc"] for charge in sorted(set(df_tmp.query("prefiltered == 1 & tp_idx == 0")["charge"].values))}
                dt_tensor_center = {charge:(df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0].ic["drift_labels"][-1] + df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0].ic["drift_labels"][0]) / 2
                 for charge in sorted(set(df_tmp.query("prefiltered == 1 & tp_idx == 0")["charge"].values))}
                rt_tensor_center = df_tmp.iloc[0]["rt_tensor_center"]

            except:

                df_tmp = pd.read_pickle(f)
                peak_error_d = {charge:df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0]["ic"].baseline_peak_error for charge in sorted(set(df_tmp.query("prefiltered == 1 & tp_idx == 0")["charge"].values))}
                idotp_d = {charge:df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0]["ic"].idotp for charge in sorted(set(df_tmp.query("prefiltered == 1 & tp_idx == 0")["charge"].values))}
                ic_auc_d = {charge:df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0]["ic"].ic_auc for charge in sorted(set(df_tmp.query("prefiltered == 1 & tp_idx == 0")["charge"].values))}
                dt_tensor_center = {charge:(df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0]["ic"].drift_labels[-1] + df_tmp.query(f"tp_idx == 0 & prefiltered == 1 & charge == {charge}").iloc[0]["ic"].drift_labels[0]) / 2
                 for charge in sorted(set(df_tmp.query("prefiltered == 1 & tp_idx == 0")["charge"].values))}
                retention_labels = df_tmp.query(f"tp_idx == 0 & prefiltered == 1").iloc[0]["ic"].retention_labels
                rt_tensor_center = (retention_labels[-1] + retention_labels[0])/2


            winner_intensity_frac = df_tmp[df_tmp["winner"] == 1]["ic_auc"].sum() / df_tmp[df_tmp["winner"] == 1][
                "tensor_auc"].sum()

            coverage_atc = len(
                df_tmp[(df_tmp["prefiltered"] == 0) & (df_tmp["winner"] == 0) & (df_tmp["ic_winner_corr"] >= 0.95)][
                    ["tp_idx", "charge"]].drop_duplicates()) / len(np.unique(df_tmp["tp_idx"].values))

            coverage_prefiltered = len(
                df_tmp[(df_tmp["prefiltered"] == 1) & (df_tmp["winner"] == 0) & (df_tmp["ic_winner_corr"] >= 0.95)][
                    ["tp_idx", "charge"]].drop_duplicates()) / len(np.unique(df_tmp["tp_idx"].values))

            complexity_atc = len(df_tmp[(df_tmp["prefiltered"] == 0) & (df_tmp["winner"] == 0)]) / len(
                np.unique(df_tmp["tp_idx"].values))

            complexity_prefiltered = len(df_tmp[(df_tmp["prefiltered"] == 1) & (df_tmp["winner"] == 0)]) / len(
                np.unique(df_tmp["tp_idx"].values))

            l.append([
                name_rt_group, library, ph, winner_intensity_frac,
                coverage_atc, coverage_prefiltered,
                complexity_atc, complexity_prefiltered,
                peak_error_d, idotp_d, ic_auc_d, dt_tensor_center, rt_tensor_center
            ])

    print("Tensor Processing Completed!")

    return pd.DataFrame(l, columns=["name_rt-group", "library", "pH", "winner_intensity_fraction",
                                    "coverage_atc", "coverage_prefiltered", "complexity_atc",
                                    "complexity_prefiltered",
                                    "peak_error", "idotp", "ic_auc", "dt_tensor_center", "rt_tensor_center"])


def get_scores(ics):
    n_tp = len(ics)

    array = np.zeros([n_tp, 9])

    for idx, ic in enumerate(ics):
        array[idx] = np.array([
            ic.dt_ground_err,
            ic.dt_ground_fit,
            ic.rt_ground_err,
            ic.rt_ground_fit,
            ic.baseline_peak_error,
            ic.auc_ground_err[0],
            ic.nearest_neighbor_correlation,
            ic.baseline_integrated_mz_rmse,
            ic.baseline_integrated_mz_FWHM
        ])

    return array.T.tolist()


def generate_extensive_dataframe_ics_scores(fs, library, ph):

    print("Extracting ICs features...")

    l = []

    for i, f in enumerate(fs):

        if i % 100 == 0:
            print(f"Processed {i} files... {len(fs) - i} to go")

        if not os.stat(f).st_size == 0:

            full_path = os.path.abspath(f)

            name_rt_group = f.split("/")[-3]

            ics = limit_read(f)

            centroids = [ic.baseline_integrated_mz_com for ic in ics]

            timepoint_idx = [ic.timepoint_idx for ic in ics]

            scores = get_scores(ics)

            l.append([name_rt_group, library, ph] + [centroids] + [timepoint_idx] + [len(ics)] + scores + [full_path])


    print("PO scores Processing Completed!")

    return pd.DataFrame(l, columns=["name_rt-group", "library", "pH", "centroids", "timepoints", "n_timepoints",
                                    "dt_ground_err", "dt_ground_fit",
                                    "rt_ground_err", "rt_ground_fit", "baseline_peak_error",
                                    "auc_ground_err", "nearest_neighbor_correlation",
                                    "baseline_integrated_mz_rmse", "baseline_integrated_mz_FWHM", "PO_winner_path"])


def consolidate_results(config,
                        winner_scores_fs,
                        tensor_df_fs,
                        winner_ics_fs,
                        output_df_path=None,
                        ):

    run_name = config["run_name"]
    library = run_name.split("_")[0]
    ph = run_name.split("_")[1]

    print(f"Using library: {library} and pH: {ph}")
    print("If these are not the intended values, please check the run_name in the config file")

    n_timepoints_filter = config["consolidate"]["n_timepoints_min"]
    po_total_score_filter = config["consolidate"]["po_total_score_max"]

    po_df_extensive = generate_extensive_dataframe_po_scores(winner_scores_fs, library, ph)
    tensor_df_extensive = extract_tensor_stats_info(tensor_df_fs, library, ph)
    ics_df_extensive = generate_extensive_dataframe_ics_scores(winner_ics_fs, library, ph)

    # Merge data
    ics_po_df_extensive = pd.merge(ics_df_extensive, po_df_extensive, left_on=["name_rt-group", "library", "pH"],
                                   right_on=["name_rt-group", "library", "pH"],
                                   how="left")

    tensor_ics_po_df_extensive = pd.merge(ics_po_df_extensive, tensor_df_extensive, left_on=["name_rt-group", "library", "pH"],
                                          right_on=["name_rt-group", "library", "pH"],
                                          how="left")

    # Check if final dataframe doesn't contain any NaN
    assert tensor_ics_po_df_extensive.isna().sum().sum() == 0

    # Save dataframe
    if n_timepoints_filter is not None and po_total_score_filter is not None:
        if output_df_path is not None:
            tensor_ics_po_df_extensive.query(f"PO_total_score < {po_total_score_filter} & n_timepoints > {n_timepoints_filter}").reset_index(drop=True).to_json(output_df_path)
        else:
            return tensor_ics_po_df_extensive.query(f"PO_total_score < {po_total_score_filter} & n_timepoints > {n_timepoints_filter}").reset_index(drop=True)
    elif output_df_path is not None:
        tensor_ics_po_df_extensive.reset_index(
            drop=True).to_json(output_df_path)
    else:
        return tensor_ics_po_df_extensive.reset_index(drop=True)




if __name__ == "__main__":

    if "snakemake" in globals():
        config = yaml.load(open(snakemake.input[0], "rb").read(), Loader=yaml.Loader)

        fs = glob.glob('resources/10_ic_time_series/*/multibody/*winner_scores_multibody.cpickle.zlib') + glob.glob('resources/10_ic_time_series/*/multibody/*_winner_multibody.df.pkl') + glob.glob('resources/10_ic_time_series/*/multibody/*_winner_multibody.cpickle.zlib')
        # flat_fs = [file for sublist in fs for file in sublist]

        winner_scores_fs = [i for i in fs if (os.stat(i).st_size > 0) and ("winner_scores_multibody.cpickle.zlib" in i)]
        tensor_df_fs = [i for i in fs if (os.stat(i).st_size > 0) and ("_winner_multibody.df.pkl" in i)]
        winner_ics_fs = [i for i in fs if (os.stat(i).st_size > 0) and ("_winner_multibody.cpickle.zlib" in i)]
        output_df_path = snakemake.output[0]

    else:

        parser = argparse.ArgumentParser()

        parser.add_argument("-c",
                            "--configfile",
                            required=True,
                            help="path/to/configfile")
        parser.add_argument("-ws",
                            "--winners_scores",
                            required=True,
                            nargs="+",
                            help="list of winners score files (pattern: *_winner_scores_multibody.cpickle.zlib)")
        parser.add_argument("-t",
                            "--tensors",
                            required=True,
                            nargs="+",
                            help="list of tensors df files (pattern: *_winner_multibody.df.pkl)")
        parser.add_argument("-wi",
                            "--winners_ics",
                            required=True,
                            nargs="+",
                            help="list of winners ics files (pattern: *_winner_multibody.cpickle.zlib)")
        parser.add_argument("-o",
                            "--output_df_path",
                            required=True,
                            help="path/to/configfile")

        args = parser.parse_args()

        config = yaml.load(open(args.configfile, "rb").read(), Loader=yaml.Loader)
        winner_scores_fs = args.winners_scores
        tensor_df_fs = args.tensors
        winner_ics_fs = args.winners_ics
        output_df_path = args.output_df_path


    consolidate_results(config,
                        winner_scores_fs,
                        tensor_df_fs,
                        winner_ics_fs,
                        output_df_path=output_df_path
                        )
