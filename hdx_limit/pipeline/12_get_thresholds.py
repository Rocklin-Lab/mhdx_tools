import os
from hdx_limit.core.io import limit_read
import numpy as np
import pandas as pd
import json
import argparse


def save_json(d, output):
    # create json object from dictionary
    json_d = json.dumps(d)
    # open file for writing, "w"
    f = open(output, "w")
    # write json object to file
    f.write(json_d)
    # close file
    f.close()

def extract_ic_features(ics):

    array = np.zeros([len(ics), 6])

    for idx, ic in enumerate(ics):
        array[idx] = np.array([
            ic.dt_ground_err,
            ic.dt_ground_fit,
            ic.rt_ground_err,
            ic.rt_ground_fit,
            ic.baseline_peak_error,
            ic.baseline_integrated_mz_rmse,
        ])

    return array.T.tolist()


def generate_df_ics_features(fs):

    cols = ["dt_ground_err", "dt_ground_fit", "rt_ground_err",
            "rt_ground_fit", "baseline_peak_error", "baseline_integrated_mz_rmse"
            ]

    l = []
    for f in fs:

        if not os.stat(f).st_size == 0:

            ics = limit_read(f)

            features = extract_ic_features(ics)

            l.append(features)

    return pd.DataFrame(l, columns=cols)


def get_flatten_values(list_of_lists):
    return np.array([i for j in list_of_lists for i in j])


def get_threshold_value(array, n_std=2):
    mean = np.mean(array)
    std = np.std(array)

    return round(mean + n_std * std, 3)


def get_threshold_dictionary(df, n_std):

    d = {}

    for col in df.columns:

        values = get_flatten_values(df[col].values)

        if col in ["dt_ground_err", "rt_ground_err", "baseline_peak_error", "baseline_integrated_mz_rmse"]:

            d[col] = get_threshold_value(values, n_std=n_std)

        else:
            d[col] = get_threshold_value(values, n_std=-n_std)


    return d


def main(fs, n_std, df_output_path=None, d_thresholds_output_path=None):

    df = generate_df_ics_features(fs)

    if df_output_path is not None:
        df.to_json(df_output_path)

    dict_thresholds = get_threshold_dictionary(df, n_std=n_std)

    if d_thresholds_output_path is not None:
        save_json(dict_thresholds, d_thresholds_output_path)


if __name__ == "__main__":

    if "snakemake" in globals():

        winner_fs = snakemake.input

        df_output_path = snakemake.output[0]
        d_thresholds_output_path = snakemake.output[1]

        n_std = snakemake.params.n_std

        main(fs=winner_fs,
             n_std=n_std,
             df_output_path=df_output_path,
             d_thresholds_output_path=d_thresholds_output_path)

    else:

        print("TODO: Implement argparse")
