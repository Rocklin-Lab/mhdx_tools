import pandas as pd
import numpy as np
from mhdx_tools.core.io import limit_read


def check_drift_labels(drift_labels, min_length=3, low_dt_value=0.2):
    """
    check if the drift labels are okay
    Args:
        drift_labels: drift labels array
        min_length: min length of the array
        low_dt_value: min dt value

    Returns:

    """
    if len(drift_labels) >= min_length:
        if drift_labels[0] > low_dt_value:
            check = True
        else:
            check = False
    else:
        check = False
    return check


def generate_dataframe_ics(configfile,
                           all_ics_inputs):

    # Create dictionary containing all ics passing idotp_cutoff
    protein_ics = {}
    for f in all_ics_inputs:
        if f.split("/")[-2:-1][0] not in protein_ics:
            ics = [ic for ic in limit_read(f) if ic.idotp >= configfile["idotp_cutoff"]]
            if len(ics) > 0:
                protein_ics[f.split("/")[-2:-1][0]] = [ics]
        else:
            ics = [ic for ic in limit_read(f) if ic.idotp >= configfile["idotp_cutoff"]]
            if len(ics) > 0:
                protein_ics[f.split("/")[-2:-1][0]].append(ics)

    # Flat list of lists of ics (all charge states will be one single list
    for key in protein_ics:
        protein_ics[key] = [i for sublist in protein_ics[key] for i in sublist]

    # Extract values for dt, rt, auc, charge and file index from each IC and store in a dataframe
    data = []
    for key in protein_ics:
        for ic in protein_ics[key]:
            if check_drift_labels(drift_labels=ic.drift_labels):
                dt = ic.drift_labels[0] + (ic.drift_labels[1] - ic.drift_labels[0]) * ic.dt_coms
                rt = ic.retention_labels[0] + (ic.retention_labels[1] - ic.retention_labels[0]) * ic.rt_com
                rt_gaussian_rmse = ic.rt_gaussian_rmse
                dt_gaussian_rmse = ic.dt_gaussian_rmse
                if dt < configfile["dt_max"]:
                    if rt < configfile["rt_max"]:
                        auc = ic.ic_auc_with_gauss_extrapol
                        charge = ic.charge_states[0]
                        file_index = configfile[0].index([i for i in configfile[0] if "_".join(
                            ic.info_tuple[0].split("/")[-1].split(".")[-5:-4][0].split("_")[-4:]) in i][0])
                        idotp = ic.idotp

                        data.append(
                            [key, ic, rt, dt, rt_gaussian_rmse, dt_gaussian_rmse, auc, charge, file_index, idotp])

    df = pd.DataFrame(data, columns=["name", "ic", "rt", "dt", "rt_gaussian_rmse", "dt_gaussian_rmse", "auc", "charge",
                                     "file_index", "idotp"])
    df["auc_log"] = 2 * np.log10(df["auc"])

    # Remove ics with bad RT/DT factorization (high gaussian_fit_rmses)
    df = df[(df["rt_gaussian_rmse"] < configfile["RT_gaussian_rmse_threshold"]) &
            (df["dt_gaussian_rmse"] < configfile["DT_gaussian_rmse_threshold"])].reset_index(drop=True)

    # Find DT weighted average
    for name, charge in set([(n, c) for (n, c) in df[["name", "charge"]].values]):
        # Remove outliers
        percentile25, percentile75 = df[(df["name"] == name) & (df["charge"] == charge)]["dt"].quantile(0.25), \
                                     df[(df["name"] == name) & (df["charge"] == charge)]["dt"].quantile(0.75)
        iqr = percentile75 - percentile25
        lb, ub = percentile25 - 1.5 * iqr, percentile75 + 1.5 * iqr
        if len(df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                df["dt"] <= configfile["dt_max"])]) > 0:
            df.loc[(df["name"] == name) & (df["charge"] == charge), "DT_weighted_avg"] = sum(
                df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"])]["dt"] *
                df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"])]["auc"]) / sum(
                df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"])]["auc"])
            # How many signals do we see? How many undeuterated files generated passing ICs?
            df.loc[df["name"] == name, "n_signals"] = len(
                df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"])])
            df.loc[df["name"] == name, "n_UN"] = len(
                set(df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"])]["file_index"].values))
            if len(df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                    df["dt"] <= configfile["dt_max"])]) > 1:
                # DT standard deviation
                df.loc[(df["name"] == name) & (df["charge"] == charge), "dt_std"] = df[(df["name"] == name) & (
                        df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (df["dt"] <= configfile[
                    "dt_max"])]["dt"].std()
                # DT weighted standard deviation
                values = df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"])]["dt"]
                weights = df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"])]["auc"]
                avg_value = df[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"])]["DT_weighted_avg"]
                df.loc[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"]), "dt_weighted_std"] = np.sqrt(
                    (weights * (values - avg_value) ** 2) / sum(weights) * (len(values) - 1))
            else:
                df.loc[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"]), "dt_std"] = 0
                df.loc[(df["name"] == name) & (df["charge"] == charge) & (df["dt"] >= lb) & (df["dt"] <= ub) & (
                        df["dt"] <= configfile["dt_max"]), "dt_weighted_std"] = 0
        else:
            df.loc[(df["name"] == name) & (df["charge"] == charge), "DT_weighted_avg"] = -1

    # Find RT weighted average
    for name in set(df["name"].values):
        # Remove outliers
        percentile25, percentile75 = df[(df["name"] == name)]["rt"].quantile(0.25), \
                                     df[(df["name"] == name)]["rt"].quantile(0.75)
        iqr = percentile75 - percentile25
        lb, ub = percentile25 - 1.5 * iqr, percentile75 + 1.5 * iqr
        df.loc[df["name"] == name, "RT_weighted_avg"] = sum(
            df[(df["name"] == name) & (df["rt"] >= lb) & (df["rt"] <= ub)]["rt"] * df[(df["name"] == name)
                                                                                      & (df["rt"] >= lb) & (
                                                                                                  df["rt"] <= ub)][
                "auc"]) / sum(df[(df["name"] == name)
                                 & (df["rt"] >= lb) & (df["rt"] <= ub)]["auc"])
        if len(df.loc[df["name"] == name, "RT_weighted_avg"]) > 1:
            # DT standard deviation
            df.loc[df["name"] == name, "rt_std"] = df[(df["name"] == name) & (df["rt"] >= lb) & (df["rt"] <= ub)][
                "rt"].std()
            # DT weighted standard deviation
            values = df[(df["name"] == name) & (df["rt"] >= lb) & (df["rt"] <= ub)]["rt"]
            weights = df[(df["name"] == name) & (df["rt"] >= lb) & (df["rt"] <= ub)]["auc"]
            avg_value = df[(df["name"] == name) & (df["rt"] >= lb) & (df["rt"] <= ub)]["RT_weighted_avg"]
            df.loc[(df["name"] == name) & (df["rt"] >= lb) & (df["rt"] <= ub),
                   "rt_weighted_std"] = np.sqrt((weights * (values - avg_value) ** 2) / sum(weights) * (len(values) - 1)
                                                )
        else:
            df.loc[(df["name"] == name) & (df["rt"] >= lb) & (df["rt"] <= ub), "rt_std"] = 0
            df.loc[(df["name"] == name) & (df["rt"] >= lb) & (df["rt"] <= ub),
                   "rt_weighted_std"] = 0

        # Compute DT weighted avg in bin dimension (this value should be used to extract tensors for consistency with
    # 5_extract_timepoint_tensor code
    df["DT_weighted_avg_bins"] = df["DT_weighted_avg"] * 200.0 / configfile["dt_max"]

    return df

def remove_duplicates_from_df(df,
                              rt_threshold=0.2,
                              dt_threshold=0.05):
    # rt_threshold: delta rt in minutes
    # dt_threshold: delta dt in as a fraction of weighted average value

    new_df = pd.DataFrame(columns=df.columns)

    for i, line in df.sort_values(by=["n_UN", "ab_cluster_total"], ascending=[False, False]).iterrows():
        if len(new_df[(new_df["sequence"] == line["sequence"]) & (new_df["charge"] == line["charge"]) & (
                abs(new_df["RT_weighted_avg"] - line["RT_weighted_avg"]) < rt_threshold) & (
                              abs(new_df["DT_weighted_avg"] - line["DT_weighted_avg"]) < dt_threshold * line[
                          "DT_weighted_avg"])]) == 0:
            new_df = pd.concat([new_df, pd.DataFrame([line])])

    new_df.drop_duplicates(subset=["name_rt-group", "charge"], ignore_index=True, inplace=True)

    return new_df