import os
from dataclasses import dataclass
import time
import sys
import copy
import psutil
import yaml
import molmass
import numpy as np
import nn_fac
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
import scipy as sp
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from scipy.stats import linregress
from Bio.SeqUtils import ProtParam
import pandas as pd
from sklearn.cluster import DBSCAN
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt


def load_names_and_seqs(names_and_seqs_path):
    # read list of all proteins in sample
    allseq = pd.read_csv(names_and_seqs_path)
    allseq["MW"] = [
        ProtParam.ProteinAnalysis(seq, monoisotopic=True).molecular_weight()
        for seq in allseq["sequence"]
    ]
    allseq["len"] = [len(seq) for seq in allseq["sequence"]]

    return allseq


def load_isotopes_file(isotopes_path):
    with open(isotopes_path) as file:
        lines = [x.strip() for x in file.readlines()]
    out = []
    for i, line in enumerate(lines):
        if line == "SCAN_START":
            RT = float(lines[i + 1].split()[2][3:-1])
            j = i + 3
            while lines[j] != "SCAN_END":
                out.append([float(x) for x in lines[j].split()] + [RT])
                j += 1

    df = pd.DataFrame(out)
    df.columns = [
        "mz_mono",
        "im_mono",
        "ab_mono_peak",
        "ab_mono_total",
        "mz_top",
        "im_top",
        "ab_top_peak",
        "ab_top_total",
        "cluster_peak_count",
        "idx_top",
        "charge",
        "mz_cluster_avg",
        "ab_cluster_peak",
        "ab_cluster_total",
        "cluster_corr",
        "noise",
        "RT"
    ]

    return df


def apply_cluster_weights(dataframe,
                          dt_weight=5,
                          rt_weight=0.6,
                          mz_weight=0.006,
                          adjusted=False):
    """Applies heuristic weights to raw physical values for cluster scoring.

    Args:
        dataframe (Pandas DataFrame): DF of charged species to reweight
        dt_weight (float): divisor for dt
        rt_weight (float): divisor for rt
        mz_weight (float): divisor for mz

    Returns:
        None

    """
    # TODO: This is not great style, this should accept 3 lists and 3 weights and return 3 new lists
    dataframe["cluster_im"] = dataframe["im_mono"] / dt_weight
    dataframe["cluster_RT"] = dataframe["RT"] / rt_weight
    if adjusted:
        dataframe["cluster_mz"] = dataframe["mz_mono_fix_round"] / mz_weight
    else:
        dataframe["cluster_mz"] = dataframe["mz_mono"] / mz_weight


def cluster_lines(dataframe, min_samples=5, eps=0.5):
    # Create dbscan object, fit, and apply cluster ids to testq lines.
    db = DBSCAN(min_samples=min_samples, eps=eps)
    db.fit(dataframe[["cluster_im", "cluster_RT", "cluster_mz", "charge"]])
    clusters = db.fit_predict(
        dataframe[["cluster_im", "cluster_RT", "cluster_mz", "charge"]])
    dataframe["cluster"] = clusters


def getnear(x, allseq, charge=None, ppm=50):
    """Creates sub-DataFrame of sumdf near a given mass

    Args:
        x (float): molecular weight of a library protein
        charge (int): charge-state filter for search
        miz (int): DEPRECATED
        ppm (float or int): Parts-per-million error-radius around base-peak m/z to search


    Returns:
        tempdf (Pandas DataFrame): DF of charged species near the given mass

    """
    subdf = allseq
    if charge != None:
        low, high = (
            ((x * charge) - (1.007825 * charge)) * ((1e6 - ppm) / 1e6),
            ((x * charge) - (1.007825 * charge)) * ((1e6 + ppm) / 1e6),
        )
        mlow, mhigh = allseq["MW"] > low, allseq["MW"] < high
        tempdf = allseq[mlow & mhigh].sort_values("MW")[[
            "MW", "name", "len", "sequence"
        ]]
        tempdf["plus%s" % int(charge)] = [
            (q + (1.007825 * charge)) / charge for q in tempdf["MW"]
        ]
        tempdf["ppm"] = [
            "%.1f" % ((1.0 - (q / x)) * 1e6)
            for q in tempdf["plus%s" % int(charge)]
        ]
        tempdf["abs_ppm"] = [
            np.abs(((1.0 - (q / x)) * 1e6))
            for q in tempdf["plus%s" % int(charge)]
        ]
        return tempdf[[
            "plus%s" % int(charge),
            "ppm",
            "abs_ppm",
            "MW",
            "name",
            "len",
            "sequence",
        ]]
    else:
        low, high = x - 1e-6 * x, x + 1e-6 * x
        mlow, mhigh = allseq["MW"] > low, allseq["MW"] < high
        tempdf = subdf[mlow & mhigh].sort_values("MW")[[
            "MW", "name", "len", "sequence"
        ]]
        tempdf["ppm"] = [ 1e6 * (x - q) / q for q in tempdf["MW"] ]
        return tempdf


def cluster_df(testq, allseq, ppm=50, adjusted=False):
    """Determine clustered charged-species signals and label with their cluster index.

    Args:
        testq (Pandas DataFrame): DF containing getnear() output
        ppm (int or float): parts-per-million error cutoff to consider a signal
        adjusted (bool): flag to alert func to use updated m/Z values

    Returns:
        sum_df (Pandas DataFrame): testq with cluster index labels added

    """
    n_ambiguous = 0
    sum_data = []
    for c in range(0, max(testq["cluster"]) + 1):
        cluster_df = testq[testq["cluster"] == c]
        charge = np.median(cluster_df["charge"])
        if adjusted:
            mz = np.average(cluster_df["mz_mono_fix_round"],
                            weights=cluster_df["ab_cluster_total"])
        else:
            mz = np.average(cluster_df["mz_mono"],
                            weights=cluster_df["ab_cluster_total"])
        RT = np.average(cluster_df["RT"],
                        weights=cluster_df["ab_cluster_total"])
        im = np.average(cluster_df["im_mono"],
                        weights=cluster_df["ab_cluster_total"])

        near = getnear(mz, allseq, charge=charge, ppm=ppm)

        if len(near) == 1:

            sum_data.append([
                near["name"].values[0],
                RT,
                im,
                sum(cluster_df["ab_cluster_total"]),
                near["MW"].values[0],
                charge,
                near["plus%s" % int(charge)].values[0],
                mz,
                near["ppm"].values[0],
                near["abs_ppm"].values[0],
                c,
            ])

        elif len(near) > 1:

            n_ambiguous += 1

    sum_df = pd.DataFrame(sum_data)
    sum_df.columns = [
        "name",
        "RT",
        "im_mono",
        "ab_cluster_total",
        "MW",
        "charge",
        "expect_mz",
        "obs_mz",
        "ppm",
        "abs_ppm",
        "cluster",
    ]
    sum_df["ppm"] = [float(x) for x in sum_df["ppm"]]

    print(f"Found {n_ambiguous} ambigous identifications...")

    return sum_df


def load_pickle_file(pickle_fpath):
    with open(pickle_fpath, "rb") as file:
        pk_object = pk.load(file)
    return pk_object


def cluster_df_hq_signals(testq,
                          allseq,
                          ppm=50,
                          intensity_threshold=1e4,
                          cluster_correlation=0.99):
    """Cluster high quality mz signals based on intensities and cluster correlation, applies cluster lables to the input DF.

    Args:
        testq (Pandas DataFrame): dataframe from imtbx
        ppm (float): ppm error to include for mz signals
        intensity_threshold (float): minimum intensity value required for mz signals
        cluster_correlation (float): cluster correlation from imtbx. higher correlation means better isotopic distribution
        adjusted (bool): Boolean to indicate if mz signals have already been corrected

    Returns:
        sum_df (Pandas DataFrame): testq with cluster lables applied
    """

    if "mz_mono_fix_lockmass" in testq.columns:
        x = "mz_mono_fix_lockmass"
    else:
        x = "mz_mono"

    hq_dataframe = testq[(testq["cluster_corr"] > cluster_correlation) &
                         (testq["ab_cluster_total"] > (intensity_threshold))]

    sum_data = []
    for c in range(0, max(hq_dataframe["cluster"]) + 1):

        cluster_df = hq_dataframe[hq_dataframe["cluster"] == c]

        if (len(cluster_df) > 0):

            charge = np.median(cluster_df["charge"])
            mz = np.average(
                cluster_df[x],
                weights=cluster_df["ab_cluster_total"]
            )
            RT = np.average(cluster_df["RT"],
                            weights=cluster_df["ab_cluster_total"])
            im = np.average(cluster_df["im_mono"],
                            weights=cluster_df["ab_cluster_total"])

            near = getnear(mz, allseq, charge=charge, ppm=ppm)

            if len(near) == 1:
                sum_data.append([
                    near["name"].values[0],
                    RT,
                    im,
                    sum(cluster_df["ab_cluster_total"]),
                    near["MW"].values[0],
                    charge,
                    near["plus%s" % int(charge)].values[0],
                    mz,
                    near["ppm"].values[0],
                    near["abs_ppm"].values[0],
                    c,
                ])

    sum_df = pd.DataFrame(sum_data)
    sum_df.columns = [
        "name",
        "RT",
        "im_mono",
        "ab_cluster_total",
        "MW",
        "charge",
        "expect_mz",
        "obs_mz",
        "ppm",
        "abs_ppm",
        "cluster",
    ]
    sum_df["ppm"] = [float(x) for x in sum_df["ppm"]]
    return sum_df


def gen_mz_error_calib_output(
        testq,
        allseq,
        calib_pk_fpath=None,
        polyfit_degree=1,
        ppm_tol=50,
        int_tol=1e4,
        cluster_corr_tol=0.99,
):
    """Generate calibration using the dataframe from imtbx.

    Args:
        testq: dataframe from imtbx
        calib_pk_fpath: pickle filepath to save calibration information
        polyfit_degree: polyfit degree
        ppm_tol: ppm tolerance for selecting mz signals for calibration
        int_tol: intensity tolerance for selecting mz signals for calibration
        cluster_corr_tol: cluster correlation tolerance for selecting mz signals for calibration

    Returns:
        calib_dict (dict): dictionary containing the dataset used for calibration, polyfit, and ppm error before and after calibration

    """

    # generate high quality cluster mz signals
    cluster_hq_df = cluster_df_hq_signals(
        testq=testq,
        allseq=allseq,
        ppm=ppm_tol,
        intensity_threshold=int_tol,
        cluster_correlation=cluster_corr_tol,
    )

    # generate calibration dictionary
    calib_dict = gen_mz_ppm_error_calib_polyfit(
        obs_mz=cluster_hq_df["obs_mz"].values,
        thr_mz=cluster_hq_df["expect_mz"].values,
        polyfit_deg=polyfit_degree,
    )

    # save calibration dictionary for further use
    if calib_pk_fpath is not None:
        save_pickle_object(calib_dict, calib_pk_fpath)

    return calib_dict


def find_offset(sum_df):
    """Returns suspected systemic ppm error and width of poi peak of run data from sum_df.

    Assumes protein of interest within +/- 50ppm of 0ppm, selects closest peak to 0 if sufficiently prominent.

    Args:
        sum_df (Pandas DataFrame): DF containing all charged species being considered

    Returns:
        offset (float): simple linear adjustment which best corrects centering of ppm gaussian around 0
        offset_peak_width (float): full-width-half-max of the ppm distribution

    """
    # Maybe make this save the distplot too.
    ppm_dist = sns.distplot(sum_df["ppm"].values).get_lines()[0].get_data()
    peaks = sp.signal.find_peaks(ppm_dist[1])[0]
    xs, ys = ppm_dist[0][peaks], ppm_dist[1][peaks]
    # If lowest ppm peak is also highest frequency within window, we hope this will be the common case in our 50 ppm initial window
    try:
        xs[np.argmin(abs(xs))] == ys[np.argmax(ys)]
    except:
        print("Error fiding offset")
        sys.exit()
    if xs[np.argmin(abs(xs))] == ys[np.argmax(ys)]:
        return xs[np.argmin(abs(xs))]
        # Lowest ppm peak is not most prominent, determine relative height of lowest ppm peak
    else:
        # If peak closer to zero is less than half the height of the more prominent peak, check larger peak"s ppm
        if ys[np.argmin(abs(xs))] < ys[np.argmax(ys)] / 2:
            # If most prominent peak is heuristically close to 0, or lowest ppm peak is relatively very small (10% of major peak): use big peak
            if (xs[np.argmax(ys)] < 25 or
                    ys[np.argmin(abs(xs))] < ys[np.argmax(ys)] / 10):
                offset = xs[np.argmax(ys)]
            else:
                offset = xs[np.argmin(abs(xs))]
        else:
            offset = xs[np.argmin(abs(xs))]

    # Having selected our offset peak, determine its 80%-max width to construct a gaussian which will give the width of our final ppm filter
    peak_widths = sp.signal.peak_widths(ppm_dist[1], peaks, rel_height=0.8)
    # This line returns the rounded value of the 80%-max width found by matching the offset to its position in xs, and feeding that index position into peaks - a list of indices, returning the peaks-list index of the xs index. The peaks index corresponds to the peak-widths index, returning the width.
    offset_peak_width = np.round(
        np.asarray(peak_widths)[
            0, list(peaks).index(list(peaks)[list(xs).index(offset)])])
    return (offset, offset_peak_width)


def gen_mz_ppm_error_calib_polyfit(obs_mz, thr_mz, polyfit_deg=1):
    """Use polyfit to generate a function to correlate observed and theoretical mz values. The function is used as calibration
    for the mz values. User can specify the degree of the polyfit.

    Args:
        obs_mz (list): observed mz values
        thr_mz (list): theoretical mz values
        polyfit_deg (int): degree for polynomial fit

    Returns:
        cal_dict (dict): dictionary containing the dataset used for calibration, polyfit, and ppm error before and after calibration

    """
    polyfit_coeffs = np.polyfit(x=obs_mz, y=thr_mz, deg=polyfit_deg)
    obs_mz_corr = apply_polyfit_cal_mz(polyfit_coeffs, obs_mz)
    ppm_error_before_corr = calc_mz_ppm_error(obs_mz, thr_mz)
    ppm_error_after_corr = calc_mz_ppm_error(obs_mz_corr, thr_mz)

    cal_dict = gen_calib_dict(
        polyfit_bool=True,
        thr_mz=thr_mz,
        obs_mz=obs_mz,
        polyfit_coeffs=polyfit_coeffs,
        polyfit_deg=polyfit_deg,
        obs_mz_corr=obs_mz_corr,
        ppm_error_before_corr=ppm_error_before_corr,
        ppm_error_after_corr=ppm_error_after_corr,
    )

    return cal_dict


def apply_polyfit_cal_mz(polyfit_coeffs, mz):
    """Apply polyfit coeff to transform the mz values.

    Args:
        polyfit_coeffs (list): polyfit coefficients
        mz (Numpy ndarray): mz values
    Returns:
        mz_corr (Numpy ndarray): transformed mz values

    """
    mz_corr = np.polyval(polyfit_coeffs, mz)
    return mz_corr


def calc_mz_ppm_error(obs_mz, thr_mz):
    """Calculate mz ppm error.

    Args:
        obs_mz (float): observed mz value for a signal
        thr_mz (float): theoreteical or expected mz value based on chemical composition

    Returns:
        ppm_err (float): ppm error between observed and theoretical m/Z

    """
    ppm_err = 1e6 * (obs_mz - thr_mz) / thr_mz
    return ppm_err


def gen_calib_dict(polyfit_bool=False, **args):
    """Generate calibration dictionary with keywords.

    Args:
        polyfit_bool (bool): flag to perform polyfit calibration
        args (key-value pairs): each additional argument creates key-value pair in the output dict, usually includes:
            thr_mz,
            obs_mz,
            polyfit_coeffs,
            polyfit_deg
            obs_mz_corr,
            ppm_error_before_corr,
            ppm_error_after_corr

    calib_dict (dict): calibration dictionary containing relevant calibration parameters and outputs

    """
    calib_dict = dict()

    calib_dict["polyfit_bool"] = polyfit_bool

    if polyfit_bool:
        for param, value in args.items():
            calib_dict[param] = value

    return calib_dict


def save_pickle_object(obj, fpath):
    """Wrapper function to output an object as a pickle.

    Args:
        obj (Any Python object): Any python object to pickle
        fpath (string): path/to/output.pickle

    Returns:
        None

    """
    with open(fpath, "wb") as outfile:
        pk.dump(obj, outfile)


def pmem(id_str):
    """Prints memory in use by process with a passed debug label.

    Args:
        id_str (str): String to prepend to memory output, for identifying origin of pmem call.

    Returns:
        None

    """
    process = psutil.Process(os.getpid())
    print(id_str + " Process Memory (GB): " +
          str(process.memory_info().rss / 1024 / 1024 / 1024))


def cal_area_under_curve_from_normal_distribution(low_bound, upper_bound, center, width):
    """Computes the cumulative distribution function for a gaussian with given center, bounds, and width.

    Args:
        low_bound (float): Lower or left bound on gaussian, unitless.
        upper_bound (float):  Upper or right bound on gaussian, unitless.
        center (float): Center of gaussian, unitless.
        width (float): Width of gaussian, unitless.

    Returns:
        auc (float): Area Under the Curve, computed cumulative distribution of specified gaussian.

    """
    lb_cdf = norm.cdf(low_bound, loc=center, scale=width)
    ub_cdf = norm.cdf(upper_bound, loc=center, scale=width)
    auc = ub_cdf - lb_cdf
    return auc


def estimate_gauss_param(y_data, x_data):
    """Estimates the parameters of a gaussian fit to a set of datapoints with x,y coordinates in x_data and y_data.

    Args:
        y_data (iterable of floats): Y values of data to fit.
        x_data (iterable of floats): X values of data to fit.

    Returns:
        init_guess (list of floats): baseline offset, amplitude, center, width.

    """
    ymax = np.max(y_data)
    maxindex = np.nonzero(y_data == ymax)[0]
    peakmax_x = x_data[maxindex][0]
    norm_arr = y_data/max(y_data)
    bins_for_width = norm_arr[norm_arr > 0.70]
    width_bin = len(bins_for_width)
    init_guess = [0, ymax, peakmax_x, width_bin]
    # bounds = ([0, 0, 0, 0], [np.inf, np.inf, len(x_data)-1, len(x_data)-1])
    return init_guess


def gauss_func(x, y0, A, xc, w):
    """Model Gaussian function to pass to scipy.optimize.curve_fit.

    Args:
        x (list of floats): X dimension values.
        y0 (float): Offset of y-values from 0.
        A (float): Amplitude of Gaussian.
        xc (float): Center of Gaussian in x dimension.
        w (float): Width or sigma of Gaussian.

    Returns:
        y (list of floats): Y-values of Gaussian function evaluated over x.

    """
    rxc = ((x - xc) ** 2) / (2 * (w ** 2))
    y = y0 + A * (np.exp(-rxc))
    return y


def adjrsquared(r2, param, num):
    """Calculates the adjusted R^2 for a parametric fit to data.

    Args:
         r2 (float): R^2 or "coefficient of determination" of a linear regression between fitted and observed values.
         param (int): Number of parameters in fitting model.
         num (int): Number of datapoints in sample.

    Returns:
        y (float): Adjusted R^2 <= R^2, increases when additional parameters improve fit more than could be expected by chance.

    """
    y = 1 - (((1 - r2) * (num - 1)) / (num - param - 1))
    return y


def fit_gaussian(x_data, y_data, data_label="dt"):
    """Performs fitting of a gaussian function to a provided data sample and computes linear regression on residuals to measure quality of fit.

    Args:
        x_data (list of floats): X dimension values for sample data.
        y_data (list of floats): Y dimension values for sample data.
        data_label (str): Label indicating origin of data in multidimensional separation (rt, dt, m/z).

    Returns:
        gauss_fit_dict (dict): Contains the following key-value pairs describing the Guassian and linear regression parameters.
            "gauss_fit_success" (bool): Boolean indicating the success (True) or failure (False) of the fitting operation.
            "y_baseline" (float): Fitted parameter for the Gaussian function"s offset from y=0.
            "y_amp" (float): Fitted parameter for the amplitude of the Gaussian function.
            "xc" (float): Fitted parameter for the center of the Gaussian in the x dimension.
            "width" (float): Fitted parameter for the x dimensional width of the Gaussian function.
            "y_fit" (list of floats): Y values of fitted Gaussian function evaluated over x.
            "fit_rmse" (float): Root-mean-square error, the standard deviation of the residuals between the fit and sample.
            "fit_lingress_slope" (float): The slope of the linear regression line over the residuals between fit and sample.
            "fit_lingress_intercept" (float): The intercept point of the line fit to the residuals.
            "fit_lingress_pvalue" (float): The p-value for a hypothesis test whose null hypothesis is that the above slope is zero
            "fit_lingress_stderr" (float): Standard error of the estimated slope under the assumption of residual normality.
            "fit_lingress_r2" (float): R^2 or "coeffiecient of determination" of linear regression over residuals.
            "fit_lingress_adj_r2" (float): Adjusted R^2, always <= R^2, decreases with extraneous parameters.
            "auc" (float): Area under the curve, cumulative distribution function of fitted gaussian evaluated over the length of x_data.

    """
    init_guess = estimate_gauss_param(y_data, x_data)
    gauss_fit_dict = dict()
    gauss_fit_dict["data_label"] = data_label
    gauss_fit_dict["gauss_fit_success"] = False
    gauss_fit_dict["xc"] = center_of_mass(y_data)[0]
    gauss_fit_dict["auc"] = 1.0
    gauss_fit_dict["fit_rmse"] = 100.0
    gauss_fit_dict["fit_linregress_r2"] = 0.0
    gauss_fit_dict["fit_lingress_adj_r2"] = 0.0

    try:
        popt, pcov = curve_fit(gauss_func, x_data, y_data, p0=init_guess, maxfev=100000)
        if popt[2] < 0:
            return gauss_fit_dict
        if popt[3] < 0:
            return gauss_fit_dict
        else:
            y_fit = gauss_func(x_data, *popt)
            fit_rmse = mean_squared_error(y_data/max(y_data), y_fit/max(y_fit), squared=False)
            slope, intercept, rvalue, pvalue, stderr = linregress(y_data, y_fit)
            adj_r2 = adjrsquared(r2=rvalue**2, param=4, num=len(y_data))
            gauss_fit_dict["gauss_fit_success"] = True
            gauss_fit_dict["y_baseline"] = popt[0]
            gauss_fit_dict["y_amp"] = popt[1]
            gauss_fit_dict["xc"] = popt[2]
            gauss_fit_dict["width"] = popt[3]
            gauss_fit_dict["y_fit"] = y_fit
            gauss_fit_dict["fit_rmse"] = fit_rmse
            gauss_fit_dict["fit_lingress_slope"] = slope
            gauss_fit_dict["fit_lingress_intercept"] = intercept
            gauss_fit_dict["fit_lingress_pvalue"] = pvalue
            gauss_fit_dict["fit_lingress_stderr"] = stderr
            gauss_fit_dict["fit_linregress_r2"] = rvalue ** 2
            gauss_fit_dict["fit_lingress_adj_r2"] = adj_r2
            gauss_fit_dict["auc"] = cal_area_under_curve_from_normal_distribution(low_bound=x_data[0],
                                                                                  upper_bound=x_data[-1],
                                                                                  center=popt[2],
                                                                                  width=popt[3])
            return gauss_fit_dict
    except:
        return gauss_fit_dict


def model_data_with_gauss(x_data, gauss_params):
    # TODO: Add docstring.

    data_length = len(x_data)
    bin_value = (x_data[-1] - x_data[0])/data_length
    center = gauss_params[2]
    data_length_half = int(data_length/2)
    low_val = center - (data_length_half * bin_value)
    new_x_data = []
    for num in range(data_length):
        val = low_val + (num * bin_value)
        new_x_data.append(val)
    new_x_data = np.array(new_x_data)
    new_gauss_data = gauss_func(new_x_data, *gauss_params)
    return new_gauss_data


@dataclass
class NNFACDATA(object):

    """
    dataclass to store nnfac data
    """

    factor_rank: int = None
    init_method: str = None
    max_iteration: int = None
    tolerance: float = None
    sparsity_coefficients: list = None
    fixed_modes: list = None
    normalize: list = None
    factors: list = None
    n_iter: int = None
    rec_errors: np.ndarray = None
    converge: bool = None


def factor_correlations(factors):
    """
    check correlactions between factors
    Args:
        factors: factors output from nnfac

    Returns: max correlation between factors

    """

    corrcoef_1 = np.corrcoef(factors[0].T)
    corrcoef_2 = np.corrcoef(factors[1].T)
    corrcoef_3 = np.corrcoef(factors[2].T)

    min_corr_mat = np.minimum(np.minimum(corrcoef_1, corrcoef_2), corrcoef_3)

    max_corr = np.max(min_corr_mat[np.where(~np.eye(min_corr_mat.shape[0], dtype=bool))])

    return max_corr


def factorize_tensor(input_grid,
                     init_method="random",
                     factors_0=[],
                     factor_rank=15,
                     n_iter_max=100000,
                     tolerance=1e-8,
                     sparsity_coefficients=[],
                     fixed_modes=[],
                     normalize=[],
                     verbose=True,
                     return_errors=True):
    """

    Args:
        input_grid: input grid for factorization
        init_method: initialization method: "random", "nndsvd", or "custom"
        factors_0: factors to iniitialize with
        factor_rank: factor rank
        n_iter_max: max iteration
        tolerance: tolerance criteria for convergence
        sparsity_coefficients: sparsity coeffs
        fixed_modes: fixing modes for W and H
        normalize: normalize boolean list
        verbose: True for printing out nnfac operation
        return_errors: True for return errors, toc, convergence etc

    Returns: nnfacdata object

    """

    # initialize the datacalss
    nnfac_output = NNFACDATA(factor_rank=factor_rank,
                             init_method=init_method,
                             max_iteration=n_iter_max,
                             tolerance=tolerance,
                             sparsity_coefficients=sparsity_coefficients,
                             fixed_modes=fixed_modes,
                             normalize=normalize)

    # factorize
    factor_out = nn_fac.ntf.ntf(tensor=input_grid,
                                rank=factor_rank,
                                init=init_method,
                                factors_0=factors_0,
                                n_iter_max=n_iter_max,
                                tol=tolerance,
                                sparsity_coefficients=sparsity_coefficients,
                                fixed_modes=fixed_modes,
                                normalize=normalize,
                                verbose=verbose,
                                return_costs=return_errors)

    # store relevant data

    if return_errors:

        nnfac_output.factors = factor_out[0]
        nnfac_output.rec_errors = factor_out[1]
        nnfac_output.n_iter = len(factor_out[1])
        nnfac_output.converge = False
        if abs(factor_out[1][-2] - factor_out[1][-1]) < tolerance:
            nnfac_output.converge = True

    else:
        nnfac_output.factors = factor_out

    return nnfac_output


def gen_factors_with_corr_check(input_grid,
                                init_method="random",
                                factors_0=[],
                                max_num_factors=15,
                                n_iter_max=100000,
                                tolerance=1e-8,
                                sparsity_coefficients=[],
                                fixed_modes=[],
                                normalize=[],
                                verbose=False,
                                return_errors=True,
                                corr_threshold=0.17):
    """
    generate factors with reducing the rank while checking factor correlations
    Args:
        input_grid: input grid for factorization
        init_method: initialization method: "random", "nndsvd", or "custom"
        factors_0: factors to iniitialize with
        max_num_factors: max number of factor rank
        n_iter_max: max iteration
        tolerance: tolerance criteria for rec error for convergence
        sparsity_coefficients: sparsity coeffs
        fixed_modes: fixing modes for W and H
        normalize: normalize boolean list for modes (l2 normalization)
        verbose: True for printing out nnfac operation
        return_errors: True for return errors, toc, convergence etc
        corr_threshold: factor correlation threshold. factors have to have a correlation smaller than this value

    Returns:

    """

    last_corr_check = 1.0
    max_num_factors += 1

    factor_output = None

    while max_num_factors > 2 and last_corr_check > corr_threshold:

        max_num_factors -= 1

        pmem("Factorize: %s # Factors (Start)" % max_num_factors)

        factor_output = factorize_tensor(input_grid=input_grid,
                                         init_method=init_method,
                                         factors_0=factors_0,
                                         factor_rank=max_num_factors,
                                         n_iter_max=n_iter_max,
                                         tolerance=tolerance,
                                         sparsity_coefficients=sparsity_coefficients,
                                         fixed_modes=fixed_modes,
                                         normalize=normalize,
                                         verbose=verbose,
                                         return_errors=return_errors)

        pmem("Factorize: %s # Factors (End)" % max_num_factors)

        if max_num_factors > 1:
            last_corr_check = factor_correlations(factor_output.factors)

    return factor_output


def calculate_theoretical_isotope_dist_from_sequence(sequence, n_isotopes=None):
    """Calculate theoretical isotope distribtuion from the given one-letter sequence of a library protein.

    Args:
        sequence (string): sequence in one letter code
        n_isotopes (int): number of isotopes to include. If none, includes all

    Return:
        isotope_dist (numpy ndarray): resulting theoretical isotope distribution

    """
    seq_formula = molmass.Formula(sequence)
    isotope_dist = np.array([x[1] for x in seq_formula.spectrum().values()])
    isotope_dist = isotope_dist / max(isotope_dist)
    if n_isotopes:
        if n_isotopes < len(isotope_dist):
            isotope_dist = isotope_dist[:n_isotopes]
        else:
            fill_arr = np.zeros(n_isotopes - len(isotope_dist))
            isotope_dist = np.append(isotope_dist, fill_arr)
    return isotope_dist

def calculate_empirical_isotope_dist_from_integrated_mz(integrated_mz_array,
                                                        n_isotopes=None):
    """Calculate the isotope distribution from the integrated mz intensitities.

    Args:
        integrated_mz_values (Numpy ndarray): array of integrated mz intensitites
        n_isotopes (int): number of isotopes to include. If none, includes all
    Returns:
        isotope_dist (Numpy ndarray): isotope distribution with magnitude normalized to 1

    """
    isotope_dist = integrated_mz_array / max(integrated_mz_array)
    if n_isotopes:
        isotope_dist = isotope_dist[:n_isotopes]
    return isotope_dist


def calculate_isotope_dist_dot_product(sequence, undeut_integrated_mz_array):
    """Calculate dot product between theoretical isotope distribution from the sequence and experimental integrated mz array.

    Args:
        sequence (string): single-letter sequence of the library protein-of-interest
        undeut_integrated_mz_array (Numpy ndarray): observed integrated mz array from an undeuterated .mzML
    Returns:
        dot_product (float): result of dot product between theoretical and observed integrated-m/Z, from [0-1]

    """
    theo_isotope_dist = calculate_theoretical_isotope_dist_from_sequence(
        sequence=sequence)
    emp_isotope_dist = calculate_empirical_isotope_dist_from_integrated_mz(
        integrated_mz_array=undeut_integrated_mz_array)
    min_length = min([len(theo_isotope_dist), len(emp_isotope_dist)])
    dot_product = np.linalg.norm(
        np.dot(theo_isotope_dist[0:min_length], emp_isotope_dist[0:min_length])
    ) / np.linalg.norm(theo_isotope_dist) / np.linalg.norm(emp_isotope_dist)
    return dot_product


class Preprocessing:

    def __init__(self):

        self.info = "Preprocessing class. Subclasses defined to handle raw data or intermediate data and return protein identification dataframe"

    class IMTBX:

        def __init__(self
                     ):

            self.imtbx_df = None
            self.precorrection_df = None
            self.post_lockmass_df = None
            self.post_protein_polyfit_df = None
            self.offset_df = None
            self.configfile = None
            self.calib_dict_lockmass = None
            self.calib_dict_protein_polyfit = None
            self.for_kde_df = None

        def load_imtbx(self,
                       configfile_path,
                       names_and_seqs_path,
                       isotopes_path,
                       lockmass_calibration_dict=None,
                       protein_polyfit_output=None,
                       output_path=None):

            # Load config file
            # Load isotopes file
            # Load names and sequences

            self.configfile = yaml.load(open(configfile_path, "rb").read(), Loader=yaml.Loader)

            df_imtbx = load_isotopes_file(isotopes_path)
            allseq = load_names_and_seqs(names_and_seqs_path)

            # Make buffer of df
            testq = copy.deepcopy(df_imtbx)

            # Cluster lines
            apply_cluster_weights(testq, adjusted=False)
            cluster_lines(testq)

            self.precorrection_df = cluster_df(testq, allseq, ppm=50, adjusted=False)

            # Apply lockmass calibration
            if lockmass_calibration_dict is not None:

                print("Running lockmass_calibration...")

                runtime = self.configfile["runtime"]
                self.calib_dict_lockmass = load_pickle_file(lockmass_calibration_dict)
                testq["mz_mono_fix_lockmass"] = 0
                if self.calib_dict_lockmass[0]["polyfit_deg"] == 0:
                    delta = int(runtime / len(self.calib_dict_lockmass))
                    for i, rt in enumerate(range(0, runtime, delta)):
                        testq.loc[(testq["RT"] >= rt) & (testq["RT"] <= rt + delta), "mz_mono_fix_lockmass"] = \
                            self.calib_dict_lockmass[i]["polyfit_coeffs"] * testq[(testq["RT"] >= rt) &
                                                                                  (testq["RT"] <= rt + delta)][
                                "mz_mono"].values
                else:
                    delta = int(runtime / len(self.calib_dict_lockmass))
                    for i, rt in enumerate(range(0, runtime, delta)):
                        testq.loc[
                            (testq["RT"] >= rt) & (testq["RT"] <= rt + delta), "mz_mono_fix_lockmass"] = np.polyval(
                            self.calib_dict_lockmass[i]["polyfit_coeffs"], testq[(testq["RT"] >= rt) &
                                                                                 (testq["RT"] <= rt + delta)][
                                "mz_mono"].values)
                testq["mz_mono_fix_round_lockmass"] = np.round(testq["mz_mono_fix_lockmass"].values, 3)

                testq["mz_mono_fix"] = testq["mz_mono_fix_lockmass"]
                testq["mz_mono_fix_round"] = testq["mz_mono_fix_round_lockmass"]

                self.post_lockmass_df = cluster_df(testq, allseq, ppm=50, adjusted=True)

                # Cluster lines
                apply_cluster_weights(testq, adjusted=True)
                cluster_lines(testq)

                # Apply protein polyfit calibration
            if self.configfile["protein_polyfit"]:

                print("Running protein_polyfit...")

                if protein_polyfit_output is None:
                    print("Missing protein_calibration_outpath")
                    sys.exit()

                self.calib_dict_protein_polyfit = gen_mz_error_calib_output(
                    testq=testq,
                    allseq=allseq[~allseq["name"].str.contains("decoy")],
                    calib_pk_fpath=protein_polyfit_output,
                    polyfit_degree=self.configfile["polyfit_deg"],
                    ppm_tol=self.configfile["ppm_tolerance"],
                    int_tol=self.configfile["intensity_tolerance"],
                    cluster_corr_tol=self.configfile["cluster_corr_tolerance"]
                )

                if "mz_mono_fix_lockmass" in testq.columns:
                    x = "mz_mono_fix_lockmass"
                else:
                    x = "mz_mono"

                testq["mz_mono_fix_protein_polyfit"] = apply_polyfit_cal_mz(
                    polyfit_coeffs=self.calib_dict_protein_polyfit["polyfit_coeffs"], mz=testq[x])
                testq["mz_mono_fix_round_protein_polyfit"] = np.round(testq["mz_mono_fix_protein_polyfit"].values, 3)

                testq["mz_mono_fix"] = testq["mz_mono_fix_protein_polyfit"]
                testq["mz_mono_fix_round"] = testq["mz_mono_fix_round_protein_polyfit"]

                self.post_protein_polyfit_df = cluster_df(testq, allseq, ppm=50, adjusted=True)

            if (not self.configfile["lockmass"]) and (not self.configfile["protein_polyfit"]):

                offset, offset_peak_width = find_offset(self.precorrection_df)
                if offset > 0:
                    testq["mz_mono_fix_offset"] = [
                        x * (1000000 - offset) / (1000000) for x in testq["mz_mono"]
                    ]
                    testq["mz_mono_fix_offset_round"] = np.round(testq["mz_mono_fix_offset"].values,
                                                          3)
                else:
                    testq["mz_mono_fix_offset"] = [
                        x * (1000000 + offset) / (1000000) for x in testq["mz_mono"]
                    ]
                    testq["mz_mono_fix_offset_round"] = np.round(testq["mz_mono_fix_offset"].values,
                                                          3)

                testq["mz_mono_fix"] = testq["mz_mono_fix_offset"]
                testq["mz_mono_fix_round"] = testq["mz_mono_fix_offset"]

                self.offset_df = cluster_df(testq, allseq, ppm=50, adjusted=True)

                # ppm_refilter = math.ceil(offset_peak_width / 2)

            # Cluster lines
            apply_cluster_weights(testq, adjusted=True)
            cluster_lines(testq)

            self.for_kde_df = cluster_df(testq, allseq, ppm=20, adjusted=True)

            self.imtbx_df = cluster_df(testq, allseq, ppm=self.configfile["ppm_refilter"], adjusted=True)

            # send sum_df to main output
            if output_path is not None:
                self.imtbx_df.to_csv(output_path, index=False)

        def plot_kde(self,
                     dpi=300,
                     output_path=None):

            if self.configfile is None:
                print("Configfile not present. Exiting...")
                sys.exit()

            sns.set_context("talk", font_scale=0.8)

            fig, ax = plt.subplots(2, 3, figsize=(15, 6), dpi=200)

            if self.precorrection_df is not None:
                sns.kdeplot(self.precorrection_df[~self.precorrection_df.name.str.contains("decoy")].ppm, ax=ax[0][0],
                            c="blue", label="no correction")
                sns.kdeplot(self.precorrection_df[self.precorrection_df.name.str.contains("decoy")].ppm, ax=ax[0][1],
                            c="blue", label="no correction")
                sns.kdeplot(self.precorrection_df.ppm, ax=ax[0][2], c="blue", label="no correction")
            if self.post_lockmass_df is not None:
                sns.kdeplot(self.post_lockmass_df[~self.post_lockmass_df.name.str.contains("decoy")].ppm, ax=ax[0][0],
                            c="orange", label="lockmass")
                sns.kdeplot(self.post_lockmass_df[self.post_lockmass_df.name.str.contains("decoy")].ppm, ax=ax[0][1],
                            c="orange", label="lockmass")
                sns.kdeplot(self.post_lockmass_df.ppm, ax=ax[0][2], c="orange", label="lockmass")
            if self.post_protein_polyfit_df is not None:
                sns.kdeplot(self.post_protein_polyfit_df[~self.post_protein_polyfit_df.name.str.contains("decoy")].ppm,
                            ax=ax[0][0], c="green", label="protein polyfit")
                sns.kdeplot(self.post_protein_polyfit_df[self.post_protein_polyfit_df.name.str.contains("decoy")].ppm,
                            ax=ax[0][1], c="green", label="protein polyfit")
                sns.kdeplot(self.post_protein_polyfit_df.ppm, ax=ax[0][2], c="green", label="protein polyfit")
            if self.offset_df is not None:
                sns.kdeplot(self.offset_df[~self.offset_df.name.str.contains("decoy")].ppm,
                            ax=ax[0][0], c="green", label="offset correction")
                sns.kdeplot(self.offset_df[self.offset_df.name.str.contains("decoy")].ppm,
                            ax=ax[0][1], c="green", label="offset correction")
                sns.kdeplot(self.offset_df.ppm, ax=ax[0][2], c="green", label="offset correction")

            for x in ax[0]:
                x.axvline(0, c="red", ls="--")

            ax[0][0].legend(prop={"size": 12}, frameon=False, bbox_to_anchor=(-0.25, 1))

            ax[0][0].text(0, 1.05, "No decoys", transform=ax[0][0].transAxes)
            ax[0][1].text(0, 1.05, "Only decoys", transform=ax[0][1].transAxes)
            ax[0][2].text(0, 1.05, "Everything", transform=ax[0][2].transAxes)

            sns.kdeplot(self.imtbx_df[~self.imtbx_df.name.str.contains("decoy")].ppm, c="black", label="No decoys",
                        ls="-", ax=ax[1][0])
            sns.kdeplot(self.imtbx_df[self.imtbx_df.name.str.contains("decoy")].ppm, c="black", label="Only decoys",
                        ls="--", ax=ax[1][0])
            ax[1][0].legend(prop={"size": 12}, frameon=False, bbox_to_anchor=(-0.25, 1))
            ax[1][0].text(0, 1.05, "Filtered", transform=ax[1][0].transAxes)

            decoy_level = self.configfile["decoy_level"]

            xs, ys1, ys2 = [], [], []
            for ppm in np.arange(1, 21, 1):
                mask = abs(self.for_kde_df.ppm) < ppm
                xs.append(ppm)
                ys1.append(
                    len(set(self.for_kde_df[(self.for_kde_df.name.str.contains("decoy")) & mask].name)) / decoy_level)
                ys2.append(len(set(self.for_kde_df[(~self.for_kde_df.name.str.contains("decoy")) & mask].name)))

            ys1 = np.array(ys1)
            ys2 = np.array(ys2)

            ax[1][1].scatter(xs, ys1 / ys2, c="blue", edgecolors="black", s=50)
            ax[1][1].set_xlim(20, 0)
            ax[1][1].axvline(self.configfile["ppm_refilter"], c="red", ls="--")
            ax[1][1].set_xlabel("ppm threshold")
            ax[1][1].set_ylabel("FDR")

            ax[1][2].scatter(xs, ys2, c="blue", edgecolors="black", s=50)
            ax[1][2].set_xlim(20, 0)
            ax[1][2].axvline(self.configfile["ppm_refilter"], c="red", ls="--")
            ax[1][2].set_xlabel("ppm threshold")
            ax[1][2].set_ylabel("# identifications")

            plt.tight_layout()

            if output_path is not None:
                plt.savefig(output_path, dpi=dpi, format="pdf", bbox_inches="tight")
                plt.close()
            else:
                plt.show()


class DataTensor:
    """A container for LC-IMS-MS data that includes smoothing and factorization methods.

    Attributes:
        source_file (str): Path of DataTensor"s parent resources/tensors/.cpickle.zlib file.
        tensor_idx (int): Index of the DataTensor in a concatenated DataTensor (concatenation of tensors is deprecated, this value always 0).
        timepoint_idx (int): Index of the DataTensors HDX timepoint in config["timepoints"].
        name (str): Name of DataTensor"s rt-group.
        total_mass_window (int): Magnitude of DataTensor"s m/Z dimension in bins.
        n_concatenated (int): Number of DataTensors combined to make the instance DataTensor. Deprecated, value always 1.
        charge_states (list with one int): Net positive charge on protein represented in DataTensor, format left over from concatenation.
        integrated_mz_limits (numpy 2D array): Values for low and high m/Z limits of integration around expected peak centers.
        bins_per_isotope_peak (int): Number of integrated m/Z bins for a signal to be considered as an IsotopeCluster.
        normalization_factor (float): Divisor for the integrated m/Z intensity of any IsotopeCluster from a parent DataTensor instance. 
        rts (numpy array): Intensity summed over each rt bin.
        dts (numpy array): Intensity summed over each dt bin.
        seq_out (numpy array): Intensity summed over each m/Z bin in a flat array.
        int_seq_out (numpy array): Intensities integrated over all dimensions in a flat array. (TODO: Review, not sure about this)
        int_seq_out_float (numpy array of float64):  Intensities integrated over all dimensions in a flat array, cast as float64.
        int_grid_out (numpy array): int_seq_out reshaped to a 3D array by rt, dt, and m/Z dimension magnitudes.
        int_gauss_grids (numpy array): int_grid_out after applying gaussian smoothing.
        concat_dt_idxs (list of ints): Deprecated - Indices marking the boundaries between different DataTensors" concatenated DT dimensions.
        concatenated_grid (numpy array): Deprecated - int_grid_out from multiple DataTensors concatenated along the dt axis.
        retention_labels (list of floats): Mapping of DataTensor"s RT bins to corresponding absolute retention time in minutes.
        drift_labels (list of floats): Mapping of DataTensor"s DT bins to corresponding absolute drift time in miliseconds.
        mz_labels (list of floats): Mapping of DataTensor"s m/Z bins to corresponding m/Z.
        full_grid_out (numpy array): Reprofiled tensor with intensities integrated within bounds defined in integrated_mz_limits.
        full_gauss_grids (numpy array): full_grid_out after the application of gaussian smoothing to the RT and DT dimensions.
        factors (list of Factor objects): List of Factor objects resulting from the factorize method.

    """

    def __init__(self, source_file, tensor_idx, timepoint_idx, name,
                 total_mass_window, n_concatenated, charge_states, integrated_mz_limits, bins_per_isotope_peak,
                 normalization_factor, **kwargs):
        """Initializes an instance of the DataTensor class from 

        Args:
            source_file (str): Path of DataTensor"s parent resources/tensors/.cpickle.zlib file.
            tensor_idx (int): Deprecated - Index of this tensor in a concatenated tensor, now always 0.
            timepoint_idx (int): Index of tensor"s source timepoint in config["timepoints"].
            name (str): Name of rt-group DataTensor is a member of.
            total_mass_window (int): Magnitude of DataTensor"s m/Z dimension in bins.
            n_concatenated (int): Number of DataTensors combined to make the instance DataTensor. Deprecated value always 1. 
            charge_states (list with one int): Net positive charge on protein represented in DataTensor, format left over from concatenation.
            integrated_mz_limits (numpy 2D array): Values for low and high m/Z limits of integration around expected peak centers.
            bins_per_isotope_peak (int):  Number of integrated m/Z bins for a signal to be considered as an IsotopeCluster.
            normalization_factor (float): Divisor for the integrated m/Z intensity of any IsotopeCluster from a parent DataTensor instance. 

        Keyword Arguments: 
            rts (numpy array): Intensity summed over each rt bin, from source tensor file.
            dts (numpy array): Intensity summer over each dt bin, from source tensor file.
            seq_out (numpy array): Intensity summed over each m/Z bin in a flat array, from source tensor file.
            int_seq_out (numpy array): Intensities integrated over all dimensions in a flat array. (TODO: Review, not sure about this)
            concat_dt_idxs (list of ints): Deprecated - Indices marking the boundaries between different DataTensors" concatenated DT dimensions.
            concatenated_grid (numpy array): Deprecated - int_grid_out from multiple DataTensors concatenated along the dt axis.

        """
        self.source_file = source_file
        self.tensor_idx = tensor_idx
        self.timepoint_idx = timepoint_idx
        self.name = name
        self.total_mass_window = total_mass_window
        self.n_concatenated = n_concatenated
        self.charge_states = charge_states
        self.integrated_mz_limits = integrated_mz_limits
        self.bins_per_isotope_peak = bins_per_isotope_peak
        self.normalization_factor = normalization_factor
        self.factors = None

        if kwargs is not None:
            kws = list(kwargs.keys())
            if "rts" in kws:
                self.rts = np.array(kwargs["rts"])
            if "dts" in kws:
                self.dts = np.array(kwargs["dts"])
            if "seq_out" in kws:
                self.seq_out = np.array(kwargs["seq_out"], dtype=object)
            if "int_seq_out" in kws and kwargs["int_seq_out"] is not None:
                self.int_seq_out = np.array(kwargs["int_seq_out"])
                self.int_seq_out_float = self.int_seq_out.astype("float64")
                self.int_grid_out = np.reshape(
                    self.int_seq_out_float, (len(self.rts), len(self.dts), 50))
                self.int_gauss_grids = self.gauss(self.int_grid_out)
            if "concat_dt_idxs" in kws:
                self.concat_dt_idxs = kwargs["concat_dt_idxs"]
            if "concatenated_grid" in kws:
                self.concatenated_grid = kwargs["concatenated_grid"]

        # Normal case, single tensor.
        if self.n_concatenated == 1:

            (
                self.retention_labels,
                self.drift_labels,
                self.mz_labels,
                self.bins_per_isotope_peak,
                self.full_grid_out,
            ) = self.sparse_to_full_tensor_reprofile((self.rts, self.dts, self.seq_out),
                                                     self.integrated_mz_limits,
                                                     self.bins_per_isotope_peak)
            self.full_gauss_grids = self.gauss(self.full_grid_out)
            self.tensor_auc = (np.sum(self.full_grid_out) / self.normalization_factor)[0]
            self.tensor_gauss_auc = (np.sum(self.full_gauss_grids) / self.normalization_factor)[0]

        # Handle concatenated tensor case, check for required inputs TODO: Remove? Deprecated, but Gabe has said to keep this before, ask again.
        else:
            if not all("dts" in kwargs and "rts" in kwargs and
                       "lows" in kwargs and "highs" in kwargs and
                       "concatenated_grid" in kwargs and
                       "abs_mz_low" in kwargs and "concat_dt_idxs" in kwargs):

                print("Concatenated Tensor Missing Required Values")
                sys.exit()


    def sparse_to_full_tensor_reprofile(self, data, integrated_mz_limits, bins_per_isotope_peak = 7, ms_resolution=25000):
        """Takes the raw m/z data and turns it into a tensor of profiled data that can be factorized.
        
           The raw data is only profiled at the specific locations where isotopic peaks are expected. The locations of
           these peaks are defined by integrated_mz_limits, and each peak will be profiled at bins_per_isotope_peak
           number of points, with the first and last points equal to the integrated_mz_limits for that peak. 

        Args:
            data (tuple of 3 list-likes): Tuple containing the absolute retention times for the tensor, the absolute
                drift times for the tensor, and the raw data for all the scans that will go in the tensor.
            integrated_mz_limits (list of list-likes of floats): The upper and lower bounds on integration surrounding each
                m/Z-peak center, a list of list-likes of length 2.
            bins_per_isotope_peak (int): Number of points to use to profile each isotope peak.
            ms_resolution (int): Resolution to use in profiling the raw peaks. This determines the width of the gaussian
                used to profile each peak.

        Returns:
            retention_labels (list of floats): List of the absolute LC-RT values associated with each RT bin.
            drift_labels (list of floats):  List of the absolute IMS-DT values associated with each DT bin.
            mz_bin_centers (list of floats):  List of the absolute m/Z values associated with each m/Z bin.
            bins_per_isotope_peak (int): Number of points to use to profile each isotope peak.
            tensor3_out (numpy array): Full and reprofiled 3D tensor of LC-IMS-MS data.

        """
        retention_labels, drift_labels, sparse_data = data

        
        FWHM = np.average(integrated_mz_limits) / ms_resolution
        gaussian_scale = FWHM / 2.355 # A gaussian with scale = standard deviation = 1 has FWHM 2.355.
        
        mz_bin_centers = np.ravel([np.linspace(lowlim, highlim, bins_per_isotope_peak) for lowlim, highlim in integrated_mz_limits])
        tensor3_out = np.zeros((len(retention_labels), len(drift_labels), len(mz_bin_centers)))

        scan = 0
        for i in range(len(retention_labels)):
            for j in range(len(drift_labels)):
                n_peaks = len(sparse_data[scan])
                gaussians = norm(loc=sparse_data[scan][:,0],scale=gaussian_scale)
                resize_gridpoints = np.resize(mz_bin_centers, (n_peaks, len(mz_bin_centers) )).T
                eval_gaussians = gaussians.pdf(resize_gridpoints) * sparse_data[scan][:,1] * gaussian_scale

                tensor3_out[i][j] = np.sum(eval_gaussians,axis=1) 
                scan += 1
                
        return retention_labels, drift_labels, mz_bin_centers, bins_per_isotope_peak, tensor3_out


    def gauss(self, grid, rt_sig=3, dt_sig=1):
        """Applies a gaussian filter to the first two dimensions of a 3D tensor.

        Args:
            grid (numpy array): 3D tensor of LC-IMS-MS data, self.full_grid_out.
            rt_sig (int or float): Gaussian sigma for smoothing function on LC-RT dimension, default 3.
            dt_sig (int or float): Gaussian sigma for smoothing function on IMS-DT dimension, defualt 1.

        Returns:
            gauss_grid (type): Input tensor after applying gaussian smoothing.

        """

        gauss_grid = np.zeros(np.shape(grid))
        for i in range(np.shape(grid)[2]):
            gauss_grid[:, :, i] = gaussian_filter(grid[:, :, i],
                                                  (rt_sig, dt_sig))
        return gauss_grid

    # TODO: This isn"t great style, make this take the tensor as input and return the factors.
    def factorize(self,
                  max_num_factors=15,
                  init_method="nndsvd",
                  factors_0=[],
                  fixed_modes=[],
                  sparsity_coeffs=[],
                  normalize=[],
                  niter_max=100000,
                  tol=1e-8,
                  factor_corr_threshold=0.17):
        """Performs the non-negative PARAFAC on tensor, implemented by nn-fac python library, saves to self.Factors.

        Args:
            n_factors (int): The number of factors used to decompose the input tensor.
            new_mz_len (int): Number of bins desired in output tensor m/Z dimension, performs interpolation.
            gauss_params (tuple of 2 ints): Two values indicating the width of smoothing in LC-RT and IMS-DT dimensions respectively.

        Returns:
            None
        """
        # Test factorization starting at n_factors = 15 and counting down, keep factorization that has no factors with correlation greater than 0.2 in any dimension.

        t = time.time()
        pmem("Factorize: start_function")
        # print("Filtering... T+"+str(t-t0))
        # handle concatenation and intetrpolfilter option
        if self.n_concatenated != 1:
            #code handing n_concatenated != 1 needs  to be re-written from scratch
            grid, lows, highs, concat_dt_idxs = (
                self.concatenated_grid,
                self.concat_dt_idxs,
            )
        else:
            concat_dt_idxs = None
        #     if gauss_params != None:
        #         grid = self.gauss(self.full_grid_out, gauss_params[0],
        #                           gauss_params[1])
        #     else:
        #         grid = self.full_grid_out
        #
        # grid = self.full_gauss_grids

        pmem("Factorize: Stage0")

        factor_output = gen_factors_with_corr_check(input_grid=self.full_gauss_grids,
                                                    init_method=init_method,
                                                    factors_0=factors_0,
                                                    max_num_factors=max_num_factors,
                                                    n_iter_max=niter_max,
                                                    tolerance=tol,
                                                    sparsity_coefficients=sparsity_coeffs,
                                                    fixed_modes=fixed_modes,
                                                    normalize=normalize,
                                                    verbose=False,
                                                    return_errors=True,
                                                    corr_threshold=factor_corr_threshold)

        # delete the factors attribute from the class to save the metadata
        factor_output_metadata = copy.deepcopy(factor_output)
        delattr(factor_output_metadata, "factors")

        pmem("Factorize: Gen Factor Object List")

        factor_list = []
        for num in range(factor_output.factor_rank):
            pmem("Factorize: Gen Factor # %s start" % num)
            factor_obj = Factor(source_file=self.source_file,
                                tensor_idx=self.tensor_idx,
                                timepoint_idx=self.timepoint_idx,
                                name=self.name,
                                charge_states=self.charge_states,
                                rts=factor_output.factors[0].T[num],
                                dts=factor_output.factors[1].T[num],
                                mz_data=factor_output.factors[2].T[num],
                                retention_labels=self.retention_labels,
                                drift_labels=self.drift_labels,
                                mz_labels=self.mz_labels,
                                factor_idx=num,
                                n_factors=factor_output.factor_rank,
                                nnfac_output=factor_output_metadata,
                                bins_per_isotope_peak=self.bins_per_isotope_peak,
                                n_concatenated=self.n_concatenated,
                                concat_dt_idxs=concat_dt_idxs,
                                tensor_auc=self.tensor_auc,
                                tensor_gauss_auc=self.tensor_gauss_auc,
                                normalization_factor=self.normalization_factor)
            pmem("Factorize: Gen Factor # %s end" % num)
            factor_list.append(factor_obj)

        pmem("Factorize: Appended factors to a list")

        self.factors = factor_list

        pmem("Factorize: End of function")



        # n_itr = 2
        #
        # last_corr_check = 1.0
        # n_factors += 1
        # while n_factors > 2 and last_corr_check > 0.17:
        #     n_factors -= 1
        #     pmem(str(n_itr) + " " + str(n_factors) + " Factors " + " Start")
        #     t1 = time.time()
        #     # print("Starting "+str(nf)+" Factors... T+"+str(t1-t))
        #     nnf1 = ntf.ntf(grid, n_factors)
        #     pmem(str(n_itr) + " " + str(n_factors) + " Factors " + " End")
        #     n_itr += 1
        #     t2 = time.time()
        #     # print("Factorization Duration: "+str(t2-t1))
        #
        #     if n_factors > 1:
        #         last_corr_check = corr_check(nnf1)
        #
        # pmem(str(n_itr) + " Post-Factorization")
        # n_itr += 1
        # # Create Factor objects
        # factors = []
        # t = time.time()
        # # print("Saving Factor Objects... T+"+str(t-t0))
        # for i in range(n_factors):
        #     pmem(str(n_itr) + " Start Factor " + str(i))
        #     n_itr += 1
        #     factors.append(
        #         Factor(
        #             source_file=self.source_file,
        #             tensor_idx=self.tensor_idx,
        #             timepoint_idx=self.timepoint_idx,
        #             name=self.name,
        #             charge_states=self.charge_states,
        #             rts=nnf1[0].T[i],
        #             dts=nnf1[1].T[i],
        #             mz_data=nnf1[2].T[i],
        #             retention_labels=self.retention_labels,
        #             drift_labels=self.drift_labels,
        #             mz_labels=self.mz_labels,
        #             factor_idx=i,
        #             n_factors=n_factors,
        #             bins_per_isotope_peak = self.bins_per_isotope_peak,
        #             n_concatenated=self.n_concatenated,
        #             concat_dt_idxs=concat_dt_idxs,
        #             tensor_auc=self.tensor_auc,
        #             normalization_factor=self.normalization_factor
        #         ))
        #     pmem(str(n_itr) + " End Factor " + str(i))
        #     n_itr += 1
        # pmem(str(n_itr) + " Factor Initialization End")
        # n_itr += 1
        # self.factors = factors
        # pmem(str(n_itr) + " Script End")
        # # t = time.time()
        # # print("Done: T+"+str(t-t0))


class Factor:
    """A container for a factor output from DataTensor.factorize(), may contain the isolated signal from a single charged species.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function"s ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute"s declaration (see __init__ method below).

    Attributes:
        source_file (str): Path of DataTensor"s parent resources/tensors/.cpickle.zlib file.
        tensor_idx (int): Index of Factor"s parent DataTensor in a concatenated DataTensor. Deprecated, this value always 0.
        timepoint_idx (int): Index of the Factor"s HDX timepoint in config["timepoints"].
        name (str): Name of Factor"s rt-group.
        charge_states (list with one int): Net positive charge on protein represented in parent DataTensor.
        integrated_mz_limits (numpy 2D array): Values for low and high m/Z limits of integration around expected peak centers.
        rts (numpy array): Intensity of Factor summed over each rt bin.
        dts (numpy array): Intensity of Factor summed over each dt bin.
        mz_data (numpy array): Intensity of Factor summed over each m/Z bin in a flat array.
        retention_labels (list of floats): Mapping of RT bins to corresponding absolute retention time in minutes.
        drift_labels (list of floats): Mapping of DT bins to corresponding absolute drift time in miliseconds.
        mz_labels (list of floats): Mapping of m/Z bins to corresponding m/Z.
        factor_idx (int): Index of instance Factor in DataTensor.factors list.
        bins_per_isotope_peak (int): Number of integrated m/Z bins for a signal to be considered as an IsotopeCluster.
        n_concatenated (int): Number of DataTensors combined to make the instance DataTensor. Deprecated, value always 1.
        concat_dt_idxs (list of ints): Deprecated - Indices marking the boundaries between different DataTensors" concatenated DT dimensions.
        normalization_factor (float): Divisor for the integrated m/Z intensity of any IsotopeCluster from a parent DataTensor instance.
        integrated_mz_data (numpy array): Coarsened version of mz_data. Number of bins to sum per index is determined by bins_per_isotope_peak.
        max_rtdt (float): Product of maximal values from rts and dts.
        outer_rtdt (float): Sum of outer product of rts and dts, multiplied by the sum of mz_data to find the magnitude of the Factor.
        integrated_mz_baseline (numpy array): Baseline signal to subtract from mz_data. Deprecated.
        baseline_subtracted_integrated_mz (numpy array): Deprecated, copy of integrated_mz_data.
        rt_gauss_fit_success (bool): Boolean indicating success of Gaussian fit operation on rts. True => success, False => failure.
        rt_auc (float): Cumulative distribution function of Gaussian fit to rts evaluated between estimated bounds.
        rt_com (float): Computed center-of-mass of the Gaussian fit to rts. 
        rt_gaussian_rmse (float): Root-mean-square error, the standard deviation of the residuals between fitted values and rts.
        rt_gauss_fit_r2 (float): R^2 or "coeffiecient of determination" of linear regression over residuals between fitted values and rts.
        dt_gauss_fit_success (bool): Boolean indicating success of Gaussian fit operation on dts. True => success, False => failure.
        dt_auc (float): Cumulative distribution function of Gaussian fit to dts evaluated between estimated bounds.
        dt_com (float): Computed center-of-mass of the Gaussian fit to dts. 
        dt_gaussian_rmse (float): Root-mean-square error, the standard deviation of the residuals between fitted values and dts.
        dt_gauss_fit_r2 (float): R^2 or "coeffiecient of determination" of linear regression over residuals between fitted values and dts.
        isotope_clusters (list of IsotopeCluster objects): Contains IsotopeCluster objects made from candidate signals in integrated_mz_data

    """
    def __init__(
        self,
        source_file,
        tensor_idx,
        timepoint_idx,
        name,
        charge_states,
        rts,
        dts,
        mz_data,
        retention_labels,
        drift_labels,
        mz_labels,
        factor_idx,
        n_factors,
        nnfac_output,
        bins_per_isotope_peak,
        n_concatenated,
        concat_dt_idxs,
        tensor_auc,
        tensor_gauss_auc,
        normalization_factor
    ):
        """Creates an instance of the Factor class from one factor of a PARAFAC run.

        Args:
            source_file (str): Path of DataTensor"s parent resources/tensors/.cpickle.zlib file.
            tensor_idx (int): Index of Factor"s parent DataTensor in a concatenated DataTensor. Deprecated, this value always 0.
            timepoint_idx (int): Index of the Factor"s HDX timepoint in config["timepoints"].
            name (str): Name of Factor"s rt-group.
            charge_states (list with one int): Net positive charge on protein represented in parent DataTensor.
            integrated_mz_limits (numpy 2D array): Values for low and high m/Z limits of integration around expected peak centers.
            rts (numpy array): Intensity of Factor summed over each rt bin.
            dts (numpy array): Intensity of Factor summed over each dt bin.
            mz_data (numpy array): Intensity of Factor summed over each m/Z bin in a flat array.
            retention_labels (list of floats): Mapping of RT bins to corresponding absolute retention time in minutes.
            drift_labels (list of floats): Mapping of DT bins to corresponding absolute drift time in miliseconds.
            mz_labels (list of floats): Mapping of m/Z bins to corresponding m/Z.
            factor_idx (int): Index of instance Factor in DataTensor.factors list.
            bins_per_isotope_peak (int): Number of integrated m/Z bins for a signal to be considered as an IsotopeCluster.
            n_concatenated (int): Number of DataTensors combined to make the instance DataTensor. Deprecated, value always 1.
            concat_dt_idxs (list of ints): Deprecated - Indices marking the boundaries between different DataTensors" concatenated DT dimensions.
            normalization_factor (float): Divisor for the integrated m/Z intensity of any IsotopeCluster from a parent DataTensor instance.

        """
        self.source_file = source_file
        self.tensor_idx = tensor_idx
        self.timepoint_idx = timepoint_idx
        self.name = name
        self.charge_states = charge_states
        self.rts = rts
        self.dts = dts
        self.mz_data = mz_data
        self.retention_labels = retention_labels
        self.drift_labels = drift_labels
        self.mz_labels = mz_labels
        self.auc = sum(mz_data)
        self.tensor_auc = tensor_auc
        self.tensor_gauss_auc = tensor_gauss_auc
        self.factor_idx = factor_idx
        self.n_factors = n_factors
        self.bins_per_isotope_peak = bins_per_isotope_peak
        self.n_concatenated = n_concatenated
        self.concat_dt_idxs = concat_dt_idxs
        self.normalization_factor = normalization_factor
        self.nnfac_output = nnfac_output
        self.isotope_clusters = []

        ###Compute Instance Values###

        # Integrate within expected peak bounds.
        self.integrated_mz_data = np.sum(np.reshape(mz_data, (-1, self.bins_per_isotope_peak)), axis=1)

        # This can be a shared function
        #self.integrated_mz_baseline = peakutils.baseline(
        #    np.asarray(self.integrated_mz_data),
        #    6)  # 6 degree curve seems to work well
        #       
        #self.baseline_subtracted_integrated_mz = (self.integrated_mz_data -
        #                                          self.integrated_mz_baseline)
 
        # fit factor rts and dts to gaussian
        rt_gauss_fit = fit_gaussian(np.arange(len(self.rts)), self.rts, data_label="rt")
        dt_gauss_fit = fit_gaussian(np.arange(len(self.dts)), self.dts, data_label="dt")

        self.rt_gauss_fit_success = rt_gauss_fit["gauss_fit_success"]
        self.rt_auc = rt_gauss_fit["auc"]
        self.rt_com = rt_gauss_fit["xc"]
        self.rt_gaussian_rmse = rt_gauss_fit["fit_rmse"]
        self.rt_gauss_fit_r2 = rt_gauss_fit["fit_linregress_r2"]

        self.dt_gauss_fit_success = dt_gauss_fit["gauss_fit_success"]
        self.dt_auc = dt_gauss_fit["auc"]
        self.dt_com = dt_gauss_fit["xc"]
        self.dt_gaussian_rmse = dt_gauss_fit["fit_rmse"]
        self.dt_gauss_fit_r2 = dt_gauss_fit["fit_linregress_r2"]

        # calculate max rtdt and outer rtdt based on gauss fits
        if rt_gauss_fit["gauss_fit_success"]:
            gauss_params = [rt_gauss_fit["y_baseline"], rt_gauss_fit["y_amp"], rt_gauss_fit["xc"], rt_gauss_fit["width"]]
            rt_fac = model_data_with_gauss(np.arange(len(self.rts)), gauss_params)
        else:
            rt_fac = self.rts

        if dt_gauss_fit["gauss_fit_success"]:
            gauss_params = [dt_gauss_fit["y_baseline"], dt_gauss_fit["y_amp"], dt_gauss_fit["xc"], dt_gauss_fit["width"]]
            dt_fac = model_data_with_gauss(np.arange(len(self.dts)), gauss_params)
        else:
            dt_fac = self.rts

        # self.max_rtdt = max(self.rts) * max(self.dts)
        # self.outer_rtdt = sum(sum(np.outer(self.rts, self.dts)))

        self.max_rtdt_old = max(self.rts) * max(self.dts)
        self.outer_rtdt_old = sum(sum(np.outer(self.rts, self.dts)))

        self.max_rtdt = max(rt_fac) * max(dt_fac)
        self.outer_rtdt = sum(sum(np.outer(rt_fac, dt_fac)))

        # calculate factor auc with and without gauss extrapolated data
        self.factor_auc = (sum(self.mz_data) * self.outer_rtdt_old / self.normalization_factor)[0]
        self.factor_auc_with_gauss_extrapol = (sum(self.mz_data) * self.outer_rtdt / self.normalization_factor)[0]

        # assign mean rt and dt values
        self.rt_mean = np.mean(np.arange(len(self.rts)))
        self.dt_mean = np.mean(np.arange(len(self.dts)))

        ## old protocol.
        # Writes to self.isotope_clusters
        # self.find_isotope_clusters()  # heuristic height value, should be high-level param TODO - Will require passage through DataTensor class

        ## now generates self.isotope_clusters upon calling the function self.find_isotope_clusters

    def rel_height_peak_bounds(self, centers, norm_integrated_mz, baseline_threshold=0.15, rel_ht_threshold=0.2):
        """Determines upper and lower bounds of IsotopeClusters within a Factor.

        Args:
            centers (type): List of peak positions in integrated m/Z from scipy.signal.find_peaks().
            norm_integrated_mz (numpy array): Integrated m/Z of Factor normalized for signal centeredness and timepoint variations.
            baseline_threshold (float): Minimum ratio of peak height to maximum height for a peak to be considered as an IC. 
            rel_ht_threshold (float): Ratio of point height to peak height that a point must exceed to be part of an IC.

        Returns:
            out (list of tuple pairs of ints): List of tuples with low and high bounding indices for ICs in the centers list.
        """
        out = []
        for center in centers:
            #REVIEW: Is this what this is supposed to be doing? Should be > max(norm_integrated_mz) * baseline_threshold? 
            if norm_integrated_mz[center] > baseline_threshold:
                i, j = center, center
                cutoff = norm_integrated_mz[center] * rel_ht_threshold
                while center - i <= 10 and i - 1 != -1:
                    i -= 1
                    if norm_integrated_mz[i] < cutoff:
                        break
                while j - center <= 10 and j + 1 != len(norm_integrated_mz):
                    j += 1
                    if norm_integrated_mz[j] < cutoff:
                        break
                out.append((i, j))
        return out

    def find_isotope_clusters(self,
                              prominence=0.15,
                              width_val=3,
                              rel_height_filter=True,
                              baseline_threshold=0.15,
                              rel_height_threshold=0.10,
                              calculate_idotp=False,
                              sequence=None):
        """Identifies portions of the integrated mz dimension that look "isotope-cluster-like", saves in isotope_clusters.

        Args:
            prominence (float): Ratio of array"s maximum intesity that a peak must surpass to be considered.
            width_val (int): Minimum width to consider a peak to be an isotope cluster.
            rel_height_filter (bool): Switch to apply relative height filtering, True applies filter. 
            baseline_threshold (float): Minimum height of peak to consider it to be an isotope cluster.
            rel_height_threshold (float): Proportion determining the minimum intensity for bins near a peak to be part of an isotope cluster.

        Returns:
            None

        """
        self.isotope_clusters = []
        norm_integrated_mz = self.integrated_mz_data/max(self.integrated_mz_data)
        peaks, feature_dict = find_peaks(norm_integrated_mz,
                                         prominence=prominence,
                                         width=width_val)
        if len(peaks) == 0:
            ic_idxs = [(0, len(self.integrated_mz_data)-1)]
            int_mz_width = [2]
            #return
        else:
            int_mz_width = [
                feature_dict["widths"][i]
                for i in range(len(peaks))
                if
                feature_dict["left_bases"][i] < feature_dict["right_bases"][i]
                if feature_dict["right_bases"][i] -
                feature_dict["left_bases"][i] > 4
            ]
            ic_idxs = [
                (feature_dict["left_bases"][i], feature_dict["right_bases"][i])
                for i in range(len(peaks))
                if
                feature_dict["left_bases"][i] < feature_dict["right_bases"][i]
                if feature_dict["right_bases"][i] -
                feature_dict["left_bases"][i] > 4
            ]
            if rel_height_filter:
                height_filtered = self.rel_height_peak_bounds(centers=peaks,
                                                              norm_integrated_mz=norm_integrated_mz,
                                                              baseline_threshold=baseline_threshold,
                                                              rel_ht_threshold=rel_height_threshold)
                ic_idxs = height_filtered

        cluster_idx = 0
        for integrated_indices, integrated_mz_width in zip(ic_idxs, int_mz_width):
            if integrated_indices != None:
                newIC = IsotopeCluster(integrated_mz_peak_width=integrated_mz_width,
                                       charge_states=self.charge_states,
                                       factor_mz_data=copy.deepcopy(self.mz_data),
                                       name=self.name,
                                       source_file=self.source_file,
                                       tensor_idx=self.tensor_idx,
                                       timepoint_idx=self.timepoint_idx,
                                       n_factors=self.n_factors,
                                       factor_idx=self.factor_idx,
                                       cluster_idx=cluster_idx,
                                       low_idx = self.bins_per_isotope_peak * integrated_indices[0],
                                       high_idx = self.bins_per_isotope_peak * (integrated_indices[1] + 1),
                                       rts=self.rts,
                                       dts=self.dts,
                                       rt_mean=self.rt_mean,
                                       dt_mean=self.dt_mean,
                                       rt_gauss_fit_success=self.rt_gauss_fit_success,
                                       dt_gauss_fit_success=self.dt_gauss_fit_success,
                                       rt_gaussian_rmse=self.rt_gaussian_rmse,
                                       dt_gaussian_rmse=self.dt_gaussian_rmse,
                                       rt_com=self.rt_com,
                                       dt_coms=self.dt_com,
                                       rt_auc=self.rt_auc,
                                       dt_auc=self.dt_auc,
                                       retention_labels=self.retention_labels,
                                       drift_labels=self.drift_labels,
                                       mz_labels=self.mz_labels,
                                       bins_per_isotope_peak=self.bins_per_isotope_peak,
                                       max_rtdt=self.max_rtdt,
                                       outer_rtdt=self.outer_rtdt,
                                       max_rtdt_old=self.max_rtdt_old,
                                       outer_rtdt_old=self.outer_rtdt_old,
                                       n_concatenated=self.n_concatenated,
                                       concat_dt_idxs=self.concat_dt_idxs,
                                       normalization_factor=self.normalization_factor,
                                       tensor_auc=self.tensor_auc,
                                       tensor_gauss_auc=self.tensor_gauss_auc,
                                       factor_auc=self.factor_auc,
                                       factor_auc_with_gauss_extrapol=self.factor_auc_with_gauss_extrapol,
                                       nnfac_output=self.nnfac_output)

                # calculate idotp if set to true and save it in the class
                if calculate_idotp:
                    newIC.idotp = calculate_isotope_dist_dot_product(sequence=sequence,
                                                                     undeut_integrated_mz_array=newIC.baseline_integrated_mz)
                self.isotope_clusters.append(newIC)
                cluster_idx += 1
        return


class IsotopeCluster:
    """Contains a portion of Factor.integrated_mz_data identified to have isotope-cluster-like characteristics, stores data of parent factor.

    Attributes:
        integrated_mz_peak_width (int): Number of isotope peaks estimated to be included in the isotope cluster. 
        charge_states (list with one int): Net positive charge on protein represented in parent DataTensor.
        factor_mz_data (numpy array): The full mz_data of the parent Factor.
        name (str): Name of IC"s rt-group.
        source_file (str): Path of DataTensor"s parent resources/tensors/.cpickle.zlib file.
        tensor_idx (int): Index of the IC"s parent DataTensor in a concatenated DataTensor. Deprecated, this value always 0.
        timepoint_idx (int): Index of the IC"s HDX timepoint in config["timepoints"].
        n_factors (int): Number of factors used in decomposition of IC"s parent DataTensor.
        factor_idx (int): Index of IC"s parent Factor in its parent DataTensor.factors.
        cluster_idx (int): Index of IC in parent Factor"s Factor.isotope_clusters.
        low_idx (int): Lower bound index of IC in integrated m/Z dimension.
        high_idx (int): Upper bound index of IC in integrated m/Z dimension.
        rts (numpy array): Intensity of IC"s parent Factor summed over each rt bin.
        dts (numpy array): Intensity of IC"s parent Factor summed over each dt bin.
        rt_mean (float):
        dt_mean (float):
        rt_gauss_fit_success (bool): Boolean indicating success of Gaussian fit operation on rts. True => success, False => failure.
        rt_auc (float): Cumulative distribution function of Gaussian fit to rts evaluated between estimated bounds.
        rt_com (float): Computed center-of-mass of the Gaussian fit to rts. 
        rt_gaussian_rmse (float): Root-mean-square error, the standard deviation of the residuals between fitted values and rts.
        rt_gauss_fit_r2 (float): R^2 or "coeffiecient of determination" of linear regression over residuals between fitted values and rts.
        dt_gauss_fit_success (bool): Boolean indicating success of Gaussian fit operation on dts. True => success, False => failure.
        dt_auc (float): Cumulative distribution function of Gaussian fit to dts evaluated between estimated bounds.
        dt_com (float): Computed center-of-mass of the Gaussian fit to dts. 
        dt_gaussian_rmse (float): Root-mean-square error, the standard deviation of the residuals between fitted values and dts.
        dt_gauss_fit_r2 (float): R^2 or "coeffiecient of determination" of linear regression over residuals between fitted values and dts.
        retention_labels (list of floats): Mapping of RT bins to corresponding absolute retention time in minutes.
        drift_labels (list of floats): Mapping of DT bins to corresponding absolute drift time in miliseconds.
        mz_labels (list of floats): Mapping of m/Z bins to corresponding m/Z.
        bins_per_isotope_peak (int): Number of integrated m/Z bins for a signal to be considered as an IsotopeCluster.
        max_rtdt (float):
        outer_rt_dt (float):
        max_rtdt_old(float):
        outer_rtdt_old (float):
        n_concatenated (int): Number of DataTensors combined to make the instance DataTensor. Deprecated, value always 1.
        concat_dt_idxs (list of ints): Deprecated - Indices marking the boundaries between different DataTensors" concatenated DT dimensions.
        normalization_factor (float): Divisor for the integrated m/Z intensity of any IsotopeCluster from a parent DataTensor instance.
        cluster_mz_data (numpy array): 
        auc (float): 
        baseline (float): Deprecated, always 0
        baseline_subtracted_mz (numpy array): Deprecated, always a copy of self.cluster_mz_data.
        baseline_auc (float): Deprecated, always a copy of auc.
        log_baseline_auc (float): Base 10 log of baseline_auc.
        baseline_max_peak_height (float): Deprecated, max value of baseline_subtracted_mz

        # TODO: Document this line.
        peak_error (float): Average of the absolute distances of peak centers from expected centers of integration bounds.
        baseline_peak_error (float): Deprecated, copy of peak_error.

    """
    def __init__(
        self,
        integrated_mz_peak_width,
        charge_states,
        factor_mz_data,
        name,
        source_file,
        tensor_idx,
        timepoint_idx,
        n_factors,
        factor_idx,
        cluster_idx,
        low_idx,
        high_idx,
        rts,
        dts,
        rt_mean,
        dt_mean,
        rt_gauss_fit_success,
        dt_gauss_fit_success,
        rt_gaussian_rmse,
        dt_gaussian_rmse,
        rt_com,
        dt_coms,
        rt_auc,
        dt_auc,
        retention_labels,
        drift_labels,
        mz_labels,
        bins_per_isotope_peak,
        max_rtdt,
        outer_rtdt,
        max_rtdt_old,
        outer_rtdt_old,
        n_concatenated,
        concat_dt_idxs,
        normalization_factor,
            tensor_auc,
            tensor_gauss_auc,
            factor_auc,
            factor_auc_with_gauss_extrapol,
            nnfac_output,
    ):
        """Creates an instance of the IsotopeCluster class from a portion of Factor.mz_data.

        Args:
            integrated_mz_peak_width (int): Number of isotope peaks estimated to be included in the isotope cluster. 
            charge_states (list with one int): Net positive charge on protein represented in parent DataTensor.
            factor_mz_data (numpy array): The full mz_data of the parent Factor.
            name (str): Name of IC"s rt-group.
            source_file (str): Path of DataTensor"s parent resources/tensors/.cpickle.zlib file.
            tensor_idx (int): Index of the IC"s parent DataTensor in a concatenated DataTensor. Deprecated, this value always 0.
            timepoint_idx (int): Index of the IC"s HDX timepoint in config["timepoints"].
            n_factors (int): Number of factors used in decomposition of IC"s parent DataTensor.
            factor_idx (int): Index of IC"s parent Factor in its parent DataTensor.factors.
            cluster_idx (int): Index of IC in parent Factor"s Factor.isotope_clusters.
            low_idx (int): Lower bound index of IC in integrated m/Z dimension.
            high_idx (int): Upper bound index of IC in integrated m/Z dimension.
            rts (numpy array): Intensity of IC"s parent Factor summed over each rt bin.
            dts (numpy array): Intensity of IC"s parent Factor summed over each dt bin.
            rt_mean (float):
            dt_mean (float):
            rt_gauss_fit_success (bool): Boolean indicating success of Gaussian fit operation on rts. True => success, False => failure.
            rt_auc (float): Cumulative distribution function of Gaussian fit to rts evaluated between estimated bounds.
            rt_com (float): Computed center-of-mass of the Gaussian fit to rts. 
            rt_gaussian_rmse (float): Root-mean-square error, the standard deviation of the residuals between fitted values and rts.
            rt_gauss_fit_r2 (float): R^2 or "coeffiecient of determination" of linear regression over residuals between fitted values and rts.
            dt_gauss_fit_success (bool): Boolean indicating success of Gaussian fit operation on dts. True => success, False => failure.
            dt_auc (float): Cumulative distribution function of Gaussian fit to dts evaluated between estimated bounds.
            dt_com (float): Computed center-of-mass of the Gaussian fit to dts. 
            dt_gaussian_rmse (float): Root-mean-square error, the standard deviation of the residuals between fitted values and dts.
            dt_gauss_fit_r2 (float): R^2 or "coeffiecient of determination" of linear regression over residuals between fitted values and dts.
            retention_labels (list of floats): Mapping of RT bins to corresponding absolute retention time in minutes.
            drift_labels (list of floats): Mapping of DT bins to corresponding absolute drift time in miliseconds.
            mz_labels (list of floats): Mapping of m/Z bins to corresponding m/Z.
            bins_per_isotope_peak (int): Number of integrated m/Z bins for a signal to be considered as an IsotopeCluster.
            max_rtdt (float):
            outer_rt_dt (float):
            max_rtdt_old(float):
            outer_rtdt_old (float):
            n_concatenated (int): Number of DataTensors combined to make the instance DataTensor. Deprecated, value always 1.
            concat_dt_idxs (list of ints): Deprecated - Indices marking the boundaries between different DataTensors" concatenated DT dimensions.
            normalization_factor (float): Divisor for the integrated m/Z intensity of any IsotopeCluster from a parent DataTensor instance.
        """

        # Compute baseline_integrated_mz_com, baseline_integrated_mz_std, baseline_integrated_mz_FWHM, and baseline_integrated_mz_rmse from gaussian fit.
        # Define functions for Gaussian fit.
        def gaussian_function(x, H, A, x0, sigma):
            """Description of function.

            Args:
                arg_name (type): Description of input variable.

            Returns:
                out_name (type): Description of any returned objects.

            """
            return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        def gauss_fit(x, y):
            """Description of function.

            Args:
                arg_name (type): Description of input variable.

            Returns:
                out_name (type): Description of any returned objects.

            """
            mean = sum(x * y) / sum(y)
            sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
            nonzeros = [index for index, value in enumerate(list(y)) if value != 0]
            popt, pcov = curve_fit(gaussian_function, x, y, p0=[0, max(y), mean, sigma], bounds=([0, 0, nonzeros[0], 0],
                                                                                                 [np.inf, np.inf,
                                                                                                  nonzeros[-1],
                                                                                                  np.inf]))
            return popt

        def params_from_gaussian_fit(self):
            """Description of function.

            Args:
                arg_name (type): Description of input variable.

            Returns:
                out_name (type): Description of any returned objects.

            """
            try:
                x_data = [i for i in range(len(self.baseline_integrated_mz))]
                y_data = self.baseline_integrated_mz
                H, A, x0, sigma = gauss_fit(x_data, y_data)
                y_gaussian_fit = gaussian_function(x_data, *gauss_fit(x_data, y_data))
                rmse = mean_squared_error(y_data / max(y_data), y_gaussian_fit / max(y_gaussian_fit), squared=False)
                com = x0
                std = sigma
                FWHM = 2 * np.sqrt(2 * np.log(2)) * std
                return rmse, com, std, FWHM
            except:
                return 100, 100, 100, 100
            # Define functions for Gaussian fit END

        def peak_error_ppm(mz_labels_array, isotope_peak_array):
            errors = []
            for x, y in zip(mz_labels_array, isotope_peak_array):
                center_idx = int((len(x)-1)/2)
                if sum(y) > 0:
                    mean = np.average(x, weights=y)
                    sigma = np.sqrt(np.sum(y*(x-mean)**2)/np.sum(y))
                    try:
                        popt, pcov = curve_fit(gaussian_function, x, y, p0=[0, max(y), mean, sigma],
                                               bounds=([0, 0, x[0], 0],[np.inf, np.inf, x[-1], np.inf]))
                        errors.append(np.abs(sum(y) * (popt[2] - x[center_idx]) * 1e6 / x[(center_idx)]))
                    except:
                        print("PEAK ERROR FAILED", x, y, mean, sigma)
                        errors.append(30*sum(y))
            avg_err_ppm = np.sum(errors)/np.sum(isotope_peak_array)
            return avg_err_ppm

        self.integrated_mz_peak_width = integrated_mz_peak_width
        self.charge_states = charge_states
        self.factor_mz_data = factor_mz_data
        self.name = name
        self.source_file = source_file
        self.tensor_idx = tensor_idx
        self.timepoint_idx = timepoint_idx
        self.n_factors = n_factors
        self.factor_idx = factor_idx
        self.cluster_idx = cluster_idx
        self.low_idx = low_idx
        self.high_idx = high_idx
        self.rts = rts
        self.dts = dts
        self.rt_mean = rt_mean
        self.dt_mean = dt_mean
        self.rt_gauss_fit_success = rt_gauss_fit_success
        self.dt_gauss_fit_success = dt_gauss_fit_success
        self.rt_gaussian_rmse = rt_gaussian_rmse
        self.dt_gaussian_rmse = dt_gaussian_rmse
        self.rt_com = rt_com
        self.dt_coms = dt_coms
        self.rt_auc = rt_auc
        self.dt_auc = dt_auc
        self.retention_labels = retention_labels
        self.drift_labels = drift_labels
        self.mz_labels = mz_labels
        self.bins_per_isotope_peak = bins_per_isotope_peak
        self.max_rtdt_old = max_rtdt_old
        self.outer_rtdt_old = outer_rtdt_old
        self.max_rtdt = max_rtdt
        self.outer_rtdt = outer_rtdt
        self.n_concatenated = n_concatenated
        self.concat_dt_idxs = concat_dt_idxs
        self.normalization_factor = normalization_factor
        self.tensor_auc = tensor_auc
        self.tensor_gauss_auc = tensor_gauss_auc
        self.factor_auc = factor_auc
        self.factor_auc_with_gauss_extrapol = factor_auc_with_gauss_extrapol
        self.nnfac_output = nnfac_output

        # set an idotp variable
        self.idotp = None

        # Prune factor_mz to get window around cluster that is consistent between charge-states.
        self.cluster_mz_data = copy.deepcopy(self.factor_mz_data)
        self.cluster_mz_data[0:self.low_idx] = 0
        self.cluster_mz_data[self.high_idx:] = 0

        # Integrate area of IC and normalize according the TIC counts.
        self.auc = sum(self.cluster_mz_data) * self.outer_rtdt / self.normalization_factor

        # todo: we could replace the above auc to be a float number rather than an array
        self.ic_auc = (sum(self.cluster_mz_data) * self.outer_rtdt_old / self.normalization_factor)[0]
        self.ic_auc_with_gauss_extrapol = (sum(self.cluster_mz_data) * self.outer_rtdt / self.normalization_factor)[0]

        # Reshape cluster m/Z data by expected number of bins per isotope.
        isotope_peak_array = np.reshape(self.cluster_mz_data, (-1, self.bins_per_isotope_peak))
        mz_peak_array = np.reshape(self.mz_labels, (-1, self.bins_per_isotope_peak))
        self.baseline = 0
        self.baseline_subtracted_mz = self.cluster_mz_data 
        self.baseline_auc = self.auc
        self.log_baseline_auc = np.log(self.baseline_auc)
        self.baseline_max_peak_height = max(self.baseline_subtracted_mz)
        self.baseline_integrated_mz = np.sum(isotope_peak_array, axis=1)

        # Takes the average of the absolute distances of peak centers from expected centers of integration bounds.
        # self.peak_error = np.average(np.abs(np.argmax(isotope_peak_array,axis=1) - ((self.bins_per_isotope_peak - 1)/2))
        #                     / ((self.bins_per_isotope_peak - 1)/2), weights=self.baseline_integrated_mz)
        self.peak_error = peak_error_ppm(mz_peak_array, isotope_peak_array)
        self.baseline_peak_error = self.peak_error

        # Cache int_mz and rt scoring values.

        
        
        self.baseline_integrated_mz_norm = self.baseline_integrated_mz / np.linalg.norm(
            self.baseline_integrated_mz)
        #self.baseline_integrated_mz_com = center_of_mass(
        #    self.baseline_integrated_mz)[
        #        0]  # COM in IC integrated bin dimension
        #self.baseline_integrated_mz_std = (np.average(
        #    (np.arange(len(self.baseline_integrated_mz)) -
        #     self.baseline_integrated_mz_com)**2,
        #    weights=self.baseline_integrated_mz,
        #)**0.5)
        self.baseline_integrated_mz_rmse, self.baseline_integrated_mz_com, self.baseline_integrated_mz_std, self.baseline_integrated_mz_FWHM = params_from_gaussian_fit(self)

        self.rt_norm = self.rts / np.linalg.norm(self.rts)

        # rt_com is pre calculated during factor data class
        # self.rt_com = center_of_mass(self.rts)[0]

        # Cache DT values
        # If DT is concatenated, return list of coms and norms of single rts relative to bin numbers, a single_dt distribution starts at 0. If only one charge state, return list of len=1
        if self.concat_dt_idxs is not None:
            single_dts = []
            # generate list of single dts
            single_dts.append(self.dts[:self.concat_dt_idxs[0]])
            for i in range(len(self.charge_states) - 1):
                single_dts.append(
                    self.dts[self.concat_dt_idxs[i]:self.concat_dt_idxs[i + 1]])

            self.single_dts = single_dts

            ## dt coms is pre calculated during factor data class
            # self.dt_coms = [center_of_mass(dt)[0] for dt in single_dts]

            self.dt_norms = [dt / np.linalg.norm(dt) for dt in single_dts]
        else:
            ## dt coms is pre calculated during factor data class
            # self.dt_coms = [center_of_mass(self.dts)[0]]
            self.dt_norms = [self.dts / np.linalg.norm(self.dts)]

        if self.n_concatenated == 1:
            self.abs_mz_com = np.average(self.mz_labels, weights=self.cluster_mz_data)
        else:
            self.abs_mz_com = "Concatenated, N/A, see IC.baseline_integrated_mz_com"

        # format useful values to be read by pandas
        self.info_tuple = (
            self.
            source_file,  # Filename of data used to create parent DataTensor
            self.tensor_idx,  # Library master list row of parent-DataTensor
            self.n_factors,  # Number of factors in parent decomposition
            self.
            factor_idx,  # Index of IC parent-factor in DataTensor.factors[]
            self.cluster_idx,  # Index of IC in parent-factor.isotope_clusters[]
            self.charge_states,  # List of charge states in IC
            self.
            n_concatenated,  # number of source tensors IC parent-DataTensor was made from
            self.low_idx,  # Low bin index corresponding to Factor-level bins
            self.high_idx,  # High bin index corresponding to Factor-level bins
            self.baseline_auc,  # Baseline-subtracted AUC (BAUC)
            self.baseline_auc,  # Baseline-subtracted grate area sum (BGS) #gabe 210507: duplicating baseline_auc for now bc we got rid of grate sum
            self.
            baseline_peak_error,  # Baseline-subtracted version of peak-error (BPE)
            self.
            baseline_integrated_mz_com,  # Center of mass in added mass units
            self.
            abs_mz_com,  # IsotopicCluster center-of-mass in absolute m/z dimension
            self.rts,  # Array of RT values
            self.
            dts,  # Array of DT values, if a tensor is concatenated,this is taken from the last tensor in the list, can be seen in tensor_idx
            np.arange(0, len(self.baseline_integrated_mz), 1),
            self.
            baseline_integrated_mz,  # Array of baseline-subtracted integrated mz intensity values
        )

        # instantiate to make available for setting in PathOptimizer
        self.bokeh_tuple = None  # bokeh plot info tuple
        self.single_sub_scores = None  # differences between winning score and score if this IC were substituted, list of signed values
        self.undeut_ground_dot_products = None
