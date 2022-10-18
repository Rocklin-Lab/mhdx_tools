# Rewrite calibration code

import pymzml
import pickle as pk
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from pathlib import Path
import argparse

def check_dir(path):
    """
    Create directory from path if doesn"t exist
    """
    if not os.path.isdir(os.path.dirname(path)) and len(os.path.dirname(path)) != 0:
        os.makedirs(os.path.dirname(path))

def save_pickle_object(obj, fpath):
    """
    Save pickle object file
    """
    with open(fpath, "wb") as outfile:
        pk.dump(obj, outfile)


def load_pickle_file(pickle_fpath):
    """
    Load pickle object file
    """
    with open(pickle_fpath, "rb") as file:
        pk_object = pk.load(file)
    return pk_object


def get_mzs_thr(lockmass_compound, m0=300, m1=2000):
    """
    Get exact masses for given reference compound extracting and mz limits
    """
    if lockmass_compound == "GluFibFragments":
        # Glufib fragment masses
        mz_centers = np.array([72.081300, 120.081300, 175.119500, 187.071900,
                               246.156600, 333.188600, 382.172600, 497.199600,
                               627.325400, 684.346900, 813.389500, 942.432100,
                               1056.475000, 1171.502000, 1285.544800])
        return mz_centers[(mz_centers >= m0) & (mz_centers <= m1)]
    elif lockmass_compound == "SodiumFormate":
        # Sodium Formate masses
        mz_centers = np.array([90.977190, 158.964613, 226.952035, 294.939457, 362.926880, 430.914302, 498.901724,
                               566.889146, 634.876569, 702.863991, 770.851413, 838.838836, 906.826258, 974.813680,
                               1042.801103, 1110.788525, 1178.775947, 1246.763369, 1314.750792, 1382.738214,
                               1450.725636, 1518.713059,
                               1586.700481, 1654.687903, 1722.675326, 1790.662748, 1858.650170, 1926.637592,
                               1994.625015, 2062.612437,
                               2130.599859, 2198.587282, 2266.574704, 2334.562126, 2402.549549, 2470.536971,
                               2538.524393, 2606.511815,
                               2674.499238, 2742.486660, 2810.474082, 2878.461505, 2946.448927])
        return mz_centers[(mz_centers >= m0) & (mz_centers <= m1)]
    elif lockmass_compound == "GluFibPrecursor":
        # Glufib precursor
        return np.array([785.84265])
    elif lockmass_compound == "LeuEnkPrecursor":
        # LeuEnk precursor
        return np.array([556.2771])
    else:
        print("LockMass compound not found. Check get_mz_centers function!")
        exit()


def generate_tensor(mzml_gz_path):
    """
    Generate mz_bins and 2d tensor for lockmass
    """

    msrun = pymzml.run.Reader(mzml_gz_path)

    scan_times = []
    raw_scans = []

    for scan in msrun:
        if scan.id_dict["function"] == 2:
            scan_times.append(scan.scan_time_in_minutes())
            spectrum = np.array(scan.peaks("raw")).astype(np.float32)
            raw_scans.append(spectrum[spectrum[:, 1] > 10])

    mzs = []
    for scan in raw_scans:
        mzs = np.unique(list(mzs) + list(scan[:, 0]))
    tensor = np.zeros(shape=(len(scan_times), len(mzs)))
    for i, scan in enumerate(raw_scans):
        tensor[i, np.searchsorted(mzs, scan[:, 0])] = scan[:, 1]
    tensor = tensor.reshape(len(scan_times), len(mzs))

    return np.array(scan_times), mzs, tensor


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
    popt, pcov = curve_fit(gaussian_function, x, y, p0=[0, max(y), mean, sigma])

    return popt


def get_closest_peaks(mz_thr, mzs, exp_spec, ppm_radius=100, min_intensity=1e3):
    """
    Fit gaussian to experimental peaks closest to theoretical ones
    """
    thr_peaks, exp_peaks = [], []
    for mz in mz_thr:
        mz_low, mz_high = mz - mz * ppm_radius / 1e6, mz + mz * ppm_radius / 1e6
        mask = (mzs >= mz_low) & (mzs <= mz_high)
        try:
            H, A, x0, sigma = gauss_fit(mzs[mask], exp_spec[mask])
        except:
            print("Parameters not found!")
            x0 = mz
            A = 0
        if abs((x0 - mz) * 1e6 / mz) <= ppm_radius and A >= min_intensity:
            thr_peaks.append(mz)
            exp_peaks.append(x0)

    return thr_peaks, exp_peaks


def rmse_from_gaussian_fit(mzs, obs_spec):
    """
    Get rmse from distribution
    """
    try:
        xdata = mzs
        ydata = obs_spec
        y_gaussian_fit = gaussian_function(xdata, *gauss_fit(xdata, ydata))
        rmse = mean_squared_error(ydata / max(ydata), y_gaussian_fit / max(y_gaussian_fit), squared=False)
        return rmse
    except:
        return 100


def generate_thr_exp_pairs(scan_times,
                           mzs,
                           tensor,
                           mzs_thr,
                           runtime=30,
                           time_bins=6,
                           min_intensity=1e3,
                           ppm_radius=100,
                           output_extracted_signals=None,
                           ):
    """
    Generate dictionary containing theoretical and experimental peaks above
    intensity threshold and below error threshold.
    """
    thr_exp_pairs = {}

    time_step = int(runtime / time_bins)

    if output_extracted_signals is not None:
        fig, ax = plt.subplots(len(mzs_thr), time_bins, figsize=(3 * time_bins, 2*len(mzs_thr)), dpi=200)

    for i, t in enumerate(range(0, runtime, time_step)):
        keep = (scan_times >= t) & (scan_times < t + time_step)
        obs_spec = np.sum(tensor[keep], axis=0)
        thr_peaks, obs_peaks = get_closest_peaks(mzs_thr, mzs, obs_spec,
                                                 ppm_radius=ppm_radius, min_intensity=min_intensity)
        thr_exp_pairs[i] = [thr_peaks, obs_peaks]

        if output_extracted_signals is not None and len(mzs_thr) > 1:
            for j, mz_thr in enumerate(mzs_thr):
                mz_low = mz_thr - mz_thr * ppm_radius / 1e6
                mz_high = mz_thr + mz_thr * ppm_radius / 1e6
                mask = (mzs >= mz_low) & (mzs <= mz_high)
                ax[j][i].plot(mzs[mask], np.sum(tensor[keep], axis=0)[mask], c="orange")
                try:
                    H, A, x0, sigma = gauss_fit(mzs[mask], obs_spec[mask])
                except:
                    x0 = mz_thr
                    A = 0
                ax[j][i].plot(np.linspace(mzs[mask][0], mzs[mask][-1], 50),
                              gaussian_function(np.linspace(mzs[mask][0], mzs[mask][-1], 50), H, A, x0, sigma),
                              c="blue")
                ax[j][i].axvline(mz_thr, ls="--", c="red")
                ax[j][i].axvline(x0, ls="--", c="blue")
                ax[j][i].set_yticks([])
                ax[j][i].set_xticks([mz_thr])
                rmse = rmse_from_gaussian_fit(mzs[mask], obs_spec[mask])
                if A > min_intensity and rmse < 0.1:
                    ax[j][i].text(0.98, 0.9, "I=%.2e" % A, horizontalalignment="right",
                                  transform=ax[j][i].transAxes)
                    ax[j][i].text(0.98, 0.8, "mz_err=%.1f ppm" % ((x0 - mz_thr) * 1e6 / mz_thr),
                                  horizontalalignment="right",
                                  transform=ax[j][i].transAxes)
                    ax[j][i].text(0.98, 0.7, "rmse_fit=%.2f" % rmse, horizontalalignment="right",
                                  transform=ax[j][i].transAxes)
                else:
                    ax[j][i].text(0.98, 0.9, "I=%.2e" % A, horizontalalignment="right",
                                  transform=ax[j][i].transAxes, c="red")
                    ax[j][i].text(0.98, 0.8, "mz_err=%.1f ppm" % ((x0 - mz_thr) * 1e6 / mz_thr),
                                  horizontalalignment="right",
                                  transform=ax[j][i].transAxes, c="red")
                    ax[j][i].text(0.98, 0.7, "rmse_fit=%.2f" % rmse, horizontalalignment="right",
                                  transform=ax[j][i].transAxes, c="red")
                if j == 0:
                    ax[j][i].text(0.02, 0.9, "t=%i-%imin" % (t, t + time_step), horizontalalignment="left",
                                  transform=ax[j][i].transAxes, color="blue")

        if output_extracted_signals is not None and len(mzs_thr) == 1:
            for mz_thr in mzs_thr:
                mz_low = mz_thr - mz_thr * ppm_radius / 1e6
                mz_high = mz_thr + mz_thr * ppm_radius / 1e6
                mask = (mzs >= mz_low) & (mzs <= mz_high)
                ax[i].plot(mzs[mask], np.sum(tensor[keep], axis=0)[mask], c="orange")
                try:
                    H, A, x0, sigma = gauss_fit(mzs[mask], obs_spec[mask])
                except:
                    x0 = mz_thr
                    A = 0
                ax[i].plot(np.linspace(mzs[mask][0], mzs[mask][-1], 50),
                              gaussian_function(np.linspace(mzs[mask][0], mzs[mask][-1], 50), H, A, x0, sigma),
                              c="blue")
                ax[i].axvline(mz_thr, ls="--", c="red")
                ax[i].axvline(x0, ls="--", c="blue")
                ax[i].set_yticks([])
                ax[i].set_xticks([mz_thr])
                rmse = rmse_from_gaussian_fit(mzs[mask], obs_spec[mask])
                if A > min_intensity and rmse < 0.1:
                    ax[i].text(0.98, 0.9, "I=%.2e" % A, horizontalalignment="right",
                                  transform=ax[i].transAxes)
                    ax[i].text(0.98, 0.8, "mz_err=%.1f ppm" % ((x0 - mz_thr) * 1e6 / mz_thr),
                                  horizontalalignment="right",
                                  transform=ax[i].transAxes)
                    ax[i].text(0.98, 0.7, "rmse_fit=%.2f" % rmse, horizontalalignment="right",
                                  transform=ax[i].transAxes)
                else:
                    ax[i].text(0.98, 0.9, "I=%.2e" % A, horizontalalignment="right",
                                  transform=ax[i].transAxes, c="red")
                    ax[i].text(0.98, 0.8, "mz_err=%.1f ppm" % ((x0 - mz_thr) * 1e6 / mz_thr),
                                  horizontalalignment="right",
                                  transform=ax[i].transAxes, c="red")
                    ax[i].text(0.98, 0.7, "rmse_fit=%.2f" % rmse, horizontalalignment="right",
                                  transform=ax[i].transAxes, c="red")

                ax[i].text(0.02, 0.9, "t=%i-%imin" % (t, t + time_step), horizontalalignment="left",
                                  transform=ax[i].transAxes, color="blue")

    if output_extracted_signals is not None:
        fig.tight_layout()
        fig.savefig(output_extracted_signals, dpi=300, format="pdf")
        plt.close("all")

    return thr_exp_pairs


def generate_lockmass_calibration_dict(thr_exp_pairs, polyfit_deg, lockmass_compound,
                                       output_pk=None, output_kde=None):
    cal_dict = {}

    if lockmass_compound != "GluFibPrecursor" and lockmass_compound != "LeuEnkPrecursor":

        for idx in thr_exp_pairs.keys():
            thr_mz, obs_mz = np.array(thr_exp_pairs[idx])
            polyfit_coeffs = np.polyfit(x=obs_mz, y=thr_mz, deg=polyfit_deg)
            obs_mz_corr = np.polyval(polyfit_coeffs, obs_mz)
            ppm_error_before_corr = 1e6 * (obs_mz - thr_mz) / thr_mz
            ppm_error_after_corr = 1e6 * (obs_mz_corr - thr_mz) / thr_mz
            cal_dict[idx] = {"polyfit_bool": True, "thr_mz": thr_mz, "obs_mz": obs_mz, "polyfit_coeffs": polyfit_coeffs,
                             "polyfit_deg": polyfit_deg, "obs_mz_corr": obs_mz_corr,
                             "ppm_error_before_corr": ppm_error_before_corr,
                             "ppm_error_after_corr": ppm_error_after_corr}

        if output_pk is not None:
            save_pickle_object(cal_dict, output_pk)

        # TO BE REMOVED LATER - AF
        if output_kde is not None:
            sns.set_context("talk")
            colors = ["black", "red", "blue", "green", "orange", "green", "cyan", "yellow"]
            fig_kde, ax_kde = plt.subplots(1,1)
            for i, key in enumerate(cal_dict.keys()):
                sns.kdeplot(cal_dict[key]["ppm_error_before_corr"], label="%i-%imin"%(6*key, 6*key+6), ax=ax_kde,
                            color=colors[i])
                sns.kdeplot(cal_dict[key]["ppm_error_after_corr"], ax=ax_kde, ls="--", color=colors[i])
            ax_kde.legend(loc=2)
            ax_kde.set_xlabel("ppm error")
            fig_kde.savefig(output_kde, dpi=200, format="pdf")
            plt.close()
        return cal_dict

    else:
        for idx in thr_exp_pairs.keys():
            thr_mz, obs_mz = np.array(thr_exp_pairs[idx])
            coeff = thr_mz / obs_mz
            obs_mz_corr = coeff * obs_mz
            ppm_error_before_corr = 1e6 * (obs_mz - thr_mz) / thr_mz
            ppm_error_after_corr = 1e6 * (obs_mz_corr - thr_mz) / thr_mz

            cal_dict[idx] = {"polyfit_bool": True, "thr_mz": thr_mz, "obs_mz": obs_mz, "polyfit_coeffs": coeff,
                             "polyfit_deg": 0, "obs_mz_corr": obs_mz_corr,
                             "ppm_error_before_corr": ppm_error_before_corr,
                             "ppm_error_after_corr": ppm_error_after_corr}

        if output_pk is not None:
            save_pickle_object(cal_dict, output_pk)

        return cal_dict


def plot_degrees(thr_exp_pairs,
                 polyfit_deg,
                 lockmass_compound,
                 runtime,
                 time_bins,
                 ppm_radius,
                 output_pk=None,
                 output_degrees=None,
                 output_kde=None):

    fig, ax = plt.subplots(8, 1, figsize=(6, 30))

    for idx, deg in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
        if deg == polyfit_deg:
            cal_dict = generate_lockmass_calibration_dict(thr_exp_pairs,
                                                          polyfit_deg=deg,
                                                          lockmass_compound=lockmass_compound,
                                                          output_pk=output_pk, output_kde=output_kde)
        else:
            cal_dict = generate_lockmass_calibration_dict(thr_exp_pairs,
                                                          polyfit_deg=deg,
                                                          lockmass_compound=lockmass_compound,
                                                          output_pk=None, output_kde=None)

        time_step = int(runtime / time_bins)

        t = 0
        err_before = 0
        err_after = 0
        for key in cal_dict:
            ax[idx].scatter(cal_dict[key]["thr_mz"], cal_dict[key]["ppm_error_before_corr"],
                            label="%i-%imin" % (t, t + time_step))
            ax[idx].scatter(cal_dict[key]["thr_mz"], cal_dict[key]["ppm_error_after_corr"], marker="x")
            xs = np.linspace(50, 2000, 1000)
            ys = np.polyval(cal_dict[key]["polyfit_coeffs"], xs)
            ax[idx].plot(xs, (ys - xs) * 1e6 / xs, "--")
            err_before += np.mean(cal_dict[key]["ppm_error_before_corr"])
            err_after += np.mean(cal_dict[key]["ppm_error_after_corr"])
            t += time_step
        err_before = err_before / len(cal_dict)
        err_after = err_after / len(cal_dict)
        ax[idx].text(0.05, 0.9, "degree=%i" % (deg), transform=ax[idx].transAxes, fontsize=12)

        ax[idx].text(0.05, 0.82, "avg_err_before=%.2f" % err_before,
                     transform=ax[idx].transAxes, fontsize=12)
        ax[idx].text(0.05, 0.74, "avg_err_after=%.2f" % err_after,
                     transform=ax[idx].transAxes, fontsize=12)
        ax[idx].set_ylabel("ppm error")
        ax[idx].set_xlabel("m/z")
        ax[idx].set_ylim(-ppm_radius, ppm_radius)
        ax[idx].legend(loc=1, fontsize=11)

    fig.tight_layout()

    if output_degrees is not None:
        fig.savefig(output_degrees, dpi=300, format="pdf")
    else:
        plt.show()


def main(mzml_gz_path,
         lockmass_compound,
         runtime,
         time_bins,
         min_intensity,
         ppm_radius,
         polyfit_deg,
         output_extracted_signals=None,
         output_pk=None,
         output_degrees=None,
         output_kde=None,
         ):

    if output_extracted_signals is not None:
        check_dir(output_extracted_signals)
    if output_pk is not None:
        check_dir(output_pk)
    if output_degrees is not None:
        check_dir(output_degrees)
    if output_kde is not None:
        check_dir(output_kde)

    scan_times, mzs, tensor = generate_tensor(mzml_gz_path=mzml_gz_path)

    mzs_thr = get_mzs_thr(lockmass_compound=lockmass_compound)

    thr_exp_pairs = generate_thr_exp_pairs(scan_times,
                                           mzs,
                                           tensor,
                                           mzs_thr,
                                           runtime=runtime,
                                           time_bins=time_bins,
                                           min_intensity=min_intensity,
                                           ppm_radius=ppm_radius,
                                           output_extracted_signals=output_extracted_signals)

    if lockmass_compound != "GluFibPrecursor" and lockmass_compound != "LeuEnkPrecursor":
        plot_degrees(thr_exp_pairs,
                     polyfit_deg=polyfit_deg,
                     lockmass_compound=lockmass_compound,
                     runtime=runtime,
                     time_bins=time_bins,
                     ppm_radius=ppm_radius,
                     output_pk=output_pk,
                     output_degrees=output_degrees,
                     output_kde=output_kde)

    else:
        generate_lockmass_calibration_dict(thr_exp_pairs,
                                       polyfit_deg=polyfit_deg,
                                       lockmass_compound=lockmass_compound,
                                       output_pk=output_pk, output_kde=None)


if __name__ == "__main__":
    # If the snakemake global object is present, save expected arguments from snakemake to be passed to main().
    if "snakemake" in globals():
        mzml_gz_path = snakemake.input[0]
        configfile = yaml.load(open(snakemake.input[1], "rb").read(), Loader=yaml.Loader)
        output_pk = snakemake.output[0]
        output_extracted_signals = snakemake.output[1]
        output_degrees = snakemake.output[2]
        output_kde = snakemake.output[3]

        if configfile["runtime"] is not None:
            runtime = int(configfile["runtime"])
        if configfile["time_bins"] is not None:
            time_bins = int(configfile["time_bins"])
        if configfile["min_intensity"] is not None:
            min_intensity = float(configfile["min_intensity"])
        if configfile["ppm_lockmass_radius"] is not None:
            ppm_lockmass_radius = int(configfile["ppm_lockmass_radius"])
        if configfile["polyfit_deg"] is not None:
            polyfit_deg = int(configfile["polyfit_deg"])

        if configfile["lockmass_compound"] == "GluFibPrecursor" or configfile["lockmass_compound"] == "LeuEnkPrecursor":
            if output_degrees is not None:
                Path(output_degrees).touch()
            if output_kde is not None:
                Path(output_kde).touch()


        main(mzml_gz_path=mzml_gz_path,
             lockmass_compound=configfile["lockmass_compound"],
             runtime=runtime,
             time_bins=time_bins,
             min_intensity=min_intensity,
             ppm_radius=ppm_lockmass_radius,
             polyfit_deg=polyfit_deg,
             output_extracted_signals=output_extracted_signals,
             output_pk=output_pk,
             output_degrees=output_degrees,
             output_kde=output_kde
             )

    else:
        # CLI context, set expected arguments with argparse module.
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-mzml_gz_path",
            "--mzml_gz_path", help="path/to/file.mzML.gz")
        parser.add_argument(
            "-lm",
            "--lockmass_compound",
            default=None,
            help=
            "LockMass compound. Choose from: SodiumFormate, GluFibPrecursor, GluFibFragments, LeuEnkPrecursor"
        )
        parser.add_argument(
            "-runtime",
            "--runtime",
            default=30,
            help=
            "chromatographic runtime"
        )
        parser.add_argument(
            "-time_bins",
            "--time_bins",
            default=None,
            help=
            "Into how many bins the chromatographic run should be divided."
        )
        parser.add_argument(
            "-min_int",
            "--min_intensity",
            default=None,
            help=
            "Minumum intensity so a peak is considered for calibration."
        )
        parser.add_argument(
            "-ppm_lockmass_radius",
            "--ppm_lockmass_radius",
            default=50,
            help=
            "ppm window to reprofile peak"
        )
        parser.add_argument(
            "-degree",
            "--polyfit_deg",
            default=None,
            help=
            "Polynomial degree for calibration curve"
        )
        parser.add_argument(
            "-s",
            "--output_extracted_signals",
            default=None,
            help=
            "Output name for pdf file containing fit analysis"
        )
        parser.add_argument(
            "-o",
            "--output_pk",
            default=None,
            help=
            "Output name for pickle file containing the dictionary"
        )
        parser.add_argument(
            "-d",
            "--output_degrees",
            default=None,
            help=
            "Output name for pdf file containing fit analysis"
        )
        parser.add_argument(
            "-k",
            "--output_kde",
            default=None,
            help=
            "Output name for pdf file containing kde error distribution"
        )


        args = parser.parse_args()

        if args.runtime is not None:
            args.runtime = int(args.runtime)
        if args.time_bins is not None:
            args.time_bins = int(args.time_bins)
        if args.min_intensity is not None:
            args.min_intensity = float(args.min_intensity)
        if args.ppm_lockmass_radius is not None:
            args.ppm_lockmass_radius = int(args.ppm_lockmass_radius)
        if args.polyfit_deg is not None:
            args.polyfit_deg = int(args.polyfit_deg)

        main(mzml_gz_path=args.mzml_gz_path,
             lockmass_compound=args.lockmass_compound,
             runtime=args.runtime,
             time_bins=args.time_bins,
             min_intensity=args.min_intensity,
             ppm_radius=args.ppm_lockmass_radius,
             polyfit_deg=args.polyfit_deg,
             output_extracted_signals=args.output_extracted_signals,
             output_pk=args.output_pk,
             output_degrees=args.output_degrees,
             output_kde=args.output_kde
             )