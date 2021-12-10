import pymzml
import _pickle as cpickle
import pickle as pk
import psutil
import time
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import norm
from scipy.signal import find_peaks
import glob as glob
import argparse


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

def get_mz_centers(m0, m1, lockmass_compound):
    """
    Get exact masses for given reference compound extracting and mz limits
    """
    if lockmass_compound == 'GluFibFragments':
        # Glufib fragment masses
        mz_centers = np.array([72.081300, 120.081300, 175.119500, 187.071900,
                               246.156600, 333.188600, 382.172600, 497.199600,
                               627.325400, 684.346900, 813.389500, 942.432100,
                               1056.475000, 1171.502000, 1285.544800])
        return mz_centers[(mz_centers >= m0) & (mz_centers <= m1)]
    elif lockmass_compound == 'SodiumFormate':
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
    elif lockmass_compound == 'GluFibPrecursor':
        # Glufib precursor
        return np.array([785.84265])
    else:
        print('LockMass compound not found. Check get_mz_centers function!')
        exit()


def generate_tensor(mzml_gz_path, ppm_radius, bins_per_isotope_peak, ms_resolution, m0, m1, lockmass_compound):

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

    scan_times = np.array(scan_times)
    scan_numbers = np.arange(0, len(scan_times))

    mz_centers = get_mz_centers(m0, m1, lockmass_compound)

    low_mz_limits = mz_centers * ((1000000.0 - ppm_radius) / 1000000.0)
    high_mz_limits = mz_centers * ((1000000.0 + ppm_radius) / 1000000.0)
    integrated_mz_limits = np.stack((low_mz_limits, high_mz_limits)).T

    FWHM = np.average(integrated_mz_limits) / ms_resolution
    gaussian_scale = FWHM / 2.355
    mz_bin_centers = np.ravel(
        [np.linspace(lowlim, highlim, bins_per_isotope_peak) for lowlim, highlim in integrated_mz_limits])
    tensor2_out = np.zeros((len(scan_times), len(mz_bin_centers)))

    scan = 0
    for i in range(len(scan_times)):
        n_peaks = len(raw_scans[i])
        gaussians = norm(loc=raw_scans[i][:, 0], scale=gaussian_scale)
        resize_gridpoints = np.resize(mz_bin_centers, (n_peaks, len(mz_bin_centers))).T
        eval_gaussians = gaussians.pdf(resize_gridpoints) * raw_scans[i][:, 1] * gaussian_scale

        tensor2_out[i] = np.sum(eval_gaussians, axis=1)
        scan += 1

    return mz_bin_centers, tensor2_out


def find_nearest(obs_mz, mz_centers):
    """
    Find closes signal to center
    """
    mz_centers_subset = []
    for mz in obs_mz:
        idx = (np.abs(mz_centers - mz)).argmin()
        mz_centers_subset.append(mz_centers[idx])
    return np.array(mz_centers_subset)

def generate_lockmass_calibration_dict(mz_bins, tensor2_out, time_bins, polyfit_deg, m0, m1,
                                       lockmass_compound, outputname):

    mz_centers = get_mz_centers(m0, m1, lockmass_compound)

    idx = 0

    cal_dict = {}

    if lockmass_compound != 'GluFibPrecursor':

        for i in range(0, len(tensor2_out), int(len(tensor2_out) / time_bins)):
            spectrum = np.sum(tensor2_out[i:i + int(len(tensor2_out) / time_bins)], axis=0)
            obs_mz = mz_bins[find_peaks(spectrum)[0]]
            thr_mz = find_nearest(obs_mz, mz_centers)
            polyfit_coeffs = np.polyfit(x=obs_mz, y=thr_mz, deg=polyfit_deg)
            obs_mz_corr = np.polyval(polyfit_coeffs, obs_mz)
            ppm_error_before_corr = 10 ** 6 * (obs_mz - thr_mz) / thr_mz
            ppm_error_after_corr = 10 ** 6 * (obs_mz_corr - thr_mz) / thr_mz
            #             print(time, np.mean(ppm_error_before_corr), np.mean(ppm_error_after_corr))

            cal_dict[idx] = {'polyfit_bool': True, 'thr_mz': thr_mz, 'obs_mz': obs_mz, 'polyfit_coeffs': polyfit_coeffs,
                        'polyfit_deg': polyfit_deg, 'obs_mz_corr': obs_mz_corr,
                        'ppm_error_before_corr': ppm_error_before_corr,
                        'ppm_error_after_corr': ppm_error_after_corr}
            idx += 1

        if outputname is not None:
            save_pickle_object(cal_dict, 'resources/1_calibration/' + outputname + '_mz_calib_dict.pk')
        else:
            return cal_dict

    else:
        for i in range(0, len(tensor2_out), int(len(tensor2_out) / time_bins)):
            spectrum = np.sum(tensor2_out[i:i + int(len(tensor2_out) / time_bins)], axis=0)
            obs_mz = mz_bins[find_peaks(spectrum)[0]]
            thr_mz = find_nearest(obs_mz, mz_centers)
            coeff = thr_mz / obs_mz
            obs_mz_corr = coeff * obs_mz
            ppm_error_before_corr = 10 ** 6 * (obs_mz - thr_mz) / thr_mz
            ppm_error_after_corr = 10 ** 6 * (obs_mz_corr - thr_mz) / thr_mz
            #             print(time, np.mean(ppm_error_before_corr), np.mean(ppm_error_after_corr))

            cal_dict[idx] = {'polyfit_bool': True, 'thr_mz': thr_mz, 'obs_mz': obs_mz, 'polyfit_coeffs': coeff,
                        'polyfit_deg': 0, 'obs_mz_corr': obs_mz_corr,
                        'ppm_error_before_corr': ppm_error_before_corr,
                        'ppm_error_after_corr': ppm_error_after_corr}

            idx += 1

        if outputname is not None:
            save_pickle_object(cal_dict, 'resources/0_calibration/' + outputname + '_mz_calib_dict.pk')
        else:
            return cal_dict

def plot_degrees(mz_bins, tensor2_out, time_bins, m0, m1, lockmass_compound, runtime, outputname):
    sns.set_context('talk')
    fig, ax = plt.subplots(8, 1, figsize=(6, 30))

    for idx, deg in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
        cal_dict = generate_lockmass_calibration_dict(mz_bins=mz_bins, tensor2_out=tensor2_out, time_bins=time_bins,
                                                      m0=m0, m1=m1, lockmass_compound=lockmass_compound,
                                                      polyfit_deg=deg, outputname=None)
        delta = int(runtime / time_bins)

        t = 0
        err_before = 0
        err_after = 0
        for key in cal_dict:
            ax[idx].scatter(cal_dict[key]['thr_mz'], cal_dict[key]['ppm_error_before_corr'], label='%i-%imin' % (t, t + delta))
            ax[idx].scatter(cal_dict[key]['thr_mz'], cal_dict[key]['ppm_error_after_corr'], marker='x')
            xs = np.linspace(50, 2000, 1000)
            ys = np.polyval(cal_dict[key]['polyfit_coeffs'], xs)
            ax[idx].plot(xs, (ys - xs) * 1e6 / xs, '--')
            err_before += np.mean(cal_dict[key]['ppm_error_before_corr'])
            err_after += np.mean(cal_dict[key]['ppm_error_after_corr'])
            t += delta
        err_before = err_before / len(cal_dict)
        err_after = err_after / len(cal_dict)
        ax[idx].text(0.05, 0.9, 'degree=%i' % (deg), transform=ax[idx].transAxes, fontsize=12)

        ax[idx].text(0.05, 0.82, 'avg_err_before=%.2f' %err_before,
                     transform=ax[idx].transAxes, fontsize=12)
        ax[idx].text(0.05, 0.74, 'avg_err_after=%.2f' %err_after,
                        transform=ax[idx].transAxes, fontsize=12)
        ax[idx].set_ylabel('ppm error')
        ax[idx].set_xlabel('m/z')
        ax[idx].set_ylim(-50, 50)
        ax[idx].legend(loc=1, fontsize=11)

    plt.tight_layout()

    plt.savefig('results/plots/preprocessing/0_calibration_' + outputname + '_degrees.pdf', dpi=300, format='pdf')


def main(mzml_gz_path=None,
         m0=None,
         m1=None,
         lockmass_compound=None,
         time_bins=None,
         ppm_radius=None,
         bins_per_isotopic_peak=None,
         ms_resolution=None,
         polyfit_deg=None,
         outputname=None,
         runtime=None,
         ):

    mz_bins, tensor2_out = generate_tensor(mzml_gz_path=mzml_gz_path, m0=m0, m1=m1, ppm_radius=ppm_radius,
                                           bins_per_isotope_peak=bins_per_isotopic_peak, ms_resolution=ms_resolution,
                                           lockmass_compound=lockmass_compound)

    generate_lockmass_calibration_dict(mz_bins=mz_bins, tensor2_out=tensor2_out, time_bins=time_bins,
                                       polyfit_deg=polyfit_deg, m0=m0, m1=m1,
                                       lockmass_compound=lockmass_compound, outputname=outputname)

    if lockmass_compound != 'GluFibPrecursor':
        plot_degrees(mz_bins=mz_bins, tensor2_out=tensor2_out, time_bins=time_bins, m0=m0, m1=m1,
                     lockmass_compound=lockmass_compound, runtime=runtime, outputname=outputname)


if __name__ == "__main__":
    # If the snakemake global object is present, save expected arguments from snakemake to be passed to main().
    if "snakemake" in globals():
        mzml_gz_path = snakemake.input[0]
        configfile = yaml.load(open(snakemake.input[1], "rb").read(), Loader=yaml.Loader)
        outputname = mzml_gz_path.split('/')[-1]

        if configfile['polyfit_deg'] is not None:
            polyfit_deg = int(configfile['polyfit_deg'])
        if configfile['m0'] is not None:
            m0 = float(configfile['m0'])
        if configfile['m1'] is not None:
            m1 = float(configfile['m1'])
        if configfile['time_bins'] is not None:
            time_bins = int(configfile['time_bins'])
        if configfile['ms_resolution'] is not None:
            ms_resolution = int(configfile['ms_resolution'])
        if configfile['ppm_lockmass_radius'] is not None:
            ppm_lockmass_radius = int(configfile['ppm_lockmass_radius'])
        if configfile['bins_per_isotopic_peak'] is not None:
            bins_per_isotopic_peak = int(configfile['bins_per_isotopic_peak'])
        if configfile['runtime'] is not None:
            runtime = int(configfile['runtime'])


        main(mzml_gz_path=mzml_gz_path,
             m0=m0,
             m1=m1,
             lockmass_compound=configfile['lockmass_compound'],
             time_bins=time_bins,
             polyfit_deg=polyfit_deg,
             outputname=outputname,
             ms_resolution=ms_resolution,
             ppm_radius=ppm_lockmass_radius,
             bins_per_isotopic_peak=bins_per_isotopic_peak,
             runtime=runtime,
             )
    else:
        # CLI context, set expected arguments with argparse module.
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-mzml_gz_path",
            "--mzml_gz_path", help="path/to/file.mzML.gz")
        parser.add_argument(
            "-m0",
            "--m0",
            default=None,
            help=
            "lower mz bound")
        parser.add_argument(
            "-m1",
            "--m1",
            default=None,
            help=
            "high mz bound"
        )
        parser.add_argument(
            "-lm",
            "--lockmass_compound",
            default=None,
            help=
            "LockMass compound. Choose from: SodiumFormate, GluFibPrecursor, GluFibFragments"
        )
        parser.add_argument(
            "-time_bins",
            "--time_bins",
            default=None,
            help=
            "Into how many bins the chromatographic run should be divided."
        )
        parser.add_argument(
            "-degree",
            "--polyfit_deg",
            default=None,
            help=
            "Polynomial degree for calibration curve"
        )
        parser.add_argument(
            "-o",
            "--outputname",
            default=None,
            help=
            "Output name for pickle file containing the dictionary"
        )
        parser.add_argument(
            "-resolution",
            "--ms_resolution",
            default=12500,
            help=
            "Resolution to reprofile peak"
        )
        parser.add_argument(
            "-ppm_lockmass_radius",
            "--ppm__lockmass_radius",
            default=50,
            help=
            "ppm window to reprofile peak"
        )
        parser.add_argument(
            "-bins_per_isotopic_peak",
            "--bins_per_isotopic_peak",
            default=50,
            help=
            "ppm window to reprofile peak"
        )
        parser.add_argument(
            "-runtime",
            "--runtime",
            default=25,
            help=
            "chromatographic runtime"
        )
        args = parser.parse_args()

        if args.m0 is not None:
            args.m0 = float(args.m0)
        if args.m1 is not None:
            args.m1 = float(args.m1)
        if args.time_bins is not None:
            args.time_bins = int(args.time_bins)
        if args.ms_resolution is not None:
            args.ms_resolution = int(args.ms_resolution)
        if args.ppm_lockmass_radius is not None:
            args.ppm_lockmass_radius = int(args.ppm_lockmass_radius)
        if args.bins_per_isotopic_peak is not None:
            args.bins_per_isotopic_peak = int(args.bins_per_isotopic_peak)
        if args.runtime is not None:
            args.runtime = int(args.runtime)



        main(mzml_gz_path=args.mzml_gz_path,
             m0=args.m0,
             m1=args.m1,
             lockmass_compound=args.lockmass_compound,
             time_bins=args.time_bins,
             polyfit_deg=args.polyfit_deg,
             outputname=args.outputname,
             ms_resolution=args.ms_resolution,
             ppm_radius=args.ppm_lockmass_radius,
             bins_per_isotopic_peak=args.bins_per_isotopic_peak,
             runtime=args.runtime,
             )