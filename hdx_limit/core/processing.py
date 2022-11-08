import os
import sys
import psutil
import copy
import math
import molmass
import numpy as np
import pandas as pd
import glob
import pickle as pk
from hdx_limit.core import io, datatypes
from hdx_limit.core.plot_factor_data import plot_factor_data_from_data_tensor
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


def filter_factors_on_rt_dt_gauss_fit(factor_list, rt_r2_cutoff=0.90, dt_r2_cutoff=0.90):
    """Filters Factors based on quality of RT and DT gaussian fit.

    New factor list is created if the Factor DT and RT gaussian fit 
    r^2 values are high. If none of the Factors pass the filtering 
    criteria, returns original Factor list.

    Args:
        factor_list (list of Factor objects): A list of datatypes.Factor objects to be filtered by their rt and dt gaussian character.
        rt_r2_cutoff (float): Minimum rt gaussian fit r^2 value for Factor to be considered of passing quality. Default = 0.9.
        dt_r2_cutoff (float): Minimum dt gaussian fit r^2 value for Factor to be considered of passing quality. Default = 0.9.

    Returns:
        new_factor_list (list of Factor objects): List of factors filtered for rt and dt gaussian likeness, 
            return original factor list if fit quality scores are low.

    """
    filtered_factors = []

    for factor in factor_list:
        if factor.rt_gauss_fit_r2 >= rt_r2_cutoff:
            if factor.dt_gauss_fit_r2 >= dt_r2_cutoff:
                filtered_factors.append(factor)

    new_factor_list = filtered_factors
    if len(filtered_factors) == 0:
        new_factor_list = factor_list

    return new_factor_list


def create_factor_data_object(data_tensor, gauss_params, timepoint_label=None):
    """Stores DataTensor and subsequent Factor object attributes to dictionary.

        Args:
            data_tensor (DataTensor): hdx_limit.core.datatypes.DataTensor that has run the factorize() function.
            gauss_params (tuple of two floats): Gaussian smoothing factors for rt and dt dimensions, in order.
            timepoint_label (obj): A label for identifying the hdx timepoint of the represented tensor, any type. 

        Returns:
            factor_data_dict (dict): A dictionary containing attributes from the parent DataTensor and its Factors.

    """
    factor_data_dict = {
        "name": data_tensor.DataTensor.name,
        "charge_state": data_tensor.DataTensor.charge_states[0],
        "timepoint_index": data_tensor.DataTensor.timepoint_idx,
        "timepoint_label": timepoint_label,
        "retention_labels": data_tensor.DataTensor.retention_labels,
        "drift_labels": data_tensor.DataTensor.drift_labels,
        "mz_labels": data_tensor.DataTensor.mz_labels,
        "bins_per_isotope_peak": data_tensor.DataTensor.bins_per_isotope_peak,
        "tensor_3d_grid": data_tensor.DataTensor.full_grid_out,
        "gauss_params": gauss_params,
        "num_factors": len(data_tensor.DataTensor.factors),
        "factors": []
    }

    for num, factor in enumerate(data_tensor.DataTensor.factors):
        factor_dict = {
            "factor_num": factor.factor_idx,
            "factor_dt": factor.dts,
            "factor_rt": factor.rts,
            "factor_mz": factor.mz_data,
            "factor_integrated_mz": factor.integrated_mz_data
        }
        factor_data_dict["factors"].append(factor_dict)

    return factor_data_dict


def generate_tensor_factors(tensor_fpath,
                            library_info_df,
                            timepoint_index,
                            gauss_params,
                            mz_centers,
                            normalization_factor,
                            num_factors_guess=5,
                            init_method="nndsvd",
                            niter_max=100000,
                            tol=1e-8,
                            factor_corr_threshold=0.17,
                            factor_output_fpath=None,
                            factor_plot_output_path=None,
                            timepoint_label=None,
                            filter_factors=False,
                            factor_rt_r2_cutoff=0.90,
                            factor_dt_r2_cutoff=0.90):
    """Generates a DataTensor from values extracted from a .mzML, along with several analytical parameters.

        Args:
            tensor_fpath (str): /path/to/file of values extracted from .mzML.
            library_info_df (pandas DataFrame): Open DataFrame from library_info.json.
            timepoint_index (int): Index of hdx_timepoint in config.yaml["timepoints"] list.
            gauss_params (tuple of two floats): Gaussian smoothing factors for rt and dt dimensions, in order.
            mz_centers (list of floats): List containing the expected centers of isotopic peaks for a given protein.
            normalization_factor (float): Factor to multiply signal by to allow comparison with signals from other MS-runs.
            n_factors (int): Starting number of Factors to decompose the DataTensor into, 
                if resulting Factors are too correlated the number is iteratively decreased.
            factor_output_fpath (str): Optional argument to define an output path for the factor_data_dict.
            factor_plot_output_path (str): Optional argument to define an output path for a plot of Factor information.
            timepoint_label (obj): Type flexible descriptor of hdx timepoint.
            filter_factors (bool): Option to filter factors by quality of rt and dt gaussian character.
            factor_rt_r2_cutoff (float): Minimum rt gaussian fit r^2 value for Factor to be considered of passing quality. Default = 0.9.
            factor_dt_r2_cutoff (float): Minimum dt gaussian fit r^2 value for Factor to be considered of passing quality. Default = 0.9.

        Returns:
            data_tensor (DataTensor): Returns the DataTensor created from tensor_fpath.

    """
    # Memory use calculation.
    process = psutil.Process(os.getpid())

    print("Pre-Tensor-Initialization: " + str(process.memory_info().rss /
                                              (1024 * 1024 * 1024)))

    data_tensor = TensorGenerator(filename=tensor_fpath,
                                  library_info=library_info_df,
                                  timepoint_index=timepoint_index,
                                  mz_centers=mz_centers,
                                  normalization_factor=normalization_factor
                                  )

    print("Post-Tensor-Pre-Factor-Initialization: " + str(process.memory_info().rss /
                                                          (1024 * 1024 * 1024)))

    print("Factorizing ... ")

    data_tensor.DataTensor.factorize(num_factors_guess=num_factors_guess,
                                     init_method=init_method,
                                     niter_max=niter_max,
                                     tol=tol,
                                     factor_corr_threshold=factor_corr_threshold)

    print("Post-Factorization: " + str(process.memory_info().rss /
                                       (1024 * 1024 * 1024)))

    data_tensor.gauss_params = gauss_params

    if filter_factors:
        filtered_factors = filter_factors_on_rt_dt_gauss_fit(factor_list=data_tensor.DataTensor.factors,
                                                             rt_r2_cutoff=factor_rt_r2_cutoff,
                                                             dt_r2_cutoff=factor_dt_r2_cutoff)
        data_tensor.DataTensor.factors = filtered_factors

    if factor_output_fpath != None:
        factor_data_dictionary = create_factor_data_object(data_tensor=data_tensor,
                                                           gauss_params=gauss_params,
                                                           timepoint_label=timepoint_label)
        io.limit_write(factor_data_dictionary, factor_output_fpath)

    if factor_plot_output_path != None:
        plot_factor_data_from_data_tensor(data_tensor=data_tensor,
                                          output_path=factor_plot_output_path)

    return data_tensor


class TensorGenerator:
    """A class that generates required inputs to create a DataTensor object.

    Attributes:
        Class:
            hd_mass_diff (float): Mass difference between protium and deuterium.
            c13_mass_diff (float): Mass difference between Carbon-12 and Carbon-13.
        Instance:
            filename (str): path/to/file containing information extracted from .mzML for a specific library protein.
            timepoint_index (int): Index of this tensor"s hdx_timepoint in config.yaml["timepoints"]. 
            library_info (Pandas DataFrame): Open DataFrame of library_info.json. 
            mz_centers (list of floats): List of expected isotopic peak centers in m/Z for a given protein.
            normalization_factor (float): Factor to multiply signal by to allow comparison with signals from other MS-runs.
            low_mass_margin (int): Number of mass bins to prepend to the window of m/Z displayed in plots.
            high_mass_margin (int): Number of mass bins to append to the window of m/Z displayed in plots
            ppm_radius (int): Area around expected isotpe peak center to include in integration, defined in ppm-error.
            bins_per_isotope_peak (int): Number of bins to use in representing each isotope peak.
            tensor (numpy ndarray): Raw values extracted from .mzML. 
            name (str): Name of rt-group charged species represented belongs to.
            charge (int): Charge state of species.
            lib_idx (int): Row index of species represented in library_info.
            my_row (pd.DataFrame): Single-row slice for charged species from library_info.
            max_peak_center (int): Number of backbone hydrogens available for deuteration, determined by sequence length.
            total_isotopes (int): Number of isotope-peak-widths from the base-peak to max_peak_center + high_mass_margin.
            total_mass_window (int): Number of isotope-peak-widths for final integrated m/Z dimension. Equal to total_isotopes + low_mass_margin.
            mz_lows (list of floats): List of m/Z positions for lower bounds of m/Z integration windows centered around mz_centers.
            mz_highs (list of floats): List of m/Z positions for upper bounds of m/Z integration windows centered around mz_centers.
            integrated_mz_limits (list of two lists of floats): Lists of high and low integration bounds around mz_centers including error bounding.
            DataTensor (DataTensor): DataTensor object resulting from __init__ method.


    """
    hd_mass_diff = 1.006277
    c13_mass_diff = 1.00335

    def __init__(self, filename, timepoint_index, library_info, mz_centers, normalization_factor,
                 **kwargs):
        """Initializes the TensorGenerator object to create a DataTensor.

        Args:
            filename (str): path/to/file containing information extracted from .mzML for a specific library protein.
            timepoint_index (int): Index of this tensor"s hdx_timepoint in config.yaml["timepoints"]. 
            library_info (Pandas DataFrame): Open DataFrame of library_info.json. 
            mz_centers (list of floats): List of expected isotopic peak centers in m/Z for a given protein.
            normalization_factor (float): Factor to multiply signal by to allow comparison with signals from other MS-runs.
            low_mass_margin (int, optional): Number of mass bins to prepend to the window of m/Z displayed in plots.
            high_mass_margin (int, optional): Number of mass bins to append to the window of m/Z displayed in plots
            ppm_radius (int, optional): Area around expected isotpe peak center to include in integration, defined in ppm-error.
            bins_per_isotope_peak (int, optional): Number of bins to use in representing each isotope peak.

        """
        self.filename = filename
        self.timepoint_index = timepoint_index
        self.library_info = library_info
        self.mz_centers = mz_centers
        self.normalization_factor = normalization_factor

        if (
                kwargs is not None
        ):
            for key in kwargs.keys():
                setattr(self, key, kwargs[key])

        if not hasattr(self, "low_mass_margin"):
            self.low_mass_margin = 10
        if not hasattr(self, "high_mass_margin"):
            self.high_mass_margin = 17
        if not hasattr(self, "ppm_radius"):
            self.ppm_radius = 30
        if not hasattr(self, "bins_per_isotope_peak"):
            self.bins_per_isotope_peak = 7

        self.tensor = io.limit_read(self.filename)
        self.name = filename.split("/")[
            -2]  # Expects format: path/to/{rt-group-name}/{rt-group-name}_{charge}_{file.mzML.gz}.cpickle.zlib.
        self.charge = int([item[6:] for item in filename.split("/")[-1].split("_") if "charge" in item][
                              0])  # Finds by keyword and strips text.
        self.lib_idx = self.library_info.loc[
            (library_info["name"] == self.name) & (library_info["charge"] == self.charge)].index
        self.my_row = self.library_info.loc[
            (library_info["name"] == self.name) & (library_info["charge"] == self.charge)]
        self.max_peak_center = len(self.library_info.loc[self.library_info["name"] == self.name]["sequence"].values[0])
        self.total_isotopes = self.max_peak_center + self.high_mass_margin
        self.total_mass_window = self.low_mass_margin + self.total_isotopes

        self.mz_lows = self.my_row["expect_mz"].values[0] - (
                self.low_mass_margin / self.charge)
        self.mz_highs = self.my_row["expect_mz"].values[0] + (
                self.total_isotopes / self.charge)

        low_mz_limits = [center * ((1000000.0 - self.ppm_radius) / 1000000.0) for center in self.mz_centers]
        high_mz_limits = [center * ((1000000.0 + self.ppm_radius) / 1000000.0) for center in self.mz_centers]

        self.integrated_mz_limits = np.stack((low_mz_limits, high_mz_limits)).T

        # Instantitate DataTensor
        self.DataTensor = datatypes.DataTensor(
            source_file=self.filename,
            tensor_idx=self.lib_idx,
            timepoint_idx=self.timepoint_index,
            name=self.name,
            total_mass_window=self.total_mass_window,
            n_concatenated=1,
            charge_states=[self.charge],
            rts=self.tensor[0],
            dts=self.tensor[1],
            seq_out=self.tensor[2],
            int_seq_out=None,
            integrated_mz_limits=self.integrated_mz_limits,
            bins_per_isotope_peak=self.bins_per_isotope_peak,
            normalization_factor=self.normalization_factor
        )

        # self.DataTensor.lows = searchsorted(self.DataTensor.mz_labels,
        #                                    self.low_lims)
        # self.DataTensor.highs = searchsorted(self.DataTensor.mz_labels,
        #                                     self.high_lims)

        # Consider separating factorize from init
        # self.DataTensor.factorize(gauss_params=(3,1))


class PathOptimizer:
    """Generates a best-estimate mass-addition timeseries from candidate IsotopeClusters from each hdx timepoint.

    Optimizes a timeseries of IsotopeClusters by bootstrapping a set of starting timeseries
    based on plausible trajectories though the integrated mz dimension. Then iteratively uses 
    a set of timeseries scoring functions to make the best single substitution until each starting 
    series is optimized for score. Timeseries with best score at the end of all minimizations is 
    selected as the winning path, which is output along with the alternatives for each timepoint.

    Attributes:
        baseline_peak_error_weight (float): Weight coefficient for the baseline_peak_error score.
        delta_mz_rate_backward_weight (float): Weight coefficient for the delta_mz_rate_backward score.
        delta_mz_rate_forward_weight (float): Weight coefficient for the delta_mz_rate_forward score.
        dt_ground_rmse_weight (float): Weight coefficient for the dt_ground_rmse score.
        dt_ground_fit_weight (float): Weight coefficient for the dt_ground_fit score.
        rt_ground_fit_weight (float): Weight coefficient for the rt_ground_fit score.
        rt_ground_rmse_weight (float): Weight coefficient for the rt_ground_rmse score.
        auc_ground_rmse_weight (float): Weight coefficient for the auc_ground_rmse score.
        rmses_sum_weight (float): Weight coefficient for the rmses_sum score.
        int_mz_FWHM_rmse_weight (float): Weight coefficient for the int_mz_FWHM_rmse score.
        nearest_neighbor_penalty_weight (float): Weight coefficient for the nearest_neighbor_penalty score.
        name (str): Name of rt-group represented.
        all_tp_clusters (list of strings): List of filenames for ICs from all charge states in all timepoints.
        library_info (Pandas DataFrame): Open DataFrame of library_info.json.
        prefilter (int): Indicates wether or not to filter all_tp_clusters before optimization, 1 will prefilter, 0 will not. Default 1.
        timepoints (dict): Dictionary with "timepoints" key containing a list of integers representing hdx times in seconds,
                            and a key for each integer in that list corresponding to lists of hdx-timepoint replicate filepaths.
        n_undeut_runs (int): Number of undeuterated replicates.
        max_peak_center (int): Number of deuteration prone backbone atoms as determined by sequence length.
        old_data_dir (str): Path to directory containing Gabe"s data dictionary pickles. For development use - remove for release.
        old_files (list of strings): List of Gabe"s dict.pickle file paths from old_data_dir. For development use - remove for release.
        rt_com_cv (float): The coefficient of variation (std.dev./mean) in center of mass for the rt dimension.
        dt_com_cv (float): The coefficient of variation (std.dev./mean) in the center of mass for the dt dimension.
        rt_error_rmse (float): Root-mean-squared error of rt centers of mass from undeuterated. Deprecated.
        dt_error_rmse (float): Root-mean-squared error of dt centers of mass from undeuterated. Deprecated.
        prefiltered_ics (list of lists of IsotopeClusters): IsotopeClusters for all timepoints after prefiltering.
    
    """

    def __init__(self,
                 name,
                 all_tp_clusters,
                 library_info,
                 timepoints,
                 n_undeut_runs,
                 user_prefilter,
                 thresholds,
                 pareto_prefilter=True,
                 old_data_dir=None,
                 use_rtdt_recenter=True,
                 **kwargs):
        """Initializes an instance of PathOptimizer, performs preprocessing of inputs so the returned object is ready for optimization.

        The __init__ method sets instance attributes before selecting an undeuterated signal for each charge state of the rt-group being
        included in the PathOptimizer object. These "ground truth" undeuterated signals are selected based on their similarity to the 
        theoretical undeuterated isotope distribution for the protein"s sequence. Deuterated signals are scored by their multidimensional 
        agreement with these undeuterated signals and those scores are saved for use in optimization. A weak pareto dominance filter can
        optionally be applied to IsotopeClusters of the same timepoint before creating a set of bootstrapped mass-addition timeseries based 
        on plausible deuteration curves populated by signals that most closely match the mass uptake at each timepoint. The bootstrapped 
        timeseries are independently optimized with a greedy optimization scheme that makes the single best replacement for the timeseries
        at each iteration based on a set of weighted score terms. The optimization ends when no improvement can be made with any single
        substitution, and the best scoring timeseries overall is kept as the winner.

        Args:
            name (str): Name of rt-group represented.
            all_tp_clusters (list of strings): List of filenames for ICs from all charge states in all timepoints.
            library_info (Pandas DataFrame): Open DataFrame of library_info.json.
            timepoints (dict): Dictionary with "timepoints" key containing a list of integers representing hdx times in seconds,
                                and a key for each integer in that list corresponding to lists of hdx-timepoint replicate filepaths.
            n_undeut_runs (int): Number of undeuterated replicates.
            prefilter (int): Indicates wether or not to filter all_tp_clusters before optimization, 1 will prefilter, 0 will not. Default 1.
            
        """
        # Set score weights
        self.baseline_peak_error_weight = 10  # 100 before
        self.delta_mz_rate_backward_weight = 0.165
        self.delta_mz_rate_forward_weight = 0.162
        self.dt_ground_rmse_weight = 7.721
        self.dt_ground_fit_weight = 13.277
        self.rt_ground_fit_weight = 1.304
        self.rt_ground_rmse_weight = 3.859
        self.auc_ground_rmse_weight = 5.045
        self.rmses_sum_weight = 0.242
        self.int_mz_FWHM_rmse_weight = 0.072
        self.nearest_neighbor_penalty_weight = 0.151

        self.name = name
        self.all_tp_clusters = all_tp_clusters
        self.library_info = library_info
        self.timepoints = timepoints
        self.n_undeut_runs = n_undeut_runs
        self.thresholds = thresholds

        self.old_data_dir = old_data_dir
        self.old_files = None

        self.rt_com_cv = None
        self.dt_com_cv = None
        self.rt_error_rmse = None
        self.dt_error_rmse = None

        self.use_rtdt_center = use_rtdt_recenter

        self.select_undeuterated() # selects best undeuterated per charge state

        self.prefiltered_ics = self.prefilter_based_on_charge() # prefilters ics with charges not present in the undeuterated ics set

        self.precalculate_fit_to_ground() # compute ics features, dt_err, rt_err, dt_fit and rt_fit

        self.first_center = self.undeuts[0].baseline_integrated_mz_com
        self.max_peak_center = len(
            self.library_info.loc[self.library_info["name"] ==
                                  self.name]["sequence"].values[0]
        ) - self.library_info.loc[self.library_info["name"] ==
                                  self.name]["sequence"].values[0][2:].count("P") - 2 + self.first_center

        self.gather_old_data()

        if user_prefilter:
            self.filters_from_user()
            if pareto_prefilter and len(self.prefiltered_ics) >= self.thresholds["min_timepoints"]:
                self.prefiltered_ics = self.weak_pareto_dom_filter()
        elif pareto_prefilter:
            self.prefiltered_ics = self.weak_pareto_dom_filter()
        self.generate_sample_paths()

    def select_undeuterated(self):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """

        """
        Selects undeuterated isotope cluster which best matches theoretically calculated isotope distribution for POI sequence, for each observed charge state of the POI
        all_tp_clusters = TensorGenerator attribute, e.g. T1 = TensorGenerator(...); select_undeuterated(T1.all_tp_clusters)
        n_undeut_runs = number of undeuterated HDX runs included in the "library_info" master csv
        """

        self.undeuts = [ic for ic in self.all_tp_clusters[0] if (ic.idotp > 0.96) and
                        (ic.auc > 10) and (np.sum(ic.rts) > 0) and (np.sum(ic.dts) > 0)]

        l = []
        for ic in self.undeuts:
            ic.dt_ground_err = 100 * abs(ic.dt_coms - len(ic.drift_labels) / 2) * (
                    ic.drift_labels[1] - ic.drift_labels[0]) / ic.drift_labels[
                                   len(ic.drift_labels) // 2]
            ic.rt_ground_err = abs(ic.rt_com - len(ic.retention_labels) / 2) * (
                    ic.retention_labels[1] - ic.retention_labels[0]) * 60

            # Append ic [0], error as a function of deviation of dt and rt center [1], and charge state [2]
            l.append([ic, ic.rt_ground_err / ic.retention_labels[len(ic.retention_labels) // 2] + ic.dt_ground_err,
                      ic.charge_states[0]])

        # Convert in numpy array
        array = np.array(l)

        out, charge_fits = {}, {}
        for charge in np.unique(array[:, 2]):
            tmp_array = array[array[:, 2] == charge]
            best_idx = np.argmin(tmp_array[:, 1])
            out[charge] = tmp_array[best_idx][0]
            charge_fits[charge] = tmp_array[best_idx][0].idotp

        self.undeut_grounds = out
        self.undeut_ground_dot_products = charge_fits

        self.undeuts = [self.undeut_grounds[charge] for charge in self.undeut_grounds]

    def prefilter_based_on_charge(self):

        atc = [self.undeuts]
        for tp in self.all_tp_clusters[1:]:
            ics = []
            for ic in tp:
                if ic.charge_states[0] in self.undeut_grounds:
                    ics.append(ic)
            atc.append(ics)

        return atc

    def precalculate_fit_to_ground(self,
                                   prefiltered_ics=None,
                                   undeut_grounds=None,
                                   ):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        if prefiltered_ics is None:
            prefiltered_ics = self.prefiltered_ics
        if undeut_grounds is None:
            undeut_grounds = self.undeut_grounds

        for timepoint in prefiltered_ics:
            for ic in timepoint:

                undeut = undeut_grounds[ic.charge_states[0]]

                # rt_ground_err: retention time error in seconds
                # dt_ground_err: drift time error in percentage of deviation * 100

                if self.use_rtdt_center:

                    ic.dt_ground_err = 100 * abs(ic.dt_coms - len(undeut.drift_labels) / 2) * (
                            undeut.drift_labels[1] - undeut.drift_labels[0]) / undeut.drift_labels[
                                           len(undeut.drift_labels) // 2]
                    ic.rt_ground_err = abs(ic.rt_com - len(undeut.retention_labels) / 2) * (
                            undeut.retention_labels[1] - undeut.retention_labels[0]) * 60

                else:
                    ic.dt_ground_err = 100 * abs(ic.dt_coms - undeut.dt_coms) * (
                            undeut.drift_labels[1] - undeut.drift_labels[0]) / undeut.drift_labels[
                                           len(undeut.drift_labels) // 2]
                    ic.rt_ground_err = abs(ic.rt_com - undeut.rt_com) * (
                            undeut.retention_labels[1] - undeut.retention_labels[0]) * 60

                ic.auc_ground_err = ic.log_baseline_auc - undeut.log_baseline_auc
                ic.dt_ground_fit = max(
                    np.correlate(undeut.dt_norms[0], ic.dt_norms[0], mode="full"))
                ic.rt_ground_fit = max(np.correlate(undeut.rt_norm, ic.rt_norm, mode="full"))

                # these are pre calculated in the factor class
                # ic.dt_gaussian_rmse = self.rmse_from_gaussian_fit(ic.dt_norms[0])
                # ic.rt_gaussian_rmse = self.rmse_from_gaussian_fit(ic.rt_norm)

                ic.log_baseline_auc_diff = ic.log_baseline_auc - undeut.log_baseline_auc

    def filters_from_user(self):
        """Description of function.

        Args:
        arg_name (type): Description of input variable.

        Returns:
        out_name (type): Description of any returned objects.

        """

        filtered_atc = [
            [ic for ic in ics if (ic.baseline_peak_error <= self.thresholds["baseline_peak_error"] and
                                  ic.dt_ground_err <= self.thresholds["dt_ground_err"] and
                                  ic.dt_ground_fit >= self.thresholds["dt_ground_fit"] and
                                  ic.rt_ground_err <= self.thresholds["rt_ground_err"] and
                                  ic.rt_ground_fit >= self.thresholds["rt_ground_fit"] and
                                  ic.baseline_integrated_mz_rmse <= self.thresholds[
                                      "baseline_integrated_rmse"] and
                                  ic.baseline_integrated_mz_FWHM >= self.thresholds[
                                      "baseline_integrated_FWHM"] and
                                  ic.nearest_neighbor_correlation >= self.thresholds["nearest_neighbor_correlation"] and
                                  ic.charge_states[0] in self.undeut_grounds
                                  and ic.baseline_integrated_mz_com <= self.max_peak_center)
             ] for ics in self.prefiltered_ics[1:]
        ] #if ics[0].timepoint_idx in self.timepoints] # TODO not sure the impact of removing this statement
        filtered_atc = np.array([self.undeuts] + filtered_atc, dtype=object)
        filtered_indexes = np.array([True if len(ics) > 0 else False for ics in filtered_atc])
        self.prefiltered_ics = list(filtered_atc[filtered_indexes])
        self.timepoints = list(np.array(self.timepoints)[filtered_indexes])

    def weak_pareto_dom_filter(self):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        self.prefiltered_ics[0] = [self.undeut_grounds[charge] for charge in self.undeut_grounds]
        out = []
        for tp in self.prefiltered_ics:
            tp_buffer = []
            for ic1 in tp:
                ic1_int_mz_com = np.round(ic1.baseline_integrated_mz_com)
                compare_flag = False
                for ic2 in tp:
                    if (
                            np.round(ic2.baseline_integrated_mz_com) == ic1_int_mz_com and
                            ic2.rt_ground_err ** 2 < ic1.rt_ground_err ** 2 and
                            ic2.dt_ground_err ** 2 < ic1.dt_ground_err ** 2 and
                            ic2.baseline_peak_error < ic1.baseline_peak_error and
                            ic2.rt_ground_fit > ic1.rt_ground_fit and
                            ic2.dt_ground_fit > ic1.dt_ground_fit and
                            ic2.auc_ground_err ** 2 < ic1.auc_ground_err ** 2
                    ):
                        compare_flag = True
                        break
                if not compare_flag:
                    tp_buffer.append(ic1)
            out.append(tp_buffer)
        return out

    def gather_old_data(self):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        if self.old_data_dir is not None:
            self.old_files = sorted([
                fn for fn in glob.iglob(self.old_data_dir + "*.pickle")
                if "_".join(self.name.split("/")[-1].split("_")[:3]) in fn
            ])
            self.old_data = []
            for fn in self.old_files:
                ts = pk.load(open(fn, "rb"))
                ts["charge"] = int(fn.split(".")[-3][-1])
                ts["delta_mz_rate"] = self.gabe_delta_mz_rate(ts["centroid"])
                ts["major_species_widths"] = [
                    len(np.nonzero(x)[0])
                    for x in ts["major_species_integrated_intensities"]
                ]
                self.old_data.append(ts)

    def gaussian_function(self, x, H, A, x0, sigma):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def gauss_fit(self, x, y):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        nonzeros = [index for index, value in enumerate(list(y)) if value != 0]
        popt, pcov = curve_fit(self.gaussian_function, x, y, p0=[0, max(y), mean, sigma],
                               bounds=([0, 0, nonzeros[0], 0], [np.inf, np.inf, nonzeros[-1], np.inf]))
        return popt

    def rmse_from_gaussian_fit(self, distribution):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        try:
            xdata = [i for i in range(len(distribution))]
            ydata = distribution
            y_gaussian_fit = self.gaussian_function(xdata, *self.gauss_fit(xdata, ydata))
            rmse = mean_squared_error(ydata / max(ydata), y_gaussian_fit / max(y_gaussian_fit), squared=False)
            return rmse
        except:
            return 100

    def generate_sample_paths(self):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        starts = np.linspace(0, 0.7, 8)
        sample_paths = []
        for start in starts:
            slopes = np.logspace(-0.5, 1.8, int(16 * (1.0 - start)))
            for slope in slopes:
                sample_paths.append(self.clusters_close_to_line(start, slope))

        self.sample_paths = [list(path) for path in set(sample_paths)]

    def clusters_close_to_line(
            self,
            start,
            slope,
            undeut_grounds=None,
            prefiltered_ics=None,
            max_peak_center=None,
    ):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        if undeut_grounds is None:
            undeut_grounds = self.undeut_grounds
        if prefiltered_ics is None:
            prefiltered_ics = self.prefiltered_ics
        if max_peak_center is None:
            max_peak_center = self.max_peak_center

        path = []
        uds = []
        fits = []
        # pick best-fitting undeut charge-state for all paths
        for key in undeut_grounds.keys():
            uds.append(undeut_grounds[key])
            fits.append(undeut_grounds[key].idotp)
        path.append(uds[np.argmax(fits)])

        # Relies on use of prefiltered_ics naming convention, when prefilter = 0, all_tp_clusters should have the first n_undeut_runs collapsed into a single list and be named and passed as prefiltered_ics
        xs = np.arange(len(prefiltered_ics))
        expected_centers = 3 + ((start + (1.0 - start) *
                                 (1 - np.exp(-xs * slope / len(xs)))) *
                                (max_peak_center - 3) * 0.85)

        # prefiltered_ics always has length n_deut_runs+1, all undeuterated are collapsed into PO.prefiltered_ics[0]
        for tp in range(1, len(xs)):
            try:
                peak_dist_rankings = []
                for ic in prefiltered_ics[tp]:
                    if ic.baseline_integrated_mz_com > max_peak_center:
                        continue
                    peak_dist_rankings.append((
                        (abs(ic.baseline_integrated_mz_com -
                             expected_centers[tp])),
                        ic,
                    ))
                peak_dist_rankings = sorted(peak_dist_rankings,
                                            key=lambda x: x[0])
                path.append(peak_dist_rankings[0][1])
            except:
                if len(peak_dist_rankings) == 0:
                    l = len(prefiltered_ics[tp])
                    if l > 0:
                        if l == 1:
                            # pick 0
                            path.append(prefiltered_ics[tp][0])
                        else:
                            # pick lowest mz
                            path.append(
                                sorted(
                                    prefiltered_ics[tp],
                                    key=lambda ic: ic.
                                        baseline_integrated_mz_com,
                                )[0])
                    else:
                        # No ics in tp, print message and append path[-1]
                        import os
                        import sys

                        # print("len(PO.prefiltered_ics["+str(tp)+"]) == 0")
                        path.append(path[-1])

        return tuple(path)

    def optimize_paths_multi(self, sample_paths=None, prefiltered_ics=None):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        # Main function of PO, returns the best-scoring HDX IC time-series "path" of a set of bootstrapped paths.

        if sample_paths is None:
            sample_paths = self.sample_paths
        if prefiltered_ics is None:
            prefiltered_ics = self.prefiltered_ics

        final_paths = []
        for sample in sample_paths:
            current = copy.copy(sample)
            edited = True
            while edited:

                edited = False
                n_changes = 0
                ic_indices = []
                alt_paths = []

                for tp in range(1, len(current)):
                    for ic in prefiltered_ics[tp]:
                        buffr = copy.copy(current)
                        buffr[tp] = ic
                        alt_paths.append(buffr)

                # Decorate alt_paths
                combo_scoring = []
                for path in alt_paths:
                    combo_scoring.append(self.combo_score_multi(path))

                if min(combo_scoring) < self.combo_score_multi(current):
                    current = alt_paths[combo_scoring.index(min(combo_scoring))]
                    n_changes += 1
                    edited = True

                current_score = self.combo_score_multi(current)

                if edited == False:
                    final_paths.append(current)

        final_scores = []
        for path in final_paths:
            final_scores.append(self.combo_score_multi(path))

        # This order must be maintained, self.winner must exist before calling find_runners; winner and runners are both needed for set_bokeh tuple
        self.winner = final_paths[final_scores.index(min(final_scores))]
        self.winner_scores = self.report_score_multi(self.winner)
        self.find_runners_multi()
        self.set_bokeh_tuples()
        self.filter_runners()
        # Compute the coefficient of variation (std.dev./mean) for RT and DT dimensions.
        self.rt_com_cv = (np.var([ic.rt_com for ic in self.winner if ic.rt_com is not None]) **
                          0.5) / np.mean([ic.rt_com for ic in self.winner if ic.rt_com is not None])
        self.dt_com_cv = (np.var([
            np.mean(ic.dt_coms) for ic in self.winner if ic.dt_coms is not None
        ]) ** 0.5) / np.mean([np.mean(ic.dt_coms) for ic in self.winner if ic.dt_coms is not None])

    def find_runners_multi(self):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        # sets self.runners atr. sorts "runner-up" single substitutions for each tp by score, lower is better.
        winner = self.winner
        prefiltered_ics = self.prefiltered_ics

        runners = []
        for tp in range(len(winner)):

            alt_paths = []
            for ic in prefiltered_ics[tp]:
                if ic is not winner[tp]:
                    buffr = copy.copy(winner)
                    buffr[tp] = ic
                    alt_paths.append(buffr)

            combo_scoring = []
            for path in alt_paths:
                combo_scoring.append(self.combo_score_multi(path))

            out_buffer = []
            for i in range(len(combo_scoring)):
                min_idx = combo_scoring.index(min(combo_scoring))
                out_buffer.append(alt_paths[min_idx][tp])
                alt_paths.pop(min_idx)
                combo_scoring.pop(min_idx)
            runners.append(out_buffer)

        self.runners = runners

    def optimize_paths_mono(self, sample_paths=None, prefiltered_ics=None):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        # Main function of PO, returns the best-scoring HDX IC time-series "path" of a set of bootstrapped paths.

        if sample_paths is None:
            sample_paths = self.sample_paths
        if prefiltered_ics is None:
            prefiltered_ics = self.prefiltered_ics

        final_paths = []
        for sample in sample_paths[:1]:
            current = copy.copy(sample)
            edited = True
            while edited:

                edited = False
                n_changes = 0
                ic_indices = []
                alt_paths = []

                for tp in range(1, len(current)):
                    for ic in prefiltered_ics[tp]:
                        buffr = copy.copy(current)
                        buffr[tp] = ic
                        alt_paths.append(buffr)

                # Decorate alt_paths
                combo_scoring = []
                for pth in alt_paths:
                    combo_scoring.append(self.combo_score_mono(pth))

                if min(combo_scoring) < self.combo_score_mono(current):
                    current = alt_paths[combo_scoring.index(min(combo_scoring))]
                    n_changes += 1
                    edited = True

                current_score = self.combo_score_mono(current)

                if edited == False:
                    final_paths.append(current)
        final_scores = []
        for pth in final_paths:
            final_scores.append(self.combo_score_mono(pth))

        # This order must be maintained, self.winner must exist before calling find_runners; winner and runners are both needed for set_bokeh tuple
        self.winner = final_paths[final_scores.index(min(final_scores))]
        self.winner_scores = self.report_score_mono(self.winner)
        self.find_runners_mono()
        self.set_bokeh_tuples()
        self.filter_runners()
        self.rt_com_cv = (np.var([ic.rt_com for ic in self.winner if ic.rt_com is not None]) **
                          0.5) / np.mean([ic.rt_com for ic in self.winner if ic.rt_com is not None])
        self.dt_com_cv = (np.var([
            np.mean(ic.dt_coms) for ic in self.winner if ic.dt_coms is not None
        ]) ** 0.5) / np.mean([np.mean(ic.dt_coms) for ic in self.winner if ic.dt_coms is not None])
        # Doesn"t return, only sets PO attributes

    def find_runners_mono(self):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        # sets self.runners atr. sorts "runner-up" single substitutions for each tp by score, lower is better.
        winner = self.winner
        prefiltered_ics = self.prefiltered_ics

        runners = []
        for tp in range(len(winner)):

            alt_paths = []
            for ic in prefiltered_ics[tp]:
                if ic is not winner[tp]:
                    buffr = copy.copy(winner)
                    buffr[tp] = ic
                    alt_paths.append(buffr)

            combo_scoring = []
            for pth in alt_paths:
                combo_scoring.append(self.combo_score_mono(pth))

            out_buffer = []
            for i in range(len(combo_scoring)):
                min_idx = combo_scoring.index(min(combo_scoring))
                out_buffer.append(alt_paths[min_idx][tp])
                alt_paths.pop(min_idx)
                combo_scoring.pop(min_idx)
            runners.append(out_buffer)

        self.runners = runners

    def set_bokeh_tuples(self):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """

        # Sets IC.bokeh_tuple to be passed to bokeh for display through the HoverTool
        # Winners store the full values of the winning series scores
        # Runners store the differences between the winning scores and the score if they were to be substituted

        def score_dict(series):
            """Description of function.

            Args:
                arg_name (type): Description of input variable.

            Returns:
                out_name (type): Description of any returned objects.

            """
            return {
                "delta_mz_rate_backward":
                    self.delta_mz_rate(series)[0] * self.delta_mz_rate_backward_weight,
                "delta_mz_rate_afterward":
                    self.delta_mz_rate(series)[1] * self.delta_mz_rate_forward_weight,
                "dt_ground_rmse":
                    self.dt_ground_rmse(series) * self.dt_ground_rmse_weight,
                "rt_ground_rmse":
                    self.rt_ground_rmse(series) * self.rt_ground_rmse_weight,
                "dt_ground_fit":
                    self.dt_ground_fit(series) * self.dt_ground_fit_weight,
                "rt_ground_fit":
                    self.rt_ground_fit(series) * self.rt_ground_fit_weight,
                "baseline_peak_error":
                    self.baseline_peak_error(series) *
                    self.baseline_peak_error_weight,
                "auc_ground_rmse":
                    self.auc_ground_rmse(series) * self.auc_ground_rmse_weight,
                "rmses_sum":
                    self.rmses_sum(series) * self.rmses_sum_weight,
                "int_mz_FWHM_rmse":
                    self.int_mz_FWHM_rmse(series) * self.int_mz_FWHM_rmse_weight,
                "nearest_neighbor_penalty":
                    self.nearest_neighbor_penalty(series) * self.nearest_neighbor_penalty_weight,
            }

        def score_diff(winner_scores, substituted_scores):
            """Description of function.

            Args:
                arg_name (type): Description of input variable.

            Returns:
                out_name (type): Description of any returned objects.

            """
            return (
                winner_scores["delta_mz_rate_backward"] -
                substituted_scores["delta_mz_rate_backward"],
                winner_scores["delta_mz_rate_afterward"] -
                substituted_scores["delta_mz_rate_afterward"],
                winner_scores["dt_ground_rmse"] -
                substituted_scores["dt_ground_rmse"],
                winner_scores["rt_ground_rmse"] -
                substituted_scores["rt_ground_rmse"],
                winner_scores["dt_ground_fit"] -
                substituted_scores["dt_ground_fit"],
                winner_scores["rt_ground_fit"] -
                substituted_scores["rt_ground_fit"],
                winner_scores["baseline_peak_error"] -
                substituted_scores["baseline_peak_error"],
                winner_scores["auc_ground_rmse"] -
                substituted_scores["auc_ground_rmse"],
                winner_scores["rmses_sum"]
                - substituted_scores["rmses_sum"],
                winner_scores["int_mz_FWHM_rmse"]
                - substituted_scores["int_mz_FWHM_rmse"],
                winner_scores["nearest_neighbor_penalty"]
                - substituted_scores["nearest_neighbor_penalty"],

                sum([winner_scores[key] for key in winner_scores.keys()]) -
                sum([
                    substituted_scores[key]
                    for key in substituted_scores.keys()
                ]),
            )

        winner = self.winner
        runners = self.runners
        winner_scores = score_dict(winner)

        # Winners store absolute values of scores

        for ic in winner:
            ic.bokeh_tuple = ic.info_tuple + (
                ic.rt_ground_err,
                ic.dt_ground_err,
                winner_scores["delta_mz_rate_backward"],
                winner_scores["delta_mz_rate_afterward"],
                winner_scores["dt_ground_rmse"],
                winner_scores["rt_ground_rmse"],
                winner_scores["dt_ground_fit"],
                winner_scores["rt_ground_fit"],
                winner_scores["baseline_peak_error"],
                winner_scores["auc_ground_rmse"],
                winner_scores["rmses_sum"],
                winner_scores["int_mz_FWHM_rmse"],
                winner_scores["nearest_neighbor_penalty"],
                0,
            )

        # Runners store the differences between the winning scores and their scores when substituted
        # Negative values here mean the Runner causes that score to be worse, as all scores are low-is-better and the diffs are calculated winner-substituted
        for tp in range(len(runners)):
            for ic in runners[tp]:
                substituted_series = copy.copy(winner)
                substituted_series[tp] = ic
                substituted_scores = score_dict(substituted_series)
                ic.bokeh_tuple = (ic.info_tuple +
                                  (ic.rt_ground_err, ic.dt_ground_err) +
                                  score_diff(winner_scores, substituted_scores))

    def filter_runners(self, n_runners=5):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        filtered_runners = []
        for tp in self.runners:
            if len(tp) > n_runners:
                # Sum score diffs and sort index array in descending order by score value
                # Put first n_runners ics into filtered_runners
                tp_scores = [sum(ic.bokeh_tuple[-8:]) for ic in tp]
                tp_idxs = list(range(len(tp)))
                hi_to_lo = sorted(tp_idxs,
                                  key=lambda idx: tp_scores[idx],
                                  reverse=True)
                filtered_runners.append([
                    tp[idx]
                    for idx in hi_to_lo
                    if hi_to_lo.index(idx) < n_runners
                ])
            else:
                # if len(tp)<n_runners just append tp to filtered_runners
                filtered_runners.append(tp)
        self.filtered_runners = filtered_runners

    def calculate_theoretical_isotope_dist_from_sequence(self, sequence, n_isotopes=None):
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

    def calculate_empirical_isotope_dist_from_integrated_mz(self, integrated_mz_array,
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

    def calculate_isotope_dist_dot_product(self, sequence, undeut_integrated_mz_array):
        """Calculate dot product between theoretical isotope distribution from the sequence and experimental integrated mz array.
        
        Args:
            sequence (string): single-letter sequence of the library protein-of-interest
            undeut_integrated_mz_array (Numpy ndarray): observed integrated mz array from an undeuterated .mzML
        Returns:
            dot_product (float): result of dot product between theoretical and observed integrated-m/Z, from [0-1]

        """
        theo_isotope_dist = self.calculate_theoretical_isotope_dist_from_sequence(
            sequence=sequence)
        emp_isotope_dist = self.calculate_empirical_isotope_dist_from_integrated_mz(
            integrated_mz_array=undeut_integrated_mz_array)
        min_length = min([len(theo_isotope_dist), len(emp_isotope_dist)])
        dot_product = np.linalg.norm(
            np.dot(theo_isotope_dist[0:min_length], emp_isotope_dist[0:min_length])
        ) / np.linalg.norm(theo_isotope_dist) / np.linalg.norm(emp_isotope_dist)
        return dot_product

    ##########################################################################################################################################################################################################################################
    ### Scoring Functions for PathOptimizer ##################################################################################################################################################################################################
    ##########################################################################################################################################################################################################################################

    def delta_mz_rate(self, ics, timepoints=None):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        # Two penalizations are computed: [0] if the ic is too fast (sd) and [1] if the ic goes backwards (back)

        if timepoints is None:
            timepoints = self.timepoints

        backward = 0
        forward = 0
        previous_rate = max(
            [(ics[1].baseline_integrated_mz_com - ics[0].baseline_integrated_mz_com) / (timepoints[1] - timepoints[0]),
             0.1])

        for i in range(2, len(ics)):
            # if previous_rate == 0: diagnostic for /0 error
            new_com = ics[i].baseline_integrated_mz_com
            if new_com < ics[
                i - 1].baseline_integrated_mz_com:  # if we went backwards
                backward += (100 *
                             (new_com - ics[i - 1].baseline_integrated_mz_com) ** 2.0
                             )  # penalize for going backwards
                new_com = (
                        ics[i - 1].baseline_integrated_mz_com + 0.01
                )  # pretend we went forwards for calculating current rate
            if round(timepoints[i] - timepoints[i - 1], 2) <= 0.1:
                continue
            current_rate = max([
                (new_com - ics[i - 1].baseline_integrated_mz_com), 0.1
            ]) / (timepoints[i] - timepoints[i - 1])
            if (current_rate / previous_rate) > 1.2:
                forward += (current_rate / previous_rate) ** 2.0
            previous_rate = current_rate
        return backward / len(ics), forward / len(ics)

    def dt_ground_rmse(
            self, ics
    ):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        # rmse penalizes strong single outliers, score is minimized - lower is better
        return math.sqrt(sum([ic.dt_ground_err ** 2 for ic in ics]) / len(ics))

    def rt_ground_rmse(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        return math.sqrt(sum([ic.rt_ground_err ** 2 for ic in ics]) / len(ics))

    def dt_ground_fit(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        return sum([(1.0 / ic.dt_ground_fit) for ic in ics])

    def rt_ground_fit(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        return sum([(1.0 / ic.rt_ground_fit) for ic in ics])

    def baseline_peak_error(self, ics):  # Use RMSE instead TODO
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        # returns avg of peak_errors from baseline subtracted int_mz -> minimize score
        return np.average([ic.baseline_peak_error for ic in ics])

    def auc_ground_rmse(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        return np.sqrt(np.mean([ic.auc_ground_err ** 2 for ic in ics]))

    def auc_rmse(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        sd = 0
        for ic in ics:
            sd += ic.log_baseline_auc_diff ** 2
        return math.sqrt(np.mean(sd))

    def rmses_sum(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        rmses = 0
        for ic in ics:
            rmses += 100 * ic.baseline_integrated_mz_rmse
        return rmses

    def int_mz_FWHM_rmse(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        sd = 0
        for i in range(2, len(ics)):
            sd += (
                          ics[i].baseline_integrated_mz_FWHM - ics[i - 1].baseline_integrated_mz_FWHM
                  ) ** 2.0

        return math.sqrt(sd)

    def nearest_neighbor_penalty(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        nn_penalty = 0
        for ic in ics:
            nn_penalty += 100 * (
                np.min([abs(1.0 - ic.nearest_neighbor_correlation), 0.5])
            ) ** 2.0
        return nn_penalty

    # Eventually put defaults here as else statements
    def set_score_weights(
            self,
            baseline_peak_error_weight=None,
            delta_mz_rate_backward_weight=None,
            delta_mz_rate_forward_weight=None,
            dt_ground_fit_weight=None,
            rt_ground_rmse_weight=None,
            dt_ground_rmse_weight=None,
            auc_ground_rmse_weight=None,
            rmses_sum_weight=None,
            int_mz_FWHM_rmse_weight=None,
            nearest_neighbor_penalty_weight=None,
            rt_ground_fit_weight=None
    ):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """

        if baseline_peak_error_weight != None:
            self.baseline_peak_error_weight = baseline_peak_error_weight
        if delta_mz_rate_backward_weight != None:
            self.delta_mz_rate_backward_weight = delta_mz_rate_backward_weight
        if delta_mz_rate_forward_weight != None:
            self.delta_mz_rate_forward_weight = delta_mz_rate_forward_weight
        if rt_ground_fit_weight != None:
            self.rt_ground_fit_weight = rt_ground_fit_weight
        if dt_ground_fit_weight != None:
            self.dt_ground_fit_weight = dt_ground_fit_weight
        if rt_ground_rmse_weight != None:
            self.rt_ground_rmse_weight = rt_ground_rmse_weight
        if dt_ground_rmse_weight != None:
            self.dt_ground_rmse_weight = dt_ground_rmse_weight
        if auc_ground_rmse_weight != None:
            self.auc_ground_rmse_weight = auc_ground_rmse_weight
        if rmses_sum_weight != None:
            self.rmses_sum_weight = rmses_sum_weight
        if int_mz_FWHM_rmse_weight != None:
            self.int_mz_FWHM_rmse_weight = int_mz_FWHM_rmse_weight
        if nearest_neighbor_penalty_weight != None:
            self.nearest_neighbor_penalty_weight = nearest_neighbor_penalty_weight

    def combo_score_multi(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        return sum([
            self.baseline_peak_error_weight * self.baseline_peak_error(ics),
            self.delta_mz_rate_backward_weight * self.delta_mz_rate(ics)[0],
            self.delta_mz_rate_forward_weight * self.delta_mz_rate(ics)[1],
            self.dt_ground_rmse_weight * self.dt_ground_rmse(ics),
            self.dt_ground_fit_weight * self.dt_ground_fit(ics),
            self.rt_ground_fit_weight * self.rt_ground_fit(ics),
            self.rt_ground_rmse_weight * self.rt_ground_rmse(ics),
            self.auc_ground_rmse_weight * self.auc_ground_rmse(ics),
            self.rmses_sum_weight * self.rmses_sum(ics),
            self.int_mz_FWHM_rmse_weight * self.int_mz_FWHM_rmse(ics),
            self.nearest_neighbor_penalty_weight * self.nearest_neighbor_penalty(ics)
        ])

    def combo_score_mono(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        return sum([
            self.baseline_peak_error_weight * self.baseline_peak_error(ics),
            self.dt_ground_rmse_weight * self.dt_ground_rmse(ics),
            self.dt_ground_fit_weight * self.dt_ground_fit(ics),
            self.rt_ground_fit_weight * self.rt_ground_fit(ics),
            self.rt_ground_rmse_weight * self.rt_ground_rmse(ics),
            self.auc_ground_rmse_weight * self.auc_ground_rmse(ics),
            self.rmses_sum_weight * self.rmses_sum(ics),
            self.nearest_neighbor_penalty_weight * self.nearest_neighbor_penalty(ics),
        ])

    def report_score_multi(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        # TODO Add additional scores to this function

        return {
            "baseline_peak_error": (
                self.baseline_peak_error_weight,
                self.baseline_peak_error(ics),
            ),
            "delta_mz_rate_backard":
                (self.delta_mz_rate_backward_weight, self.delta_mz_rate(ics)[0]),
            "delta_mz_rate_foward":
                (self.delta_mz_rate_forward_weight, self.delta_mz_rate(ics)[1]),
            "dt_ground_rmse": (self.dt_ground_rmse_weight,
                               self.dt_ground_rmse(ics)),
            "dt_ground_fit":
                (self.dt_ground_fit_weight, self.dt_ground_fit(ics)),
            "rt_ground_fit":
                (self.rt_ground_fit_weight, self.rt_ground_fit(ics)),
            "rt_ground_rmse": (self.rt_ground_rmse_weight,
                               self.rt_ground_rmse(ics)),
            "auc_ground_rmse": (self.auc_ground_rmse_weight,
                                self.auc_ground_rmse(ics)),
            "rmses_sum": (self.rmses_sum_weight,
                          self.rmses_sum(ics)),
            "int_mz_FWHM_rmse": (self.int_mz_FWHM_rmse_weight,
                                 self.int_mz_FWHM_rmse(ics)),
            "nearest_neighbor_penalty": (self.nearest_neighbor_penalty_weight,
                                         self.nearest_neighbor_penalty(ics)),
        }

    def report_score_mono(self, ics):
        """Description of function.

        Args:
            arg_name (type): Description of input variable.

        Returns:
            out_name (type): Description of any returned objects.

        """
        # TODO Add additional scores to this function

        return {
            "baseline_peak_error": (
                self.baseline_peak_error_weight,
                self.baseline_peak_error(ics),
            ),
            "dt_ground_rmse": (self.dt_ground_rmse_weight,
                               self.dt_ground_rmse(ics)),
            "dt_ground_fit":
                (self.dt_ground_fit_weight, self.dt_ground_fit(ics)),
            "rt_ground_fit":
                (self.rt_ground_fit_weight, self.rt_ground_fit(ics)),
            "rt_ground_rmse": (self.rt_ground_rmse_weight,
                               self.rt_ground_rmse(ics)),
            "auc_ground_rmse": (self.auc_ground_rmse_weight,
                                self.auc_ground_rmse(ics)),
            "rmses_sum": (self.rmses_sum_weight,
                          self.rmses_sum(ics)),
            "int_mz_FWHM_rmse": (self.int_mz_FWHM_rmse_weight,
                                 self.int_mz_FWHM_rmse(ics)),
            "nearest_neighbor_penalty": (self.nearest_neighbor_penalty_weight,
                                         self.nearest_neighbor_penalty(ics)),
        }
