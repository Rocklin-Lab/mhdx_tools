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
import os
import sys
import yaml
import psutil
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from hdx_limit.core.plot_ics_data import plot_ics_from_ic_list
from hdx_limit.core.processing import TensorGenerator, generate_tensor_factors
from hdx_limit.core.io import limit_write


def main(library_info_path,
         tensor_input_path,
         timepoints_dict,
         normalization_factor,
         isotope_clusters_out_path=None,
         factor_out_path=None,
         factor_plot_output_path=None,
         ic_plot_output_path=None,
         return_flag=False,
         gauss_params=(3, 1),
         n_factors=15,
         init_method='nndsvd',
         niter_max=100000,
         tol=1e-8,
         factor_corr_threshold=0.17,
         filter_factors=False,
         factor_rt_r2_cutoff=0.91,
         factor_dt_r2_cutoff=0.91,
         ic_peak_prominence=0.10,
         auto_ic_peak_width=True,
         ic_peak_width=2,
         ic_rel_height_filter=True,
         ic_rel_height_filter_baseline=0.10,
         ic_rel_height_threshold=0.10,
         use_rtdt_recenter=False):
    """Performs factorization to deconvolute tensor, identifies IsotopeCluster objects, and can return and/or write output list of IsotopeClusters.

    Args:
        library_info_path (str): A path/to/library_info.json.
        tensor_input_path (str): A path/to/tensor.cpickle.zlib.
        timepoints_dict (dict): Dictionary with 'timepoints' key containing list of hdx timepoints in integer seconds, 
            which are keys mapping to lists of each timepoint's replicate .mzML filenames. 
        isotope_clusters_out_path (str): path/to/file for main output - list of IsotopeClusters objects.
        return_flag (bool): Option to return output in python, for notebook context.
        gauss_params (tuple of ints/floats): Gaussian smoothing parameters in LC-RT and IMS-DT dimensions (rt_sigma, dt_sigma).

    Returns:
        out_dict (dict): dictionary containing TensorGenerator object
    
    """
    out_dict = {}
    library_info = pd.read_json(library_info_path)
    my_name = tensor_input_path.split("/")[-2] # Name from protein directory.
    my_charge = int([item[6:] for item in tensor_input_path.split("/")[-1].split("_") if "charge" in item][0]) # Finds by keyword and strip text.
    if use_rtdt_recenter:
        my_row = library_info.loc[(library_info["name_recentered"] == my_name) & (library_info["charge"] == my_charge)]
    else:
        my_row = library_info.loc[(library_info["name"]==my_name) & (library_info["charge"]==my_charge)]
    my_centers = my_row["mz_centers"].values
    centers = my_centers[0]

    # Finds timepoint of passed filename by config comparison.
    for tp in timepoints_dict["timepoints"]:
        for fn in timepoints_dict[tp]:
            if fn in tensor_input_path:
                my_tp = tp

    data_tensor = generate_tensor_factors(tensor_fpath=tensor_input_path,
                                          library_info_df=library_info,
                                          timepoint_index=my_tp,
                                          gauss_params=gauss_params,
                                          n_factors=n_factors,
                                          init_method=init_method,
                                          niter_max=niter_max,
                                          tol=tol,
                                          factor_corr_threshold=factor_corr_threshold,
                                          mz_centers=centers,
                                          normalization_factor=normalization_factor,
                                          factor_output_fpath=factor_out_path,
                                          factor_plot_output_path=factor_plot_output_path,
                                          timepoint_label=None,
                                          filter_factors=filter_factors,
                                          factor_rt_r2_cutoff=factor_rt_r2_cutoff,
                                          factor_dt_r2_cutoff=factor_dt_r2_cutoff,
                                          use_rtdt_recenter=use_rtdt_recenter)

    # set up vars for calculating idotp if 0 timepoint index
    calc_idotp = False
    prot_seq = None
    if my_tp == 0:
        calc_idotp = True
        prot_seq = my_row['sequence'].values[0]

    all_ics = []

    ic_peak_width_auto = 0.8 * my_row['integrated_mz_width'].values[0]

    for factor in data_tensor.DataTensor.factors:

        # Generate isotope cluster class.

        if auto_ic_peak_width:
            factor.find_isotope_clusters(prominence=ic_peak_prominence,
                                         width_val=ic_peak_width_auto,
                                         rel_height_filter=ic_rel_height_filter,
                                         baseline_threshold=ic_rel_height_filter_baseline,
                                         rel_height_threshold=ic_rel_height_threshold,
                                         calculate_idotp=calc_idotp,
                                         sequence=prot_seq)
        else:
            factor.find_isotope_clusters(prominence=ic_peak_prominence,
                                         width_val=ic_peak_width,
                                         rel_height_filter=ic_rel_height_filter,
                                         baseline_threshold=ic_rel_height_filter_baseline,
                                         rel_height_threshold=ic_rel_height_threshold,
                                         calculate_idotp=calc_idotp,
                                         sequence=prot_seq)

        for ic in factor.isotope_clusters:
            all_ics.append(ic)

    if isotope_clusters_out_path is not None:
        limit_write(all_ics, isotope_clusters_out_path)

    if ic_plot_output_path is not None:
        plot_ics_from_ic_list(all_ics, ic_plot_output_path)

    if return_flag:
        out_dict["TensorGenerator"] = data_tensor
        return out_dict


if __name__ == "__main__":

    # Set expected command line arguments.
    parser = argparse.ArgumentParser(
        description=
        "Accepts tensor as input, factorizes and saves IsotopeClusters from resulting Factors"
    )
    parser.add_argument("library_info_path", help="path/to/library_info.json")
    parser.add_argument(
        "tensor_input_path",
        help="path/to/file.cpickle.zlib for tensor to factorize")
    parser.add_argument(
        "timepoints_yaml",
        help=
        "path/to/file.yaml containing list of hdx timepoints in integer seconds which are also keys mapping to lists of each timepoint's .mzML file, can pass config/config.yaml - for Snakemake context"
    )
    parser.add_argument(
        "-o",
        "--isotope_clusters_out_path",
        help="path/to/output.cpickle.zlib, list of IsotopeClusters")
    parser.add_argument(
        "-of",
        "--factor_data_out_path",
        help="path/to/output.cpickle.zlib.factor, FactorData")
    parser.add_argument(
        "-po",
        "--factor_plot_out_path",
        help="path/to/output.cpickle.zlib.factor.pdf, FactorData Plot output .pdf")
    parser.add_argument(
        "-po_ic",
        "--ic_plot_out_path",
        help="path/to/output.cpickle.zlib.ic.pdf, ICData Plot output .pdf")
    parser.add_argument(
        "-p",
        "--normalization_factors_path",
        help="path/to/normalization_factors.csv")
    parser.add_argument(
        "-g",
        "--gauss_params",
        type=tuple,
        default=(3, 1),
        help="determines intensity of gaussian smoothing in rt and dt dimensions"
    )
    parser.add_argument(
        "-n",
        "--n_factors",
        type=int,
        default=15,
        help="maximum number of factors for factorization of the data tensor"
    )
    args = parser.parse_args()

    # Open timepoints .yaml into dict for main().
    config_dict = yaml.load(open(args.timepoints_yaml, 'rb'), Loader=yaml.Loader)

    use_rtdt_recenter = config_dict["use_rtdt_recenter"]

    filter_factors = config_dict["filter_factor"]
    factor_rt_r2_cutoff = config_dict["factor_rt_r2_cutoff"]
    factor_dt_r2_cutoff = config_dict["factor_dt_r2_cutoff"]

    max_num_factors = config_dict["max_num_factors"]
    factor_init_method = config_dict["init_method"]
    factor_niter_max = config_dict["n_iter_max"]
    factor_tol = config_dict["tolerance"]
    factor_corr_thres = config_dict["factor_corr_threshold"]

    ic_peak_prom = config_dict["ic_peak_prominence"]
    ic_peak_width = config_dict["ic_peak_width"]
    auto_ic_peak_width = config_dict["auto_ic_peak_width"]
    ic_rel_ht_filter = config_dict["ic_rel_height_filter"]
    ic_rel_ht_baseline = config_dict["ic_rel_height_filter_baseline"]
    ic_rel_ht_threshold = config_dict["ic_rel_height_threshold"]

    normalization_factors = pd.read_csv(args.normalization_factors_path)
    my_mzml = [filename for timepoint in config_dict["timepoints"] for filename in config_dict[timepoint] if filename in args.tensor_input_path][0]
    normalization_factor = normalization_factors.loc[normalization_factors["mzml"]==my_mzml]["normalization_factor"].values

    main(library_info_path=args.library_info_path,
         tensor_input_path=args.tensor_input_path,
         timepoints_dict=config_dict,
         normalization_factor=normalization_factor,
         isotope_clusters_out_path=args.isotope_clusters_out_path,
         factor_out_path=args.factor_data_out_path,
         factor_plot_output_path=args.factor_plot_out_path,
         ic_plot_output_path=args.ic_plot_out_path,
         gauss_params=args.gauss_params,
         n_factors=max_num_factors,
         init_method=factor_init_method,
         niter_max=factor_niter_max,
         tol=factor_tol,
         factor_corr_threshold=factor_corr_thres,
         filter_factors=filter_factors,
         factor_rt_r2_cutoff=factor_rt_r2_cutoff,
         factor_dt_r2_cutoff=factor_dt_r2_cutoff,
         ic_peak_prominence=ic_peak_prom,
         ic_peak_width=ic_peak_width,
         auto_ic_peak_width=auto_ic_peak_width,
         ic_rel_height_filter=ic_rel_ht_filter,
         ic_rel_height_filter_baseline=ic_rel_ht_baseline,
         ic_rel_height_threshold=ic_rel_ht_threshold,
         use_rtdt_recenter=use_rtdt_recenter)
