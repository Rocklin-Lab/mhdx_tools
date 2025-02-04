import os
import os.path
import yaml
import argparse
import pandas as pd
from mhdx_tools.core.plot_factor_data import plot_factor_data_from_data_tensor
from mhdx_tools.core.plot_ics_data import plot_ics_from_ic_list
from mhdx_tools.core.processing import generate_tensor_factors
from mhdx_tools.core.io import limit_write


def main(library_info_path,
         tensor_input_path,
         configfile,
         normalization_factor,
         isotope_clusters_out_path=None,
         factor_output_path=None,
         factor_plot_output_path=None,
         ic_plot_output_path=None,
         return_flag=False,
         gauss_params=(3, 1),
         num_factors_guess=5,
         init_method="nndsvd",
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
         ic_rel_height_threshold=0.10
         ):
    """Performs factorization to deconvolute tensor, identifies IsotopeCluster objects, and can return and/or write output list of IsotopeClusters.

    Args:
        library_info_path (str): A path/to/library_info.json.
        tensor_input_path (str): A path/to/tensor.cpickle.zlib.
        configfile (dict): Dictionary with 'timepoints' key containing list of hdx timepoints in integer seconds, 
            which are keys mapping to lists of each timepoint's replicate .mzML filenames. 
        isotope_clusters_out_path (str): path/to/file for main output - list of IsotopeClusters objects.
        return_flag (bool): Option to return output in python, for notebook context.
        gauss_params (tuple of ints/floats): Gaussian smoothing parameters in LC-RT and IMS-DT dimensions (rt_sigma, dt_sigma).

    Returns:
        out_dict (dict): dictionary containing TensorGenerator object
    
    """
    out_dict = {}
    library_info = pd.read_json(library_info_path)
    my_name = tensor_input_path.split("/")[-2]  # Name from protein directory.
    my_charge = int([item[6:] for item in tensor_input_path.split("/")[-1].split("_") if "charge" in item][
                        0])  # Finds by keyword and strip text.
    my_row = library_info.loc[(library_info["name_rt-group"] == my_name) & (library_info["charge"] == my_charge)]
    my_centers = my_row["mz_centers"].values
    centers = my_centers[0]

    my_tp_ind = None # if timepoint not found in config file, put None
    # Finds timepoint of passed filename by config comparison.
    for tp in configfile["timepoints"]:
        for fn in configfile[tp]:
            if fn in tensor_input_path:
                my_tp = tp
                my_tp_ind = configfile["timepoints"].index(my_tp)

    # set up vars for calculating idotp if 0 timepoint index
    calc_idotp = False
    prot_seq = None
    if my_tp == 0:
        calc_idotp = True
        prot_seq = my_row['sequence'].values[0]

    data_tensor = generate_tensor_factors(tensor_fpath=tensor_input_path,
                                          library_info_df=library_info,
                                          timepoint_index=my_tp,
                                          tp_ind=my_tp_ind,
                                          gauss_params=gauss_params,
                                          num_factors_guess=num_factors_guess,
                                          init_method=init_method,
                                          niter_max=niter_max,
                                          tol=tol,
                                          factor_corr_threshold=factor_corr_threshold,
                                          mz_centers=centers,
                                          normalization_factor=normalization_factor,
                                          factor_output_fpath=factor_output_path,
                                          factor_plot_output_path=factor_plot_output_path,
                                          timepoint_label=None,
                                          filter_factors=filter_factors,
                                          factor_rt_r2_cutoff=factor_rt_r2_cutoff,
                                          factor_dt_r2_cutoff=factor_dt_r2_cutoff
                                          )

    all_ics = []

    ic_peak_width_auto = 0.8 * my_row['integrated_mz_width'].values[0]

    for ind, factor in enumerate(data_tensor.DataTensor.factors):

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

    if factor_plot_output_path is not None:

        # Check if the output file exists and remove it
        if os.path.exists(factor_plot_output_path):
            os.remove(factor_plot_output_path)

        if calc_idotp:

            max_idotp_list = []
            for ind, factor in enumerate(data_tensor.DataTensor.factors):
                idotp_list = [x.idotp for x in factor.isotope_clusters]
                if len(idotp_list) > 0:
                    max_idotp_list.append(max(idotp_list))
                else:
                    max_idotp_list.append(None)

            plot_factor_data_from_data_tensor(data_tensor=data_tensor,
                                              idotp_list=max_idotp_list,
                                              output_path=factor_plot_output_path)

        else:
            plot_factor_data_from_data_tensor(data_tensor=data_tensor,
                                              idotp_list=None,
                                              output_path=factor_plot_output_path)

    if ic_plot_output_path is not None:
        plot_ics_from_ic_list(all_ics, ic_plot_output_path)

    if return_flag:
        out_dict["TensorGenerator"] = data_tensor
        return out_dict


if __name__ == "__main__":

    if "snakemake" in globals():

        configfile_path = snakemake.input[0]
        library_info_path = snakemake.input[1]
        normalization_factors_path = snakemake.input[2]

        tensor_input_paths = [f for f in snakemake.input if ".gz.cpickle.zlib" in f]
        isotope_clusters_out_paths = [f for f in snakemake.output if ".gz.cpickle.zlib" in f]

        # Open timepoints .yaml into dict for main().
        configfile = yaml.load(open(configfile_path, 'rb'), Loader=yaml.Loader)

        filter_factors = configfile["filter_factor"]
        factor_rt_r2_cutoff = configfile["factor_rt_r2_cutoff"]
        factor_dt_r2_cutoff = configfile["factor_dt_r2_cutoff"]

        num_factors_guess = configfile["num_factors_guess"]
        factor_init_method = configfile["init_method"]
        factor_niter_max = configfile["n_iter_max"]
        factor_tol = configfile["tolerance"]
        factor_corr_thres = configfile["factor_corr_threshold"]

        gauss_params = (configfile["gauss_params"][0], configfile["gauss_params"][1])

        ic_peak_prom = configfile["ic_peak_prominence"]
        ic_peak_width = configfile["ic_peak_width"]
        auto_ic_peak_width = configfile["auto_ic_peak_width"]
        ic_rel_ht_filter = configfile["ic_rel_height_filter"]
        ic_rel_ht_baseline = configfile["ic_rel_height_filter_baseline"]
        ic_rel_ht_threshold = configfile["ic_rel_height_threshold"]

        normalization_factors = pd.read_csv(normalization_factors_path)

        for tensor_input_path, isotope_clusters_out_path in zip(tensor_input_paths, isotope_clusters_out_paths):

            my_mzml = [f for tp in configfile["timepoints"] for f in configfile[tp] if f in tensor_input_path][0]
            normalization_factor = normalization_factors.loc[normalization_factors["mzml"] == my_mzml][
                "normalization_factor"].values

            factor_data_output_path = isotope_clusters_out_path.replace("resources/9_subtensor_ics",
                                                                        "results/factors").replace(".cpickle.zlib",
                                                                                                   ".factor.cpickle.zlib") if \
                configfile[
                    "save_factor_data"] else None
            factor_plot_output_path = isotope_clusters_out_path.replace("resources/9_subtensor_ics",
                                                                        "results/plots/factors").replace(
                ".cpickle.zlib", ".cpickle.zlib.factor.pdf") if configfile[
                "save_factor_plot"] else None
            ic_plot_output_path = isotope_clusters_out_path.replace("resources/9_subtensor_ics",
                                                                    "results/plots/ics").replace(".cpickle.zlib",
                                                                                                 ".cpickle.zlib.ics.pdf") \
                if configfile["save_ic_plot"] else None

            if configfile["save_factor_data"]:
                if not os.path.isdir(os.path.dirname(factor_data_output_path)):
                    os.makedirs(os.path.dirname(factor_data_output_path), exist_ok=True)
            if configfile["save_factor_plot"]:
                if not os.path.isdir(os.path.dirname(factor_plot_output_path)):
                    os.makedirs(os.path.dirname(factor_plot_output_path), exist_ok=True)
            if configfile["save_ic_plot"]:
                if not os.path.isdir(os.path.dirname(ic_plot_output_path)):
                    os.makedirs(os.path.dirname(ic_plot_output_path), exist_ok=True)

            print(f"Processing {tensor_input_path}...")

            main(library_info_path=library_info_path,
                 tensor_input_path=tensor_input_path,
                 configfile=configfile,
                 normalization_factor=normalization_factor,
                 isotope_clusters_out_path=isotope_clusters_out_path,
                 factor_output_path=factor_data_output_path,
                 factor_plot_output_path=factor_plot_output_path,
                 ic_plot_output_path=ic_plot_output_path,
                 gauss_params=gauss_params,
                 num_factors_guess=num_factors_guess,
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
                 ic_rel_height_threshold=ic_rel_ht_threshold
                 )

    else:

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
            "--factor_output_path",
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
        configfile = yaml.load(open(args.timepoints_yaml, 'rb'), Loader=yaml.Loader)

        filter_factors = configfile["filter_factor"]
        factor_rt_r2_cutoff = configfile["factor_rt_r2_cutoff"]
        factor_dt_r2_cutoff = configfile["factor_dt_r2_cutoff"]

        num_factors_guess = configfile["num_factors_guess"]
        factor_init_method = configfile["init_method"]
        factor_niter_max = configfile["n_iter_max"]
        factor_tol = configfile["tolerance"]
        factor_corr_thres = configfile["factor_corr_threshold"]

        ic_peak_prom = configfile["ic_peak_prominence"]
        ic_peak_width = configfile["ic_peak_width"]
        auto_ic_peak_width = configfile["auto_ic_peak_width"]
        ic_rel_ht_filter = configfile["ic_rel_height_filter"]
        ic_rel_ht_baseline = configfile["ic_rel_height_filter_baseline"]
        ic_rel_ht_threshold = configfile["ic_rel_height_threshold"]

        normalization_factors = pd.read_csv(args.normalization_factors_path)
        my_mzml = [filename for timepoint in configfile["timepoints"] for filename in configfile[timepoint] if
                   filename in args.tensor_input_path][0]
        normalization_factor = normalization_factors.loc[normalization_factors["mzml"] == my_mzml][
            "normalization_factor"].values

        main(library_info_path=args.library_info_path,
             tensor_input_path=args.tensor_input_path,
             configfile=configfile,
             normalization_factor=normalization_factor,
             isotope_clusters_out_path=args.isotope_clusters_out_path,
             factor_output_path=args.factor_output_path,
             factor_plot_output_path=args.factor_plot_out_path,
             ic_plot_output_path=args.ic_plot_out_path,
             gauss_params=args.gauss_params,
             num_factors_guess=args.num_factors_guess,
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
             ic_rel_height_threshold=ic_rel_ht_threshold
             )
