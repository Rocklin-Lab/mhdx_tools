import sys
import glob
import yaml
import molmass
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import find_peaks
from mhdx_tools.core.plot_ics_data import plot_ics_from_ic_list
from mhdx_tools.core.processing import generate_tensor_factors
from mhdx_tools.core.io import limit_write


def cum_peak_gaps_from_sequence(sequence):
    """Determine expected cumulative mass distances between isotopes of a given protein sequence.
    Later divided by charge and added to observed m/Z.

    Args:
        sequence (str): Single letter amino acid sequence of a protein.

    Returns:
        cumulative_peak_gaps (list): List of floats containing cumulative expected mass differences between isotopes.

    """
    deut = molmass.Formula('D')
    hyd = molmass.Formula('H')

    n_exchangable = len(sequence) - 2 - sequence[2:].count('P')

    pep_seq = 'peptide(' + sequence + ')'
    f=molmass.Formula(pep_seq)
    start = min([x[0] for x in f.spectrum().values()])-0.5
    n_isotopes = int(len(f.spectrum(minfract=0.0001)))

    x = np.linspace(start,start+n_isotopes,1000000)
    y=np.zeros((len(x)))

    for k in range(n_isotopes):
        f=molmass.Formula(pep_seq)
        if k > 0: f = f - (k * hyd) + (k * deut)
        peaks = [(x[0], x[1]) for x in f.spectrum().values()]

        for peak in peaks:
            y += norm.pdf(x,loc=peak[0], scale=peak[0]*(20/1000000)) * peak[1]

    peak_pos = find_peaks(y)[0]
    peak_gaps = x[peak_pos][1:] - x[peak_pos][0:-1]
    peak_gaps = list(peak_gaps) + ([1.00627301] * (n_exchangable - len(peak_gaps) + int(n_isotopes / 2)))

    return np.array([0] + list(np.cumsum(peak_gaps)))


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


def gen_ics_list(library_info_df,
                 undeut_tensor_path_list,
                 mz_centers,
                 normalization_factor,
                 prot_sequence,
                 factor_output_path_list=None,
                 factor_plot_output_path_list=None,
                 ics_plot_output_path_list=None,
                 timepoint_index=0,
                 num_factors_guess=5,
                 init_method='nndsvd',
                 niter_max=100000,
                 tol=1e-8,
                 factor_corr_threshold=0.17,
                 gauss_params=(3, 1),
                 filter_factors=False,
                 factor_rt_r2_cutoff=0.91,
                 factor_dt_r2_cutoff=0.91,
                 ic_peak_prominence=0.10,
                 ic_peak_width=2,
                 ic_rel_height_filter=False,
                 ic_rel_height_filter_baseline=0.10,
                 ic_rel_height_threshold=0.10):
    """Instantiates TensorGenerator and factorizes.
    
    Args:
        library_info_df (Pandas DataFrame): library info data frame
        undeut_tensor_path_list (List of strs): undeuterated tensor file path list
        timepoint_index (int): time point index
        n_factors (int): number of factors for factorization
        gauss_params (tuple of two ints): Paramters for gaussian smoothing of DataTensor, format: (RT, DT); default: (3, 1).
        filter_factors (bool): Boolean switch for filtering factors by gaussian fits in RT and DT dimensions.
        factor_dt_r2_cutoff (float): Minimum R^2 for gaussian fit to pass filter in DT dimension.
        factor_rt_r2_cutoff (float): Minimum R^2 for gaussian fit to pass filter in RT dimension
        ic_peak_prominence (float): Minimum ratio of intensity between an isotope-cluster-like signal and highest m/Z peak.
        ic_peak_width (int): Minumum m/Z width for an isotope-cluster-like signal to be condsidered.
        ic_rel_height_filter (bool): Boolean switch to filter IsotopeClusters by their relative height.
        ic_rel_height_filter_baseline (float): Minimum ratio of intensity between total factor intensity and isotope-cluster-like signal. 
        ic_rel_height_threshold (float): Bounding criteria for a single IsotopeCluster, ratio of signal to peak height to be accepted as part of IC.

    Return:
        undeut_ics_list (list): list of all IsotopeCluster objects from factorized tensors
        data_tensor_list (list): list of all DataTensor objects made from path_list

    """
    undeut_ics_list = []

    # Handle optional factor arguments in control structure.
    if factor_output_path_list is None:
        factor_output_path_list = [None for i in undeut_tensor_path_list]
    if factor_plot_output_path_list is None:
        factor_plot_output_path_list = [None for i in undeut_tensor_path_list]
    if ics_plot_output_path_list is None:
        ics_plot_output_path_list = [None for i in undeut_tensor_path_list]

    for undeut_tensor_path, factor_output_path, factor_plot_output_path, ic_plot_output_path in zip(
        undeut_tensor_path_list, factor_output_path_list, factor_plot_output_path_list, ics_plot_output_path_list):
        # Generate DataTensor from extracted data.
        data_tensor = generate_tensor_factors(tensor_fpath=undeut_tensor_path,
                                              library_info_df=library_info_df,
                                              timepoint_index=timepoint_index,
                                              gauss_params=gauss_params,
                                              num_factors_guess=num_factors_guess,
                                              init_method=init_method,
                                              niter_max=niter_max,
                                              tol=tol,
                                              factor_corr_threshold=factor_corr_threshold,
                                              mz_centers=mz_centers,
                                              normalization_factor=normalization_factor,
                                              factor_output_fpath=factor_output_path,
                                              factor_plot_output_path=factor_plot_output_path,
                                              timepoint_label=None,
                                              filter_factors=filter_factors,
                                              factor_rt_r2_cutoff=factor_rt_r2_cutoff,
                                              factor_dt_r2_cutoff=factor_dt_r2_cutoff)

        # Generate Factors and IsotopeClusters from DataTensor. calculate idotp here
        ics_per_tensor_file = []
        for factor in data_tensor.DataTensor.factors:
            factor.find_isotope_clusters(prominence=ic_peak_prominence,
                                         width_val=ic_peak_width,
                                         rel_height_filter=ic_rel_height_filter,
                                         baseline_threshold=ic_rel_height_filter_baseline,
                                         rel_height_threshold=ic_rel_height_threshold,
                                         calculate_idotp=True,
                                         sequence=prot_sequence)
            for isotope_cluster in factor.isotope_clusters:
                undeut_ics_list.append(isotope_cluster)
                ics_per_tensor_file.append(isotope_cluster)

        if ic_plot_output_path is not None:
            plot_ics_from_ic_list(list_of_ics=ics_per_tensor_file,
                                  output_path=ic_plot_output_path)

    return undeut_ics_list


def main(library_info_path,
         normalization_factor,
         undeut_tensor_path_list,
         factor_output_path_list=None,
         factor_plot_output_path_list=None,
         output_path=None,
         all_clusters_output=None,
         ics_plot_output_path_list=None,
         return_flag=None,
         num_factors_guess=5,
         init_method='nndsvd',
         niter_max=100000,
         tol=1e-8,
         factor_corr_threshold=0.17,
         gauss_params=(3, 1),
         filter_factors=False,
         factor_rt_r2=0.91,
         factor_dt_r2=0.91,
         ic_peak_prominence=0.15,
         ic_peak_width=3,
         ic_rel_height_filter=True,
         ic_rel_height_filter_baseline=0.15,
         ic_rel_height_threshold=0.10):
    """Compares each undeuterated replicate of a charge state to its theoretical distribution as a measure of signal quality.

    Args:
        library_info_path (string): path/to/library_info.json
        undeut_tensor_path_list (list of strings): list of paths/to/files.cpickle.zlib
        output_path (string): path/to/output.csv
        return_flag (bool): option to return output in python, for notebook context
        n_factors (int): high number of factors to start factorization with
        gauss_params (tuple of floats): gaussian smoothing parameters in tuple (rt-sigma, dt-sigma), default (3,1)
        filter_factors (bool): Boolean switch for filtering factors by gaussian fits in RT and DT dimensions.
        factor_dt_r2_cutoff (float): Minimum R^2 for gaussian fit to pass filter in DT dimension.
        factor_rt_r2_cutoff (float): Minimum R^2 for gaussian fit to pass filter in RT dimension
        ic_peak_prominence (float): Minimum ratio of intensity between an isotope-cluster-like signal and highest m/Z peak.
        ic_peak_width (int): Minumum m/Z width for an isotope-cluster-like signal to be condsidered.
        ic_rel_height_filter (bool): Boolean switch to filter IsotopeClusters by their relative height.
        ic_rel_height_filter_baseline (float): Minimum ratio of intensity between total factor intensity and isotope-cluster-like signal. 
        ic_rel_height_threshold (float): Bounding criteria for a single IsotopeCluster, ratio of signal to peak height to be accepted as part of IC.
    
    Returns:
        out_dict (dict): containing - 
            iso_cluster_list (list): list of IsotopeCluster objects produced from factorized input tensors
            data_tensor_list (list): list of DataTensor objects produced from input tensor paths
            idotp_list (list): list of resulting idotps for charge states
            integrated_mz_list (list): list containing integrated mzs of IsotopeClusters

    """
    print(undeut_tensor_path_list)
    library_info = pd.read_json(library_info_path)
    prot_name = undeut_tensor_path_list[0].split("/")[-2] # Name from protein directory.
    prot_charge = int([item[6:] for item in undeut_tensor_path_list[0].split("/")[-1].split("_") if "charge" in item][0]) # Finds by keyword and strip text.
    my_info = library_info.loc[(library_info["name_rt-group"]==prot_name) & (library_info["charge"]==prot_charge)]
    prot_seq = my_info["sequence"].values[0]
    print("Protein Sequence: "+prot_seq)
    prot_cum_peak_gaps = cum_peak_gaps_from_sequence(prot_seq)
    print("Cumulative Peak Gaps: "+str(prot_cum_peak_gaps))
    print("Protein Charge States: "+str(my_info["charge"].values[0]))
    print("Observed m/Z: "+str(my_info['obs_mz'].values[0]))
    mz_centers = (prot_cum_peak_gaps/prot_charge) + my_info['expect_mz'].values[0] # replaced from obs_mz to expect_mz

    iso_clusters_list = gen_ics_list(library_info_df=library_info,
                                     undeut_tensor_path_list=undeut_tensor_path_list,
                                     factor_output_path_list=factor_output_path_list,
                                     factor_plot_output_path_list=factor_plot_output_path_list,
                                     ics_plot_output_path_list=ics_plot_output_path_list,
                                     prot_sequence=prot_seq,
                                     mz_centers=mz_centers,
                                     normalization_factor=normalization_factor,
                                     num_factors_guess=num_factors_guess,
                                     init_method=init_method,
                                     niter_max=niter_max,
                                     tol=tol,
                                     factor_corr_threshold=factor_corr_threshold,
                                     gauss_params=gauss_params,
                                     filter_factors=filter_factors,
                                     factor_dt_r2_cutoff=factor_dt_r2,
                                     factor_rt_r2_cutoff=factor_rt_r2,
                                     ic_peak_prominence=ic_peak_prominence,
                                     ic_peak_width=ic_peak_width,
                                     ic_rel_height_filter=ic_rel_height_filter,
                                     ic_rel_height_filter_baseline=ic_rel_height_filter_baseline,
                                     ic_rel_height_threshold=ic_rel_height_threshold)

    idotp_list = [x.idotp for x in iso_clusters_list]
    integrated_mz_width_list = [x.integrated_mz_peak_width for x in iso_clusters_list]
    theo_mz_dist = calculate_theoretical_isotope_dist_from_sequence(sequence=prot_seq)


    # idotp_list, integtd_mz_list, integt_mz_width_list, theo_mz_dist = calc_dot_prod_for_isotope_clusters(
    #     sequence=prot_seq, undeut_isotope_clusters=iso_clusters_list)

    if len(idotp_list) > 0:
        max_idotp_idx = np.argmax(np.array(idotp_list))
        max_idotp = np.array(idotp_list)[max_idotp_idx]
        int_mz_width = np.array(integrated_mz_width_list)[max_idotp_idx]
    else: 
        max_idotp = 0
        int_mz_width = 0

    if all_clusters_output is not None:
        limit_write(iso_clusters_list, all_clusters_output)

    if output_path is not None:
        pd.DataFrame({
            "idotp": max_idotp,
            "integrated_mz_width": int_mz_width,
            "mz_centers": [mz_centers], # Cast as nested list to force into single index.
            "theor_mz_dist": [theo_mz_dist]
            }, index=[0]).to_json(output_path)

    if return_flag is not None:
        return {
            "iso_clusters_list": iso_clusters_list,
            "idotp_list": idotp_list,
        }


if __name__ == "__main__":

    if "snakemake" in globals():
        library_info_path = snakemake.input[0]
        config_dict = yaml.load(open(snakemake.input[1], 'rb'), Loader=yaml.Loader)
        normalization_factors = pd.read_csv(snakemake.input[2])
        undeut_tensor_path_list = snakemake.input[3:]
        my_mzml = [filename for timepoint in config_dict["timepoints"] for filename in config_dict[timepoint] if filename in undeut_tensor_path_list[0]][0]
        print(undeut_tensor_path_list[0])
        print(my_mzml)
        normalization_factor = normalization_factors.loc[normalization_factors["mzml"]==my_mzml]["normalization_factor"].values
        output_path = snakemake.output[0]
        all_clusters_output = snakemake.output[1]
        factor_output_path_list = [item for item in snakemake.output if item.endswith(".factor")]
        factor_plot_output_path_list = [item for item in snakemake.output if item.endswith(".factor.pdf")]
        if factor_plot_output_path_list == []:
            factor_plot_output_path_list = None
        if factor_output_path_list == []:
            factor_output_path_list = None
        filter_factors = config_dict["filter_factor"]
        factor_rt_r2_cutoff = config_dict["factor_rt_r2_cutoff"]
        factor_dt_r2_cutoff = config_dict["factor_dt_r2_cutoff"]
        ic_peak_prom = config_dict["ic_peak_prominence"]
        ic_peak_width = config_dict["ic_peak_width"]
        ic_rel_ht_filter = config_dict["ic_rel_height_filter"]
        ic_rel_ht_baseline = config_dict["ic_rel_height_filter_baseline"]
        ic_rel_ht_threshold = config_dict["ic_rel_height_threshold"]
        num_factors_guess = config_dict["num_factors_guess"]
        factor_init_method = config_dict["init_method"]
        factor_niter_max = config_dict["n_iter_max"]
        factor_tol = config_dict["tolerance"]
        factor_corr_thres = config_dict["factor_corr_threshold"]

        main(library_info_path=library_info_path,
             normalization_factor=normalization_factor,
             num_factors_guess=num_factors_guess,
             init_method=factor_init_method,
             niter_max=factor_niter_max,
             tol=factor_tol,
             factor_corr_threshold=factor_corr_thres,
             undeut_tensor_path_list=undeut_tensor_path_list,
             factor_output_path_list=factor_output_path_list,
             output_path=output_path,
             factor_plot_output_path_list=factor_plot_output_path_list,
             filter_factors=filter_factors,
             factor_rt_r2=factor_rt_r2_cutoff,
             factor_dt_r2=factor_dt_r2_cutoff,
             ic_peak_prominence=ic_peak_prom,
             ic_peak_width=ic_peak_width,
             ic_rel_height_filter=ic_rel_ht_filter,
             ic_rel_height_filter_baseline=ic_rel_ht_baseline,
             ic_rel_height_threshold=ic_rel_ht_threshold,
             all_clusters_output=all_clusters_output)

    else:
        parser = argparse.ArgumentParser(
            description=
            "Checks observed undeuterated signal against theoretical isotopic distribution, returns dataframe with highest idotp of all undeut"
        )
        parser.add_argument("library_info_path", help="path/to/library_info.json")
        parser.add_argument("config_file_path", help='path/to/config.yaml')
        parser.add_argument("normalization_factors", help="path/to/normalization_factors.csv")
        parser.add_argument(
            "-l",
            "--undeut_tensor_path_list",
            nargs="+",
            help=
            "list of paths to undeuterated tensor outputs from extract_tensors.py")
        parser.add_argument("-d", "--input_directory", help="path/to/dir/ containing undeuterated tensor inputs")
        parser.add_argument("-r", "--rt_group_name", help="rt-group name to capture for idotp check")
        parser.add_argument("-o", "--output_path", help="path/to/file for main .json output")
        parser.add_argument("-f", "--factor_output_path_list", nargs="+", help="list of paths/to/file for factor data .factor output")
        parser.add_argument("-p", "--factor_plot_output_path_list", nargs="+", help="list of paths/to/files for factor data plot output .pdf")
        parser.add_argument(
            "-n",
            "--n_factors",
            default=15,
            help=
            "high number of factors to use in non_negative_parafac decomposition, counts down until correlation constraint is reached - see DataTensor.factorize()"
        )
        parser.add_argument(
            "-g",
            "--gauss_params",
            type=tuple,
            default=(3, 1),
            help="parameters for smoothing rt and dt dimensions"
        )
        parser.add_argument(
            "-a",
            "--all_clusters_output",
            help="all isotopic clusters for specific ic"
        )
        args = parser.parse_args()
        if args.undeut_tensor_path_list is None:
            if args.input_directory is None or args.rt_group_name is None:
                parser.print_help()
                sys.exit()
            library_info = pd.read_json(args.library_info_path)
            args.undeut_tensor_path_list = [fn for i in library_info.loc[library_info["name"]==args.rt_group_name].index.values for fn in glob.glob(args.input_directory+str(i)+"/*.zlib")]

        config_dict = yaml.load(open(args.config_file_path, 'rb'), Loader=yaml.Loader)

        filter_factors = config_dict["filter_factor"]
        factor_rt_r2_cutoff = config_dict["factor_rt_r2_cutoff"]
        factor_dt_r2_cutoff = config_dict["factor_dt_r2_cutoff"]

        num_factors_guess = config_dict["num_factors_guess"]
        factor_init_method = config_dict["init_method"]
        factor_niter_max = config_dict["n_iter_max"]
        factor_tol = config_dict["tolerance"]
        factor_corr_thres = config_dict["factor_corr_threshold"]

        ic_peak_prom = config_dict["ic_peak_prominence"]
        ic_peak_width = config_dict["ic_peak_width"]
        ic_rel_ht_filter = config_dict["ic_rel_height_filter"]
        ic_rel_ht_baseline = config_dict["ic_rel_height_filter_baseline"]
        ic_rel_ht_threshold = config_dict["ic_rel_height_threshold"]

        normalization_factors = pd.read_csv(args.normalization_factors)
        my_mzml = [filename for timepoint in config_dict["timepoints"] for filename in config_dict[timepoint] if filename in args.undeut_tensor_path_list[0]][0]
        normalization_factor = normalization_factors.loc[normalization_factors["mzml"]==my_mzml]["normalization_factor"].values

        main(library_info_path=args.library_info_path,
             normalization_factor=normalization_factor,
             num_factors_guess=num_factors_guess,
             init_method=factor_init_method,
             niter_max=factor_niter_max,
             tol=factor_tol,
             factor_corr_threshold=factor_corr_thres,
             undeut_tensor_path_list=args.undeut_tensor_path_list,
             factor_output_path_list=args.factor_output_path_list,
             output_path=args.output_path,
             factor_plot_output_path_list=args.factor_plot_output_path_list,
             filter_factors=filter_factors,
             factor_rt_r2=factor_rt_r2_cutoff,
             factor_dt_r2=factor_dt_r2_cutoff,
             ic_peak_prominence=ic_peak_prom,
             ic_peak_width=ic_peak_width,
             ic_rel_height_filter=ic_rel_ht_filter,
             ic_rel_height_filter_baseline=ic_rel_ht_baseline,
             ic_rel_height_threshold=ic_rel_ht_threshold,
             all_clusters_output=args.all_clusters_output)
