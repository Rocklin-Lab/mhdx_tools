import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from hdx_limit.core.io import limit_read, limit_write, check_for_create_dirs


def optimize_paths_inputs(library_info_path,
                          input_directory_path,
                          configfile):
    """Generate explicit PathOptimizer input paths for one rt_group.

    Args:
        library_info_path (str): path/to/checked_library_info.json
        input_directory_path (str): /path/to/dir/ to prepend to each outpath
        rt_group_name (str): value from 'name' column of library_info
        configfile (dict): dictionary containing list of hdx timepoints in seconds, where each timepoint is also an integer key corresponding to that timepoint's .mzML filenames

    Returns:
        name_inputs (list of strings): flat list of all IsotopeCluster inputs to PathOptimizer

    """
    name_inputs = []
    library_info = pd.read_json(library_info_path)
    charges = library_info.loc[library_info["name"]==name]['charge'].values
    for key in configfile["timepoints"]:
        if len(configfile[key]) > 1:
            for file in configfile[key]:
                for charge in charges:
                    name_inputs.append(
                        input_directory_path
                        + name
                        + "_"
                        + "charge"
                        + str(charge)
                        + "_"
                        + file
                        +".cpickle.zlib"
                    )
        else:
            file = configfile[key][0]
            for charge in charges:
                name_inputs.append(
                    input_directory_path
                    + name
                    + "_"
                    + "charge"
                    + str(charge)
                    + "_"
                    + file
                    + ".cpickle.zlib")

    return name_inputs

def gen_correlate_matrix(list_of_arrays):
    """Takes a list of 1D signal arrays and generates a correlation matrix for those signals.
    Args:
        list_of_arrays (list of number arrays): A flat iterable of signals (arrays) to be compared.

    Returns:
        corr_matrix (numpy array): A correlation matrix containing correlation coefficients for each pair of signal arrays.

    """
    corr_matrix = np.zeros((len(list_of_arrays), len(list_of_arrays)))
    for ind1, arr1 in enumerate(list_of_arrays):
        for ind2, arr2 in enumerate(list_of_arrays):
            corr_matrix[ind1, ind2] = max(np.correlate(arr1 / np.linalg.norm(arr1), arr2 / np.linalg.norm(arr2)))

    return corr_matrix

def main(configfile,
         all_ic_input_paths=None,
         all_timepoint_clusters_out_path=None):


    atc = []
    suffix = ".mzML"
    for tp in configfile["timepoints"]:
        tp_buf = []
        for fn in configfile[tp]:
            for file in all_ic_input_paths:
                if fn[:-len(suffix)] in file:
                    ics = limit_read(file)
                    for ic in ics:
                        tp_buf.append(ic)

        atc.append(tp_buf)

    for ics in atc:
        all_baseline_integrated_mz = []
        all_rts = []
        charge_list = []
        for ic in ics:
            all_baseline_integrated_mz.append(ic.baseline_integrated_mz)
            all_rts.append(ic.rts)
            charge_list.append(ic.charge_states[0])

        charge_list = np.array(charge_list)
        mz_corrmat = gen_correlate_matrix(all_baseline_integrated_mz)
        rt_corrmat = gen_correlate_matrix(all_rts)
        minimum_corrmat = np.minimum(mz_corrmat, rt_corrmat)

        for column, ic in enumerate(ics):
            min_corr_list = minimum_corrmat[column][charge_list != ic.charge_states[0]]
            if len(min_corr_list) != 0:
                ic.nearest_neighbor_correlation = max(min_corr_list)
            else:
                ic.nearest_neighbor_correlation = 0

    if all_timepoint_clusters_out_path is not None:
        limit_write(atc, all_timepoint_clusters_out_path)


if __name__ == "__main__":

    # Checks for Snakemake context and parses arguments.
    if "snakemake" in globals():
        configfile = yaml.load(open(snakemake.input[0], "rb").read(), Loader=yaml.Loader)
        all_ic_input_paths = snakemake.input[1:]
        all_timepoint_clusters_out_path = snakemake.output[0]

        main(configfile=configfile,
             all_ic_input_paths=all_ic_input_paths,
             all_timepoint_clusters_out_path=all_timepoint_clusters_out_path)

    else:
        # Configure command line argument parser.
        parser = argparse.ArgumentParser(
            description=
            "Collect all subtensors and write a all timepoints cluster file"
        )
        # Inputs
        parser.add_argument("library_info_path", help="path/to/checked_library_info.json")
        parser.add_argument(
            "configfile_yaml",
            help=
            "path/to/file.yaml containing list of hdx timepoints in integer seconds which are also keys mapping to lists of each timepoint's .mzML file, can pass config/config.yaml - for Snakemake context"
        )
        parser.add_argument(
            "-i",
            "--all_ic_input_paths",
            nargs="*",
            help=
            "structured 2D list of extracted IsotopeCluster objects from each tensor included in the rt_group."
        )
        parser.add_argument(
            "-d",
            "--input_directory_path",
            help=
            "path/to/directory to search for relevant files if assembling filenames automatically, requires --rt_group_name"
        )
        parser.add_argument(
            "-n",
            "--rt_group_name",
            help=
            "rt-group name to use for generating relevant tensor files, requires --input_directory_path"
        )

        args = parser.parse_args()

        # Opens configfile .yaml and generates explicit inputs.
        configfile = yaml.load(open(args.configfile_yaml, "rb").read(), Loader=yaml.Loader)
        if args.all_ic_input_paths is None:
            if args.input_directory_path is not None and args.rt_group_name is not None:
                args.all_ic_input_paths = optimize_paths_inputs(
                    args.library_info_path, args.input_directory_path,
                    args.rt_group_name, configfile)
            else:
                parser.print_help()
                sys.exit()

        output_paths = [
            args.all_timepoint_clusters_out_path]

        check_for_create_dirs(output_paths)

        main(configfile=configfile,
             all_ic_input_paths=args.all_ic_input_paths,
             all_timepoint_clusters_out_path=args.all_timepoint_clusters_out_path,
             )
