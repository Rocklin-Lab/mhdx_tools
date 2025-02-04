import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from mhdx_tools.core.io import limit_read, limit_write, check_for_create_dirs, optimize_paths_inputs
from mhdx_tools.core.processing import PathOptimizer
from mhdx_tools.core.gjr_plot import plot_gjr_
from mhdx_tools.core.ajf_plot import plot_ajf_


def check_for_create_files(paths):
    for path in paths:
        if path is not None:
            Path(path).touch()


def write_baseline_integrated_mz_to_csv(path_object_list, output_path, norm_dist=True, return_flag=False):
    """
    write the integrated_mz_distribution for each timepoint to a .csv file
    :param ic_object_list: ic object list
    :param output_path: .csv output file path
    :param norm_dist: bool. Whether to normalize the integrated_mz_distribution
    :return: None
    """
    timepoint_list = [ic.timepoint_idx for ic in path_object_list]
    timepoint_index_list = [ic.tp_ind for ic in path_object_list]
    timepoint_str = ",".join([str(x) for x in timepoint_list])
    tp_idx_str = ",".join([str(x) for x in timepoint_index_list])
    header1 = "#tp_ind," + tp_idx_str + "\n"
    header2 = "#tp," + timepoint_str + "\n"

    if norm_dist:
        integrated_mz_distribution_list = [ic.baseline_integrated_mz/max(ic.baseline_integrated_mz) for ic in path_object_list]
    else:
        integrated_mz_distribution_list = [ic.baseline_integrated_mz for ic in path_object_list]
    integrated_mz_distribution_array = np.array(integrated_mz_distribution_list)

    data_string = ""
    for ind, array in enumerate(integrated_mz_distribution_array.T):
        arr_str = ",".join([str(x) for x in array])
        data_string += "{},{}\n".format(ind, arr_str)

    with open(output_path, "w") as outfile:
        outfile.write(header1 + header2 + data_string)

    if return_flag:
        out_dict = dict()
        out_dict["integrated_mz_distribution_array"] = integrated_mz_distribution_array
        out_dict["timepoint_list"] = timepoint_list
        out_dict["timepoint_index_list"] = timepoint_index_list
        return out_dict


def evaluate_list_score_with_removal(evaluation_function, original_list, threshold=1, min_timepoints=5):
    """
    Evaluates the impact of removing one item at a time from the list on the score.

    Parameters:
    - evaluation_function: The function that evaluates the list and returns a score.
                          It should take a list of objects as input and return a score.
    - original_list: The original list of objects to be evaluated.

    Returns:
    - A dictionary containing the index of the removed item as key and the corresponding
      score without that item as the value.
    """

    print(f"Len original list: {len(original_list)}")

    if len(original_list) < min_timepoints:
        return []

    original_score = evaluation_function(original_list)

    # print(f"[0, {len(original_list)}, {original_score}],")

    for i, item in enumerate(original_list):

        if item.timepoint_idx == 0:
            continue

        modified_list = original_list[:i] + original_list[i + 1:]
        score_without_item = evaluation_function(modified_list)

        # print(f"[{i}, {len(original_list)}, {score_without_item}],")

        if original_score - score_without_item > threshold:
            # n += 1
            # print(
            #     f"Previous score: {original_score}, new score {score_without_item}. Improvement of {1 - score_without_item / original_score}. N_change {n} ")
            return evaluate_list_score_with_removal(evaluation_function, modified_list, threshold=threshold, min_timepoints=min_timepoints)

    # print(f"No modification led to significant improvement. score: {original_score}")

    return original_list #, n

def evaluate_and_terminate(ics, configfile, name, monobody_path_arguments, multibody_path_arguments):
    if len(ics) < configfile["thresholds"]["min_timepoints"]:
        print("Not enough timepoints with ics to evaluate path. Creating empty files...")
        check_for_create_files(monobody_path_arguments)
        check_for_create_files(multibody_path_arguments)

        with open("joberr.out", "a") as f:
            f.write(f"{name}\tNot enough timepoints with ics to evaluate path. Creating empty files...\n")

        sys.exit()

def load_json(fpath):
    f = open(fpath)
    d = json.load(f)
    f.close()
    return d


def main(library_info_path,
         configfile,
         all_timepoints_clusters_input_path=None,
         monobody_return_flag=False,
         multibody_return_flag=False,
         rt_group_name=None,
         old_data_dir=None,
         prefiltered_ics_out_path=None,
         mono_path_plot_out_path=None,
         mono_html_plot_out_path=None,
         mono_winner_out_path=None,
         mono_runner_out_path=None,
         mono_undeut_ground_out_path=None,
         mono_winner_scores_out_path=None,
         mono_rtdt_com_cvs_out_path=None,
         mono_winner_csv_out_path=None,
         multi_path_plot_out_path=None,
         multi_html_plot_out_path=None,
         multi_winner_out_path=None,
         multi_runner_out_path=None,
         multi_undeut_ground_out_path=None,
         multi_winner_scores_out_path=None,
         multi_rtdt_com_cvs_out_path=None,
         multi_winner_csv_out_path=None,
         ajf_plot_out_path=None,
         dictionary_thresholds_path=None,
         ):
    """Uses PathOptimzier class to generate best-estimate hdx-timeseries of IsotopeClusters for a given library protein.

    Args:
        library_info_path (str): path/to/checked_library_info.json
        all_ic_input_paths (list of strings): list of paths/to/files.cpickle.zlib for all lists of IsotopeClusters from generate_tensor_ics.py
        configfile (dict): dictionary with "timepoints" key containing list of hdx timepoints in integer seconds, which are keys mapping to lists of each timepoint"s replicate .mzML filenames
        monobody_return_flag: option to return monobody output in python, for notebook context, can be combined with multibody_return_flag.
        multibody_return_flag: option to return multibody output in python, for notebook context, can be combined with monobody_return_flag.
        rt_group_name (str): library_info["name"] value
        old_data_dir (str): path/to/dir to provide comparison to GJR formatted results
        all_timepoint_clusters_out_path (str): path/to/file to output all clusters collected for PathOptimizer as a nested list.
        prefiltered_ics_out_path (str): path/to/file to output ICs selected from prefiltering as a nested list.
        mono_path_plot_out_path (str): path/to/dir for pdf plot of monobody scoring results
        mono_winner_out_path (str): path/to/file for winning path from monobody scoring
        mono_runner_out_path (str): path/to/file for top n_runners paths from monobody scoring
        mono_undeut_ground_out_path (str): path/to/file for undeuterated ground-truth IsotopeClusters from monobody scoring
        mono_winner_scores_out_path (str): path/to/file for monobody scoring winning path score values 
        mono_rtdt_com_cvs_out_path (str): path/to/file for rt and dt correlation values from monobody scoring
        multi_path_plot_out_path (str): path/to/dir for pdf plot of multibody scoring results
        multi_winner_out_path (str): path/to/file for winning path from multibody scoring
        multi_runner_out_path (str): path/to/file for top n_runners paths from multibody scoring
        multi_undeut_ground_out_path (str): path/to/file for undeuterated ground-truth IsotopeClusters from multibody scoring
        multi_winner_scores_out_path (str): path/to/file for multibody winning path score values 
        multi_rtdt_com_cvs_out_path (str): path/to/file for rt and dt correlation values from multibody scoring

    Returns:
        out_dict (dict): dictionary containing "path_optimizer" key, and corresponding PathOptimizer object 

    """
    monobody_path_arguments = [
                                    mono_path_plot_out_path,
                                    mono_html_plot_out_path,
                                    mono_winner_out_path, 
                                    mono_runner_out_path, 
                                    mono_undeut_ground_out_path, 
                                    mono_winner_scores_out_path, 
                                    mono_rtdt_com_cvs_out_path,
                                    mono_winner_csv_out_path,
                                    ajf_plot_out_path,
                                ]

    multibody_path_arguments = [
                                    multi_path_plot_out_path,
                                    multi_html_plot_out_path,
                                    multi_winner_out_path, 
                                    multi_runner_out_path, 
                                    multi_undeut_ground_out_path, 
                                    multi_winner_scores_out_path, 
                                    multi_rtdt_com_cvs_out_path,
                                    multi_winner_csv_out_path,
                                ]

    check_for_create_dirs(monobody_path_arguments)
    check_for_create_dirs(multibody_path_arguments)

    out_dict = {}
    library_info = pd.read_json(library_info_path)
    if not "name_rt-group" in library_info.columns: library_info["name_rt-group"] = library_info["name"]


    if rt_group_name is None:
        name = all_timepoints_clusters_input_path[0].split("/")[-2]
    else:
        name = rt_group_name

    # Load dictionary of thresholds if any. Update values on configfile["thresholds"]
    # if dictionary_thresholds_path is not None:
    #     dictionary_thresholds = load_json(dictionary_thresholds_path)

        # # for key in dictionary_thresholds:
        # for key in ["dt_ground_err", "rt_ground_err", "baseline_peak_error", "baseline_integrated_mz_rmse"]:
        #     configfile["thresholds"][key] = dictionary_thresholds[key]

    atc = limit_read(all_timepoints_clusters_input_path)

    # Populate ics with tp_ind
    for ics in atc:
        for ic in ics:
            ic.tp_ind = configfile["timepoints"].index(ic.timepoint_idx)

    if len([ic for ic in atc[0] if ic.idotp > configfile["thresholds"]["idotp_cutoff"]]) == 0:
        print("No tp=0 with idotp greater than threshold found. Creating empty files...")
        check_for_create_files(monobody_path_arguments)
        check_for_create_files(multibody_path_arguments)
        check_for_create_files([prefiltered_ics_out_path])

        with open("joberr.out", "a") as f:
            f.write(f"{name}\tNo tp=0 with idotp greater than threshold found. Creating empty files...\n")

        sys.exit()

    p1 = PathOptimizer(
        name,
        atc,
        library_info,
        user_prefilter=configfile["user_prefilter"],
        thresholds=configfile["thresholds"],
        pareto_filter=configfile["pareto_prefilter"],
        # timepoints=configfile["timepoints"],
        # n_undeut_runs=len(configfile[0]),
        old_data_dir=old_data_dir,
        use_rtdt_recenter=configfile["use_rtdt_recenter"],
    )

    limit_write(p1.all_tp_clusters, all_timepoints_clusters_input_path)

    # Starting condition output arguments.
    if prefiltered_ics_out_path is not None:
        limit_write(p1.prefiltered_ics, prefiltered_ics_out_path)

    # Check if number of timepoints in prefiltered set is greater than threshold
    evaluate_and_terminate(p1.prefiltered_ics, configfile, name, monobody_path_arguments, multibody_path_arguments)

    # if len(p1.prefiltered_ics) < configfile["thresholds"]["min_timepoints"]:
    #     print("Not enough timepoints with ics to evaluate path. Creating empty files...")
    #     check_for_create_files(monobody_path_arguments)
    #     check_for_create_files(multibody_path_arguments)
    #
    #     with open("joberr.out", "a") as f:
    #         f.write(f"{name}\tNot enough timepoints with ics to evaluate path. Creating empty files...\n")
    #
    #     sys.exit()

    # Checks if arguments require monobody scoring run.
    if (any(arg is not None for arg in monobody_path_arguments)) or (monobody_return_flag is not False):

        print("Running monobody scoring...")

        p1.optimize_paths_mono()

        p1.winner = evaluate_list_score_with_removal(p1.combo_score_mono, p1.winner, threshold=1)
        # Check if number of timepoints in prefiltered set is greater than threshold
        evaluate_and_terminate(p1.winner, configfile, name, monobody_path_arguments, multibody_path_arguments)

        if monobody_return_flag is not False:
            out_dict["monobody_winner"] = p1.winner
            out_dict["monobody_runners"] = p1.runners
            out_dict["monobody_undeut_grounds"] = [p1.undeut_grounds, p1.undeut_ground_dot_products]
            out_dict["monobody_winner_scores"] = p1.winner_scores
            out_dict["monobody_rt_dt_com_cvs"] = [p1.rt_com_cv, p1.dt_com_cv]

        if mono_path_plot_out_path is not None:
            undeut_grounds = [p1.undeut_grounds, p1.undeut_ground_dot_products]
            plot_gjr_(winner=p1.winner,
                      undeut_grounds=undeut_grounds,
                      output_path=mono_path_plot_out_path,
                      prefix=name)
        if mono_winner_out_path is not None:
            limit_write(p1.winner, mono_winner_out_path)
        if mono_runner_out_path is not None:
            limit_write(p1.runners, mono_runner_out_path)
        if mono_undeut_ground_out_path is not None:
            limit_write([p1.undeut_grounds, p1.undeut_ground_dot_products],
                        mono_undeut_ground_out_path)
        if mono_winner_scores_out_path is not None:
            limit_write(p1.winner_scores, mono_winner_scores_out_path)
        if mono_rtdt_com_cvs_out_path is not None:
            limit_write([p1.rt_com_cv, p1.dt_com_cv], mono_rtdt_com_cvs_out_path)
        if mono_winner_csv_out_path is not None:
            write_baseline_integrated_mz_to_csv(p1.winner, mono_winner_csv_out_path)

    # Checks if arguments require multibody scoring run.
    if (any(arg is not None for arg in multibody_path_arguments)) or (multibody_return_flag is not False):

        print("Running multibody scoring...")

        p1.optimize_paths_multi()

        # Remove low quality ICs from winner list.
        p1.winner = [ic for ic in p1.winner if ic.baseline_integrated_mz_rmse < 0.1]
        # Check if number of timepoints in prefiltered set is greater than threshold
        evaluate_and_terminate(p1.winner, configfile, name, monobody_path_arguments, multibody_path_arguments)
        # Remove outliers from winner list.
        p1.winner = evaluate_list_score_with_removal(p1.combo_score_multi, p1.winner, threshold=1, min_timepoints=5)
        # Check if number of timepoints in prefiltered set is greater than threshold
        evaluate_and_terminate(p1.winner, configfile, name, monobody_path_arguments, multibody_path_arguments)
        # Re-score winner list.
        p1.winner_scores = p1.report_score_multi(p1.winner)

        if multibody_return_flag is not False:
            out_dict["multibody_winner"] = p1.winner
            out_dict["multibody_runners"] = p1.runners
            out_dict["multibody_undeut_grounds"] = [p1.undeut_grounds, p1.undeut_ground_dot_products]
            out_dict["multibody_winner_scores"] = p1.winner_scores
            out_dict["multibody_rt_dt_com_cvs"] = [p1.rt_com_cv, p1.dt_com_cv]

        if multi_path_plot_out_path is not None:
            undeut_grounds = [p1.undeut_grounds, p1.undeut_ground_dot_products]
            plot_gjr_(winner=p1.winner,
                      undeut_grounds=undeut_grounds,
                      output_path=multi_path_plot_out_path,
                      prefix=name)
        if ajf_plot_out_path is not None:
            plot_ajf_(configfile=configfile,
                      atc=atc,
                      prefiltered_ics=p1.prefiltered_ics,
                      winner=p1.winner,
                      output_plot_path=ajf_plot_out_path)
        if multi_winner_out_path is not None:
            limit_write(p1.winner, multi_winner_out_path)
        if multi_runner_out_path is not None:
            limit_write(p1.runners, multi_runner_out_path)
        if multi_undeut_ground_out_path is not None:
            limit_write([p1.undeut_grounds, p1.undeut_ground_dot_products],
                        multi_undeut_ground_out_path)
        if multi_winner_scores_out_path is not None:
            limit_write(p1.winner_scores, multi_winner_scores_out_path)
        if multi_rtdt_com_cvs_out_path is not None:
            limit_write([p1.rt_com_cv, p1.dt_com_cv], multi_rtdt_com_cvs_out_path)
        if multi_winner_csv_out_path is not None:
            write_baseline_integrated_mz_to_csv(p1.winner, multi_winner_csv_out_path)

if __name__ == "__main__":

    # Checks for Snakemake context and parses arguments.
    if "snakemake" in globals():

        library_info_path = snakemake.input[0]
        configfile = yaml.load(open(snakemake.input[1], "rb").read(), Loader=yaml.Loader)
        all_timepoints_clusters_input_path = snakemake.input[2]

        rt_group_name = snakemake.params.rt_group_name
        old_data_dir = None
        mono_html_plot_out_path = None
        multi_html_plot_out_path = None
        ajf_plot_out_path = None

        if snakemake.params.tmp:
            dictionary_thresholds_path = None
            prefiltered_ics_out_path = None
            mono_path_plot_out_path = None
            mono_winner_out_path = None
            mono_runner_out_path = None
            mono_undeut_ground_out_path = None
            mono_winner_scores_out_path = None
            mono_rtdt_com_cvs_out_path = None
            mono_winner_csv_out_path = None
            multi_winner_out_path = snakemake.output[0]
            multi_path_plot_out_path = None
            multi_runner_out_path = None
            multi_undeut_ground_out_path = None
            multi_winner_scores_out_path = snakemake.output[1]
            multi_rtdt_com_cvs_out_path = None
            multi_winner_csv_out_path = None
        else:
            prefiltered_ics_out_path = snakemake.output[0]
            mono_path_plot_out_path = snakemake.output[1]
            mono_winner_out_path = snakemake.output[2]
            mono_runner_out_path = snakemake.output[3]
            mono_undeut_ground_out_path = snakemake.output[4]
            mono_winner_scores_out_path = snakemake.output[5]
            mono_rtdt_com_cvs_out_path = snakemake.output[6]
            mono_winner_csv_out_path = snakemake.output[7]
            multi_path_plot_out_path = snakemake.output[8]
            multi_winner_out_path = snakemake.output[9]
            multi_runner_out_path = snakemake.output[10]
            multi_undeut_ground_out_path = snakemake.output[11]
            multi_winner_scores_out_path = snakemake.output[12]
            multi_rtdt_com_cvs_out_path = snakemake.output[13]
            multi_winner_csv_out_path = snakemake.output[14]



        main(library_info_path=library_info_path,
             configfile=configfile,
             all_timepoints_clusters_input_path=all_timepoints_clusters_input_path,
             old_data_dir=old_data_dir,
             rt_group_name=rt_group_name,
             prefiltered_ics_out_path=prefiltered_ics_out_path,
             mono_path_plot_out_path=mono_path_plot_out_path,
             mono_html_plot_out_path=mono_html_plot_out_path,
             mono_winner_out_path=mono_winner_out_path,
             mono_runner_out_path=mono_runner_out_path,
             mono_undeut_ground_out_path=mono_undeut_ground_out_path,
             mono_winner_scores_out_path=mono_winner_scores_out_path,
             mono_rtdt_com_cvs_out_path=mono_rtdt_com_cvs_out_path,
             mono_winner_csv_out_path=mono_winner_csv_out_path,
             multi_path_plot_out_path=multi_path_plot_out_path,
             multi_html_plot_out_path=multi_html_plot_out_path,
             multi_winner_out_path=multi_winner_out_path,
             multi_runner_out_path=multi_runner_out_path,
             multi_undeut_ground_out_path=multi_undeut_ground_out_path,
             multi_winner_scores_out_path=multi_winner_scores_out_path,
             multi_rtdt_com_cvs_out_path=multi_rtdt_com_cvs_out_path,
             multi_winner_csv_out_path=multi_winner_csv_out_path,
             ajf_plot_out_path=None,
             # dictionary_thresholds_path=dictionary_thresholds_path,
             )

    else:

        # Configure command line argument parser.
        parser = argparse.ArgumentParser(
            description=
            "Generate a best-estimate HDX-timeseries of IsotopeClusters for a given library protein"
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
            "--all_timepoints_clusters_input_path",
            help=
            "all timepoints clusters cpickle file"
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
        parser.add_argument(
            "-g",
            "--old_data_dir",
            help=
            "directory containing Gabe's pickled output files, using this option prints old data on html plots"
        )
        
        # Starting condition output arguments.
        parser.add_argument("--prefiltered_ics_out_path",
                            help="path/to/file to output ICs passing prefiltering as a nested list")

        # Monobody scoring outputs.
        parser.add_argument("--mono_html_plot_out_path",
                            help="path/to/file for .html plot of results")
        parser.add_argument(
            "--mono_winner_out_path",
            help="path/to/file to save winning IsotopeCluster objects")
        parser.add_argument("--mono_runner_out_path",
                            help="path/to/file to save runner-up IsotopeClusters")
        parser.add_argument(
            "--mono_undeut_ground_out_path",
            help=
            "path/to/file to save selected highest-confidence undeuterated IsotopeClusters"
        )
        parser.add_argument("--mono_winner_scores_out_path",
                            help="path/to/file to save winning path IC scores")
        parser.add_argument("--mono_rtdt_com_cvs_out_path",
                            help="path/to/file to save rt/dt error measurement")
        parser.add_argument("--mono_path_plot_out_path",
                            help="path/to/file to save path plot .pdf")
        parser.add_argument("--mono_winner_csv_out_path",
                            help="path/to/file to save path to .csv file")
        
        # Multibody scoring outputs.
        parser.add_argument("--multi_html_plot_out_path",
                            help="path/to/file for .html plot of results")
        parser.add_argument(
            "--multi_winner_out_path",
            help="path/to/file to save winning IsotopeCluster objects")
        parser.add_argument("--multi_runner_out_path",
                            help="path/to/file to save runner-up IsotopeClusters")
        parser.add_argument(
            "--multi_undeut_ground_out_path",
            help=
            "path/to/file to save selected highest-confidence undeuterated IsotopeClusters"
        )
        parser.add_argument("--multi_winner_scores_out_path",
                            help="path/to/file to save winning path IC scores")
        parser.add_argument("--multi_rtdt_com_cvs_out_path",
                            help="path/to/file to save rt/dt error measurement")
        parser.add_argument("--multi_path_plot_out_path",
                            help="path/to/file to save path plot .pdf")
        parser.add_argument("--multi_winner_csv_out_path",
                            help="path/to/file to save path to .csv file")
        parser.add_argument("--ajf_plot_out_path",
                            default=None,
                            help="path/to/ajf_plot file")
        parser.add_argument("--dictionary_thresholds_path",
                            default=None,
                            help="path/to/dictionary_thresholds_path file")

        args = parser.parse_args()

        # Opens configfile .yaml and generates explicit inputs.
        configfile = yaml.load(open(args.configfile_yaml, "rb").read(), Loader=yaml.Loader)
        if args.all_ic_input_paths is None:
            if args.input_directory_path is not None and args.rt_group_name is not None:
                args.all_ic_input_paths = optimize_paths_inputs(args.library_info_path, args.input_directory_path,
                                                                args.rt_group_name, configfile)
            else:
                parser.print_help()
                sys.exit()


        main(library_info_path=args.library_info_path,
             all_timepoints_clusters_input_path=args.all_timepoints_clusters_input_path,
             configfile=configfile,
             rt_group_name=args.rt_group_name,
             old_data_dir=args.old_data_dir,
             prefiltered_ics_out_path=args.prefiltered_ics_out_path,
             mono_path_plot_out_path=args.mono_path_plot_out_path,
             mono_html_plot_out_path=args.mono_html_plot_out_path,
             mono_winner_out_path=args.mono_winner_out_path,
             mono_runner_out_path=args.mono_runner_out_path,
             mono_undeut_ground_out_path=args.mono_undeut_ground_out_path,
             mono_winner_scores_out_path=args.mono_winner_scores_out_path,
             mono_rtdt_com_cvs_out_path=args.mono_rtdt_com_cvs_out_path,
             mono_winner_csv_out_path=args.multi_winner_cvs_out_path,
             multi_path_plot_out_path=args.multi_path_plot_out_path,
             multi_html_plot_out_path=args.multi_html_plot_out_path,
             multi_winner_out_path=args.multi_winner_out_path,
             multi_runner_out_path=args.multi_runner_out_path,
             multi_undeut_ground_out_path=args.multi_undeut_ground_out_path,
             multi_winner_scores_out_path=args.multi_winner_scores_out_path,
             multi_rtdt_com_cvs_out_path=args.multi_rtdt_com_cvs_out_path,
             multi_winner_csv_out_path=args.multi_winner_cvs_out_path,
             ajf_plot_out_path=args.ajf_plot_out_path,
             dictionary_thresholds_path=args.dictionary_thresholds_path,
             )
