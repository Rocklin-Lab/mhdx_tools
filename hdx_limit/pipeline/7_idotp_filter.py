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
import sys
import glob
import argparse
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def main(library_info_path,
         all_idotp_inputs,
         library_info_out_path=None,
         plot_out_path=None,
         return_flag=False,
         idotp_cutoff=0.99):
    """Reads all library_info index idotp_check.csv files and returns or saves a list of indices with idotp >= idotp_cutoff.

    Args:
        library_info_path (str): path/to/library_info.json
        all_idotp_inputs (list of strings): list of all input IsotopeCluster-list filepaths
        indices_out_path (str): path/to/filter_passing_indices.csv
        library_info_out_path (str): path/to/checked_library_info.json
        plot_out_path (str): path/to/file.png for idotp_distribution plot
        return_flag (bool): option to return a dictionary of outputs in python context
        idotp_cutoff (float): inclusive lower-bound on idotp [0,1] to be considered for evaluation, default=0.95

    Returns:
        out_df (Pandas DataFrame): Dataframe with all information from library_info and idotp_check for all idotp filter passing rows.

    """
    library_info = pd.read_json(library_info_path)
    out_df = pd.DataFrame(columns=list(library_info.columns)+['idotp', 'integrated_mz_width', 'mz_centers', 'theor_mz_dist'])
    idotps = []
    # Opens each idotp_check dataframe, if idotp>=cutoff adds computed values to row and appends row to output.
    for idpc in all_idotp_inputs:
        open_idpc = pd.read_json(idpc)
        idotps.append(open_idpc["idotp"].values)
        if open_idpc["idotp"].values>=0.99:
            my_name = idpc.split("/")[-2]
            my_charge = int([item[6:] for item in idpc.split("/")[-1].split("_") if "charge" in item][0])
            my_row = library_info.loc[(library_info["name"]==my_name) & (library_info["charge"]==my_charge)]
            for column in open_idpc.columns:
                my_row[column] = open_idpc[column].values
            out_df = out_df.append(my_row)

    if library_info_out_path is not None:
        out_df.to_json(library_info_out_path)

    if plot_out_path is not None:
        sns.displot(idotps)
        plt.axvline(idotp_cutoff, 0, 1)
        plt.savefig(plot_out_path)

    if return_flag:
        return out_df


if __name__ == "__main__":
    # If the snakemake global object is present, save expected arguments from snakemake to be passed to main().
    if "snakemake" in globals():
        library_info_path = snakemake.input.pop(0)
        all_idotp_inputs = snakemake.input

        library_info_out_path =  snakemake.output[0]
        plot_out_path = snakemake.output[1]


        main(library_info_path=library_info_path,
             all_idotp_inputs=all_idotp_inputs,
             library_info_out_path=library_info_out_path,
             plot_out_path=plot_out_path)
    else:
        # CLI context, set expected arguments with argparse module.
        parser = argparse.ArgumentParser(
            description=
            "Reads all rt-group idotp csvs and returns or saves a list of indices with idotp >= idotp_cutoff."
        )
        parser.add_argument("library_info_path", help="path/to/library_info.json")
        parser.add_argument("-i",
                            "--all_idotp_inputs",
                            help="list of all idotp check .json outputs to be read")
        parser.add_argument("-d",
                            "--input_dir_path",
                            help="path/to/dir/ containing idotp_check.csv files")
        parser.add_argument("-l",
                            "--library_info_out_path",
                            help="path/to/checked_library_info.json")
        parser.add_argument("--p",
                            "--plot_out_path",
                            help="path/to/idotp_distribution.png")
        parser.add_argument(
            "-c",
            "--idotp_cutoff",
            type=float,
            default=0.99,
            help=
            "lower limit on dot-product between theoretical integrated m/z of POI and int. m/z of observed signal in question. Float in range [0,1], default 0.95 "
        )
        args = parser.parse_args()

        if args.all_idotp_inputs is None and args.input_dir_path is None:
            parser.print_help()
            sys.exit()

        if args.all_idotp_inputs is None and args.input_dir_path is not None:
            args.all_idotp_inputs = sorted(
                list(glob.glob(args.input_dir_path + "*idotp_check.csv")))

        all_idotp_inputs = args.all_idotp_inputs.split(' ')

        main(library_info_path=args.library_info_path,
             all_idotp_inputs=args.all_idotp_inputs,
             library_info_out_path = args.library_info_out_path,
             plot_out_path=args.plot_out_path,
             idotp_cutoff=args.idotp_cutoff)
