import os
import shutil
import argparse
import yaml
import pandas as pd


def main(input_paths, output_paths):
    """Moves idotp_check passing tensors to a new directory to simplify downstream input calling.

    Args:
        input_paths (list of strings): Sorted list of paths/to/inputs in the same order as the output list.
        output_paths (list of strings): Sorted list of paths/to/outputs in the same order as the input list.

    Returns:
        None

    """
    for fin, fout in zip(input_paths, output_paths):
        os.makedirs(os.path.dirname(fout), exist_ok=True)
        shutil.copy(fin, fout)


if __name__ == "__main__":

    if "snakemake" in globals():
        config = yaml.load(open(snakemake.input[0], "rb").read(), Loader=yaml.Loader)
        library_info = pd.read_json(snakemake.input[1])

        # Makes two zippable lists that are used for extract_tensors: repeated rt_group_names and their corresponding charges in order.
        zippable_names = list(library_info["name"].values)
        zippable_charges = list(library_info["charge"].values)

        # Creates three zippable lists that are used for mv_passing_tensors: rt-group names, charges, and undeut_mzmls.
        mv_passing_tensors_zippable_names = []
        mv_passing_tensors_zippable_charges = []
        mv_passing_tensors_zippable_undeut_mzmls = []
        for name, charge in zip(zippable_names, zippable_charges):
            for undeut_mzml in config[0]:
                mv_passing_tensors_zippable_names.append(name)
                mv_passing_tensors_zippable_charges.append(charge)
                mv_passing_tensors_zippable_undeut_mzmls.append(undeut_mzml)

        input_paths = [
            "resources/5_tensors/" + name + "/" + name + "_charge" + str(charge) + "_" + mzml + ".gz.cpickle.zlib"
            for (name, charge, mzml) in zip(mv_passing_tensors_zippable_names,
                                            mv_passing_tensors_zippable_charges,
                                            mv_passing_tensors_zippable_undeut_mzmls)
            ]

        output_paths = ["resources/8_passing_tensors/" + name + "/" + name + "_charge" + str(
            charge) + "_" + mzml + ".gz.cpickle.zlib"
                        for (name, charge, mzml) in zip(mv_passing_tensors_zippable_names,
                                                        mv_passing_tensors_zippable_charges,
                                                        mv_passing_tensors_zippable_undeut_mzmls)
                        ]
        main(input_paths, output_paths)

    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("configfile", help="path/to/config.yaml")
        parser.add_argument("library_info_path", help="path/to/checked_library_info.json")
        # parser.add_argument("input_dir_path",
        #                     help="path/to/input_dir, inputs are globbed from this path, I don't work right now",
        #                     default="resources/")
        # parser.add_argument("output_dir_path",
        #                     help="path/to/output_dir, outputs are globbed from this path, I don't work right now")
        args = parser.parse_args()

        config = yaml.load(open(args.configfile, "rb").read(), Loader=yaml.Loader)
        library_info = pd.read_json(args.library_info_path)

        # Makes two zippable lists that are used for extract_tensors: repeated rt_group_names and their corresponding charges in order.
        zippable_names = list(library_info["name"].values)
        zippable_charges = list(library_info["charge"].values)

        # Creates three zippable lists that are used for mv_passing_tensors: rt-group names, charges, and undeut_mzmls.
        mv_passing_tensors_zippable_names = []
        mv_passing_tensors_zippable_charges = []
        mv_passing_tensors_zippable_undeut_mzmls = []
        for name, charge in zip(zippable_names, zippable_charges):
            for undeut_mzml in config[0]:
                mv_passing_tensors_zippable_names.append(name)
                mv_passing_tensors_zippable_charges.append(charge)
                mv_passing_tensors_zippable_undeut_mzmls.append(undeut_mzml)

        input_paths = ["resources/5_tensors/" + name + "/" + name + "_charge" + str(charge) + "_" + mzml + ".gz.cpickle.zlib"
                       for (name, charge, mzml) in zip(mv_passing_tensors_zippable_names,
                                                       mv_passing_tensors_zippable_charges,
                                                       mv_passing_tensors_zippable_undeut_mzmls)
                       ]

        output_paths = ["resources/8_passing_tensors/" + name + "/" + name + "_charge" + str(charge) + "_" + mzml + ".gz.cpickle.zlib"
                       for (name, charge, mzml) in zip(mv_passing_tensors_zippable_names,
                                                       mv_passing_tensors_zippable_charges,
                                                       mv_passing_tensors_zippable_undeut_mzmls)
                       ]
        main(input_paths, output_paths)