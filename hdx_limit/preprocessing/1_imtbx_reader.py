from hdx_limit.core.datatypes import Preprocessing

import argparse
import yaml

def main(configfile_path,
         isotopes_path,
         names_and_seqs_path,
         lockmass_calibration_dict=None,
         output_path=None,
         protein_polyfit_output=None,
         kde_output_path=None):

    imtbx = Preprocessing.IMTBX()

    imtbx.load_imtbx(configfile_path=configfile_path,
                     isotopes_path=isotopes_path,
                     names_and_seqs_path=names_and_seqs_path,
                     output_path=output_path,
                     protein_polyfit_output=protein_polyfit_output,
                     lockmass_calibration_dict=lockmass_calibration_dict)

    if kde_output_path is not None:
        imtbx.plot_kde(output_path=kde_output_path)


if __name__ == '__main__':

    if "snakemake" in globals():

        configfile_path = snakemake.input[0]
        isotopes_path = snakemake.input[1]
        names_and_seqs_path = snakemake.input[2]
        output_path = snakemake.output[0]
        kde_output_path = snakemake.output[1]

        configfile = yaml.load(open(configfile_path, "rb").read(), Loader=yaml.Loader)

        undeut_fn = isotopes_path.split("/")[-1].strip(".peaks.isotopes")
        if configfile["lockmass"]:
            lockmass_calibration_dict = f"resources/0_calibration/{undeut_fn}_mz_calib_dict.pk"
        else:
            lockmass_calibration_dict = None

        if configfile["protein_polyfit"]:
            protein_polyfit_outpath = f"resources/1_imtbx/{undeut_fn}_mz_calib_dict.pk"
        else:
            protein_polyfit_outpath = None

        # lockmass_calibration_dict = snakemake.input[3] if any('mz_calib_dict.pk' in i for i in snakemake.input) else None
        #
        # protein_polyfit_outpath = snakemake.output[2] if any('mz_calib_dict.pk' in i for i in snakemake.output) else None

        main(configfile_path=configfile_path,
             isotopes_path=isotopes_path,
             names_and_seqs_path=names_and_seqs_path,
             output_path=output_path,
             kde_output_path=kde_output_path,
             protein_polyfit_output=protein_polyfit_outpath,
             lockmass_calibration_dict=lockmass_calibration_dict)

    else:
        # Sets expected command line arguments.
        parser = argparse.ArgumentParser(
            description=
            "Reads an imtbx .peaks.isotopes file and creates an intermediate list of identified charged species to be used by 4_make_library_master_list.py"
        )
        parser.add_argument(
            "-c",
            "--configfile_path",
            help="/path/to/configfile"
        )
        parser.add_argument(
            "-i",
            "--isotopes_path",
            help="path/to/.peaks.isotopes file from undeuterated mzml"
        )
        parser.add_argument(
            "-s",
            "--names_and_seqs_path",
            help="path/to/.csv with names and sequences of library proteins"
        )
        parser.add_argument(
            "-l",
            "--lockmass",
            default=None,
            help="path/to/.pk with lockmass calibration"
        )
        parser.add_argument(
            "-p",
            "--protein_polyfit_output",
            default=None,
            help="path/to/.pk for protein polyfit calibration"
        )
        parser.add_argument("-o",
                            "--output_path",
                            default=None,
                            help="path/to/_intermediate.csv main output file"
                            )
        parser.add_argument("-k",
                            "--kde_output_path",
                            default=None,
                            help="path/to/*kde.pdf plot"
                            )

        args = parser.parse_args()

        main(configfile_path=args.configfile_path,
             isotopes_path=args.isotopes_path,
             names_and_seqs_path=args.names_and_seqs_path,
             lockmass_calibration_dict=args.lockmass,
             output_path=args.output_path,
             kde_output_path=args.kde_output_path,
             protein_polyfit_output=args.protein_polyfit_output,
             )



