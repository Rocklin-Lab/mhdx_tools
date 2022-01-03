import glob as glob
import shutil
import yaml
from pathlib import Path


def main(configfile_yaml,
         outputfile=outputfile):
    configfile = yaml.load(open(configfile_yaml, "rb").read(), Loader=yaml.Loader)
    if configfile['keep_mzml_gz']:
        files_to_delete = glob.glob('resources/5_tensors/*') + glob.glob('resources/6_idotp_check/*') + \
                          glob.glob('resources/8_passing_tensors/*') + glob.glob('resources/9_subtensor_ics/*') +\
                          glob.glob('results/plots/factors/*') + glob.glob('results/plots/ics/*')

        for file in files_to_delete:
            shutil.rmtree(file)
    else:
        files_to_delete = glob.glob('resources/5_tensors/*') + glob.glob('resources/6_idotp_check/*') + \
                          glob.glob('resources/8_passing_tensors/*') + glob.glob('resources/9_subtensor_ics/*') \
                          + glob.glob('results/plots/factors/*') + glob.glob('results/plots/ics/*') + \
                          glob.glob('resources/2_mzml_gz/*gz')
        for file in files_to_delete:
            shutil.rmtree(file)

    Path(outputfile).touch()

if __name__ == "__main__":

    if "snakemake" in globals():
        configfile_yaml = snakemake.input[0]
        outputfile = snakemake.output[0]
        main(configfile_yaml=configfile_yaml)
    else:
        parser = argparse.ArgumentParser(
            description=
            "Delete all non essential files"
        )
        parser.add_argument("configfile_yaml",  help="path/to/file.yaml")
        parser.add_argument("outputfile", help="path/to/file.yaml")
        args = parser.parse_args()
        main(configfile_yaml=args.configfile_yaml,
             outputfile=args.outputfile)



