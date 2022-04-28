import yaml
import os
from hdx_limit.core.ajf_plot import plot_ajf_
from hdx_limit.core.io import limit_read
import argparse

if __name__ == '__main__':

    if "snakemake" in globals():
        configfile = yaml.load(open(snakemake.input[0], "rb").read(), Loader=yaml.Loader)
        atc_path = snakemake.input[1]
        prefiltered_ics_path = snakemake.input[2]
        winner_multi_path = snakemake.input[3]
        winner_mono_path = snakemake.input[4]
        output_multi = snakemake.output[0]
        output_mono = snakemake.output[1]

        df_output_multi = None
        df_output_mono = None

        if not os.stat(atc_path).st_size == 0:
            atc = limit_read(atc_path)
        else:
            atc = None
            print('ATC is NONE')
        if not os.stat(prefiltered_ics_path).st_size == 0:
            prefiltered_ics = limit_read(prefiltered_ics_path)
        else:
            prefiltered_ics = None
            print('PREFILTERED IS NONE')
        if not os.stat(winner_multi_path).st_size == 0:
            winner_multi = limit_read(winner_multi_path)
            df_output_multi = winner_multi_path.replace('.cpickle.zlib', '.df.pkl')
        else:
            winner_multi = None
            print('WINNER MULTI IS NONE')
        if not os.stat(winner_mono_path).st_size == 0:
            winner_mono = limit_read(winner_mono_path)
            df_output_mono = winner_mono_path.replace('.cpickle.zlib', '.df.pkl')
        else:
            winner_mono = None
            print('WINNER MONO IS NONE')

        plot_ajf_(configfile=configfile,
                  atc=atc,
                  prefiltered_ics=prefiltered_ics,
                  winner=winner_multi,
                  output_path=output_multi,
                  df_output_path=df_output_multi)

        plot_ajf_(configfile=configfile,
                  atc=atc,
                  prefiltered_ics=prefiltered_ics,
                  winner=winner_mono,
                  output_path=output_mono,
                  df_output_path=df_output_mono)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c",
            "--configfile",
            help=
            "Configfile path. /config/config.yaml"
        )
        parser.add_argument(
            "-a",
            "--atc",
            help=
            "All timepoint clusters (or all prefiltered ics)"
        )
        parser.add_argument(
            "-f",
            "--prefiltered_ics",
            help=
            "Prefiltered ics"
        )
        parser.add_argument(
            "-w_mono",
            "--winner_mono",
            help=
            "Winner path monobody"
        )
        parser.add_argument(
            "-w_multi",
            "--winner_multi",
            help=
            "Winner path multibody"
        )
        parser.add_argument(
            "-o_mono",
            "--output_mono",
            help=
            "Output path monobody"
        )
        parser.add_argument(
            "-o_multi",
            "--output_multi",
            help=
            "Output path multibody"
        )

        args = parser.parse_args()

        configfile = yaml.load(open(args.configfile, "rb").read(), Loader=yaml.Loader)

        df_output_mono = None
        df_output_multi = None

        if not os.stat(args.atc).st_size == 0:
            atc = limit_read(args.atc)
        else:
            atc = None
            print('ATC is NONE')
        if not os.stat(args.prefiltered_ics).st_size == 0:
            prefiltered_ics = limit_read(args.prefiltered_ics)
        else:
            prefiltered_ics = None
            print('PREFILTERED IS NONE')
        if not os.stat(args.winner_mono).st_size == 0:
            winner_mono = limit_read(args.winner_mono)
            df_output_mono = args.winner_mono.replace('.cpickle.zlib', '.df.pkl')
        else:
            winner_mono = None
            print('WINNER MONO IS NONE')
        if not os.stat(args.winner_multi).st_size == 0:
            winner_multi = limit_read(args.winner_multi)
            df_output_multi = args.winner_multi.replace('.cpickle.zlib', '.df.pkl')
        else:
            winner_multi = None
            print('WINNER MULTI IS NONE')

        plot_ajf_(configfile=configfile,
                  atc=atc,
                  prefiltered_ics=prefiltered_ics,
                  winner=winner_mono,
                  output_path=args.output_mono,
                  df_output_path=df_output_mono)

        plot_ajf_(configfile=configfile,
                  atc=atc,
                  prefiltered_ics=prefiltered_ics,
                  winner=winner_multi,
                  output_path=args.output_multi,
                  df_output_path=df_output_multi)
