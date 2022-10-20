from hdx_limit.auxiliar.plots import generate_benchmark_stats_plot
import argparse


def main(benchmark_folder,
         output_path=None,
         dpi=300):

    generate_benchmark_stats_plot(benchmark_folder=benchmark_folder,
                                  output_path=output_path,
                                  dpi=dpi)


if __name__ == "__main__":

    if snakemake in globals():

        benchmark_folder = snakemake.input[0]
        output_path = snakemake.output[0]

        main(benchmark_folder=benchmark_folder,
             output_path=output_path)

    else:

        parser = argparse.ArgumentParser(
            description=
            "Generate stats for computational resources used during pipeline"
        )
        parser.add_argument(
            "-i",
            "--benchmark_folder",
            help="path/to/benchmark_folder")
        parser.add_argument(
            "-o",
            "--output_path",
            help="path/to/pdf")

        args = parser.parse_args()

        main(benchmark_folder=args.benchmark_folder,
             output_path=args.output_path)
