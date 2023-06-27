import os
import argparse
from pymzml.utils.utils import index_gzip
from pymzml.run import Reader


def main(mzml_path, delete_source=False, out_path=None):
    """Create an indexed, gzipped mzML.gz file from a .mzML file.
    
    Args:
        mzml_path (string): path/to/file.mzML to be gzipped 
        out_path (string): path/to/file.mzML.gz - the gzipped .mzML output path

    Returns:
        None
        
    """
    with open(mzml_path) as fin:
        fin.seek(0, 2)
        max_offset_len = fin.tell()
        max_spec_no = Reader(mzml_path).get_spectrum_count() + 10

    index_gzip(mzml_path,
               out_path,
               max_idx=max_spec_no,
               idx_len=len(str(max_offset_len)))
    if delete_source is True:
        os.remove(mzml_path)


if __name__ == "__main__":

    # set expected command line arguments
    parser = argparse.ArgumentParser(
        description="Create an indexed gzip mzML file from a plain .mzML")
    parser.add_argument(
        "mzml_path",
        help=
        "path to .mzML input file, resources/mzml/*.mzML in pipeline context")
    parser.add_argument("-d", "--delete_source", action="store_true", help="deletes the source mzML file after writing gzipped file.")
    parser.add_argument("-o", "--out_path", help="path to .mzML.gz output file")
    args = parser.parse_args()

    main(mzml_path=args.mzml_path, delete_source=args.delete_source, out_path=args.out_path)
