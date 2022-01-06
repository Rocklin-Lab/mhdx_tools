import glob as glob
import shutil
import os
import argparse
import time

def main(delete_gz_mzml=False):

    start = time.time()

    files_to_delete = glob.glob('resources/5_tensors/*') + glob.glob('resources/6_idotp_check/*') + \
                      glob.glob('resources/8_passing_tensors/*') + glob.glob('resources/9_subtensor_ics/*') +\
                      glob.glob('*out')
    for file in files_to_delete:
        try:
            shutil.rmtree(file)
        except:
            print('%s not deleted'%file)
    for file in glob.glob('resources/10_ic_time_series/*/*/*'):
        if os.path.getsize(file) == 0:
            os.remove(file)
    if delete_gz_mzml:
        for file in glob.glob('resources/2_mzml_gz/*gz'):
            shutil.rmtree(file)

    print('Finished in: %0.2f s'%(time.time()-start))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Delete all non essential files"
    )
    parser.add_argument("--delete_gz_mzml", action='store_true', help="delete gz mzml")
    args = parser.parse_args()
    if args.delete_gz_mzml:
        main(delete_gz_mzml=args.delete_gz_mzml)
    else:
        main()



