import zlib
import _pickle as cpickle
import os
import pandas as pd

def limit_write(obj, out_path):
    """ Writes a Python object as a zlib-compressed pickle.

	Args:
		obj (any Python Object):
		out_path (string): path/to/file.cpickle.zlib to write out

	Returns:
		None

	"""
    with open(out_path, "wb+") as file:
        file.write(zlib.compress(cpickle.dumps(obj)))


def limit_read(path):
    """ Reads a Python object from a zlib-compressed pickle.

	Args:
		path (string): path/to/file.cpickle.zlib to read

	Returns:
		(obj): some python object

	"""
    return cpickle.loads(zlib.decompress(open(path, "rb").read()))


def optimize_paths_inputs(library_info_path, input_directory_path,
                          name, timepoints):
    """Generate explicit PathOptimizer input paths for one rt_group.

    Args:
        library_info_path (str): path/to/checked_library_info.json
        input_directory_path (str): /path/to/dir/ to prepend to each outpath
        name (str): value from "name" column of library_info
        timepoints (dict): dictionary containing list of hdx timepoints in seconds, where each timepoint is also an integer key corresponding to that timepoint"s .mzML filenames

    Returns:
        name_inputs (list of strings): flat list of all IsotopeCluster inputs to PathOptimizer

    """
    name_inputs = []
    library_info = pd.read_json(library_info_path)
    charges = library_info.loc[library_info["name"]==name]["charge"].values
    for key in timepoints["timepoints"]:
        if len(timepoints[key]) > 1:
            for file in timepoints[key]:
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
                    )  # TODO: This won"t work if we go to .mzML/.RAW interoperability.
        else:
            file = timepoints[key][0]
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


def idotp_check_inputs(config, rt_group_name, charge):
    """Wildcard-based input-file-path generator for idotp_check rule, writes based on rt_group and charge.

    Args:
        config (dict): Dictionary containing timepoint-wise filepaths for .mzML replicates.
        rt_group_name (str): Value in "name" field of library_info.json, shared by groups of mass-agreeing signals close in rt.
        charge (int): Net charge of signal, unique for each member of an rt-group.  

    Returns:
        inputs (list of strs): List of paths to extracted tensors from undeuterated timepoints for a given rt-group charge state.

    """
    inputs = []
    if len(config[0]) > 1:
        for undeut_fn in config[0]:
            inputs.append(
                "resources/5_tensors/" + rt_group_name + "/" + rt_group_name + "_" + "charge" + str(charge) + "_" + undeut_fn + ".gz.cpickle.zlib"
            )
    else: # Handle 0th timepoint with no replicates.
        undeut_fn = config[0][0]
        inputs.append(
            "resources/5_tensors/" + rt_group_name + "/" + rt_group_name + "_" + "charge" + str(charge) + "_" + undeut_fn + ".gz.cpickle.zlib"
        )
    return inputs


def check_for_create_dirs(path_args):
    """Check a list of output paths to see if their required directories exist, creates them if they don"t.

    Args:
        path_args (list of strings): List of output paths, can be string or None.

    Returns:
        None

    """
    directories = ["/".join(arg.split("/")[:-1]) for arg in path_args if arg is not None]
    for dir_path in directories:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


