import os
from copy import deepcopy


# NAMING CONVENTIONS
# artifact = {directory,file}
# convenvention (1): if artifact_name already exists, create extension with DEFAULT_EXTENSION+INT
# convenvention (2): if artifact_name+DEFAULT_EXTENSION+INT already exists, create extension with higher INT


def check_for_existing_artifact_name(path, artifact_name, propose_alternative_if_necessary=False, verbose=False):
    DEFAULT_EXTENSION = '.'
    original_artifact_name = deepcopy(artifact_name)
    n_occurences_artifact_name = 0
    n_occurences_artifact_name += sum([f.startswith(str(artifact_name) + DEFAULT_EXTENSION) for f in os.listdir(path)])
    if n_occurences_artifact_name >= 0:  # change to >=1 if you want to start counting from second experiment onwards
        artifact_name = str(artifact_name) + DEFAULT_EXTENSION + str(n_occurences_artifact_name)
        if verbose and propose_alternative_if_necessary: print(
            f'Artifact already existed, modified new artifact name to {artifact_name}')
    if propose_alternative_if_necessary:
        return (path, artifact_name)
    else:
        return (path, original_artifact_name)


def create_file(path=None, file_name=None, return_path=True):
    # default_op None flags assert correct user behaviour
    outpath = os.path.join(path, file_name)
    if os.path.exists(outpath):
        raise FileExistsError
    else:
        open(outpath,
             'a').close()  # os.mknod(outpath) requires root privileges: https://stackoverflow.com/questions/12654772/create-empty-file-using-python
    if return_path:
        return outpath


def create_directory(path=None, directory_name=None, return_path=True, verbose=False):
    # default_op None flags assert correct user behaviour
    outpath = os.path.join(path, directory_name)
    if os.path.exists(outpath):
        if verbose: print(f'Directory {outpath} exists, will not overwrite.')
    else:
        os.makedirs(outpath, exist_ok=False)  # will raise FileExistsError if applicable
    if return_path:
        return outpath


def create_core_routine(path, artifact_name, artifact_type='dir', propose_alternative=False, verbose=False,
                        return_path=True):
    if artifact_type not in ['file', 'dir']:
        raise NotImplementedError
    else:
        (p, a) = check_for_existing_artifact_name(path, artifact_name, \
                                                  propose_alternative_if_necessary=propose_alternative, \
                                                  verbose=verbose)
    if artifact_type == 'dir':
        outpath = create_directory(*(p, a))
    else:
        outpath = create_file(*(p, a))
    if return_path:
        return outpath


"""
# demo
#os.makedirs('./demo')
test = create_core_routine('./demo', 'asd.txt', propose_alternative_if_necessary=False, artifact_type='dir')
print(test)
"""
