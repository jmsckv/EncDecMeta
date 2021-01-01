import os
from shutil import copy2

from old_utils.paths.path_creation import create_core_routine
from old_utils.paths.path_manager import PathManager


# idea: perform preprocessing only once and share computed class weights and normalization constants via commit in git repo
# we then also have to only to synchronize $DATAPATH/meta between workstations, and can skip the preprocessing

# TODO: write function that aggregates results


def share_results_via_git(scope=['Cityscapes', 'Chargrid']):
    # workaround as problems with read-only file permissions when trying to copy via shutil:
    # open yaml as dict from one /raw, write it out here
    if scope:
        l = scope
    else:
        l = os.listdir(os.environ['DATAPATH'])
    for dataset in l:
        pm = PathManager(dataset_name=dataset)
        for calculated in ['class_weights.yaml', 'normalization_statistics.yaml']:
            try:
                src = os.path.join(pm.paths['proc_data'], calculated)
                dst_pth = os.path.join(os.environ['PYTHONPATH'], 'old_utils', 'precomputed_and_shared_results')
                dst = create_core_routine(dst_pth, dataset)  # will overwrite previous results!
                copy2(src, dst)
                print(f'Copied {calculated} for dataset {dataset}.')
            except:
                print(f'Could not copy file {calculated} for dataset {dataset}.')


if __name__ == '__main__':
    share_results_via_git()
