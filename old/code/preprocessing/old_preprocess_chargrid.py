import os
import sys

sys.path.extend(os.environ['PYTHONPATH'])

import argparse
import os
import tqdm
import numpy as np

from old_utils.paths.path_manager import PathManager
from old_utils.preprocessing.class_weighting.calculate_class_weights import calculate_class_weights

from code.model import defaults

N_CLASSES = defaults['dataset']['Chargrid']['n_cl']


def preprocess_chargrid(test=None, copy_to_destination=True, get_class_weights=False):
    # prerequisites:
    # Stefan's extraction script has been run on the original Epam dataset: https://github.wdf.sap.corp/c2c/chargrid/commit/d1fa302381d6fd756713c9e4c9aeb0a18f6c18dd
    # on the DL COE machine the dataset can be found in: /overfit_single_sample/epam_2_image/epam_v2/training_data/Data_LAT_7
    # after running the script, we have npz files in three folders which we copy to $DATAPATH/Chargrid/raw/{train,val,test}

    paths = PathManager(dataset_name='Chargrid').paths

    if copy_to_destination:
        for i in ['train', 'val', 'test']:
            print(f'\nPreprocessing data and labels for fold: {i}')
            # path_in = os.path.join(paths['raw'], i)
            path_in = os.path.join(paths['raw_data'], i)

            for f in tqdm.tqdm(os.listdir(path_in)[:test]):
                file_in = os.path.join(path_in, f)
                XY_in = np.load(file_in)
                file_out_data = os.path.join(paths['data_' + i], f)
                file_out_labels = os.path.join(paths['labels_' + i], f)
                np.savez_compressed(file_out_data, data=XY_in['ocr'])
                np.savez_compressed(file_out_labels, labels=XY_in['labels'])

    # note: script not optimized for efficiency since looping twice through labels if calculating class weights
    if get_class_weights:
        calculate_class_weights(in_path=paths['labels_train'], out_path=paths['proc_data'], dataset='Chargrid')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Chargrid')
    parser.add_argument('--scope', help='specify which parts of the preprocessing you want to be executed.')
    args = parser.parse_args()
    if not args.scope or args.scope == 'all':
        print('Executing the entire preprocessing.')
        preprocess_chargrid(copy_to_destination=True, get_class_weights=True)
    elif args.scope == 'class_weighting':
        print(f'Performing preprocessing step: {args.scope}')
        preprocess_chargrid(copy_to_destination=False, get_class_weights=True)
    else:
        raise NotImplementedError('Invalid argument.')

# other than in preprocss_cityscapes.py, we only loop once through the overfit_single_sample
# i.e. we perform copying files to target folders and calculation of class weights
