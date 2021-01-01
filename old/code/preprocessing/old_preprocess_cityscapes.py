import os
import sys

sys.path.extend(os.environ['PYTHONPATH'])

import argparse
import shutil

import tqdm
import numpy as np
from PIL import Image
from collections import namedtuple

from old_utils.preprocessing.normalization.calculate_normalization_statistics import calculate_normalization_statistics
from old_utils.preprocessing.class_weighting.calculate_class_weights import calculate_class_weights

from old_utils.paths.path_manager import PathManager

# NOTE: currently not optimized for speed since one-time cost; possible speed-ups are:
# (a) multiprocessing
# (b) not looping through images one by one when copying images from raw to meta
# (c) calculating class weights and normalization constant on the go (instead of re-reading them into memory)

################################
# GROUND TRUTH LABELS CITYSCAPES
################################

# (NOTE! this is taken from the official Cityscapes scripts:)
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # old_evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our old_evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type n_cl to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

# (NOTE! this is taken from the official Cityscapes scripts:)
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 19, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 19, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 19, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 19, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 19, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 19, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 19, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 19, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 19, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 19, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, 19, 'vehicle', 7, False, True, (0, 0, 142)),
]

# create a function which maps id to trainId:
id_to_trainId = {label.id: label.trainId for label in labels}
id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]


#######################
# GET PREPROCESS DATA
######################


def preprocesss_data_and_labels_cityscapes():
    paths = PathManager(dataset_name='Cityscapes').paths

    for i in tqdm.tqdm(['train', 'val', 'test']):
        print(f'\nPreprocessing data and labels for fold: {i}')
        img_dir = os.path.join(paths['raw_data'], 'leftImg8bit', i)
        label_dir = os.path.join(paths['raw_data'], 'gtFine', i)

        for city_dir in tqdm.tqdm(eval(i + '_dirs')):
            img_dir_path = os.path.join(img_dir, city_dir)
            label_dir_path = os.path.join(label_dir, city_dir)
            file_names = os.listdir(img_dir_path)

            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                # convert gtFine_img from id to trainId pixel values
                # store in meta/labels_data_preprocessed/{train/val}
                gtFine_img_path = os.path.join(label_dir_path, img_id + "_gtFine_labelIds.png")
                gtFine_img = Image.open(gtFine_img_path)
                label_img = id_to_trainId_map_func(gtFine_img)
                label_img = Image.fromarray(label_img.astype(np.uint8))
                key = 'labels_' + i
                out_path_label = os.path.join(paths[key], img_id + '.png')
                label_img.save(out_path_label)

                # copy 'preprocessed' image to meta/data_preprocessed/{train/val/test}
                in_path_image = os.path.join(img_dir_path, file_name)
                key = 'data_' + i
                out_path_image = os.path.join(paths[key], img_id + '.png')
                shutil.copy(in_path_image, out_path_image)  # no meta-overfit_single_sample copied


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess Cityscapes')
    parser.add_argument('--scope', help='specify which parts of the preprocessing you want to be executed.')
    args = parser.parse_args()

    paths = PathManager(dataset_name='Cityscapes').paths

    if not args.scope or args.scope == 'all':
        print('Executing the entire preprocessing.')
        preprocesss_data_and_labels_cityscapes()
        calculate_normalization_statistics(paths['data_train'], out_path=paths['proc_data'])
        calculate_class_weights(paths['labels_train'], out_path=paths['proc_data'], dataset='Cityscapes')

    elif args.scope in ['normalization', 'class_weighting']:
        print(f'Performing preprocessing step: {args.scope}')
        if args.scope == 'normalization':
            assert os.path.isdir(paths[
                                     'data_train']), f'Make sure that $DATAPATH/meta/data/train is being populated; maybe run preprocessing on raw data first?'
            calculate_normalization_statistics(paths['data_train'], out_path=paths['proc_data'])
        else:
            assert os.path.isdir(paths[
                                     'labels_train']), f'Make sure that $DATAPATH/meta/labels/train is being populated; maybe run preprocessing on raw data first?'
            calculate_class_weights(paths['labels_train'], out_path=paths['proc_data'], dataset='Cityscapes')
    else:
        raise NotImplementedError('Invalid argument.')
